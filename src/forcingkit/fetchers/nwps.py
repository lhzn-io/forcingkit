"""NOAA's Nearshore Wave Prediction System (NWPS) as surface waves (schema `nwps-waves-v1`).

Each Weather Forecast Office (WFO) runs SWAN on its own computational grids (CG1 is the office's
domain, about 1.8 km for OKX) and publishes one GRIB2 file per grid and cycle: the analysis and 144
hourly forecast steps on a regular longitude-latitude grid, with land as missing. Records here stay
on that grid, cropped to the request's bbox; nothing is regridded.

Each hour t is taken from the newest 00 or 12 UTC cycle at or before t whose file holds t, so a
window in the past is made of leads 0 to 11 from successive cycles, and hours past the newest
cycle come from its longer leads. A cycle without an analysis falls back to the previous cycle's
lead 12 for that hour. A file counts only when it decodes with all 144 forecast steps: NOMADS
answers 200 while a file is still being written, and a truncated download decodes short too.

Cycles are read from an archive first (FORCINGKIT_NWPS_ARCHIVE: a directory, `file://` or
`gs://` prefix, read only), then from NOMADS, which keeps about five days. An hour that neither
holds is an error, never a shorter record.
"""

import logging
import os
import re
import tempfile
import warnings
from dataclasses import dataclass
from typing import Callable, Iterator, Optional

import numpy as np
import pandas as pd
import requests
import xarray as xr

from forcingkit import settings

logger = logging.getLogger(__name__)

NWPS_SCHEMA = "nwps-waves-v1"

NOMADS_BASE = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/nwps/prod"
USER_AGENT = "forcingkit (+https://github.com/lhzn-io/forcingkit)"

CYCLE_HOURS = 12  # cycles at 00 and 12 UTC
FORECAST_STEPS = 144  # hourly leads 1..144 after the analysis
# Hours after its cycle time by which a file is normally complete on NOMADS. A cycle younger than
# this is not asked for, so a request repeated before then keys to the previous cycle.
POSTING_LAG_HOURS = 7
NOMADS_REACH_DAYS = 5

# NWS region of each WFO that runs NWPS, which is the first path element on NOMADS.
REGIONS = {
    **{
        w: "er"
        for w in ("car", "gyx", "box", "okx", "phi", "lwx", "akq", "mhx", "ilm", "chs")
    },
    **{
        w: "sr"
        for w in (
            "jax",
            "mlb",
            "mfl",
            "key",
            "tbw",
            "tae",
            "mob",
            "lix",
            "lch",
            "hgx",
            "crp",
            "bro",
            "sju",
        )
    },
    **{w: "wr" for w in ("sew", "pqr", "mfr", "eka", "mtr", "lox", "sgx")},
    **{w: "ar" for w in ("ajk", "aer", "alu", "afg")},
    **{w: "pr" for w in ("hfo", "gum")},
}

GRID_NAME = re.compile(r"^CG[1-9]$")

# Store name -> (GRIB shortName, units, standard_name, long_name).
FIELDS = {
    "hs": (
        "swh",
        "m",
        "sea_surface_wave_significant_height",
        "Significant height of combined wind waves and swell",
    ),
    "tp": (
        "perpw",
        "s",
        "sea_surface_wave_period_at_variance_spectral_density_maximum",
        "Peak wave period (NWPS primary wave period, PERPW)",
    ),
    "dir": (
        "dirpw",
        "degree",
        "sea_surface_wave_to_direction",
        "Primary wave direction of travel, clockwise from true north",
    ),
    "shts": (
        "shts",
        "m",
        "sea_surface_swell_wave_significant_height",
        "Significant height of total swell",
    ),
    "ws": ("ws", "m s-1", "wind_speed", "NWPS forcing wind speed at 10 m"),
    "wdir": (
        "wdir",
        "degree",
        "wind_from_direction",
        "NWPS forcing wind direction (from), clockwise from true north",
    ),
}

RECORD_DIMS: dict[str, tuple[str, ...]] = {
    **{name: ("lat", "lon") for name in FIELDS},
    "forecast_hour": (),
}

RECORD_ATTRS = {
    **{
        name: {"units": u, "standard_name": sn, "long_name": ln}
        for name, (_, u, sn, ln) in FIELDS.items()
    },
    "forecast_hour": {
        "units": "hours",
        "standard_name": "forecast_period",
        "long_name": "Lead of the record from its NWPS cycle (cycle = time - forecast_hour)",
    },
}
RECORD_ATTRS["dir"]["direction_convention"] = "to"
RECORD_ATTRS["wdir"]["direction_convention"] = "from"


def check_wfo_grid(wfo: str, grid: str) -> tuple[str, str]:
    """(wfo, grid) normalised (lower-case office, upper-case grid); ValueError if unknown."""
    w, g = wfo.lower(), grid.upper()
    if w not in REGIONS:
        raise ValueError(
            f"unknown NWPS office {wfo!r}; known: {', '.join(sorted(REGIONS))}"
        )
    if not GRID_NAME.match(g):
        raise ValueError(f"NWPS grid must be CG1 to CG9, not {grid!r}")
    return w, g


def file_name(wfo: str, grid: str, cycle: pd.Timestamp) -> str:
    return f"{wfo}_nwps_{grid}_{cycle:%Y%m%d}_{cycle:%H}00.grib2"


def nomads_url(wfo: str, grid: str, cycle: pd.Timestamp) -> str:
    return (
        f"{NOMADS_BASE}/{REGIONS[wfo]}.{cycle:%Y%m%d}/{wfo}/{cycle:%H}/{grid}/"
        f"{file_name(wfo, grid, cycle)}"
    )


def archive_paths(root: str, wfo: str, grid: str, cycle: pd.Timestamp) -> list[str]:
    """Where an archive may hold a cycle's file, in the order tried: `YYYY/MM/<name>`,
    `YYYYMMDD/HH/<name>`, then `<name>` directly under the root."""
    root = root.rstrip("/")
    name = file_name(wfo, grid, cycle)
    return [
        f"{root}/{cycle:%Y}/{cycle:%m}/{name}",
        f"{root}/{cycle:%Y%m%d}/{cycle:%H}/{name}",
        f"{root}/{name}",
    ]


def floor_cycle(t: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(t).floor(f"{CYCLE_HOURS}h")


def newest_cycle(
    end_valid: pd.Timestamp, now: Optional[pd.Timestamp] = None
) -> pd.Timestamp:
    """The newest cycle a window ending at `end_valid` may use: at or before the window's end and
    at least POSTING_LAG_HOURS old. It is part of the store's cache key, so a forecast window asked
    for again after a newer cycle has posted becomes a new store instead of a stale hit."""
    now = (
        pd.Timestamp.now(tz="UTC").tz_localize(None)
        if now is None
        else pd.Timestamp(now)
    )
    return floor_cycle(
        min(pd.Timestamp(end_valid), now - pd.Timedelta(hours=POSTING_LAG_HOURS))
    )


def naive_utc(t) -> pd.Timestamp:
    t = pd.Timestamp(t)
    if t.tzinfo is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    return t


def record_times(start_time, hours: int) -> pd.DatetimeIndex:
    """Hourly from start - 1 h to start + hours + 1 h, bracketing the run as the atmosphere does."""
    t0 = naive_utc(start_time)
    if t0 != t0.floor("h"):
        raise ValueError(f"start_time must be on the hour, not {t0}")
    return pd.date_range(
        t0 - pd.Timedelta(hours=1), t0 + pd.Timedelta(hours=hours + 1), freq="1h"
    )


@dataclass
class CycleFields:
    """One cycle's fields on the cropped grid, by lead hour (0 is the analysis, when present)."""

    cycle: pd.Timestamp
    origin: str  # where the file came from (archive path or NOMADS URL)
    lat: np.ndarray
    lon: np.ndarray
    fields: dict[int, dict[str, np.ndarray]]


@dataclass
class Crop:
    lat: slice
    lon: slice
    lat_axis: np.ndarray
    lon_axis: np.ndarray


def crop_for(lat: np.ndarray, lon: np.ndarray, bbox) -> Crop:
    """Index ranges of a regular grid covering `bbox` (min_lon, min_lat, max_lon, max_lat) with
    one cell to spare on every side. `lon` must already be in -180..180."""
    lat, lon = np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)
    if np.any(np.diff(lon) <= 0) or np.any(np.diff(lat) <= 0):
        raise ValueError("NWPS grid axes are not ascending in -180..180")
    dx, dy = float(lon[1] - lon[0]), float(lat[1] - lat[0])
    ix = np.flatnonzero((lon >= bbox[0] - dx) & (lon <= bbox[2] + dx))
    iy = np.flatnonzero((lat >= bbox[1] - dy) & (lat <= bbox[3] + dy))
    if ix.size == 0 or iy.size == 0:
        raise ValueError(
            f"bbox {list(bbox)} lies outside the NWPS grid "
            f"({lon[0]:.3f}..{lon[-1]:.3f} E, {lat[0]:.3f}..{lat[-1]:.3f} N)"
        )
    xs, ys = slice(int(ix[0]), int(ix[-1]) + 1), slice(int(iy[0]), int(iy[-1]) + 1)
    return Crop(ys, xs, lat[ys], lon[xs])


def decode_cycle(path: str, cycle: pd.Timestamp, bbox) -> Optional[CycleFields]:
    """A cycle file's fields cropped to `bbox`, or None when it does not decode with every field
    at all FORECAST_STEPS forecast steps (still being written, or truncated). Raises if the file
    is for another cycle."""
    import cfgrib

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            datasets = cfgrib.open_datasets(path, backend_kwargs={"indexpath": ""})
        except Exception as e:  # a truncated file can fail anywhere in the decoder
            logger.warning(f"NWPS {path} does not decode: {e}")
            return None
    # NWPS files decode as the analysis (no step dimension, and it may be absent) beside the
    # forecast steps; leads are collected from whichever datasets hold the wave height.
    datasets = [d for d in datasets if "swh" in d]
    if not datasets:
        logger.info(f"NWPS {path} holds no wave height; not complete")
        return None
    ref = datasets[0]
    for d in datasets:
        missing = [g for g, *_ in FIELDS.values() if g not in d]
        if missing:  # a truncated file ends part way through a step's messages
            logger.warning(f"NWPS {path} lacks {missing} in some steps; not complete")
            return None
        if naive_utc(d["time"].values) != cycle:
            raise RuntimeError(f"NWPS {path} is cycle {d['time'].values}, not {cycle}")

    lon = ((ref["longitude"].values + 180.0) % 360.0) - 180.0
    crop = crop_for(ref["latitude"].values, lon, bbox)

    def cut(da: xr.DataArray) -> np.ndarray:
        return np.asarray(da.values[..., crop.lat, crop.lon], dtype=np.float32)

    fields: dict[int, dict[str, np.ndarray]] = {}
    for d in datasets:
        steps = np.atleast_1d(d["step"].values)
        leads = (steps / np.timedelta64(1, "h")).astype(int)
        stacked = {name: cut(d[g]) for name, (g, *_) in FIELDS.items()}
        if "step" not in d.dims:
            stacked = {name: a[None] for name, a in stacked.items()}
        for i, lead in enumerate(leads):
            fields[int(lead)] = {name: a[i] for name, a in stacked.items()}
    forecast = sum(1 for lead in fields if 1 <= lead <= FORECAST_STEPS)
    if forecast != FORECAST_STEPS:
        logger.info(f"NWPS {path}: {forecast} of {FORECAST_STEPS} steps; not complete")
        return None
    for f in fields.values():
        # NWPS publishes the direction waves come from; the store holds where they travel to.
        f["dir"] = np.where(
            np.isnan(f["dir"]), np.nan, (f["dir"] + 180.0) % 360.0
        ).astype(np.float32)
    return CycleFields(cycle, path, crop.lat_axis, crop.lon_axis, fields)


def _from_archive(root: str, wfo: str, grid: str, cycle: pd.Timestamp, workdir: str):
    """(local path, archive path) of a cycle's file in the archive, or None. Read only: a `gs://`
    object is copied into `workdir`, a local file is read where it is."""
    for candidate in archive_paths(root, wfo, grid, cycle):
        if candidate.startswith("gs://"):
            try:
                import gcsfs  # type: ignore[import-not-found]
            except ImportError as e:
                raise RuntimeError(
                    "a gs:// NWPS archive needs gcsfs (pip install 'forcingkit[gcs]')"
                ) from e
            fs = gcsfs.GCSFileSystem()
            if fs.exists(candidate):
                local = os.path.join(workdir, os.path.basename(candidate))
                fs.get(candidate, local)
                return local, candidate
        else:
            path = candidate.removeprefix("file://")
            if os.path.isfile(path):
                return path, candidate
    return None


def _from_nomads(wfo: str, grid: str, cycle: pd.Timestamp, raw_dir: str):
    """(local path, URL) of a cycle's file from NOMADS, kept in `raw_dir` once it has decoded
    complete, or None on a 404 (not yet posted, or past NOMADS's reach)."""
    url = nomads_url(wfo, grid, cycle)
    path = os.path.join(raw_dir, file_name(wfo, grid, cycle))
    if os.path.isfile(path):
        return path, url
    os.makedirs(raw_dir, exist_ok=True)
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=180)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    partial = path + ".part"
    with open(partial, "wb") as f:
        f.write(resp.content)
    return partial, url


def load_cycle(
    wfo: str, grid: str, cycle: pd.Timestamp, bbox, workdir: str
) -> Optional[CycleFields]:
    """A complete cycle from the archive, else NOMADS; None if neither has it complete."""
    root = settings.nwps_archive()
    if root:
        hit = _from_archive(root, wfo, grid, cycle, workdir)
        if hit is not None:
            fields = decode_cycle(hit[0], cycle, bbox)
            if fields is not None:
                fields.origin = hit[1]
                logger.info(f"NWPS {cycle:%Y-%m-%dT%HZ} from the archive")
                return fields
            logger.warning(f"NWPS archive copy {hit[1]} is incomplete; trying NOMADS")
    hit = _from_nomads(wfo, grid, cycle, settings.cache_dir("nwps", wfo, grid))
    if hit is None:
        return None
    path, url = hit
    fields = decode_cycle(path, cycle, bbox)
    if path.endswith(".part"):
        if fields is None:
            os.remove(path)
        else:
            os.rename(path, path.removesuffix(".part"))
    if fields is not None:
        fields.origin = url
        logger.info(f"NWPS {cycle:%Y-%m-%dT%HZ} from NOMADS")
    return fields


def plan_records(
    times: pd.DatetimeIndex,
    latest: pd.Timestamp,
    load: Callable[[pd.Timestamp], Optional[CycleFields]],
    reach_message: str = "",
) -> list[tuple[pd.Timestamp, CycleFields, int]]:
    """(time, cycle, lead) for every record: the newest cycle no newer than `latest` whose file
    holds that hour. `load` returns a cycle's fields or None, and is called once per cycle."""
    loaded: dict[pd.Timestamp, Optional[CycleFields]] = {}

    def get(c: pd.Timestamp) -> Optional[CycleFields]:
        if c not in loaded:
            loaded[c] = load(c)
        return loaded[c]

    plan = []
    for t in times:
        c = min(floor_cycle(t), latest)
        chosen = None
        while (t - c) <= pd.Timedelta(hours=FORECAST_STEPS):
            lead = int((t - c) / pd.Timedelta(hours=1))
            cf = get(c)
            if cf is not None and lead in cf.fields:
                chosen = (t, cf, lead)
                break
            c -= pd.Timedelta(hours=CYCLE_HOURS)
        if chosen is None:
            raise LookupError(
                f"no NWPS cycle holds {t:%Y-%m-%dT%H:%MZ} (tried cycles back to "
                f"{c + pd.Timedelta(hours=CYCLE_HOURS):%Y-%m-%dT%HZ}). {reach_message}"
            )
        plan.append(chosen)
    return plan


def iter_waves(
    start_time: str,
    hours: int,
    bbox: list[float],
    wfo: str = "okx",
    grid: str = "CG1",
    latest_cycle: Optional[pd.Timestamp] = None,
    loader: Optional[Callable[[pd.Timestamp], Optional[CycleFields]]] = None,
) -> Iterator[tuple]:
    """Yield `("static", Dataset)`, then `("record", time, fields)` for every hour from
    start - 1 h to start + hours + 1 h. Every cycle is read before the first yield, so a window
    out of reach fails before a store is begun. `latest_cycle` caps the cycles used (the cache key
    holds it); `loader` replaces the archive and NOMADS reads, for tests."""
    wfo, grid = check_wfo_grid(wfo, grid)
    times = record_times(start_time, hours)
    latest = (
        newest_cycle(times[-1]) if latest_cycle is None else pd.Timestamp(latest_cycle)
    )
    root = settings.nwps_archive()
    reach = (
        f"NOMADS keeps about {NOMADS_REACH_DAYS} days of NWPS; older windows need an archive "
        + (
            f"(FORCINGKIT_NWPS_ARCHIVE={root} does not hold it)."
            if root
            else "(FORCINGKIT_NWPS_ARCHIVE is not set)."
        )
    )
    with tempfile.TemporaryDirectory() as workdir:
        load = loader or (lambda c: load_cycle(wfo, grid, c, bbox, workdir))
        plan = plan_records(times, latest, load, reach)

    cycles = {id(cf): cf for _, cf, _ in plan}.values()
    first = plan[0][1]
    for cf in cycles:
        if cf.lat.shape != first.lat.shape or cf.lon.shape != first.lon.shape:
            raise RuntimeError(f"NWPS cycle {cf.cycle} is on a different grid")
    # Sea is wherever any step of any cycle used has a wave height.
    wet = np.zeros(first.lat.shape + first.lon.shape, dtype=bool)
    for cf in cycles:
        for f in cf.fields.values():
            wet |= np.isfinite(f["hs"])
    used = sorted({cf.cycle for cf in cycles})
    origins = sorted({cf.origin for cf in cycles})
    static = xr.Dataset(
        {
            "wet": (
                ("lat", "lon"),
                wet.astype(np.uint8),
                {
                    "long_name": "1 where NWPS has a wave height in any record's cycle, else 0"
                },
            )
        },
        coords={
            "lat": (
                "lat",
                first.lat,
                {"units": "degrees_north", "standard_name": "latitude"},
            ),
            "lon": (
                "lon",
                first.lon,
                {"units": "degrees_east", "standard_name": "longitude"},
            ),
        },
        attrs={
            "type": "NWPS surface waves",
            "source": "NOAA NWPS (SWAN), one WFO computational grid",
            "schema": NWPS_SCHEMA,
            "wfo": wfo,
            "grid": grid,
            "crs": "EPSG:4326",
            "dx": float(first.lon[1] - first.lon[0])
            if first.lon.size > 1
            else float("nan"),
            "dy": float(first.lat[1] - first.lat[0])
            if first.lat.size > 1
            else float("nan"),
            "grid_note": "the NWPS grid cropped to the bbox plus one cell; not regridded",
            "requested_bbox": list(bbox),
            "start_time": naive_utc(start_time).strftime("%Y-%m-%dT%H:%M:%S"),
            "first_cycle": used[0].strftime("%Y-%m-%dT%HZ"),
            "last_cycle": used[-1].strftime("%Y-%m-%dT%HZ"),
            "record_cycles": [cf.cycle.strftime("%Y-%m-%dT%HZ") for _, cf, _ in plan],
            "source_urls": origins,
            "files_from_archive": sum(not o.startswith("https://") for o in origins),
            "files_from_nomads": sum(o.startswith("https://") for o in origins),
            "direction_convention": "dir: direction waves travel to; wdir: direction wind comes from",
        },
    )
    yield ("static", static)
    for t, cf, lead in plan:
        yield ("record", t, {**cf.fields[lead], "forecast_hour": np.float32(lead)})


def fetch_nwps_waves(
    start_time: str, hours: int, bbox: list[float], **kwargs
) -> xr.Dataset:
    """`iter_waves` assembled into one in-memory Dataset."""
    static = None
    times, records = [], []
    for item in iter_waves(start_time, hours, bbox, **kwargs):
        if item[0] == "static":
            static = item[1]
        else:
            times.append(item[1])
            records.append(item[2])
    assert static is not None
    return static.assign(
        {
            k: (("time",) + dims, np.stack([r[k] for r in records]).astype(np.float32))
            for k, dims in RECORD_DIMS.items()
        }
    ).assign_coords(time=times)
