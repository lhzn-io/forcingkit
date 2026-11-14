"""Hourly HF radar surface currents from the NOAA IOOS HF Radar National Network, and an archive that outlasts the
server's rolling window.

Shore-based HF radars measure radial surface currents; the HF Radar National Network (HFRNet) combines them into
hourly total vectors on fixed 1 km, 2 km and 6 km grids. NOAA's National Data Buoy Center serves the US East and Gulf
Coast totals over OPeNDAP (``hfradar_usegc_1km``, ``_2km``), but keeps only about 92 days. The values are surface
currents representative of roughly the upper 0.5 m: an observation for scoring a model's surface layer, not a
forcing.

`fetch` reads a box and a time window from the server as stored: 16-bit integers with their scale factor
(0.01) and fill value, so nothing is rounded twice. DAP2 has no signed byte, so a signed byte arrives unsigned with
``_Unsigned = "false"``; it is turned back into int8 here. The variables are ``u`` and ``v`` (eastward and northward
surface current, m/s), ``dopx``, ``dopy`` and ``hdop`` (dilution of precision, for screening the geometry),
``number_of_sites`` and ``number_of_radials``, with the grid mapping and the processing settings.

`archive` keeps a box one file per UTC day and grid (NetCDF-4), under ``settings.cache_dir("hfradar")`` unless
another store is given::

    usegc_<grid>/<box>/<YYYY>/<MM>/hfradar_usegc_<grid>_<YYYYMMDD>.nc

where ``<box>`` names the box, for example ``40.850N_41.300N_073.800W_071.800W``. A day is written once the server
has published past its end and six hours have gone by, and never rewritten; every hour the server holds for the day is
kept, with no screening. Unlike the Zarr stores beside it, this directory is an archive: the server cannot supply
those days again once they leave its window, so do not purge it. `load` reads a box and window back, decoded to m/s,
from any archived box that contains it, and fetches days the archive lacks from the server while it still holds them.

Older days come from NOAA NCEI, which archives the same hourly totals (the same variables, stored the same way, on
the same grids) as one file per hour in monthly accessions. `archive_ncei` finds a window's accessions through NCEI's
metadata search, reads each hour's file, cuts the box out and writes the same daily files, so `load` reads both
alike; their ``source_dataset`` names the accession. NCEI's accessions lag the present by a month or two; NDBC's
window covers the rest. The archive's root is ``FORCINGKIT_HFRADAR_ARCHIVE`` when set (see `settings.hfradar_archive`).

From the command line::

    python -m forcingkit.fetchers.hfradar ncei --box W S E N --start 2025-01-01 --end 2026-09-01 [--root DIR]
    python -m forcingkit.fetchers.hfradar ndbc --box W S E N [--start all] [--root DIR]

Credit: the radar sites' operators (named in HFRNet's site list; for example the University of Connecticut's LISICOS
for western Long Island Sound) and the NOAA IOOS HF Radar National Network; see ``source_credit`` in each file.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import re
import time as _time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
import requests
import xarray as xr

from forcingkit import settings

logger = logging.getLogger(__name__)

SERVER = "https://dods.ndbc.noaa.gov/thredds/dodsC"
USER_AGENT = "forcingkit (+https://github.com/lhzn-io/forcingkit)"
GRIDS = ("1km", "2km")
VARIABLES = ["u", "v", "dopx", "dopy", "hdop", "number_of_sites", "number_of_radials"]
# The grid mapping and the processing settings. (The server's radial_metadata names the radial files of its latest
# hour only, so it is left out.)
SCALARS = ["wgs84", "processing_parameters"]
LAG = timedelta(
    hours=6
)  # after a day ends, before it is written (the server posts an hour about 2 h late)
LOOKBACK_DAYS = 10  # what a routine `archive` run looks back over
CHUNK_DAYS = 8  # days per request
# Global attributes carried over from the server's dataset (not the ones describing its latest hour).
KEEP_ATTRS = [
    "Conventions",
    "title",
    "summary",
    "institution",
    "program",
    "project",
    "naming_authority",
    "license",
    "acknowledgment",
    "acknowledgement",
    "references",
    "creator_name",
    "creator_url",
    "creator_email",
    "publisher_name",
    "publisher_url",
    "publisher_email",
    "contributor_name",
    "contributor_role",
    "processing_level",
    "instrument",
    "platform",
    "product_version",
    "metadata_link",
    "keywords",
    "keywords_vocabulary",
    "standard_name_vocabulary",
]
CREDIT = (
    "HF radar sites operated by their institutions (for western Long Island Sound, the University of Connecticut's "
    "LISICOS); hourly total vectors from the NOAA IOOS HF Radar National Network (HFRNet), served by NOAA NDBC."
)
NCEI_CREDIT = CREDIT.replace("served by NOAA NDBC.", "archived by NOAA NCEI.")
NCEI = "https://www.ncei.noaa.gov"
NCEI_SEARCH = f"{NCEI}/metadata/granule/geoportal/opensearch"
NCEI_ARCHIVE = f"{NCEI}/data/oceans/archive/"
# The metadata series holding the hourly totals (the Radial series beside them hold each site's radials).
NCEI_SERIES = ("IOOS-HFRadarRTVector", "NDBC-HFRadarRTVector")
NCEI_INDEX = "ncei_accessions.json"  # accession -> its folder, beside the archive, so each is looked up once
_NCEI_TITLE = re.compile(r"during ([A-Z][a-z]+) (\d{4}) \(NCEI Accession (\d+)\)")
_NCEI_FILE = re.compile(
    r'href="(rtv-usegc-([126]km)-[^"/]*_s(\d{12})\d*_e\d+_c(\d+)\.nc)"'
)
NAME = re.compile(
    r"usegc_[126]km/\d{2}\.\d{3}[NS]_\d{2}\.\d{3}[NS]_\d{3}\.\d{3}[EW]_\d{3}\.\d{3}[EW]/\d{4}/\d{2}/"
    r"hfradar_usegc_[126]km_\d{8}\.nc"
)
_BOX = re.compile(
    r"(\d{2}\.\d{3})([NS])_(\d{2}\.\d{3})([NS])_(\d{3}\.\d{3})([EW])_(\d{3}\.\d{3})([EW])"
)

BBox = Sequence[float]  # [min_lon, min_lat, max_lon, max_lat]


class Store(Protocol):
    """Where archived days go. `has` may answer False when it cannot tell (a create-only bucket); `put` then refuses
    an existing file and returns False."""

    def has(self, name: str) -> bool: ...

    def put(self, name: str, data: bytes) -> bool: ...


class LocalStore:
    """A directory; files are created once and never replaced."""

    def __init__(self, root: str | os.PathLike | None = None):
        self.root = Path(root if root is not None else archive_root()).expanduser()

    def has(self, name: str) -> bool:
        return (self.root / name).exists()

    def put(self, name: str, data: bytes) -> bool:
        if not NAME.fullmatch(name):
            raise ValueError(f"refusing to write {name}")
        path = self.root / name
        if path.exists():
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        tmp.write_bytes(data)
        tmp.rename(path)
        return True


def archive_root() -> str:
    """The archive's default root: FORCINGKIT_HFRADAR_ARCHIVE, else the cache's ``hfradar`` directory."""
    return settings.hfradar_archive() or settings.cache_dir("hfradar")


def dataset_url(grid: str) -> str:
    return f"{SERVER}/hfradar_usegc_{grid}"


def box_key(bbox: BBox) -> str:
    """The box's directory name, to three decimals: ``40.850N_41.300N_073.800W_071.800W`` (south, north, west, east)."""
    w, s, e, n = (float(x) for x in bbox)

    def lat(v: float) -> str:
        return f"{abs(v):06.3f}{'N' if v >= 0 else 'S'}"

    def lon(v: float) -> str:
        return f"{abs(v):07.3f}{'E' if v >= 0 else 'W'}"

    return f"{lat(s)}_{lat(n)}_{lon(w)}_{lon(e)}"


def parse_box_key(key: str) -> tuple[float, float, float, float] | None:
    """The bbox ([min_lon, min_lat, max_lon, max_lat]) a directory name stands for, or None."""
    m = _BOX.fullmatch(key)
    if not m:
        return None
    s, sh, n, nh, w, wh, e, eh = m.groups()
    sign = {"N": 1, "S": -1, "E": 1, "W": -1}
    return (
        float(w) * sign[wh],
        float(s) * sign[sh],
        float(e) * sign[eh],
        float(n) * sign[nh],
    )


def day_name(grid: str, bbox: BBox, day: pd.Timestamp) -> str:
    return (
        f"usegc_{grid}/{box_key(bbox)}/{day:%Y/%m}/hfradar_usegc_{grid}_{day:%Y%m%d}.nc"
    )


def open_remote(
    grid: str = "1km", session: requests.Session | None = None
) -> xr.Dataset:
    """The server's dataset, lazily, with the stored integers left as they are."""
    if session is None:
        session = requests.Session()
        session.headers["User-Agent"] = USER_AGENT
    return xr.open_dataset(
        dataset_url(grid).replace("https://", "dap2://"),
        engine="pydap",
        session=session,
        mask_and_scale=False,
    )


def box_slices(lat: np.ndarray, lon: np.ndarray, bbox: BBox) -> tuple[slice, slice]:
    """Index slices of the grid cells whose centres fall in the box."""
    w, s, e, n = bbox
    iy = np.flatnonzero((lat >= s) & (lat <= n))
    ix = np.flatnonzero((lon >= w) & (lon <= e))
    if not len(iy) or not len(ix):
        raise ValueError("the box misses the grid")
    return slice(int(iy[0]), int(iy[-1]) + 1), slice(int(ix[0]), int(ix[-1]) + 1)


def _utc_naive(t) -> pd.Timestamp:
    ts = pd.Timestamp(t)
    return ts.tz_convert("UTC").tz_localize(None) if ts.tzinfo else ts


def ready_days(
    times: pd.DatetimeIndex,
    now: datetime,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> list[pd.Timestamp]:
    """The UTC days with hours on the server that may be archived: the server has published past the day's end and
    the lag has passed. Bounded by ``start`` and ``end`` (days, inclusive) when given."""
    if not len(times):
        return []
    last = times.max()
    now_ts = _utc_naive(now)
    days = []
    for day in pd.DatetimeIndex(times.normalize().unique()).sort_values():
        stop = day + pd.Timedelta(days=1)
        if last < stop or now_ts < stop + LAG:
            continue
        if (start is not None and day < start) or (end is not None and day > end):
            continue
        days.append(day)
    return days


def runs(days: list[pd.Timestamp], most: int = CHUNK_DAYS) -> list[list[pd.Timestamp]]:
    """Consecutive days grouped into requests of at most ``most`` days."""
    out: list[list[pd.Timestamp]] = []
    for day in days:
        if out and day - out[-1][-1] == pd.Timedelta(days=1) and len(out[-1]) < most:
            out[-1].append(day)
        else:
            out.append([day])
    return out


def _stored(var: xr.DataArray) -> tuple[np.ndarray, dict]:
    """A variable's stored values and its attributes, minus the server's private ones, with signed bytes restored."""
    values = np.asarray(var.values)
    if (
        values.dtype == np.uint8
        and str(var.attrs.get("_Unsigned", "")).lower() == "false"
    ):
        values = values.view(np.int8)
    attrs = {
        k: v
        for k, v in var.attrs.items()
        if (not k.startswith("_") or k == "_FillValue") and k != "coordinates"
    }
    return values, attrs


def _normalize(
    sub: xr.Dataset,
    grid: str,
    bbox: BBox,
    source_attrs: dict,
    fetched: datetime,
    source: str | None = None,
    credit: str = CREDIT,
) -> xr.Dataset:
    """The server's subset as a clean Dataset of stored values, with its encodings and credit. ``source`` names
    where it came from (default: NDBC's dataset)."""
    source = source or dataset_url(grid)
    out = xr.Dataset(
        coords={
            "time": sub["time"].values,
            "lat": sub["lat"].values,
            "lon": sub["lon"].values,
        }
    )
    for name in VARIABLES:
        if name in sub:
            values, attrs = _stored(sub[name])
            fill = attrs.pop("_FillValue", None)
            out[name] = (sub[name].dims, values, attrs)
            out[name].encoding.update(
                {"zlib": True, "complevel": 4, "shuffle": True, "dtype": values.dtype}
            )
            if fill is not None:
                out[name].encoding["_FillValue"] = values.dtype.type(fill)
    for name in SCALARS:
        if name in sub:
            values, attrs = _stored(sub[name])
            out[name] = ((), values, attrs)
    for c in ("lat", "lon"):
        out[c].attrs = {k: v for k, v in sub[c].attrs.items() if not k.startswith("_")}
    out["time"].attrs = {
        "standard_name": "time",
        "long_name": "time at the centre of the hour",
    }
    out["time"].encoding.update(
        {"units": "seconds since 1970-01-01", "calendar": "gregorian", "dtype": "int32"}
    )
    attrs = {k: source_attrs[k] for k in KEEP_ATTRS if k in source_attrs}
    w, s, e, n = bbox
    attrs.update(
        {
            "source_dataset": source,
            "source_credit": credit,
            "subset": f"latitude {s} to {n}, longitude {w} to {e}; values as stored on the server",
            "history": f"{fetched:%Y-%m-%dT%H:%M:%SZ} subset from {source} by forcingkit",
        }
    )
    out.attrs = attrs
    return out


def fetch(
    bbox: BBox,
    start,
    end,
    grid: str = "1km",
    src: xr.Dataset | None = None,
    session: requests.Session | None = None,
) -> xr.Dataset:
    """The box's hours in [start, end) from the server, as stored (see the module notes). Pass an open `src` to reuse
    one connection across calls. Empty in time when the server holds none of the window."""
    own = src is None
    src = src if src is not None else open_remote(grid, session)
    try:
        times = pd.DatetimeIndex(src["time"].values)
        t = np.flatnonzero((times >= _utc_naive(start)) & (times < _utc_naive(end)))
        ys, xs = box_slices(src["lat"].values, src["lon"].values, bbox)
        names = [n for n in VARIABLES + SCALARS if n in src]
        tsel = slice(int(t[0]), int(t[-1]) + 1) if len(t) else slice(0, 0)
        sub = src[names].isel(time=tsel, lat=ys, lon=xs).load()
        return _normalize(sub, grid, bbox, dict(src.attrs), datetime.now(timezone.utc))
    finally:
        if own:
            src.close()


def day_file(ds: xr.Dataset, day: pd.Timestamp) -> bytes:
    """One UTC day of a `fetch` result as NetCDF-4 bytes (the archive's file)."""
    one = ds.sel(
        time=slice(day, day + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
    ).copy()
    for name in VARIABLES:  # sel keeps the encodings, but be explicit
        if name in ds:
            one[name].encoding = dict(ds[name].encoding)
    one["time"].encoding = dict(ds["time"].encoding)
    one.attrs = dict(
        ds.attrs, subset_day=f"{day:%Y-%m-%d} UTC", subset_hours=int(one.sizes["time"])
    )
    buf = io.BytesIO()
    one.to_netcdf(buf, engine="h5netcdf")
    return buf.getvalue()


def archive(
    bbox: BBox,
    start: str | None = None,
    end: str | None = None,
    grids: Sequence[str] = GRIDS,
    store: Store | None = None,
    lookback_days: int = LOOKBACK_DAYS,
    now: datetime | None = None,
    session: requests.Session | None = None,
) -> dict[str, list[str]]:
    """Archive every ready day of the box not yet kept, per grid; return the names written per grid.

    With neither ``start`` nor ``end``, the last ``lookback_days`` days; ``start="all"`` takes every day the server
    still holds. The store defaults to ``settings.cache_dir("hfradar")``; ``session`` lets a caller send its own
    User-Agent. A request that fails is skipped and the rest carry on; the call then raises, naming the days
    missed."""
    now = now or datetime.now(timezone.utc)
    store = store if store is not None else LocalStore()
    written: dict[str, list[str]] = {}
    failed: list[str] = []
    for grid in grids:
        src = open_remote(grid, session)
        try:
            times = pd.DatetimeIndex(src["time"].values)
            if start == "all":
                first = None
            elif start:
                first = pd.Timestamp(start)
            else:
                first = _utc_naive(now).normalize() - pd.Timedelta(days=lookback_days)
            last = pd.Timestamp(end) if end else None
            todo = [
                d
                for d in ready_days(times, now, first, last)
                if not store.has(day_name(grid, bbox, d))
            ]
            written[grid] = []
            for group in runs(todo):
                try:
                    ds = fetch(
                        bbox, group[0], group[-1] + pd.Timedelta(days=1), grid, src=src
                    )
                except Exception as exc:  # carry on; a later run fills the gap while the server holds it
                    logger.warning(
                        "HF radar %s %s to %s: %s",
                        grid,
                        group[0].date(),
                        group[-1].date(),
                        exc,
                    )
                    failed.append(f"{grid} {group[0]:%Y-%m-%d}..{group[-1]:%Y-%m-%d}")
                    continue
                for day in group:
                    name = day_name(grid, bbox, day)
                    if store.put(name, day_file(ds, day)):
                        written[grid].append(name)
                        logger.info("%s written", name)
                    else:
                        logger.info("%s already kept", name)
        finally:
            src.close()
    if failed:
        raise RuntimeError(
            f"HF radar days not fetched: {', '.join(failed)} (written: {written})"
        )
    return written


def _decoded(ds: xr.Dataset) -> xr.Dataset:
    """Stored values decoded as a reader of the archive sees them (scale applied, fills as NaN)."""
    buf = io.BytesIO()
    ds.to_netcdf(buf, engine="h5netcdf")
    buf.seek(0)
    with xr.open_dataset(buf, engine="h5netcdf") as out:
        return out.load()


def _containing_boxes(root: Path, grid: str, bbox: BBox) -> list[Path]:
    """Archived box directories that contain the box, smallest first."""
    found = []
    for d in (root / f"usegc_{grid}").glob("*"):
        b = parse_box_key(d.name)
        if (
            b
            and b[0] <= bbox[0]
            and b[1] <= bbox[1]
            and b[2] >= bbox[2]
            and b[3] >= bbox[3]
        ):
            found.append(((b[2] - b[0]) * (b[3] - b[1]), d))
    return [d for _, d in sorted(found)]


def load(
    bbox: BBox,
    start,
    end,
    grid: str = "1km",
    root: str | os.PathLike | None = None,
    remote: bool = True,
    session: requests.Session | None = None,
) -> xr.Dataset:
    """The box's hours in [start, end), decoded (u and v in m/s, NaN where there is no solution): archived days from
    any archived box that contains this one, and, when ``remote``, the other days from the server while it still
    holds them (not archived by this call). Empty in time when neither has any."""
    base = Path(root if root is not None else archive_root()).expanduser()
    t0, t1 = _utc_naive(start), _utc_naive(end)
    boxes = _containing_boxes(base, grid, bbox)
    parts: list[xr.Dataset] = []
    missing: list[pd.Timestamp] = []
    for day in pd.date_range(
        t0.normalize(), (t1 - pd.Timedelta(seconds=1)).normalize(), freq="D"
    ):
        path = next(
            (
                p
                for b in boxes
                if (
                    p := b / f"{day:%Y/%m}" / f"hfradar_usegc_{grid}_{day:%Y%m%d}.nc"
                ).exists()
            ),
            None,
        )
        if path is None:
            missing.append(day)
            continue
        with xr.open_dataset(path) as ds:
            ys, xs = box_slices(ds["lat"].values, ds["lon"].values, bbox)
            parts.append(
                ds[[v for v in VARIABLES if v in ds]].isel(lat=ys, lon=xs).load()
            )
    if remote and missing:
        src = open_remote(grid, session)
        try:
            for group in runs(missing):
                got = fetch(
                    bbox, group[0], group[-1] + pd.Timedelta(days=1), grid, src=src
                )
                if got.sizes["time"]:
                    parts.append(_decoded(got[[v for v in VARIABLES if v in got]]))
        finally:
            src.close()
    if not parts:
        return xr.Dataset()
    out = xr.concat(
        parts, dim="time", data_vars="all", coords="minimal", join="exact"
    ).sortby("time")
    out = out.sel(time=slice(t0, t1 - pd.Timedelta(seconds=1)))
    out.attrs = {"source_dataset": dataset_url(grid), "source_credit": CREDIT}
    return out


# ---- NCEI's monthly archive ----


def _client(session: requests.Session | None) -> requests.Session:
    if session is None:
        session = requests.Session()
        session.headers["User-Agent"] = USER_AGENT
    return session


def _get(
    session: requests.Session, url: str, tries: int = 4, **kw
) -> requests.Response:
    """GET with a few retries and backoff (NCEI answers a burst with the odd 5xx or reset)."""
    for k in range(tries):
        try:
            r = session.get(url, timeout=120, **kw)
            if r.status_code < 500:
                r.raise_for_status()
                return r
            err: Exception = requests.HTTPError(f"{r.status_code} for {url}")
        except (requests.ConnectionError, requests.Timeout) as exc:
            err = exc
        if k < tries - 1:
            _time.sleep(2**k)
    raise err


def ncei_months(
    start, end, session: requests.Session | None = None
) -> dict[pd.Timestamp, str]:
    """NCEI's accessions of hourly totals for the months that overlap [start, end): {first of month: accession}.
    A month listed twice keeps the higher (later) accession number."""
    session = _client(session)
    first = _utc_naive(start).to_period("M").to_timestamp()
    last = (_utc_naive(end) - pd.Timedelta(seconds=1)).to_period("M").to_timestamp()
    found: dict[pd.Timestamp, str] = {}
    for series in NCEI_SERIES:
        at = 1
        while True:
            page = _get(
                session,
                NCEI_SEARCH,
                params={"q": series, "start": at, "num": 200, "f": "json"},
            ).json()
            for item in page.get("results", []):
                if not str(item.get("id", "")).startswith(series + "."):
                    continue
                m = _NCEI_TITLE.search(item.get("title", ""))
                if not m:
                    continue
                month = pd.Timestamp(f"{m.group(1)} 1 {m.group(2)}")
                if first <= month <= last and int(m.group(3)) > int(
                    found.get(month, "0")
                ):
                    found[month] = m.group(3)
            nxt = page.get("nextStart")
            if not nxt or nxt <= at or nxt > page.get("total", 0):
                break
            at = nxt
    return dict(sorted(found.items()))


def ncei_accession_dir(
    accession: str, session: requests.Session | None = None, known: dict | None = None
) -> str:
    """The folder of an accession's US East and Gulf Coast totals. NCEI files accessions under numbered ``arc``
    folders; the newest are searched first. ``known`` (accession -> folder) is consulted and updated."""
    if known is not None and accession in known:
        return known[accession]
    session = _client(session)
    arcs = sorted(
        set(re.findall(r'href="(arc\d+)/"', _get(session, NCEI_ARCHIVE).text)),
        reverse=True,
    )
    for arc in arcs:
        if f'href="{accession}/"' in _get(session, f"{NCEI_ARCHIVE}{arc}/").text:
            url = f"{NCEI_ARCHIVE}{arc}/{accession}/1.1/data/0-data/USEGC/"
            if known is not None:
                known[accession] = url
            return url
    raise RuntimeError(f"NCEI accession {accession} not found under {NCEI_ARCHIVE}")


def ncei_hours(
    folder: str, grid: str, session: requests.Session | None = None
) -> dict[pd.Timestamp, str]:
    """The hourly files of one grid in an accession's folder: {hour: URL}; a reissued hour keeps the latest file."""
    best: dict[pd.Timestamp, tuple[str, str]] = {}
    for name, g, start, created in _NCEI_FILE.findall(
        _get(_client(session), folder).text
    ):
        if g != grid:
            continue
        hour = pd.Timestamp(datetime.strptime(start, "%Y%m%d%H%M"))
        if hour not in best or created > best[hour][1]:
            best[hour] = (name, created)
    return {h: folder + n for h, (n, _) in sorted(best.items())}


def _ncei_hour(
    session: requests.Session, url: str, bbox: BBox
) -> tuple[xr.Dataset, dict]:
    """One hourly file, the box cut out, values as stored."""
    data = _get(session, url).content
    with xr.open_dataset(
        io.BytesIO(data), engine="h5netcdf", mask_and_scale=False
    ) as ds:
        ys, xs = box_slices(ds["lat"].values, ds["lon"].values, bbox)
        names = [n for n in VARIABLES + SCALARS if n in ds]
        return ds[names].isel(lat=ys, lon=xs).load(), dict(ds.attrs)


def _read_index(store: Store) -> tuple[dict, Path | None]:
    root = getattr(store, "root", None)
    if root is None:
        return {}, None
    path = Path(root) / NCEI_INDEX
    try:
        return json.loads(path.read_text()), path
    except (OSError, ValueError):
        return {}, path


def archive_ncei(
    bbox: BBox,
    start,
    end,
    grids: Sequence[str] = GRIDS,
    store: Store | None = None,
    session: requests.Session | None = None,
    workers: int = 4,
) -> dict[str, list[str]]:
    """Archive the box's days in [start, end) from NCEI's monthly accessions, per grid, skipping days already kept
    (from either source); return the names written per grid. A day is written with the hours NCEI holds for it.
    Days in months NCEI has not published are left for NDBC. A day that fails is skipped and the rest carry on; the
    call then raises, naming the days missed."""
    session = _client(session)
    store = store if store is not None else LocalStore()
    t0, t1 = _utc_naive(start).normalize(), _utc_naive(end)
    days = pd.date_range(t0, (t1 - pd.Timedelta(seconds=1)).normalize(), freq="D")
    months = ncei_months(t0, t1, session)
    known, index_path = _read_index(store)
    written: dict[str, list[str]] = {g: [] for g in grids}
    failed: list[str] = []
    for grid in grids:
        for month, accession in months.items():
            todo = [
                d
                for d in days
                if d.to_period("M").to_timestamp() == month
                and not store.has(day_name(grid, bbox, d))
            ]
            if not todo:
                continue
            try:
                folder = ncei_accession_dir(accession, session, known)
                hours = ncei_hours(folder, grid, session)
            except Exception as exc:
                logger.warning(
                    "NCEI %s %s (%s): %s", grid, f"{month:%Y-%m}", accession, exc
                )
                failed.append(f"{grid} {month:%Y-%m}")
                continue
            if index_path is not None:
                index_path.parent.mkdir(parents=True, exist_ok=True)
                index_path.write_text(json.dumps(known, indent=1, sort_keys=True))
            for day in todo:
                urls = [u for h, u in hours.items() if h.normalize() == day]
                if not urls:
                    logger.info(
                        "NCEI %s %s: no hours in accession %s",
                        grid,
                        day.date(),
                        accession,
                    )
                    continue
                try:
                    with ThreadPoolExecutor(max_workers=workers) as pool:
                        got = list(
                            pool.map(lambda u: _ncei_hour(session, u, bbox), urls)
                        )
                except Exception as exc:  # carry on; a later run fills the gap
                    logger.warning("NCEI %s %s: %s", grid, day.date(), exc)
                    failed.append(f"{grid} {day:%Y-%m-%d}")
                    continue
                sub = xr.concat(
                    [g for g, _ in got],
                    dim="time",
                    data_vars="minimal",
                    coords="minimal",
                    compat="override",
                    join="exact",
                ).sortby("time")
                ds = _normalize(
                    sub,
                    grid,
                    bbox,
                    got[0][1],
                    datetime.now(timezone.utc),
                    source=f"{NCEI}/archive/accession/{accession} ({folder})",
                    credit=NCEI_CREDIT,
                )
                name = day_name(grid, bbox, day)
                if store.put(name, day_file(ds, day)):
                    written[grid].append(name)
                    logger.info(
                        "%s written (%d h, NCEI %s)", name, len(urls), accession
                    )
    if failed:
        raise RuntimeError(
            f"HF radar days not read from NCEI: {', '.join(failed)} (written: {written})"
        )
    return written


def main(argv: Sequence[str] | None = None) -> None:
    """Archive a box from NDBC (the last ~3 months) or NCEI (older months)."""
    p = argparse.ArgumentParser(
        prog="python -m forcingkit.fetchers.hfradar", description=main.__doc__
    )
    p.add_argument("source", choices=["ndbc", "ncei"])
    p.add_argument(
        "--box", nargs=4, type=float, required=True, metavar=("W", "S", "E", "N")
    )
    p.add_argument(
        "--start", help="first day (ndbc: or 'all'; default the last 10 days)"
    )
    p.add_argument("--end", help="ncei: the day after the last; ndbc: the last day")
    p.add_argument("--grids", nargs="+", default=list(GRIDS), choices=list(GRIDS))
    p.add_argument(
        "--root",
        help="the archive's root (default: FORCINGKIT_HFRADAR_ARCHIVE, else the cache)",
    )
    p.add_argument(
        "--workers", type=int, default=4, help="ncei: hourly files read at once"
    )
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if a.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    store = LocalStore(a.root)
    if a.source == "ncei":
        if not (a.start and a.end):
            p.error("ncei needs --start and --end")
        out = archive_ncei(
            a.box, a.start, a.end, a.grids, store=store, workers=a.workers
        )
    else:
        out = archive(a.box, a.start, a.end, a.grids, store=store)
    print(json.dumps({g: len(n) for g, n in out.items()}))


if __name__ == "__main__":
    main()
