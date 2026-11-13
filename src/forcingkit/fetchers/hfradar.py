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

Credit: the radar sites' operators (named in HFRNet's site list; for example the University of Connecticut's LISICOS
for western Long Island Sound) and the NOAA IOOS HF Radar National Network; see ``source_credit`` in each file.
"""

from __future__ import annotations

import io
import logging
import os
import re
from collections.abc import Sequence
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
        self.root = Path(
            root if root is not None else settings.cache_dir("hfradar")
        ).expanduser()

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
    sub: xr.Dataset, grid: str, bbox: BBox, source_attrs: dict, fetched: datetime
) -> xr.Dataset:
    """The server's subset as a clean Dataset of stored values, with its encodings and credit."""
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
            "source_dataset": dataset_url(grid),
            "source_credit": CREDIT,
            "subset": f"latitude {s} to {n}, longitude {w} to {e}; values as stored on the server",
            "history": f"{fetched:%Y-%m-%dT%H:%M:%SZ} subset from {dataset_url(grid)} by forcingkit",
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
    base = Path(
        root if root is not None else settings.cache_dir("hfradar")
    ).expanduser()
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
