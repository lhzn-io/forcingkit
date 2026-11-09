"""HRDPS forecasts at points, kept with both times: when each forecast was issued and the time it is for.

The High Resolution Deterministic Prediction System (HRDPS) is the 2.5 km limited-area model of Environment and
Climate Change Canada's Meteorological Service of Canada (MSC). Its continental domain reaches south past the
US Northeast coast: Long Island Sound lies about 350 cells inside the southern edge and the Gulf of Maine about 460
cells inside the eastern edge. It runs at 00, 06, 12 and 18 UTC to 48 h, with hourly output from lead 0.

Source. The MSC Datamart (``https://dd.weather.gc.ca``) publishes each run as one GRIB2 file per variable, level and
lead, with no byte-range index, under a directory for the run's date:

    {base}/{YYYYMMDD}/WXO-DD/model_hrdps/continental/2.5km/{HH}/{hhh}/
        {YYYYMMDD}T{HH}Z_MSC_HRDPS_{VAR}_{LEVEL}_RLatLon0.0225_PT{hhh}H.grib2

``{base}/today/model_hrdps/...`` is an alias for the current date. The Datamart keeps 30 days of dated directories
(MSC's stated retention, which held on 2026-11-09: 2026-10-11 to 2026-11-09 were present). The HPFX server
(``https://hpfx.collab.science.gc.ca``) mirrors the same layout and retention with no 24/7 support; pass it as
``base_url`` if the Datamart is slow. No longer public archive of the GRIB2 files was found, so a record of HRDPS
scores has to be accumulated from the day collection starts. A run's 48 h files were complete about 3 h after the
run time (the 2026-11-09 12 UTC run's PT048H files were posted at 15:08 UTC).

Fields. 10 m wind is ``UGRD`` and ``VGRD`` at ``AGL-10m``; ``GUST`` at ``AGL-10m`` is the instantaneous 10 m gust
(GRIB2 parameter 0/2/22, eccodes ``i10fg``). The land-sea mask is ``LAND`` at ``Sfc`` (a 0 to 1 fraction), published
at lead 0 only. Each file is about 2 MB, so the three wind fields of one 49-lead run are about 300 MB.

Grid. A rotated latitude-longitude grid of 2540 x 1290 cells at 0.0225 degrees, with the southern pole of rotation
at 36.08852 S, 245.305142 E and no rotation angle (from the GRIB2 headers). Winds are relative to the rotated grid
(``uvRelativeToGrid = 1``); here they are rotated to east and north by the angle between each cell's rotated
longitude axis and geographic east, which across the domain runs from -54 to +31 degrees. Checked against MSC's
own 10 m wind direction (``WDIR``, degrees true) for the 2026-11-09 12 UTC run, lead 1, at every cell with a wind
over 2 m/s: the median difference was 0.07 degrees and the 99th percentile 0.44 degrees (unrotated, 24 degrees).

The value at each point is an inverse-distance-squared mean of up to four HRDPS water cells (land fraction under
0.5) within ``max_km``, as for HRRR in `hrrr_points`. Values stay in the units HRDPS publishes (m/s).

License. ECCC's Data Server End-use Licence (https://eccc-msc.github.io/open-data/licence/readme_en/) allows use,
including commercial use and redistribution, with attribution: "Data Source: Environment and Climate Change
Canada", with a link to the license where possible. It grants no right to suggest official status or endorsement.
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
import tempfile
import time
from typing import Callable, Iterable

import httpx
import numpy as np
import pandas as pd

from .hrrr_atmosphere import _read_message
from .hrrr_points import PointWeights

logger = logging.getLogger(__name__)

BASE_URL = "https://dd.weather.gc.ca"
HPFX_URL = "https://hpfx.collab.science.gc.ca"
ATTRIBUTION = "Data Source: Environment and Climate Change Canada"
LICENSE_URL = "https://eccc-msc.github.io/open-data/licence/readme_en/"

CYCLE_HOURS = (0, 6, 12, 18)
MAX_LEAD = 48
RETENTION_DAYS = 30

# Datamart variable and level of each field.
FIELDS = {
    "u10": ("UGRD", "AGL-10m"),
    "v10": ("VGRD", "AGL-10m"),
    "gust": ("GUST", "AGL-10m"),
}
DEFAULT_FIELDS = tuple(FIELDS)
UNITS = {"u10": "m s-1", "v10": "m s-1", "gust": "m s-1"}
LAND = ("LAND", "Sfc")  # published at lead 0 only


def _utc_naive(t) -> pd.Timestamp:
    t = pd.Timestamp(t)
    return t.tz_convert("UTC").tz_localize(None) if t.tzinfo is not None else t


def file_url(cycle, lead: int, var: str, level: str, base_url: str = BASE_URL) -> str:
    """The Datamart URL of one variable at one level and lead of a run."""
    cycle = _utc_naive(cycle)
    day, hh = cycle.strftime("%Y%m%d"), cycle.strftime("%H")
    name = f"{day}T{hh}Z_MSC_HRDPS_{var}_{level}_RLatLon0.0225_PT{lead:03d}H.grib2"
    return (
        f"{base_url}/{day}/WXO-DD/model_hrdps/continental/2.5km/{hh}/{lead:03d}/{name}"
    )


def field_url(cycle, lead: int, field: str, base_url: str = BASE_URL) -> str:
    var, level = FIELDS[field]
    return file_url(cycle, lead, var, level, base_url)


def is_cycle(cycle) -> bool:
    cycle = _utc_naive(cycle)
    return cycle == cycle.floor("h") and cycle.hour in CYCLE_HOURS


def _exists(client: httpx.Client, url: str) -> bool:
    r = client.head(url)
    if r.status_code == 404:
        return False
    r.raise_for_status()
    return True


def newest_cycle(
    now=None,
    fields: Iterable[str] = DEFAULT_FIELDS,
    lead: int = MAX_LEAD,
    lookback: int = 8,
    base_url: str = BASE_URL,
    exists: Callable[[str], bool] | None = None,
) -> pd.Timestamp | None:
    """The newest run at or before `now` whose `lead` file is published for every field, looking back `lookback`
    runs (two days by default); None if none is. A run's last lead is posted last, so the default asks for a
    complete run."""
    now = _utc_naive(pd.Timestamp.now(tz="UTC") if now is None else now)
    fields = tuple(fields)
    cycle = now.floor("6h")
    client = None
    if exists is None:
        client = httpx.Client(timeout=30.0, follow_redirects=True)
        exists = lambda url: _exists(client, url)  # noqa: E731
    try:
        for _ in range(lookback):
            if all(exists(field_url(cycle, lead, f, base_url)) for f in fields):
                return cycle
            cycle -= pd.Timedelta(hours=6)
    finally:
        if client is not None:
            client.close()
    return None


# Rotated latitude-longitude grids (GRIB2 template 3.1), with the southern pole of rotation at (sp_lat, sp_lon) and
# no further rotation angle. The rotated system is the geographic one turned by sp_lon about the polar axis and
# then by 90 + sp_lat degrees about the new y axis.


def _cart(lon, lat):
    lon, lat = np.deg2rad(lon), np.deg2rad(lat)
    return np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)


def geographic_to_rotated(lon, lat, sp_lat: float, sp_lon: float):
    """Geographic (lon, lat) in degrees to rotated (rlon, rlat), rlon in -180..180."""
    th = np.deg2rad(90.0 + sp_lat)
    x, y, z = _cart(np.asarray(lon, dtype=float) - sp_lon, lat)
    xr = np.cos(th) * x + np.sin(th) * z
    zr = -np.sin(th) * x + np.cos(th) * z
    return np.rad2deg(np.arctan2(y, xr)), np.rad2deg(np.arcsin(np.clip(zr, -1.0, 1.0)))


def rotated_to_geographic(rlon, rlat, sp_lat: float, sp_lon: float):
    """Rotated (rlon, rlat) in degrees to geographic (lon, lat), lon in -180..180."""
    th = np.deg2rad(90.0 + sp_lat)
    xr, y, zr = _cart(rlon, rlat)
    x = np.cos(th) * xr - np.sin(th) * zr
    z = np.sin(th) * xr + np.cos(th) * zr
    lon = np.rad2deg(np.arctan2(y, x)) + sp_lon
    return ((lon + 180.0) % 360.0) - 180.0, np.rad2deg(np.arcsin(np.clip(z, -1.0, 1.0)))


def grid_east_angle(lon, lat, sp_lat: float, sp_lon: float, eps: float = 1e-3):
    """Angle in radians, counterclockwise from geographic east, of the rotated grid's x axis at each (lon, lat)."""
    rlon, rlat = geographic_to_rotated(lon, lat, sp_lat, sp_lon)
    lon1, lat1 = rotated_to_geographic(rlon, rlat, sp_lat, sp_lon)
    lon2, lat2 = rotated_to_geographic(rlon + eps, rlat, sp_lat, sp_lon)
    dlon = ((lon2 - lon1 + 180.0) % 360.0) - 180.0
    return np.arctan2(lat2 - lat1, dlon * np.cos(np.deg2rad(lat1)))


def earth_relative_winds(u, v, angle):
    """Rotate grid-relative (u, v) to east and north, given the grid x axis `angle` from east (radians)."""
    c, s = np.cos(angle), np.sin(angle)
    return c * u - s * v, s * u + c * v


def _pole(attrs) -> tuple[float, float]:
    if float(attrs.get("GRIB_angleOfRotationInDegrees", 0.0)) != 0.0:
        raise NotImplementedError(
            "HRDPS: rotated grid with a nonzero angle of rotation"
        )
    return (
        float(attrs["GRIB_latitudeOfSouthernPoleInDegrees"]),
        float(attrs["GRIB_longitudeOfSouthernPoleInDegrees"]),
    )


_angle_cache: dict = {}


def _grid_angle(da) -> np.ndarray:
    sp = _pole(da.attrs)
    lon, lat = da["longitude"].values, da["latitude"].values
    key = (lon.shape, sp, float(lon.flat[0]), float(lat.flat[0]))
    if key not in _angle_cache:
        _angle_cache[key] = grid_east_angle(lon, lat, *sp)
    return _angle_cache[key]


RETRIES = 4


def _download(client: httpx.Client, url: str, path: str) -> bool:
    """Write `url` to `path`; False if the Datamart has no such file. A dropped connection or a 5xx reply is
    retried up to `RETRIES` times, 2, 4, 8 and 16 s apart (the Datamart closed connections during a full run)."""
    for attempt in range(RETRIES + 1):
        try:
            with client.stream("GET", url) as r:
                if r.status_code == 404:
                    return False
                r.raise_for_status()
                with open(path, "wb") as f:
                    for chunk in r.iter_bytes():
                        f.write(chunk)
            return True
        except (httpx.TransportError, httpx.HTTPStatusError) as e:
            server_side = (
                not isinstance(e, httpx.HTTPStatusError)
                or e.response.status_code >= 500
            )
            if not server_side or attempt == RETRIES:
                raise
            logger.info("HRDPS %s: %s; retrying", url.rsplit("/", 1)[-1], e)
            time.sleep(2.0 ** (attempt + 1))
    return False  # not reached


def _read_url(client, url: str, path: str):
    if not _download(client, url, path):
        return None
    try:
        return _read_message(path)
    finally:
        os.remove(path)


def _fetch_lead(cycle, lead, fields, workdir, client, base_url):
    """The requested fields of one run and lead, on the HRDPS grid; None if any is not published."""
    out = {}
    for name in fields:
        path = os.path.join(workdir, f"{name}_{cycle:%Y%m%d%H}_f{lead:03d}.grib2")
        da = _read_url(client, field_url(cycle, lead, name, base_url), path)
        if da is None:
            logger.info("HRDPS %s f%03d: %s not published", cycle, lead, name)
            return None
        out[name] = da
    return out


_land_cache: dict = {}


def _fetch_land(cycle, workdir, client, base_url):
    """The land-sea mask from a run's lead 0, kept for the session (it does not change between runs)."""
    if "land" not in _land_cache:
        path = os.path.join(workdir, f"land_{cycle:%Y%m%d%H}.grib2")
        da = _read_url(client, file_url(cycle, 0, *LAND, base_url=base_url), path)
        if da is None:
            raise FileNotFoundError(
                f"HRDPS land mask not published for the {cycle} run"
            )
        _land_cache["land"] = da
    return _land_cache["land"]


_weights_cache: dict = {}


def _weights(native, land, points, k, max_km) -> PointWeights:
    da = next(iter(native.values()))
    if da.shape != land.shape:
        raise ValueError(
            f"HRDPS: field grid {da.shape} differs from land mask grid {land.shape}"
        )
    key = (da.shape, tuple(sorted(points.items())), k, max_km)
    if key not in _weights_cache:
        _weights_cache[key] = PointWeights(
            da["longitude"].values,
            da["latitude"].values,
            land.values,
            points,
            k,
            max_km,
            label="HRDPS",
        )
    return _weights_cache[key]


def fetch_points(
    cycles: Iterable,
    leads: Iterable[int],
    points: dict[str, tuple[float, float]],
    fields: Iterable[str] = DEFAULT_FIELDS,
    k: int = 4,
    max_km: float = 6.0,
    max_workers: int = 2,
    base_url: str = BASE_URL,
) -> pd.DataFrame:
    """HRDPS forecasts at `points` ({name: (lon, lat)}) for every run and lead, as rows of
    (issue_time, valid_time, lead_h, point, field, value). Times that are not HRDPS runs, and leads outside 0 to 48,
    are skipped; so is a lead with any field not (yet) published."""
    fields = tuple(fields)
    unknown = set(fields) - set(FIELDS)
    if unknown:
        raise ValueError(f"unknown HRDPS point fields: {sorted(unknown)}")
    wind = "u10" in fields and "v10" in fields
    if ("u10" in fields or "v10" in fields) and not wind:
        raise ValueError(
            "HRDPS u10 and v10 are grid-relative: request both to rotate them to east and north"
        )
    tasks = []
    for c in cycles:
        cycle = _utc_naive(c)
        if not is_cycle(cycle):
            logger.warning("HRDPS: %s is not a 00/06/12/18 UTC run; skipped", cycle)
            continue
        tasks += [(cycle, int(lead)) for lead in leads if 0 <= int(lead) <= MAX_LEAD]
    rows = []
    with (
        tempfile.TemporaryDirectory() as workdir,
        httpx.Client(timeout=120.0, follow_redirects=True) as client,
        concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex,
    ):
        results = ex.map(
            lambda t: _fetch_lead(t[0], t[1], fields, workdir, client, base_url), tasks
        )
        for (cycle, lead), native in zip(tasks, results):
            if native is None:
                continue
            land = _fetch_land(cycle, workdir, client, base_url)
            w = _weights(native, land, points, k, max_km)
            if wind and int(native["u10"].attrs.get("GRIB_uvRelativeToGrid", 1)) == 1:
                ue, vn = earth_relative_winds(
                    native["u10"].values,
                    native["v10"].values,
                    _grid_angle(native["u10"]),
                )
                values = {"u10": w(ue), "v10": w(vn)}
            else:
                values = {}
            for name in fields:
                if name not in values:
                    values[name] = w(native[name].values)
            valid = cycle + pd.Timedelta(hours=lead)
            for name in fields:
                for p, v in values[name].items():
                    rows.append((cycle, valid, lead, p, name, v))
    return pd.DataFrame(
        rows, columns=["issue_time", "valid_time", "lead_h", "point", "field", "value"]
    )
