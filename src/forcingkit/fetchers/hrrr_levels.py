"""HRRR's lower atmosphere on its native levels (schema `hrrr-levels-v1`).

The surface delivery (`hrrr_atmosphere`, schema `hrrr-atm-v1`) carries 10 m wind and 2 m air; this
one carries the air above it: wind, vertical velocity and temperature on HRRR's lowest hybrid
levels, for vertical sections through the boundary layer and for tracing air parcels. Each hour t
is the one-hour forecast (f01) of the cycle initialised at t - 1 h, read by byte range from
`wrfnatf01`, as the surface delivery reads `wrfsfcf01`.

HRRR's hybrid levels follow the terrain near the ground and flatten with height. Over open water
the lowest 14 reach about 2.9 km (level 1 near 40 m, level 6 near 460 m, level 10 near 1.35 km).
Their heights move only a few metres with the surface pressure, so the level heights and pressures
are read once, from the first hour, and stored as static fields; that halves the download (each
CONUS message is fetched whole, about 1 to 2.4 MB, so 14 levels of U, V, T and VVEL come to about
63 MB an hour).

Vertical velocity is converted from HRRR's pressure tendency (omega, Pa/s) to m/s with the
hydrostatic relation w = -omega / (rho g), rho = p / (R_d T), using the hour's temperature and the
first hour's pressure. Winds are rotated from the Lambert grid axes to east and north. Everything is
regridded to the same regular longitude-latitude grid as the surface delivery.

A missing message or hour is an error, never a shorter record.
"""

import concurrent.futures
import logging
import os
import tempfile
from typing import Iterator

import numpy as np
import pandas as pd
import xarray as xr

from .hrrr import _fetch_s3_byte_ranges, _parse_idx, fs
from .hrrr_atmosphere import (
    HRRRRegridder,
    _read_message,
    cycle_for_valid_time,
    earth_relative_winds,
    target_axes,
)

logger = logging.getLogger(__name__)

HRRR_LEVELS_SCHEMA = "hrrr-levels-v1"
DEFAULT_LEVELS = 14
MAX_LEVELS = 50

R_DRY = 287.05  # J kg-1 K-1
G = 9.80665  # m s-2

# Fetched every hour, per level.
HOURLY = ("UGRD", "VGRD", "TMP", "VVEL")
# Fetched once (from the first hour), per level, and the ground height.
STATIC = ("HGT", "PRES")

RECORD_DIMS = {name: ("level", "lat", "lon") for name in ("u", "v", "w", "t")}

UNITS = {
    "u": ("m s-1", "eastward_wind"),
    "v": ("m s-1", "northward_wind"),
    "w": ("m s-1", "upward_air_velocity"),
    "t": ("K", "air_temperature"),
}


def s3_key(cycle: pd.Timestamp) -> str:
    return (
        f"noaa-hrrr-bdp-pds/hrrr.{cycle.strftime('%Y%m%d')}/conus/"
        f"hrrr.t{cycle.strftime('%H')}z.wrfnatf01.grib2"
    )


def level_pattern(name: str, level: int) -> str:
    """The .idx text of one variable on one hybrid level of a one-hour forecast."""
    return f":{name}:{level} hybrid level:1 hour fcst:"


SURFACE_HEIGHT_PATTERN = ":HGT:surface:1 hour fcst:"


def omega_to_w(omega, t, p):
    """Vertical velocity (m/s, upward positive) from omega (Pa/s), temperature (K), pressure (Pa)."""
    rho = np.asarray(p) / (R_DRY * np.asarray(t))
    return -np.asarray(omega) / (rho * G)


def _stack(regrid: HRRRRegridder, native: dict, name: str, levels) -> np.ndarray:
    """One variable on every level, regridded: `(level, lat, lon)`."""
    return np.stack([regrid(native[(name, k)].values) for k in levels])


def fetch_hour(
    valid_time: pd.Timestamp,
    workdir: str,
    n_levels: int,
    names: tuple[str, ...] = HOURLY,
    surface: bool = False,
    max_workers: int = 8,
) -> dict:
    """`{(name, level): DataArray}` valid at `valid_time` on HRRR's native grid, plus
    `("HGT", 0)` (the ground) when `surface`. Raises if any message is missing."""
    key = s3_key(cycle_for_valid_time(valid_time))
    try:
        with fs.open(key + ".idx", "r") as f:
            idx = f.read()
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"HRRR index missing for {valid_time}: s3://{key}.idx"
        ) from e
    jobs = [
        ((name, k), level_pattern(name, k))
        for name in names
        for k in range(1, n_levels + 1)
    ]
    if surface:
        jobs.append((("HGT", 0), SURFACE_HEIGHT_PATTERN))
    stamp = valid_time.strftime("%Y%m%d%H")

    def get(job):
        (name, k), pattern = job
        ranges = _parse_idx(idx, (pattern,))
        if len(ranges) != 1:
            raise FileNotFoundError(
                f"HRRR s3://{key}: expected one message for {pattern!r}, found {len(ranges)}"
            )
        path = os.path.join(workdir, f"{name}_{k}_{stamp}.grib2")
        _fetch_s3_byte_ranges(key, ranges, path)
        try:
            da = _read_message(path)
        finally:
            os.remove(path)
        vt = pd.Timestamp(da["valid_time"].values)
        if vt != valid_time:
            raise RuntimeError(
                f"HRRR {name} level {k} from s3://{key} is valid at {vt}, not {valid_time}"
            )
        return (name, k), da

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        return dict(ex.map(get, jobs))


def iter_levels(
    start_time: str,
    hours: int,
    bbox: list[float],
    n_levels: int = DEFAULT_LEVELS,
    resolution_deg: float = 0.03,
    margin_deg: float = 0.25,
) -> Iterator[tuple]:
    """Yield `("static", Dataset)`, then `("record", time, fields)` for every hour from
    start - 1 h to start + hours + 1 h (hours + 3 records, as the surface delivery). Fields are
    `u, v, w, t`, each `(level, lat, lon)`; the static Dataset holds the level heights `z`
    (geopotential metres above sea level), `zs` (the ground) and pressures `p`, from the first hour.
    """
    if not 1 <= n_levels <= MAX_LEVELS:
        raise ValueError(f"n_levels must be 1 to {MAX_LEVELS}, not {n_levels}")
    t0 = pd.Timestamp(start_time)
    if t0.tzinfo is not None:
        t0 = t0.tz_convert("UTC").tz_localize(None)
    times = pd.date_range(
        t0 - pd.Timedelta(hours=1), t0 + pd.Timedelta(hours=hours + 1), freq="1h"
    )
    lon_axis, lat_axis = target_axes(bbox, resolution_deg, margin_deg)
    levels = np.arange(1, n_levels + 1)

    regrid: HRRRRegridder | None = None
    p: np.ndarray | None = None
    with tempfile.TemporaryDirectory() as workdir:
        for i, t in enumerate(times):
            first = i == 0
            native = fetch_hour(
                t, workdir, n_levels, HOURLY + (STATIC if first else ()), surface=first
            )
            if first:
                u = native[("UGRD", 1)]
                a = u.attrs
                if not (
                    a.get("GRIB_gridType") == "lambert"
                    and a.get("GRIB_Latin1InDegrees") == a.get("GRIB_Latin2InDegrees")
                ):
                    raise ValueError(
                        "HRRR grid is not a tangent Lambert conformal projection: "
                        f"{a.get('GRIB_gridType')}, {a.get('GRIB_Latin1InDegrees')}, "
                        f"{a.get('GRIB_Latin2InDegrees')}"
                    )
                cone = float(np.sin(np.deg2rad(a["GRIB_Latin1InDegrees"])))
                lov = float(a["GRIB_LoVInDegrees"])
                grid_relative = int(a.get("GRIB_uvRelativeToGrid", 1)) == 1
                regrid = HRRRRegridder(
                    u["longitude"].values, u["latitude"].values, lon_axis, lat_axis
                )
                z = _stack(regrid, native, "HGT", levels)
                p = _stack(regrid, native, "PRES", levels)
                zs = regrid(native[("HGT", 0)].values)
                yield (
                    "static",
                    xr.Dataset(
                        {
                            "z": (("level", "lat", "lon"), z.astype(np.float32)),
                            "zs": (("lat", "lon"), zs.astype(np.float32)),
                            "p": (("level", "lat", "lon"), p.astype(np.float32)),
                        },
                        coords={"level": levels, "lat": lat_axis, "lon": lon_axis},
                        attrs={
                            "type": "HRRR lower atmosphere on native hybrid levels",
                            "source": "NOAA HRRR via noaa-hrrr-bdp-pds (wrfnatf01)",
                            "schema": HRRR_LEVELS_SCHEMA,
                            "requested_bbox": list(bbox),
                            "resolution_deg": resolution_deg,
                            "margin_deg": margin_deg,
                            "n_levels": n_levels,
                            "start_time": t0.strftime("%Y-%m-%dT%H:%M:%S"),
                            "first_cycle": cycle_for_valid_time(times[0]).strftime(
                                "%Y-%m-%dT%HZ"
                            ),
                            "last_cycle": cycle_for_valid_time(times[-1]).strftime(
                                "%Y-%m-%dT%HZ"
                            ),
                            "wind_frame": "earth-relative (rotated from Lambert grid axes)",
                            "level_geometry": "z (geopotential m above sea level) and p (Pa) from the first hour",
                            "vertical_velocity": "w = -omega / (rho g), rho = p / (R_d T), p from the first hour",
                        },
                    ),
                )
            assert regrid is not None and p is not None
            # Winds are rotated on the native grid (the angle depends on its longitude), then
            # regridded.
            us, vs = [], []
            for k in levels:
                uk, vk = native[("UGRD", k)].values, native[("VGRD", k)].values
                if grid_relative:
                    uk, vk = earth_relative_winds(uk, vk, regrid.native_lon, cone, lov)
                us.append(regrid(uk))
                vs.append(regrid(vk))
            tk = _stack(regrid, native, "TMP", levels)
            w = omega_to_w(_stack(regrid, native, "VVEL", levels), tk, p)
            yield (
                "record",
                t,
                {"u": np.stack(us), "v": np.stack(vs), "w": w, "t": tk},
            )


def fetch_hrrr_levels(start_time: str, hours: int, bbox: list[float], **kwargs):
    """`iter_levels` assembled into one in-memory Dataset."""
    static = None
    times, records = [], []
    for item in iter_levels(start_time, hours, bbox, **kwargs):
        if item[0] == "static":
            static = item[1]
        else:
            times.append(item[1])
            records.append(item[2])
    assert static is not None
    return static.assign(
        {
            k: (
                ("time", "level", "lat", "lon"),
                np.stack([r[k] for r in records]).astype(np.float32),
            )
            for k in RECORD_DIMS
        }
    ).assign_coords(time=times)
