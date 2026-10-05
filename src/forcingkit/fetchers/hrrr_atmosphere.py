"""HRRR as a prescribed atmosphere for an ocean model (schema `hrrr-atm-v1`).

Each hour t is taken from the one-hour forecast (f01) of the cycle initialised at t - 1 h, so every
record is a short forecast from the latest analysis, and the precipitation accumulation it carries
covers exactly the hour ending at t. Hours are chained across cycles, so any run length works.

Eight surface fields are read from `wrfsfcf01` by byte range: 10 m wind (rotated from HRRR's
Lambert-conformal grid axes to east and north), 2 m air temperature and specific humidity, surface
pressure, downwelling shortwave and longwave radiation (instantaneous at t), and the 1 h
accumulated precipitation, delivered as a rate centred on t. They are regridded to a regular
longitude-latitude grid, which is what NumericalEarth's atmosphere regridder accepts.

A missing message or hour is an error, never a shorter record.
"""

import concurrent.futures
import logging
import os
import tempfile
import warnings
from typing import Iterator

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import Delaunay

from .hrrr import _fetch_s3_byte_ranges, _parse_idx, fs
from .necofs import Barycentric

logger = logging.getLogger(__name__)

HRRR_ATM_SCHEMA = "hrrr-atm-v1"

# One message per field in wrfsfcf01; the leading and trailing colons anchor the whole field.
ATM_IDX_PATTERNS = {
    "u10": ":UGRD:10 m above ground:1 hour fcst:",
    "v10": ":VGRD:10 m above ground:1 hour fcst:",
    "t2m": ":TMP:2 m above ground:1 hour fcst:",
    "q2m": ":SPFH:2 m above ground:1 hour fcst:",
    "sp": ":PRES:surface:1 hour fcst:",
    "apcp": ":APCP:surface:0-1 hour acc fcst:",
    "dswrf": ":DSWRF:surface:1 hour fcst:",
    "dlwrf": ":DLWRF:surface:1 hour fcst:",
}

RECORD_DIMS = {
    name: ("lat", "lon")
    for name in ("u10", "v10", "t2m", "q2m", "sp", "dswrf", "dlwrf", "prate")
}

UNITS = {
    "u10": ("m s-1", "eastward_wind"),
    "v10": ("m s-1", "northward_wind"),
    "t2m": ("K", "air_temperature"),
    "q2m": ("kg kg-1", "specific_humidity"),
    "sp": ("Pa", "surface_air_pressure"),
    "dswrf": ("W m-2", "surface_downwelling_shortwave_flux_in_air"),
    "dlwrf": ("W m-2", "surface_downwelling_longwave_flux_in_air"),
    "prate": ("kg m-2 s-1", "precipitation_flux"),
}


def cycle_for_valid_time(t: pd.Timestamp) -> pd.Timestamp:
    """The cycle whose one-hour forecast is valid at `t`."""
    return pd.Timestamp(t) - pd.Timedelta(hours=1)


def s3_key(cycle: pd.Timestamp) -> str:
    return (
        f"noaa-hrrr-bdp-pds/hrrr.{cycle.strftime('%Y%m%d')}/conus/"
        f"hrrr.t{cycle.strftime('%H')}z.wrfsfcf01.grib2"
    )


def earth_relative_winds(u, v, lon, cone, lov):
    """Rotate grid-relative winds on a Lambert conformal grid to east and north.

    alpha = cone * (lon - lov), with cone = sin(standard parallel) for a tangent cone and both
    longitudes in degrees east in -180..180 (NCEP's convention, as in wgrib2).
    """
    lon = ((np.asarray(lon) + 180.0) % 360.0) - 180.0
    lov = ((lov + 180.0) % 360.0) - 180.0
    alpha = np.deg2rad(cone * (lon - lov))
    c, s = np.cos(alpha), np.sin(alpha)
    return c * u + s * v, -s * u + c * v


def _read_message(path: str) -> xr.DataArray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""})
    (name,) = list(ds.data_vars)
    return ds[name].load()


def fetch_hour(valid_time: pd.Timestamp, workdir: str) -> dict[str, xr.DataArray]:
    """The eight fields valid at `valid_time`, on HRRR's native grid. Raises if any is missing."""
    key = s3_key(cycle_for_valid_time(valid_time))
    try:
        with fs.open(key + ".idx", "r") as f:
            idx = f.read()
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"HRRR index missing for {valid_time}: s3://{key}.idx"
        ) from e
    fields = {}
    for name, pattern in ATM_IDX_PATTERNS.items():
        ranges = _parse_idx(idx, (pattern,))
        if len(ranges) != 1:
            raise FileNotFoundError(
                f"HRRR s3://{key}: expected one message for {pattern!r}, found {len(ranges)}"
            )
        path = os.path.join(workdir, f"{name}_{valid_time.strftime('%Y%m%d%H')}.grib2")
        _fetch_s3_byte_ranges(key, ranges, path)
        try:
            fields[name] = _read_message(path)
        finally:
            os.remove(path)
        vt = pd.Timestamp(fields[name]["valid_time"].values)
        if vt != valid_time:
            raise RuntimeError(
                f"HRRR {name} from s3://{key} is valid at {vt}, not {valid_time}"
            )
    return fields


class HRRRRegridder:
    """Native HRRR grid to a regular longitude-latitude grid, with weights built once."""

    def __init__(self, native_lon, native_lat, lon_axis, lat_axis):
        lon = ((np.asarray(native_lon) + 180.0) % 360.0) - 180.0
        lat = np.asarray(native_lat)
        # Native cells near the target, plus a margin of a few HRRR cells (3 km, about 0.03 deg).
        near = (
            (lon >= lon_axis[0] - 0.1)
            & (lon <= lon_axis[-1] + 0.1)
            & (lat >= lat_axis[0] - 0.1)
            & (lat <= lat_axis[-1] + 0.1)
        )
        if not near.any():
            raise ValueError("target grid lies outside the HRRR domain")
        self.near = near
        gx, gy = np.meshgrid(lon_axis, lat_axis)
        targets = np.column_stack((gx.ravel(), gy.ravel()))
        self.interp = Barycentric(
            Delaunay(np.column_stack((lon[near], lat[near]))), targets, gx.shape
        )
        self.native_lon = lon

    def __call__(self, field) -> np.ndarray:
        out = self.interp(np.asarray(field)[self.near])
        if np.isnan(out).any():
            raise RuntimeError(
                "regridded HRRR field has points outside the native subset"
            )
        return out


def target_axes(bbox, resolution_deg: float, margin_deg: float):
    lon_axis = np.arange(
        bbox[0] - margin_deg, bbox[2] + margin_deg + resolution_deg / 2, resolution_deg
    )
    lat_axis = np.arange(
        bbox[1] - margin_deg, bbox[3] + margin_deg + resolution_deg / 2, resolution_deg
    )
    return lon_axis, lat_axis


def iter_atmosphere(
    start_time: str,
    hours: int,
    bbox: list[float],
    resolution_deg: float = 0.03,
    margin_deg: float = 0.25,
    max_workers: int = 6,
) -> Iterator[tuple]:
    """Yield `("static", Dataset)`, then `("record", time, fields)` for every hour from
    start - 1 h to start + hours + 1 h, so the series brackets the run with a record to spare at
    each end. `prate` at t is the mean of the hours ending at t and at t + 1 h (a rate centred on
    t), so one more hour is fetched past the last record.
    """
    t0 = pd.Timestamp(start_time)
    if t0.tzinfo is not None:
        t0 = t0.tz_convert("UTC").tz_localize(None)
    record_times = pd.date_range(
        t0 - pd.Timedelta(hours=1), t0 + pd.Timedelta(hours=hours + 1), freq="1h"
    )
    fetch_times = record_times.append(
        pd.DatetimeIndex([record_times[-1] + pd.Timedelta(hours=1)])
    )
    lon_axis, lat_axis = target_axes(bbox, resolution_deg, margin_deg)

    regrid = None
    previous = (
        None  # (time, regridded fields) waiting for the next hour's precipitation
    )
    with (
        tempfile.TemporaryDirectory() as workdir,
        concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor,
    ):
        for b in range(0, len(fetch_times), max_workers):
            batch = list(fetch_times[b : b + max_workers])
            for t, native in zip(
                batch, executor.map(lambda t: fetch_hour(t, workdir), batch)
            ):
                u = native["u10"]
                if regrid is None:
                    a = u.attrs
                    if not (
                        a.get("GRIB_gridType") == "lambert"
                        and a.get("GRIB_Latin1InDegrees")
                        == a.get("GRIB_Latin2InDegrees")
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
                    yield (
                        "static",
                        xr.Dataset(
                            coords={"lat": lat_axis, "lon": lon_axis},
                            attrs={
                                "type": "HRRR prescribed atmosphere",
                                "source": "NOAA HRRR via noaa-hrrr-bdp-pds (wrfsfcf01)",
                                "schema": HRRR_ATM_SCHEMA,
                                "requested_bbox": list(bbox),
                                "resolution_deg": resolution_deg,
                                "margin_deg": margin_deg,
                                "start_time": t0.strftime("%Y-%m-%dT%H:%M:%S"),
                                "first_cycle": cycle_for_valid_time(
                                    fetch_times[0]
                                ).strftime("%Y-%m-%dT%HZ"),
                                "last_cycle": cycle_for_valid_time(
                                    fetch_times[-1]
                                ).strftime("%Y-%m-%dT%HZ"),
                                "wind_frame": "earth-relative (rotated from Lambert grid axes)",
                                "radiation": "instantaneous at the record time",
                                "precipitation": "rate centred on the record time: mean of the 1 h accumulations ending at t and t + 1 h",
                            },
                        ),
                    )
                ue, ve = u.values, native["v10"].values
                if grid_relative:
                    ue, ve = earth_relative_winds(ue, ve, regrid.native_lon, cone, lov)
                fields = {
                    "u10": regrid(ue),
                    "v10": regrid(ve),
                    "t2m": regrid(native["t2m"].values),
                    "q2m": regrid(native["q2m"].values),
                    "sp": regrid(native["sp"].values),
                    "dswrf": regrid(native["dswrf"].values),
                    "dlwrf": regrid(native["dlwrf"].values),
                    "apcp": regrid(native["apcp"].values),
                }
                if previous is not None:
                    pt, pf = previous
                    pf["prate"] = (pf.pop("apcp") + fields["apcp"]) / 7200.0
                    yield ("record", pt, pf)
                previous = (t, fields)


def fetch_hrrr_atmosphere(start_time: str, hours: int, bbox: list[float], **kwargs):
    """`iter_atmosphere` assembled into one in-memory Dataset."""
    static = None
    times, records = [], []
    for item in iter_atmosphere(start_time, hours, bbox, **kwargs):
        if item[0] == "static":
            static = item[1]
        else:
            times.append(item[1])
            records.append(item[2])
    assert static is not None
    return static.assign(
        {
            k: (
                ("time", "lat", "lon"),
                np.stack([r[k] for r in records]).astype(np.float32),
            )
            for k in RECORD_DIMS
        }
    ).assign_coords(time=times)
