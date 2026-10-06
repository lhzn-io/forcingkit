import xarray as xr
import pandas as pd
import logging
from dataclasses import dataclass
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)

_TDS = "https://tds.hycom.org/thredds/dodsC"

# HYCOM variable names and their canonical forcingkit names; nothing else is downloaded.
_RENAME = {
    "water_u": "u",
    "water_v": "v",
    "water_temp": "temp",
    "salinity": "salt",
    "surf_el": "zeta",
}


@dataclass(frozen=True)
class _Segment:
    """A HYCOM experiment serving times from `start` until the next segment starts."""

    start: pd.Timestamp
    name: str
    # One aggregated dataset, or one dataset per variable to be merged.
    urls: tuple[str, ...]


def _glb(path: str, start: str) -> _Segment:
    return _Segment(pd.Timestamp(start), path, (f"{_TDS}/{path}",))


# Ordered by start; the last segment is open-ended. Facts below were read from tds.hycom.org on
# 2026-10-06 (docs/source/hycom.rst has the full table).
#
# - Switch times. Where experiments overlap, the switch is at the newer one's first time step
#   (12:00 for most). GLBy0.08/expt_93.0 starts at 2018-12-04 12:00, not 00:00, and is kept to
#   its last step (2024-09-05 09:00) although ESPC-D-V02 starts on 2024-08-10 12:00.
#   GLBv0.08/expt_53.X is listed from 1994-01-01 and ends 2015-12-31 09:00. The path
#   GLBy0.08/expt_53.X answers HTTP 200 with an empty body; the data is under GLBv0.08.
# - Grids. GLBv0.08 has 0.08 degree latitude spacing between 40S and 40N and 0.04 degree
#   poleward; GLBy0.08 and ESPC-D-V02 have 0.04 degree everywhere. Both are 0.08 degree in
#   longitude, 80S to 90N, with the same 40 depths (0 to 5000 m). A window spanning the two is
#   regridded onto the grid of its first segment (an identity north of 40N).
# - Longitudes. The 5x-series experiments use -180..180, the 9x-series and ESPC-D-V02 use
#   0..360; `_fetch_hycom_data` detects this per dataset.
# - Time. All use "hours since 2000-01-01 00:00:00", 3-hourly, with occasional gaps of up to
#   51 h that are passed through unfilled. GLBv0.08/expt_93.0 steps back once.
# - ESPC-D-V02 serves u, v, temperature and salinity (3-hourly) and surface elevation (hourly)
#   as separate datasets, merged on the shared 3-hourly times.
_SEGMENTS: tuple[_Segment, ...] = (
    _glb("GLBv0.08/expt_53.X", "1994-01-01"),
    _glb("GLBv0.08/expt_56.3", "2015-12-31"),
    _glb("GLBv0.08/expt_57.2", "2016-05-01 12:00"),
    _glb("GLBv0.08/expt_92.8", "2017-02-01 12:00"),
    _glb("GLBv0.08/expt_57.7", "2017-06-01 12:00"),
    _glb("GLBv0.08/expt_92.9", "2017-10-01 12:00"),
    _glb("GLBv0.08/expt_93.0", "2018-01-01 12:00"),
    _glb("GLBy0.08/expt_93.0", "2018-12-04 12:00"),
    _Segment(
        pd.Timestamp("2024-09-05"),
        "ESPC-D-V02",
        tuple(
            f"{_TDS}/ESPC-D-V02/{var}" for var in ("u3z", "v3z", "t3z", "s3z", "ssh")
        ),
    ),
)


def get_metadata() -> dict:
    return {
        "id": "hycom",
        "name": "HYCOM Global",
        "resolution_approx_m": 9000.0,
        "type_desc": "Global regular grid",
        "domain_bbox": [-180.0, -90.0, 180.0, 90.0],
    }


def supports_bbox(bbox: list[float]) -> bool:
    return True


def _get_hycom_segment(target_dt: pd.Timestamp) -> Optional[_Segment]:
    """The HYCOM segment serving `target_dt`, or None before the first one."""
    current = None
    for seg in _SEGMENTS:
        if target_dt < seg.start:
            break
        current = seg
    return current


def _split_window(
    start_dt: pd.Timestamp, end_dt: pd.Timestamp
) -> list[tuple[_Segment, pd.Timestamp, pd.Timestamp]]:
    """Split [start_dt, end_dt] into (segment, start, end) pieces at segment boundaries.

    Empty if `start_dt` precedes the first segment.
    """
    seg = _get_hycom_segment(start_dt)
    if seg is None:
        return []
    pieces = []
    piece_start = start_dt
    for nxt in _SEGMENTS[_SEGMENTS.index(seg) + 1 :]:
        if nxt.start >= end_dt:
            break
        pieces.append((seg, piece_start, nxt.start))
        seg, piece_start = nxt, nxt.start
    pieces.append((seg, piece_start, end_dt))
    return pieces


def _normalize_lons(lons: np.ndarray) -> np.ndarray:
    """Convert -180/180 to 0/360."""
    return np.where(lons < 0, lons + 360, lons)


def fetch_hycom_boundary_conditions(
    start_date: str,
    duration_hours: int,
    bbox: list[float],
) -> Optional[xr.Dataset]:
    """
    Fetches continuous historical 3D ocean boundary conditions from HYCOM.
    """
    start_dt = pd.to_datetime(start_date).tz_localize(None)
    end_dt = start_dt + pd.Timedelta(hours=duration_hours)

    pieces = _split_window(start_dt, end_dt)
    if not pieces:
        logger.error(
            f"HYCOM has no data before {_SEGMENTS[0].start}; requested start {start_dt}."
        )
        return None

    if len(pieces) > 1:
        names = ", ".join(seg.name for seg, _, _ in pieces)
        logger.info(f"Hindcast spans HYCOM experiments; stitching {names}...")

    parts = []
    for seg, piece_start, piece_end in pieces:
        part = _fetch_segment(seg, piece_start, piece_end, bbox)
        if part is None:
            # A window with a hole is not a usable boundary condition; let the caller fail.
            logger.error(
                f"HYCOM segment {seg.name} returned nothing; dropping the window."
            )
            return None
        parts.append(part)

    if len(parts) == 1:
        return parts[0]
    return _stitch(parts)


def _fetch_segment(
    seg: _Segment,
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    bbox: list[float],
) -> Optional[xr.Dataset]:
    """Fetch one segment, merging its per-variable datasets on their shared times."""
    parts = []
    for url in seg.urls:
        part = _fetch_hycom_data(start_dt, end_dt, bbox, url)
        if part is None:
            return None
        parts.append(part)
    if len(parts) == 1:
        return parts[0]
    # The inner join keeps the 3-hourly times common to all variables (surface elevation is
    # hourly).
    return xr.merge(parts, join="inner")


def _stitch(parts: list[xr.Dataset]) -> xr.Dataset:
    """Concatenate consecutive segments in time on the grid of the first one."""
    first = parts[0]
    lat_var = "lat" if "lat" in first.coords else "latitude"
    lon_var = "lon" if "lon" in first.coords else "longitude"
    aligned = [first]
    for part in parts[1:]:
        if not (
            np.array_equal(part[lat_var].values, first[lat_var].values)
            and np.array_equal(part[lon_var].values, first[lon_var].values)
        ):
            part = part.interp({lat_var: first[lat_var], lon_var: first[lon_var]})
        aligned.append(part)
    ds = xr.concat(aligned, dim="time", data_vars="all")
    # Each piece is fetched inclusive of the boundary time, so the step at a switch appears
    # twice; keep the newer experiment's copy.
    return ds.isel(time=~ds.indexes["time"].duplicated(keep="last"))


def _fetch_hycom_data(
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    bbox: list[float],
    dataset_url: str,
    is_ic: bool = False,
) -> Optional[xr.Dataset]:
    min_lon, min_lat, max_lon, max_lat = bbox

    logger.info(f"Fetching HYCOM data from {dataset_url} for {start_dt} to {end_dt}")

    try:
        dap_url = dataset_url.replace("https://", "dap2://").replace(
            "http://", "dap2://"
        )
        ds = xr.open_dataset(dap_url, engine="pydap", decode_times=False)

        # HYCOM time axis is "hours since 2000-01-01 00:00:00"
        epoch = pd.Timestamp("2000-01-01 00:00:00")
        start_hours = (start_dt - epoch).total_seconds() / 3600.0
        end_hours = (end_dt - epoch).total_seconds() / 3600.0

        time_var = "time"
        if is_ic:
            ds_subset = ds.sel({time_var: start_hours}, method="nearest")
        else:
            # Add a small buffer to ensure we capture the boundary times. Slice by position:
            # GLBv0.08/expt_93.0 steps back once (2018-06-21 09:00 to 06:00), so its time
            # index is not monotonic and label slicing fails.
            times = ds[time_var].values
            hit = np.flatnonzero(
                (times >= start_hours - 0.1) & (times <= end_hours + 0.1)
            )
            if hit.size == 0:
                logger.error(
                    f"Requested time range out of bounds for HYCOM. Required: {start_hours} to {end_hours}."
                )
                return None
            ds_subset = ds.isel({time_var: slice(hit[0], hit[-1] + 1)})

        lon_var = "lon" if "lon" in ds.coords else "longitude"
        lat_var = "lat" if "lat" in ds.coords else "latitude"

        ds_subset = ds_subset.sel({lat_var: slice(min_lat - 0.1, max_lat + 0.1)})

        # The longitude convention varies by experiment, not by grid: the 5x-series GLBv0.08
        # experiments (53.X, 56.3, 57.2, 57.7) use -180..180, the 9x-series (92.8, 92.9,
        # 93.0) and ESPC-D-V02 use 0..360. Detect it per dataset and slice in the
        # dataset's own convention; the output is always 0..360.
        lon_hi = 360.0 if float(ds[lon_var].min()) >= 0 else 180.0
        lon_lo = lon_hi - 360.0
        ds_min_lon = (min_lon - lon_lo) % 360 + lon_lo
        ds_max_lon = (max_lon - lon_lo) % 360 + lon_lo

        # Handle wrapping if the bbox crosses the dataset's longitude seam
        if ds_min_lon > ds_max_lon:
            logger.info(
                f"BBox crosses the {lon_lo:g}/{lon_hi:g} seam, performing dual-slice and concat."
            )
            part1 = ds_subset.sel({lon_var: slice(ds_min_lon - 0.1, lon_hi)})
            part2 = ds_subset.sel({lon_var: slice(lon_lo, ds_max_lon + 0.1)})
            ds_subset = xr.concat([part1, part2], dim=lon_var, data_vars="all")
        else:
            ds_subset = ds_subset.sel(
                {lon_var: slice(ds_min_lon - 0.1, ds_max_lon + 0.1)}
            )

        # Download only the fields forcingkit uses. OPeNDAP fetches each variable in its own
        # request, and the GLB experiments also carry `tau` and four `*_bottom` fields, so this
        # halves the round trips to a server that is often slow. It also gives every
        # experiment the same variable set, which stitching requires.
        ds_subset = ds_subset[[k for k in _RENAME if k in ds_subset.data_vars]]

        logger.info("Executing OPeNDAP download for HYCOM subset...")
        ds_subset = ds_subset.compute()
        ds_subset = ds_subset.assign_coords({lon_var: ds_subset[lon_var] % 360})
        if not is_ic:
            # Restore a strictly increasing time axis within the requested range.
            ds_subset = ds_subset.sortby(time_var)
            ds_subset = ds_subset.isel(
                {time_var: ~ds_subset.indexes[time_var].duplicated()}
            )
            ds_subset = ds_subset.sel(
                {time_var: slice(start_hours - 0.1, end_hours + 0.1)}
            )

        # Rename variables to canonical names if necessary
        actual_rename = {k: v for k, v in _RENAME.items() if k in ds_subset.data_vars}
        ds_subset = ds_subset.rename(actual_rename)

        # Explicitly decode the raw float time coordinate to pandas DatetimeIndex lengths
        if time_var in ds_subset.coords:
            ds_subset[time_var] = epoch + pd.to_timedelta(
                ds_subset[time_var].values, unit="h"
            )

        return ds_subset

    except Exception as e:
        logger.error(f"Failed to fetch from HYCOM ({dataset_url}): {e}")
        return None
