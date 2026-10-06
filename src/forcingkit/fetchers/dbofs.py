"""
DBOFS (NOAA Delaware Bay Operational Forecast System) fetcher.

Provides Open Boundary Conditions (OBC) from a ROMS structured grid. Covers Delaware Bay and
the adjacent Mid-Atlantic Bight continental shelf, including offshore NJ south of 40°N. Recent
data comes from the CO-OPS FMRC aggregation; older data is the chain of hourly nowcast files,
from AWS S3 (per-day layout, from 2024-11-19) or NCEI THREDDS.

Domain: [-75.875, 37.810, -73.264, 40.206], finer than the NECOFS 132°² footprint, so DBOFS
wins the dispatcher ranking for any bbox fully contained in this region.
"""

import concurrent.futures
import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from forcingkit import settings

logger = logging.getLogger(__name__)

FMRC_URL = (
    "https://opendap.co-ops.nos.noaa.gov/thredds/dodsC/"
    "DBOFS/fmrc/Aggregated_7_day_DBOFS_Fields_Forecast_best.ncd"
)
NCEI_BASE = "https://www.ncei.noaa.gov/thredds/dodsC/model-dbofs-files"
AWS_BASE = "s3://noaa-nos-ofs-pds/dbofs/netcdf"

# AWS holds one directory per day from 2024-11-19 (that day has only the 18 UTC cycle). Before
# that, AWS has flat month directories with gaps, so those dates are read from NCEI instead.
AWS_PER_DAY_FROM = pd.Timestamp("2024-11-19")
# NCEI file names changed with the 2024-09-09 cycles.
NCEI_NEW_NAMES_FROM = pd.Timestamp("2024-09-09")


def get_metadata() -> dict:
    """Returns metadata for the DBOFS system."""
    return {
        "id": "dbofs",
        "name": "NOAA DBOFS (Delaware Bay / Offshore NJ)",
        "resolution_approx_m": 100.0,
        "type_desc": "Structured curvilinear ROMS grid",
        "domain_bbox": [-75.875, 37.810, -73.264, 40.206],
    }


def supports_bbox(bbox: list[float]) -> bool:
    """Check if the requested bbox is fully within the DBOFS active mask domain."""
    min_lon, min_lat, max_lon, max_lat = bbox
    domain_bbox = get_metadata()["domain_bbox"]
    domain_min_lon, domain_min_lat, domain_max_lon, domain_max_lat = domain_bbox

    # 1. Check strict rectangle bounds
    if not (
        min_lon >= domain_min_lon
        and max_lon <= domain_max_lon
        and min_lat >= domain_min_lat
        and max_lat <= domain_max_lat
    ):
        return False

    # 2. Check curvilinear offshore empty corner (e.g. Barnegat Light Area)
    # The active mask drops out completely for regions east of -74.0 and north of 39.03
    if min_lon > -74.0 and min_lat > 39.03:
        return False

    return True


def _to_dap_url(url: str) -> str:
    """Convert http(s) URL to pydap dap2:// scheme."""
    return url.replace("https://", "dap2://").replace("http://", "dap2://")


def _get_dbofs_url(target_dt: pd.Timestamp) -> Tuple[str, Optional[str]]:
    """
    Resolve the DBOFS access mode for a start time.

    Returns ("fmrc", url) for start times up to 6 days old, and ("archive", None) otherwise;
    archive file URLs come from `_nowcast_file_urls`.
    """
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    age_days = (now - target_dt).total_seconds() / 86400
    if 0 <= age_days <= 6:
        return ("fmrc", FMRC_URL)
    return ("archive", None)


def _nowcast_file(hour_dt: pd.Timestamp) -> Tuple[pd.Timestamp, int]:
    """
    Return (cycle, n) such that nowcast file `n00{n}` of `cycle` holds the record at `hour_dt`.

    Cycles run at 00, 06, 12 and 18 UTC, and file n001 to n006 of cycle tCCz hold hours CC-5
    to CC, so 00 UTC is n006 of t00z and 01 UTC is n001 of t06z.
    """
    hour_dt = hour_dt.floor("h")
    cycle = hour_dt.ceil("6h")
    n = 6 - int((cycle - hour_dt) / pd.Timedelta(hours=1))
    return cycle, n


def _nowcast_file_urls(hour_dt: pd.Timestamp) -> list[str]:
    """Candidate URLs for the nowcast file holding `hour_dt`, in order of preference."""
    cycle, n = _nowcast_file(hour_dt)
    day_dir, month_dir = cycle.strftime("%Y/%m/%d"), cycle.strftime("%Y/%m")
    ymd, cc = cycle.strftime("%Y%m%d"), cycle.strftime("%H")
    new_name = f"dbofs.t{cc}z.{ymd}.fields.n{n:03d}.nc"

    urls = []
    if cycle.normalize() >= AWS_PER_DAY_FROM:
        urls.append(f"{AWS_BASE}/{day_dir}/{new_name}")
    if cycle.normalize() >= NCEI_NEW_NAMES_FROM:
        urls.append(f"{NCEI_BASE}/{month_dir}/{new_name}")
    else:
        old_name = f"nos.dbofs.fields.n{n:03d}.{ymd}.t{cc}z.nc"
        urls.append(f"{NCEI_BASE}/{month_dir}/{old_name}")
    return urls


def _open_archive_file(url: str) -> xr.Dataset:
    """Open one DBOFS file lazily, so only the subset the request needs is read."""
    if url.startswith("s3://"):
        import fsspec  # type: ignore[import-untyped]

        # The file object must stay open while the dataset is lazy; xarray closes it with the
        # dataset.
        return xr.open_dataset(
            fsspec.open(url, "rb", anon=True).open(), engine="h5netcdf"
        )
    return xr.open_dataset(_to_dap_url(url), engine="pydap")


def _open_first_available(urls: list[str]) -> Optional[xr.Dataset]:
    for url in urls:
        try:
            return _open_archive_file(url)
        except Exception as e:
            logger.info(f"DBOFS file unavailable at {url}: {e}")
    return None


def _open_dbofs_dataset(
    access_mode: str,
    url_or_pattern: Optional[str],
    target_dt: pd.Timestamp,
    end_dt: Optional[pd.Timestamp] = None,
) -> Optional[xr.Dataset]:
    """
    Open DBOFS data covering [target_dt, end_dt] (FMRC or archive).

    Archive reads return None unless every hourly nowcast file in the window is found, so a
    gap falls back to the next donor rather than shortening the store.
    """
    if end_dt is None:
        end_dt = target_dt + pd.Timedelta(hours=1)
    try:
        if access_mode == "fmrc":
            logger.info(f"Opening DBOFS FMRC aggregation: {url_or_pattern}")
            assert url_or_pattern is not None
            ds = xr.open_dataset(_to_dap_url(url_or_pattern), engine="pydap")

            time_var = "time" if "time" in ds.coords else "ocean_time"
            ds = ds.sortby(time_var)

            ds_t = ds.sel({time_var: slice(target_dt, end_dt)})
            if ds_t.sizes[time_var] == 0:
                logger.warning(
                    "DBOFS FMRC: exact time range empty, fell back to nearest."
                )
                return ds.sel({time_var: target_dt}, method="nearest").expand_dims(
                    time_var
                )
            return ds_t

        if access_mode != "archive":
            logger.error(f"Unknown access_mode: {access_mode}")
            return None

        hours = list(pd.date_range(target_dt.floor("h"), end_dt.ceil("h"), freq="h"))
        logger.info(
            f"Opening {len(hours)} DBOFS nowcast files for {target_dt} to {end_dt}"
        )
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=settings.max_workers()
        ) as executor:
            opened = list(
                executor.map(
                    lambda h: _open_first_available(_nowcast_file_urls(h)), hours
                )
            )

        missing = [h for h, ds in zip(hours, opened) if ds is None]
        if missing:
            logger.error(
                f"DBOFS nowcast files missing for hours {[str(h) for h in missing]}"
            )
            return None

        datasets = [ds for ds in opened if ds is not None]
        time_var = "ocean_time" if "ocean_time" in datasets[0].dims else "time"
        return xr.concat(
            datasets,
            dim=time_var,
            data_vars="minimal",
            coords="minimal",
            compat="override",
            join="override",
        )

    except Exception as e:
        logger.error(f"Failed to open DBOFS dataset ({access_mode}): {e}")
        return None


_VAR_CANDIDATES: dict[str, list[str]] = {
    "u": ["u", "water_u", "u_eastward"],
    "v": ["v", "water_v", "v_northward"],
    "temp": ["temp", "water_temp", "temperature", "sea_water_temperature"],
    "salt": ["salt", "salinity", "sea_water_salinity"],
    "zeta": ["zeta", "sea_surface_height", "ssh"],
}


def _resolve_var(ds: xr.Dataset, role: str) -> str:
    """Return the first candidate name for *role* that exists in *ds*."""
    for name in _VAR_CANDIDATES[role]:
        if name in ds:
            return name
    raise KeyError(
        f"No variable found for role '{role}'. "
        f"Tried: {_VAR_CANDIDATES[role]}. Available: {list(ds.data_vars)}"
    )


# ROMS writes missing values as 1e37. The NCEI OPeNDAP path (pydap) does not always decode
# them, so treat anything this large as missing.
_FILL_THRESHOLD = 1e30


def _fill_to_nan(values: np.ndarray) -> np.ndarray:
    """Return `values` as Float32 with ROMS fill values replaced by NaN."""
    out = np.asarray(values, dtype=np.float32)
    return np.where(np.abs(out) < _FILL_THRESHOLD, out, np.float32(np.nan))


def _c_grid_to_rho(
    u_raw: np.ndarray,
    v_raw: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolate Arakawa C-grid u,v face values to rho-points by averaging
    adjacent pairs. Output shape always matches input shape (boundary filled
    by copy-edge extrapolation).
    """
    u_rho = np.empty_like(u_raw)
    u_rho[..., :-1] = 0.5 * (u_raw[..., :-1] + u_raw[..., 1:])
    u_rho[..., -1] = u_raw[..., -1]

    v_rho = np.empty_like(v_raw)
    v_rho[..., :-1, :] = 0.5 * (v_raw[..., :-1, :] + v_raw[..., 1:, :])
    v_rho[..., -1, :] = v_raw[..., -1, :]

    return u_rho, v_rho


def fetch_dbofs_boundary_conditions(
    start_date: str, duration_hours: int, bbox: list[float]
) -> Optional[xr.Dataset]:
    """
    Fetch 4D Ocean State (u, v) from NOAA DBOFS over a time range for OBC.

    Output dimensions: (time, depth, eta, xi) matching the standard OBC contract.

    Args:
        start_date: ISO format datetime string
        duration_hours: Duration of the boundary condition period
        bbox: [min_lon, min_lat, max_lon, max_lat]

    Returns:
        xr.Dataset with dims (time, depth, eta, xi) or None if fetch fails
    """
    min_lon, min_lat, max_lon, max_lat = bbox

    if not supports_bbox(bbox):
        logger.info(f"Bounding box {bbox} outside DBOFS domain.")
        return None

    target_dt = pd.to_datetime(start_date)
    if target_dt.tzinfo is not None:
        target_dt = target_dt.tz_convert("UTC").tz_localize(None)

    end_dt = target_dt + pd.Timedelta(hours=duration_hours)

    access_mode, url_or_pattern = _get_dbofs_url(target_dt)
    logger.info(f"Attempting DBOFS OBC ({duration_hours}h) from {access_mode.upper()}")

    ds_t = _open_dbofs_dataset(access_mode, url_or_pattern, target_dt, end_dt)
    if ds_t is None:
        return None

    try:
        lon_var = "lon_rho" if "lon_rho" in ds_t else "lon"
        lat_var = "lat_rho" if "lat_rho" in ds_t else "lat"

        lon_arr = ds_t[lon_var].values
        lat_arr = ds_t[lat_var].values

        mask_var = (
            "mask_rho" if "mask_rho" in ds_t else ("mask" if "mask" in ds_t else None)
        )
        mask_arr = ds_t[mask_var].values if mask_var else np.ones_like(lon_arr)

        valid_indices = np.where(
            (lon_arr >= min_lon)
            & (lon_arr <= max_lon)
            & (lat_arr >= min_lat)
            & (lat_arr <= max_lat)
            & (mask_arr == 1)
        )

        if len(valid_indices[0]) == 0:
            logger.warning("No valid DBOFS ocean points found in bounding box.")
            return None

        eta_min, eta_max = int(np.min(valid_indices[0])), int(np.max(valid_indices[0]))
        xi_min, xi_max = int(np.min(valid_indices[1])), int(np.max(valid_indices[1]))

        # Buffer slightly for interpolation
        eta_min = max(0, eta_min - 2)
        eta_max = min(lon_arr.shape[0] - 1, eta_max + 2)
        xi_min = max(0, xi_min - 2)
        xi_max = min(lon_arr.shape[1] - 1, xi_max + 2)

        slice_dict = {}
        for d in list(ds_t.dims):
            if "eta" in str(d):
                c_max = min(eta_max + 1, ds_t.sizes[d])
                slice_dict[str(d)] = slice(eta_min, c_max)
            elif "xi" in str(d):
                c_max = min(xi_max + 1, ds_t.sizes[d])
                slice_dict[str(d)] = slice(xi_min, c_max)

        ds_sub = ds_t.isel(slice_dict)  # type: ignore

        # Check if the bounding box yielded zero valid water points
        if any(size == 0 for size in ds_sub.sizes.values()):
            logger.warning(
                "No valid DBOFS ocean points found in bounding box (size is 0)."
            )
            return None

        logger.info("Executing OPeNDAP download for DBOFS OBC subset...")
        ds_sub = ds_sub.compute()

        u_var = _resolve_var(ds_sub, "u")
        v_var = _resolve_var(ds_sub, "v")
        logger.info(f"DBOFS OBC variable mapping: u={u_var}, v={v_var}")

        all_dims = list(ds_sub[u_var].dims)
        time_var = "time" if "time" in ds_sub.coords else "ocean_time"
        sigma_dim = None
        for candidate in ["s_rho", "sigma", "depth", "siglay"]:
            if candidate in all_dims:
                sigma_dim = candidate
                break

        if sigma_dim is None:
            logger.error(
                f"No recognized sigma dimension in DBOFS OBC. Dims: {all_dims}"
            )
            return None

        spatial_dims = [d for d in all_dims if d != time_var and d != sigma_dim]
        if len(spatial_dims) < 2:
            logger.error(f"Unexpected spatial dims after subset: {spatial_dims}")
            return None

        u_rho, v_rho = _c_grid_to_rho(
            _fill_to_nan(ds_sub[u_var].values),
            _fill_to_nan(ds_sub[v_var].values),
        )

        n_sigma = u_rho.shape[1]
        depths = np.linspace(-50, 0, n_sigma).astype(np.float32)
        out_times = ds_sub[time_var].values
        n_eta = u_rho.shape[2]
        n_xi = u_rho.shape[3]

        ds_out = xr.Dataset(
            data_vars={
                "u": (("time", "depth", "eta", "xi"), u_rho),
                "v": (("time", "depth", "eta", "xi"), v_rho),
            },
            coords={
                "time": out_times,
                "depth": depths,
                "eta": np.arange(n_eta, dtype=np.float32),
                "xi": np.arange(n_xi, dtype=np.float32),
            },
            attrs={"type": "NOAA DBOFS OBC", "source": "NOAA CO-OPS"},
        )

        logger.info(
            f"Successfully processed DBOFS OBC. "
            f"Shape: u{ds_out['u'].shape}, {len(out_times)} time steps"
        )
        return ds_out

    except Exception as e:
        logger.error(f"Failed to process DBOFS OBC data: {e}")
        return None
