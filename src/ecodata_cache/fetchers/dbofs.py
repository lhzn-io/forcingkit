"""
DBOFS (NOAA Delaware Bay Operational Forecast System) fetcher.

Provides Initial Conditions (IC) and Open Boundary Conditions (OBC) from a
ROMS structured grid via OPeNDAP. Covers Delaware Bay and the adjacent
Mid-Atlantic Bight continental shelf, including offshore NJ south of 40°N.

Domain: [-76.5, 37.5, -73.0, 40.0] - domain area 8.75°², finer than the
NECOFS/MARACOOS 132°² footprint, so DBOFS wins the dispatcher ranking for
any bbox fully contained in this region.
"""

import numpy as np
import xarray as xr
import pandas as pd
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Abort NCEI file enumeration after this many consecutive open failures.
_NCEI_CONSECUTIVE_FAIL_LIMIT = 3


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


def _get_dbofs_url(target_dt: pd.Timestamp) -> Tuple[str, str]:
    """
    Resolve DBOFS data access URL and mode.

    Returns (access_mode, url_or_pattern) where access_mode is one of:
      - "fmrc": single FMRC aggregated OPeNDAP URL (recent data, < 31 days)
      - "ncei": NCEI THREDDS file pattern (historical data, > 31 days)
    """
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    age_days = (now - target_dt).total_seconds() / 86400

    if 0 <= age_days <= 6:
        fmrc_url = (
            "https://opendap.co-ops.nos.noaa.gov/thredds/dodsC/"
            "DBOFS/fmrc/Aggregated_7_day_DBOFS_Fields_Forecast_best.ncd"
        )
        return ("fmrc", fmrc_url)

    # Historical: AWS S3 or NCEI file-per-hour. Naming convention changed 2024-09-09
    # and the archive migrated to AWS S3 starting 2024-01-01.
    if target_dt >= pd.Timestamp("2024-01-01"):
        pattern = (
            "s3://noaa-nos-ofs-pds/dbofs/netcdf/"
            "{yyyy}/{mm}/{dd}/dbofs.t{cc}z.{yyyymmdd}.fields.{type}{hhh:03d}.nc"
        )
        return ("aws_s3", pattern)
    else:
        pattern = (
            "https://www.ncei.noaa.gov/thredds/dodsC/model-dbofs-files/"
            "{yyyy}/{mm}/nos.dbofs.fields.{type}{hhh:03d}.{yyyymmdd}.t{cc}z.nc"
        )
        return ("ncei", pattern)


def _enumerate_ncei_dbofs_files(
    pattern: str, start_dt: pd.Timestamp, end_dt: pd.Timestamp
) -> list[str]:
    """
    Enumerate NCEI DBOFS file URLs for a time range.

    Picks the best 6-hourly cycle (00/06/12/18Z) and builds per-hour file URLs.
    """
    cycle_hours = [0, 6, 12, 18]
    cycle_before = None
    for ch in reversed(cycle_hours):
        test_dt = start_dt.replace(hour=ch, minute=0, second=0, microsecond=0)
        if test_dt <= start_dt:
            cycle_before = test_dt
            break

    if cycle_before is None:
        cycle_before = (start_dt - pd.Timedelta(days=1)).replace(
            hour=18, minute=0, second=0, microsecond=0
        )

    files = []
    current_dt = cycle_before
    current_cycle_dt = cycle_before

    while current_dt <= end_dt:
        if (current_dt - current_cycle_dt).total_seconds() >= 6 * 3600:
            current_cycle_dt = current_dt.replace(minute=0, second=0, microsecond=0)
            cycle_hour = (current_cycle_dt.hour // 6) * 6
            current_cycle_dt = current_cycle_dt.replace(hour=cycle_hour)

        hour_offset = int((current_dt - current_cycle_dt).total_seconds() / 3600) + 1
        forecast_or_nowcast = "f" if current_dt > current_cycle_dt else "n"

        fmt_vars = {
            "yyyy": current_cycle_dt.strftime("%Y"),
            "mm": current_cycle_dt.strftime("%m"),
            "dd": current_cycle_dt.strftime("%d"),
            "yyyymmdd": current_cycle_dt.strftime("%Y%m%d"),
            "cc": current_cycle_dt.strftime("%H"),
            "hhh": hour_offset,
            "type": forecast_or_nowcast,
        }

        files.append(pattern.format(**fmt_vars))
        current_dt += pd.Timedelta(hours=1)

    return files


def _open_dbofs_dataset(
    access_mode: str,
    url_or_pattern: str,
    target_dt: pd.Timestamp,
    end_dt: Optional[pd.Timestamp] = None,
) -> Optional[xr.Dataset]:
    """Open DBOFS dataset via OPeNDAP (FMRC or NCEI)."""
    try:
        if access_mode == "fmrc":
            logger.info(f"Opening DBOFS FMRC aggregation: {url_or_pattern}")
            dap_url = _to_dap_url(url_or_pattern)
            ds = xr.open_dataset(dap_url, engine="pydap")

            time_var = "time" if "time" in ds.coords else "ocean_time"
            ds = ds.sortby(time_var)

            if end_dt is None:
                return ds.sel({time_var: target_dt}, method="nearest")
            else:
                ds_t = ds.sel({time_var: slice(target_dt, end_dt)})
                if ds_t.sizes[time_var] == 0:
                    logger.warning(
                        "DBOFS FMRC: exact time range empty, fell back to nearest."
                    )
                    return ds.sel({time_var: target_dt}, method="nearest").expand_dims(
                        time_var
                    )
                return ds_t

        elif access_mode in ("ncei", "aws_s3"):
            mode_name = "AWS S3" if access_mode == "aws_s3" else "NCEI"
            logger.info(
                f"Enumerating DBOFS {mode_name} files from {target_dt} to {end_dt}"
            )
            if end_dt is None:
                end_dt = target_dt + pd.Timedelta(hours=1)

            files = _enumerate_ncei_dbofs_files(url_or_pattern, target_dt, end_dt)
            logger.info(f"Opening {len(files)} DBOFS {mode_name} files...")

            if access_mode == "aws_s3":
                try:
                    logger.info("Using xarray.open_mfdataset for parallel S3 access...")
                    ds_t = xr.open_mfdataset(
                        files,
                        engine="h5netcdf",
                        parallel=True,
                        storage_options={"anon": True},
                        data_vars="minimal",
                        coords="minimal",
                        compat="override",
                    )
                    return ds_t
                except Exception as e:
                    logger.error(
                        f"Failed to open/concat DBOFS S3 files via mfdataset: {e}"
                    )
                    return None
            else:
                import concurrent.futures
                import os

                datasets: list[Optional[xr.Dataset]] = [None] * len(files)
                fail_counts = [0]

                def _fetch_file(args):
                    i, f = args
                    # Optional short-circuit if another thread hit the failure limit
                    if fail_counts[0] >= _NCEI_CONSECUTIVE_FAIL_LIMIT:
                        return i, None
                    try:
                        if i % 10 == 0 or i == 1 or i == len(files):
                            logger.info(
                                f"[{i}/{len(files)}] Fetching/Opening DBOFS {mode_name} file: {f.split('/')[-1] if 's3' in f else f}"
                            )
                        ds_file = xr.open_dataset(_to_dap_url(f), engine="pydap")
                        return i, ds_file
                    except Exception as e:
                        logger.warning(
                            f"Failed to open DBOFS {mode_name} file {f}: {e}"
                        )
                        fail_counts[0] += 1
                        return i, None

                max_workers = int(os.environ.get("ECODATA_CACHE_MAX_WORKERS", 4))
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=max_workers
                ) as executor:
                    for i, ds_file in executor.map(_fetch_file, enumerate(files, 1)):
                        datasets[i - 1] = ds_file

                # Filter out failures
                opened = [ds for ds in datasets if ds is not None]

                if not opened:
                    logger.error(f"No DBOFS {mode_name} files could be opened.")
                    return None

                return xr.concat(opened, dim="time", join="override")

        else:
            logger.error(f"Unknown access_mode: {access_mode}")
            return None

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
            ds_sub[u_var].values.astype(np.float32),
            ds_sub[v_var].values.astype(np.float32),
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
