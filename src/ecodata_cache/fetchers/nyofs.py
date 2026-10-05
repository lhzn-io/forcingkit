"""
NYOFS (NOAA New York / New Jersey Operational Forecast System) fetcher.

Provides Initial Conditions (IC) and Open Boundary Conditions (OBC) from Princeton Ocean
Model (POM) structured curvilinear grid via OPeNDAP.
"""

import numpy as np
import xarray as xr
import pandas as pd
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def get_metadata() -> dict:
    """Returns metadata for the NYOFS system."""
    return {
        "id": "nyofs",
        "name": "NOAA NYOFS (NY/NJ Harbor)",
        "resolution_approx_m": 100.0,
        "type_desc": "Structured curvilinear POM grid",
        "domain_bbox": [-74.3, 40.2, -73.3, 41.1],
    }


def supports_bbox(bbox: list[float]) -> bool:
    """Check if the requested bbox is within NYOFS domain."""
    min_lon, min_lat, max_lon, max_lat = bbox
    domain_bbox = get_metadata()["domain_bbox"]
    domain_min_lon, domain_min_lat, domain_max_lon, domain_max_lat = domain_bbox

    # Request bbox must be fully contained within NYOFS domain
    if (
        min_lon < domain_min_lon
        or max_lon > domain_max_lon
        or min_lat < domain_min_lat
        or max_lat > domain_max_lat
    ):
        return False
    return True


def _to_dap_url(url: str) -> str:
    """Convert http(s) URL to pydap dap2:// scheme."""
    return url.replace("https://", "dap2://").replace("http://", "dap2://")


def _get_nyofs_url(target_dt: pd.Timestamp) -> Tuple[str, str]:
    """
    Resolve NYOFS data access URL and mode.

    Returns (access_mode, url_or_pattern) where access_mode is one of:
      - "fmrc": single FMRC aggregated OPeNDAP URL
      - "ncei": NCEI THREDDS file pattern (requires enumeration)

    FMRC is preferred for recent data (< 31 days).
    NCEI is fallback for historical data (> 31 days).
    """
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    age_days = (now - target_dt).total_seconds() / 86400

    # Prefer FMRC aggregation for recent data (< 7 days)
    if 0 <= age_days <= 6:
        fmrc_url = (
            "https://opendap.co-ops.nos.noaa.gov/thredds/dodsC/"
            "NYOFS/fmrc/Aggregated_7_day_NYOFS_Fields_Forecast_best.ncd"
        )
        return ("fmrc", fmrc_url)

    # Historical: AWS S3 or NCEI file-per-hour. Naming convention changed 2024-09-09
    # and the archive migrated to AWS S3 starting 2024-01-01.
    if target_dt >= pd.Timestamp("2024-01-01"):
        pattern = (
            "s3://noaa-nos-ofs-pds/nyofs/netcdf/"
            "{yyyy}/{mm}/{dd}/nyofs.t{cc}z.{yyyymmdd}.fields.{type}{hhh:03d}.nc"
        )
        return ("aws_s3", pattern)
    elif target_dt >= pd.Timestamp("2024-09-09"):
        # Post-Sept 9 2024 naming: nyofs.tCCz.YYYYMMDD.fields.nHHH.nc
        pattern = (
            "https://www.ncei.noaa.gov/thredds/dodsC/model-nyofs-files/"
            "{yyyy}/{mm}/{dd}/nyofs.t{cc}z.{yyyymmdd}.fields.{type}{hhh:03d}.nc"
        )
    else:
        # Legacy naming: nos.nyofs.fields.nHHH.YYYYMMDD.tCCz.nc
        pattern = (
            "https://www.ncei.noaa.gov/thredds/dodsC/model-nyofs-files/"
            "{yyyy}/{mm}/nos.nyofs.fields.{type}{hhh:03d}.{yyyymmdd}.t{cc}z.nc"
        )
    return ("ncei", pattern)


def _enumerate_ncei_nyofs_files(
    pattern: str, start_dt: pd.Timestamp, end_dt: pd.Timestamp
) -> list[str]:
    """
    Enumerate NCEI NYOFS file URLs for a time range.

    Picks the best 6-hourly cycle (00/06/12/18Z) and builds per-hour file URLs.
    Spans cycle boundaries if duration_hours > remaining hours in first cycle.
    """
    # Find the most recent 6-hourly cycle before start_dt
    cycle_hours = [0, 6, 12, 18]
    cycle_before = None
    for ch in reversed(cycle_hours):
        test_dt = start_dt.replace(hour=ch, minute=0, second=0, microsecond=0)
        if test_dt <= start_dt:
            cycle_before = test_dt
            break

    if cycle_before is None:
        # Fall back to previous day's last cycle
        cycle_before = (start_dt - pd.Timedelta(days=1)).replace(
            hour=18, minute=0, second=0, microsecond=0
        )

    # Enumerate files from cycle_before to end_dt
    files = []
    current_dt = cycle_before
    current_cycle_dt = cycle_before

    while current_dt <= end_dt:
        # Check if we've crossed into a new cycle (6 hours apart)
        if (current_dt - current_cycle_dt).total_seconds() >= 6 * 3600:
            current_cycle_dt = current_dt.replace(minute=0, second=0, microsecond=0)
            # Snap to 6-hourly boundaries
            cycle_hour = (current_cycle_dt.hour // 6) * 6
            current_cycle_dt = current_cycle_dt.replace(hour=cycle_hour)

        # Calculate file index (hour offset from cycle start)
        hour_offset = int((current_dt - current_cycle_dt).total_seconds() / 3600) + 1
        forecast_or_nowcast = "f" if current_dt > current_cycle_dt else "n"

        # Format template variables
        fmt_vars = {
            "yyyy": current_cycle_dt.strftime("%Y"),
            "mm": current_cycle_dt.strftime("%m"),
            "dd": current_cycle_dt.strftime("%d"),
            "yyyymmdd": current_cycle_dt.strftime("%Y%m%d"),
            "cc": current_cycle_dt.strftime("%H"),
            "hhh": hour_offset,
            "type": forecast_or_nowcast,
        }

        # Build URL
        url = pattern.format(**fmt_vars)
        files.append(url)

        current_dt += pd.Timedelta(hours=1)

    return files


def _open_nyofs_dataset(
    access_mode: str,
    url_or_pattern: str,
    target_dt: pd.Timestamp,
    end_dt: Optional[pd.Timestamp] = None,
) -> Optional[xr.Dataset]:
    """
    Open NYOFS dataset via OPeNDAP (FMRC or NCEI).

    Args:
        access_mode: "fmrc" or "ncei"
        url_or_pattern: Full URL (fmrc) or pattern (ncei)
        target_dt: Start datetime for slicing
        end_dt: End datetime (required for ncei, optional for fmrc)

    Returns:
        xr.Dataset or None if fetch fails
    """
    try:
        if access_mode == "fmrc":
            logger.info(f"Opening FMRC aggregation: {url_or_pattern}")
            dap_url = _to_dap_url(url_or_pattern)
            ds = xr.open_dataset(dap_url, engine="pydap")

            # Determine time variable name
            time_var = "time" if "time" in ds.coords else "ocean_time"

            # FMRC aggregations can have non-monotonic time indices; sort before slicing
            ds = ds.sortby(time_var)

            # Slice to requested time range
            if end_dt is None:
                ds_t = ds.sel({time_var: target_dt}, method="nearest")
            else:
                ds_t = ds.sel({time_var: slice(target_dt, end_dt)})
                if ds_t.sizes[time_var] == 0:
                    ds_t = ds.sel({time_var: target_dt}, method="nearest").expand_dims(
                        time_var
                    )
                    logger.warning("Exact time range empty, fell back to nearest.")

            return ds_t

        elif access_mode in ("ncei", "aws_s3"):
            mode_name = "AWS S3" if access_mode == "aws_s3" else "NCEI"
            logger.info(
                f"Enumerating NYOFS {mode_name} files from {target_dt} to {end_dt}"
            )
            if end_dt is None:
                end_dt = target_dt + pd.Timedelta(hours=1)

            files = _enumerate_ncei_nyofs_files(url_or_pattern, target_dt, end_dt)
            logger.info(f"Opening {len(files)} NYOFS {mode_name} files...")

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
                        f"Failed to open/concat NYOFS S3 files via mfdataset: {e}"
                    )
                    return None
            else:
                import concurrent.futures
                import os

                datasets: list[Optional[xr.Dataset]] = [None] * len(files)
                fail_counts = [0]

                def _fetch_file(args):
                    i, f = args
                    if fail_counts[0] >= 3:
                        return i, None
                    try:
                        if i % 10 == 0 or i == 1 or i == len(files):
                            logger.info(
                                f"[{i}/{len(files)}] Fetching/Opening NYOFS {mode_name} file: {f.split('/')[-1] if 's3' in f else f}"
                            )
                        ds_file = xr.open_dataset(_to_dap_url(f), engine="pydap")
                        return i, ds_file
                    except Exception as e:
                        logger.warning(
                            f"Failed to open NYOFS {mode_name} file {f}: {e}"
                        )
                        fail_counts[0] += 1
                        return i, None

                max_workers = int(os.environ.get("ECODATA_CACHE_MAX_WORKERS", 4))
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=max_workers
                ) as executor:
                    for i, ds_file in executor.map(_fetch_file, enumerate(files, 1)):
                        if ds_file is not None:
                            datasets[i - 1] = ds_file

                opened = [ds for ds in datasets if ds is not None]

                if not opened:
                    logger.error(f"No NYOFS {mode_name} files could be opened.")
                    return None

                # Concatenate along time dimension
                return xr.concat(opened, dim="time", join="override")

        else:
            logger.error(f"Unknown access_mode: {access_mode}")
            return None

    except Exception as e:
        logger.error(f"Failed to open NYOFS dataset ({access_mode}): {e}")
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
    available = list(ds.data_vars)
    raise KeyError(
        f"No variable found for role '{role}'. "
        f"Tried: {_VAR_CANDIDATES[role]}. Available: {available}"
    )


def _c_grid_to_rho(
    u_raw: np.ndarray,
    v_raw: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolate Arakawa C-grid u,v face values to rho-points by averaging
    adjacent pairs.  Works on already-subset arrays of any leading shape:

        u_rho[..., j, i] = 0.5 * (u[..., j, i] + u[..., j, i+1])
        v_rho[..., j, i] = 0.5 * (v[..., j, i] + v[..., j+1, i])

    The boundary column/row is filled by extrapolation (copy-edge) so the
    output shape always matches the input shape.  This is valid for both
    IC arrays (sigma, eta, xi) and OBC arrays (time, sigma, eta, xi).
    """
    u_rho = np.empty_like(u_raw)
    u_rho[..., :-1] = 0.5 * (u_raw[..., :-1] + u_raw[..., 1:])
    u_rho[..., -1] = u_raw[..., -1]  # boundary: copy edge

    v_rho = np.empty_like(v_raw)
    v_rho[..., :-1, :] = 0.5 * (v_raw[..., :-1, :] + v_raw[..., 1:, :])
    v_rho[..., -1, :] = v_raw[..., -1, :]  # boundary: copy edge

    return u_rho, v_rho


def fetch_nyofs_boundary_conditions(
    start_date: str, duration_hours: int, bbox: list[float]
) -> Optional[xr.Dataset]:
    """
    Fetch 4D Ocean State (u, v) from NOAA NYOFS over a time range for OBC.

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
        logger.info(f"Bounding box {bbox} outside NYOFS domain.")
        return None

    target_dt = pd.to_datetime(start_date)
    if target_dt.tzinfo is not None:
        target_dt = target_dt.tz_convert("UTC").tz_localize(None)

    end_dt = target_dt + pd.Timedelta(hours=duration_hours)

    access_mode, url_or_pattern = _get_nyofs_url(target_dt)
    logger.info(
        f"Attempting to fetch NYOFS OBC ({duration_hours}h) from {access_mode.upper()}"
    )

    ds_t = _open_nyofs_dataset(access_mode, url_or_pattern, target_dt, end_dt)
    if ds_t is None:
        return None

    try:
        # Spatial subsetting
        lon_var = "lon"
        lat_var = "lat"

        mask = (
            (ds_t[lon_var] >= min_lon)
            & (ds_t[lon_var] <= max_lon)
            & (ds_t[lat_var] >= min_lat)
            & (ds_t[lat_var] <= max_lat)
            & (ds_t.get("mask", 1) == 1)
        )

        ds_sub = ds_t.where(mask, drop=True)

        # Check if the bounding box yielded zero valid water points
        if any(size == 0 for size in ds_sub.sizes.values()):
            logger.warning(
                "No valid NYOFS ocean points found in bounding box (size is 0)."
            )
            return None

        logger.info("Executing OPeNDAP download for NYOFS OBC subset...")
        ds_sub = ds_sub.compute()

        # Resolve actual variable names
        u_var = _resolve_var(ds_sub, "u")
        v_var = _resolve_var(ds_sub, "v")
        logger.info(f"NYOFS OBC variable mapping: u={u_var}, v={v_var}")

        # Detect dimensions
        all_dims = list(ds_sub[u_var].dims)
        time_var = "time" if "time" in ds_sub.coords else "ocean_time"
        sigma_dim = None
        for candidate in ["sigma", "s_rho", "depth", "siglay"]:
            if candidate in all_dims:
                sigma_dim = candidate
                break

        if sigma_dim is None:
            logger.error(
                f"No recognized sigma dimension in NYOFS OBC. Dims: {all_dims}"
            )
            return None

        # Spatial dimensions
        spatial_dims = [d for d in all_dims if d != time_var and d != sigma_dim]
        if len(spatial_dims) < 2:
            logger.error(f"Unexpected spatial dims after subset: {spatial_dims}")
            return None

        eta_dim = spatial_dims[0]  # noqa: F841 - kept for readability / debug logging
        xi_dim = spatial_dims[1]  # noqa: F841

        # C-grid interpolation: vectorized over all leading dims (time, sigma, ...)
        u_raw = ds_sub[u_var].values
        v_raw = ds_sub[v_var].values
        u_rho, v_rho = _c_grid_to_rho(
            u_raw.astype(np.float32), v_raw.astype(np.float32)
        )

        n_sigma = u_rho.shape[1]
        n_eta = u_rho.shape[2]
        n_xi = u_rho.shape[3]

        # Map sigma to pseudo-depth
        n_depth = n_sigma
        depths = np.linspace(-50, 0, n_depth).astype(np.float32)

        # Get time coordinates
        out_times = ds_sub[time_var].values

        # Build output dataset
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
            attrs={"type": "NOAA NYOFS OBC", "source": "NOAA CO-OPS"},
        )

        logger.info(
            f"Successfully processed NYOFS OBC data. "
            f"Shape: u{ds_out['u'].shape}, {len(out_times)} time steps"
        )
        return ds_out

    except Exception as e:
        logger.error(f"Failed to process NYOFS OBC data: {e}")
        return None
