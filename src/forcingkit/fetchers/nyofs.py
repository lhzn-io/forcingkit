"""
NYOFS (NOAA New York / New Jersey Operational Forecast System) fetcher.

Provides Open Boundary Conditions (OBC) from the Princeton Ocean Model (POM) coarse
curvilinear grid. Recent data comes from the CO-OPS FMRC aggregation; older data is the chain
of per-cycle nowcast files, from AWS S3 (per-day layout, from 2024-11-19) or NCEI THREDDS.
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
    "NYOFS/fmrc/Aggregated_7_day_NYOFS_Fields_Forecast_best.ncd"
)
NCEI_BASE = "https://www.ncei.noaa.gov/thredds/dodsC/model-nyofs-files"
AWS_BASE = "s3://noaa-nos-ofs-pds/nyofs/netcdf"

# AWS holds one directory per day from 2024-11-19 (that day has only the 23 UTC cycle). Before
# that, AWS has flat month directories with gaps, so those dates are read from NCEI instead.
AWS_PER_DAY_FROM = pd.Timestamp("2024-11-19")
# NCEI file names changed with the 2024-09-09 cycles.
NCEI_NEW_NAMES_FROM = pd.Timestamp("2024-09-09")

# Each nowcast file holds the 6 hourly records up to its cycle time: t05z holds 00 to 05 UTC.
NOWCAST_CYCLE_HOURS = (5, 11, 17, 23)


def get_metadata() -> dict:
    """Returns metadata for the NYOFS system."""
    return {
        "id": "nyofs",
        "name": "NOAA NYOFS (NY/NJ Harbor)",
        "resolution_approx_m": 100.0,
        "type_desc": "Structured curvilinear POM grid",
        # Extent of the coarse grid (land cells included), so a box NYOFS ranks first for has
        # NYOFS cells under it.
        "domain_bbox": [-74.475, 40.389, -73.743, 40.940],
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


def _get_nyofs_url(target_dt: pd.Timestamp) -> Tuple[str, Optional[str]]:
    """
    Resolve the NYOFS access mode for a start time.

    Returns ("fmrc", url) for start times up to 6 days old, and ("archive", None) otherwise;
    archive file URLs come from `_nowcast_file_urls`.
    """
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    age_days = (now - target_dt).total_seconds() / 86400
    if 0 <= age_days <= 6:
        return ("fmrc", FMRC_URL)
    return ("archive", None)


def _nowcast_cycle(hour_dt: pd.Timestamp) -> pd.Timestamp:
    """Return the cycle whose nowcast file holds the record at `hour_dt` (a whole hour)."""
    cycle_hour = next(c for c in NOWCAST_CYCLE_HOURS if c >= hour_dt.hour)
    return hour_dt.replace(hour=cycle_hour, minute=0, second=0, microsecond=0)


def _nowcast_cycles(start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> list[pd.Timestamp]:
    """Cycles whose nowcast files together cover every hour from `start_dt` to `end_dt`."""
    hours = pd.date_range(start_dt.floor("h"), end_dt.ceil("h"), freq="h")
    return sorted({_nowcast_cycle(h) for h in hours})


def _nowcast_file_urls(cycle_dt: pd.Timestamp) -> list[str]:
    """Candidate URLs for one cycle's coarse-grid nowcast file, in order of preference."""
    day_dir, month_dir = cycle_dt.strftime("%Y/%m/%d"), cycle_dt.strftime("%Y/%m")
    ymd, cc = cycle_dt.strftime("%Y%m%d"), cycle_dt.strftime("%H")
    new_name = f"nyofs.t{cc}z.{ymd}.fields.nowcast.nc"

    urls = []
    if cycle_dt.normalize() >= AWS_PER_DAY_FROM:
        urls.append(f"{AWS_BASE}/{day_dir}/{new_name}")
    if cycle_dt.normalize() >= NCEI_NEW_NAMES_FROM:
        urls.append(f"{NCEI_BASE}/{month_dir}/{new_name}")
    else:
        old_name = f"nos.nyofs.fields.nowcast.{ymd}.t{cc}z.nc"
        urls.append(f"{NCEI_BASE}/{month_dir}/{old_name}")
    return urls


def _open_archive_file(url: str) -> xr.Dataset:
    """Open one NYOFS file. AWS copies are netCDF3 classic, which h5netcdf cannot read."""
    if url.startswith("s3://"):
        import fsspec  # type: ignore[import-untyped]

        with fsspec.open(url, "rb", anon=True) as f:
            # scipy reads netCDF3 from a file object into memory; a nowcast file is about 6 MB.
            return xr.open_dataset(f, engine="scipy").load()
    return xr.open_dataset(_to_dap_url(url), engine="pydap")


def _open_first_available(urls: list[str]) -> Optional[xr.Dataset]:
    for url in urls:
        try:
            return _open_archive_file(url)
        except Exception as e:
            logger.info(f"NYOFS file unavailable at {url}: {e}")
    return None


def _round_to_hour(ds: xr.Dataset, time_var: str) -> xr.Dataset:
    """Snap record times to whole hours; NYOFS stores them with up to 15 s of jitter."""
    times = pd.DatetimeIndex(ds[time_var].values).round("h")
    return ds.assign_coords({time_var: times})


def _open_nyofs_dataset(
    access_mode: str,
    url_or_pattern: Optional[str],
    target_dt: pd.Timestamp,
    end_dt: Optional[pd.Timestamp] = None,
) -> Optional[xr.Dataset]:
    """
    Open NYOFS data covering [target_dt, end_dt].

    Args:
        access_mode: "fmrc" or "archive"
        url_or_pattern: FMRC URL (fmrc); unused for archive
        target_dt: Start datetime for slicing
        end_dt: End datetime (defaults to one hour after target_dt)

    Returns:
        xr.Dataset or None if fetch fails. Archive reads return None unless every nowcast file
        in the window is found, so a gap falls back to the next donor rather than shortening
        the store.
    """
    if end_dt is None:
        end_dt = target_dt + pd.Timedelta(hours=1)
    try:
        if access_mode == "fmrc":
            logger.info(f"Opening FMRC aggregation: {url_or_pattern}")
            assert url_or_pattern is not None
            ds = xr.open_dataset(_to_dap_url(url_or_pattern), engine="pydap")

            time_var = "time" if "time" in ds.coords else "ocean_time"

            # FMRC aggregations can have non-monotonic time indices; sort before slicing
            ds = _round_to_hour(ds.sortby(time_var), time_var)

            ds_t = ds.sel({time_var: slice(target_dt, end_dt)})
            if ds_t.sizes[time_var] == 0:
                ds_t = ds.sel({time_var: target_dt}, method="nearest").expand_dims(
                    time_var
                )
                logger.warning("Exact time range empty, fell back to nearest.")
            return ds_t

        if access_mode != "archive":
            logger.error(f"Unknown access_mode: {access_mode}")
            return None

        cycles = _nowcast_cycles(target_dt, end_dt)
        logger.info(
            f"Opening {len(cycles)} NYOFS nowcast files for {target_dt} to {end_dt}"
        )
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=settings.max_workers()
        ) as executor:
            opened = list(
                executor.map(
                    lambda c: _open_first_available(_nowcast_file_urls(c)), cycles
                )
            )

        missing = [c for c, ds in zip(cycles, opened) if ds is None]
        if missing:
            logger.error(
                f"NYOFS nowcast files missing for cycles {[str(c) for c in missing]}"
            )
            return None

        datasets = [ds for ds in opened if ds is not None]
        ds = xr.concat(
            datasets,
            dim="time",
            data_vars="minimal",
            coords="minimal",
            compat="override",
            join="override",
        )
        ds = _round_to_hour(ds, "time")
        return ds.sel(time=slice(target_dt.floor("h"), end_dt.ceil("h")))

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


def fetch_nyofs_boundary_conditions(
    start_date: str, duration_hours: int, bbox: list[float]
) -> Optional[xr.Dataset]:
    """
    Fetch 4D Ocean State (u, v) from NOAA NYOFS over a time range for OBC.

    Output dimensions: (time, depth, eta, xi) matching the standard OBC contract.
    NYOFS u and v are earth-referenced (eastward, northward) and co-located with lon/lat, so
    they are passed through without any face-to-centre averaging.

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

    access_mode, url = _get_nyofs_url(target_dt)
    logger.info(
        f"Attempting to fetch NYOFS OBC ({duration_hours}h) from {access_mode.upper()}"
    )

    ds_t = _open_nyofs_dataset(access_mode, url, target_dt, end_dt)
    if ds_t is None:
        return None

    try:
        u_var = _resolve_var(ds_t, "u")
        v_var = _resolve_var(ds_t, "v")
        time_var = "time" if "time" in ds_t.coords else "ocean_time"

        all_dims = list(ds_t[u_var].dims)
        sigma_dim = next(
            (d for d in ("sigma", "s_rho", "depth", "siglay") if d in all_dims), None
        )
        if sigma_dim is None:
            logger.error(
                f"No recognized sigma dimension in NYOFS OBC. Dims: {all_dims}"
            )
            return None
        spatial_dims = [d for d in all_dims if d not in (time_var, sigma_dim)]
        if len(spatial_dims) != 2:
            logger.error(f"Unexpected spatial dims in NYOFS OBC: {spatial_dims}")
            return None
        eta_dim, xi_dim = spatial_dims

        lon = np.asarray(ds_t["lon"].values)
        lat = np.asarray(ds_t["lat"].values)
        wet = (
            np.asarray(ds_t["mask"].values) == 1
            if "mask" in ds_t
            else np.ones_like(lon, dtype=bool)
        )
        inside = (
            (lon >= min_lon)
            & (lon <= max_lon)
            & (lat >= min_lat)
            & (lat <= max_lat)
            & wet
        )
        rows, cols = np.nonzero(inside)
        if rows.size == 0:
            logger.warning("No valid NYOFS ocean points found in bounding box.")
            return None

        # The bounding rectangle of the wet cells inside the box; cells outside the box or on
        # land within it are NaN.
        window = {
            eta_dim: slice(int(rows.min()), int(rows.max()) + 1),
            xi_dim: slice(int(cols.min()), int(cols.max()) + 1),
        }
        keep = inside[window[eta_dim], window[xi_dim]]

        logger.info("Reading NYOFS OBC subset...")
        ds_sub = ds_t[[u_var, v_var]].isel(window).compute()
        if ds_sub.indexes[time_var].has_duplicates:
            # The FMRC "best" series repeats some hours (times differing only in jitter), one
            # copy of which can be all fill; keep the first non-missing value of each hour.
            ds_sub = ds_sub.groupby(time_var).first()
        u = np.where(keep, ds_sub[u_var].values, np.nan).astype(np.float32)
        v = np.where(keep, ds_sub[v_var].values, np.nan).astype(np.float32)

        n_sigma, n_eta, n_xi = u.shape[1], u.shape[2], u.shape[3]
        out_times = ds_sub[time_var].values

        ds_out = xr.Dataset(
            data_vars={
                "u": (("time", "depth", "eta", "xi"), u),
                "v": (("time", "depth", "eta", "xi"), v),
            },
            coords={
                "time": out_times,
                # Placeholder for the sigma levels (legacy layout), not true depths.
                "depth": np.linspace(-50, 0, n_sigma).astype(np.float32),
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
