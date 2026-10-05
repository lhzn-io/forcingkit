import xarray as xr
import pandas as pd
import logging
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)


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


def _get_hycom_url(target_dt: pd.Timestamp) -> str:
    """
    Returns the appropriate HYCOM OPeNDAP URL based on the target date.
    """
    # expt_93.0 covers 2018-12-04 to present
    switch_date = pd.Timestamp("2018-12-04").tz_localize(None)

    if target_dt >= switch_date:
        return "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0"
    else:
        # Fallback to reanalysis expt_53.X series
        # Note: In production, this might need further refinement based on specific 53.X sub-experiments
        return "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_53.X"


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

    # Check if we need to stitch multiple experiments
    switch_date = pd.Timestamp("2018-12-04").tz_localize(None)

    if start_dt < switch_date and end_dt > switch_date:
        logger.info(
            "Hindcast spans HYCOM experiment boundary. Stitching expt_53.X and expt_93.0..."
        )
        ds_old = _fetch_hycom_data(
            start_dt, switch_date, bbox, _get_hycom_url(start_dt)
        )
        ds_new = _fetch_hycom_data(switch_date, end_dt, bbox, _get_hycom_url(end_dt))

        if ds_old is None or ds_new is None:
            return ds_old or ds_new

        return xr.concat([ds_old, ds_new], dim="time")

    dataset_url = _get_hycom_url(start_dt)
    return _fetch_hycom_data(start_dt, end_dt, bbox, dataset_url)


def _fetch_hycom_data(
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    bbox: list[float],
    dataset_url: str,
    is_ic: bool = False,
) -> Optional[xr.Dataset]:
    min_lon, min_lat, max_lon, max_lat = bbox

    # Normalize longitudes for HYCOM (0 to 360)
    hycom_min_lon = min_lon if min_lon >= 0 else 360 + min_lon
    hycom_max_lon = max_lon if max_lon >= 0 else 360 + max_lon

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
            # Add a small buffer to ensure we capture the boundary times
            ds_subset = ds.sel({time_var: slice(start_hours - 0.1, end_hours + 0.1)})
            if ds_subset[time_var].size == 0:
                logger.error(
                    f"Requested time range out of bounds for HYCOM. Required: {start_hours} to {end_hours}."
                )
                return None

        lon_var = "lon" if "lon" in ds.coords else "longitude"
        lat_var = "lat" if "lat" in ds.coords else "latitude"

        # Handle 0-360 wrapping if bbox crosses prime meridian
        if hycom_min_lon > hycom_max_lon:
            logger.info(
                "BBox crosses 0/360 boundary, performing dual-slice and concat."
            )
            part1 = ds_subset.sel({lon_var: slice(hycom_min_lon - 0.1, 360.0)})
            part2 = ds_subset.sel({lon_var: slice(0.0, hycom_max_lon + 0.1)})
            ds_subset = xr.concat([part1, part2], dim=lon_var)
        else:
            ds_subset = ds_subset.sel(
                {
                    lat_var: slice(min_lat - 0.1, max_lat + 0.1),
                    lon_var: slice(hycom_min_lon - 0.1, hycom_max_lon + 0.1),
                }
            )

        logger.info("Executing OPeNDAP download for HYCOM subset...")
        ds_subset = ds_subset.compute()

        # Rename variables to canonical names if necessary
        rename_map = {
            "water_u": "u",
            "water_v": "v",
            "water_temp": "temp",
            "salinity": "salt",
            "surf_el": "zeta",
        }
        actual_rename = {
            k: v for k, v in rename_map.items() if k in ds_subset.data_vars
        }
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
