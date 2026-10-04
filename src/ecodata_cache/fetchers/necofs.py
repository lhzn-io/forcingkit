import warnings
import logging
import pandas as pd
import numpy as np
import xarray as xr
import os
from typing import Optional
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator


def _require_all(results: list[Optional[np.ndarray]], label: str) -> list[np.ndarray]:
    """Narrow a pre-allocated result list once every slot has been filled.

    The per-timestep lists start as `[None] * nt` and are filled by index from the
    thread pool. An exception inside a worker surfaces through future.result() and
    aborts the whole chunk via the caller's handler, so reaching this point means
    no slot is still None. Checking it here turns a broken invariant into a named
    diagnostic instead of an opaque dtype error from np.stack further downstream.
    """
    missing = [i for i, r in enumerate(results) if r is None]
    if missing:
        raise RuntimeError(f"NECOFS {label}: timesteps {missing} were never populated")
    return [r for r in results if r is not None]


logger = logging.getLogger(__name__)


def get_metadata() -> dict:
    return {
        "id": "necofs",
        "name": "NECOFS FVCOM GOM7",
        "resolution_approx_m": 200.0,
        "type_desc": "Unstructured Triangular Mesh",
        "domain_bbox": [-77.0, 35.0, -65.0, 46.0],
    }


def supports_bbox(bbox: list[float]) -> bool:
    min_lon, min_lat, max_lon, max_lat = bbox
    # Rough check for GOM3 bounds (NECOFS FVCOM)
    if max_lat < 35.0 or min_lat > 46.0 or max_lon < -77.0 or min_lon > -65.0:
        return False
    return True


NECOFS_GOM7_URL = "http://www.smast.umassd.edu:8080/thredds/dodsC/models/fvcom/NECOFS/Forecasts/NECOFS_GOM7_FORECAST.nc"


def get_necofs_url(target_dt: pd.Timestamp) -> str:
    """
    Determine best NECOFS GOM7 URL by falling back to daily history archives.
    SMAST daily archives (e.g. 2026_03_03.nc) contain [Mar 2 01:00 to Mar 3 00:00].
    """
    import requests

    # NECOFS GOM7 daily history files contain data for the PREVIOUS day up to 00:00 of the CURRENT day.
    # Therefore to get data forward-looking from target_dt, we must ALWAYS fetch the file for the NEXT day.
    file_dt = target_dt.normalize() + pd.Timedelta(days=1)

    date_str = file_dt.strftime("%Y_%m_%d")
    history_url = f"http://www.smast.umassd.edu:8080/thredds/dodsC/models/fvcom/NECOFS/Archive/necofs_history/NECOFS_GOM7_{date_str}.nc"

    try:
        resp = requests.get(history_url + ".dds", timeout=5)
        if resp.status_code == 200:
            logger.info(f"Using historic NECOFS GOM7 archive: {history_url}")
            return history_url
    except requests.RequestException:
        pass

    logger.info("Falling back to NECOFS GOM7 rolling forecast.")
    return NECOFS_GOM7_URL


def fetch_necofs_initial_conditions(
    target_date: str, bbox: list
) -> Optional[xr.Dataset]:
    """
    Fetches the 3D unstructured ocean state from NECOFS (GOM3),
    and regrids it to a structured Z-level grid suitable for Oceananigans.

    Args:
        target_date: ISO 8601 datestring (e.g. "2026-03-04T00:00:00Z")
        bbox: [min_lon, min_lat, max_lon, max_lat]
    """
    min_lon, min_lat, max_lon, max_lat = bbox

    # Rough check for GOM3 bounds (NECOFS FVCOM)
    if max_lat < 35.0 or min_lat > 46.0 or max_lon < -77.0 or min_lon > -65.0:
        logger.info(
            f"Bounding box {bbox} is completely outside NECOFS GOM3 domain. Aborting fetch."
        )
        return None

    # Enforce UTC timezone naivety
    target_dt = pd.to_datetime(target_date)
    if target_dt.tzinfo is not None:
        target_dt = target_dt.tz_convert("UTC").tz_localize(None)

    logger.info("Attempting to fetch initial conditions from NECOFS GOM3...")

    try:
        # Determine the correct URL based on target date (history archive vs rolling forecast)
        dap_url = get_necofs_url(target_dt)

        # Pydap might hang on SMAST, so we should rely on dispatcher catching/logging it,
        # or we could use requests to check if it's alive first.
        import requests

        try:
            # Quick alive check
            resp = requests.get(dap_url + ".dds", timeout=10)
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.warning(f"NECOFS server appears unreachable at {dap_url}: {e}")
            return None

        # Open the dataset
        # We must disable decode_times because GOM7 contains broken Itime/Itime2 variables
        # specifying "msec since 00:00:00" which cftime cannot parse.
        # We load without decoding, drop them, and decode the rest properly.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ds_raw = xr.open_dataset(dap_url, engine="pydap", decode_times=False)
        drop_vars = [v for v in ["Itime", "Itime2"] if v in ds_raw.variables]
        ds = xr.decode_cf(ds_raw.drop_vars(drop_vars))

        # Extract unstructured coordinate variables
        lon = ds["lon"].values
        lat = ds["lat"].values
        lonc = ds["lonc"].values
        latc = ds["latc"].values

        _h = ds["h"].values  # noqa: F841 - needed for future depth mapping
        _siglay = ds["siglay"].values  # noqa: F841 - shape: (siglay, node)

        # Check nodes within bbox + buffer
        buffer = 0.05
        node_mask = (
            (lon >= min_lon - buffer)
            & (lon <= max_lon + buffer)
            & (lat >= min_lat - buffer)
            & (lat <= max_lat + buffer)
        )

        if not np.any(node_mask):
            logger.warning("No NECOFS nodes found within the padded bounding box.")
            return None

        # Temporal sub-selection
        try:
            ds_t = ds.sel(time=target_dt, method="nearest")
        except KeyError:
            logger.error(f"Cannot find time {target_dt} in NECOFS GOM3.")
            return None

        # Load required fields into memory (for the region or full domain depending on size)
        # Fetching full domain of one timestep might be faster/easier than fancy indexing with pydap
        logger.info("Downloading variables from NECOFS...")
        try:
            # We explicitly need node variables: temp, salinity, zeta
            # And element variables: u, v (and ww if available, but we'll stick to u, v)
            vars_to_get = ["temp", "salinity", "zeta", "u", "v"]
            ds_sub = ds_t[vars_to_get].compute()
            zeta = ds_sub["zeta"].values
            temp = ds_sub["temp"].values
            salt = ds_sub["salinity"].values
            u = ds_sub["u"].values
            v = ds_sub["v"].values
        except Exception as e:
            logger.error(f"Failed to extract variables from NECOFS: {e}")
            return None

        # Determine dimensions of regular grid
        # 0.002 degrees ~ roughly 200 meters resolution matching coastal NECOFS
        d_spacing = 0.002
        lon_rho = np.arange(min_lon, max_lon, d_spacing)
        lat_rho = np.arange(min_lat, max_lat, d_spacing)
        xi_rho = np.arange(len(lon_rho))
        eta_rho = np.arange(len(lat_rho))

        # Create target structured meshgrid
        lon_grid, lat_grid = np.meshgrid(lon_rho, lat_rho)

        logger.info("Building FVCOM interpolators for regridding...")

        # Node-based interpolator (temp, salt, zeta - defined at mesh nodes)
        pts_node = np.column_stack((lon, lat))
        tri_node = Delaunay(pts_node)

        # Element-based interpolator (u, v - defined at element centroids)
        pts_elem = np.column_stack((lonc, latc))
        tri_elem = Delaunay(pts_elem)

        target_pts = np.column_stack((lon_grid.ravel(), lat_grid.ravel()))

        logger.info("Interpolating variables to structured grid...")

        def interpolate_nodes(field_vals):
            interp = LinearNDInterpolator(tri_node, field_vals)
            return interp(target_pts).reshape(lon_grid.shape)

        def interpolate_elems(field_vals, strict_mask):
            interp = LinearNDInterpolator(tri_elem, field_vals)
            val = interp(target_pts).reshape(lon_grid.shape)
            val[strict_mask] = np.nan
            return val

        # 2D field
        zeta_interp = interpolate_nodes(zeta)
        land_mask = np.isnan(zeta_interp)

        s_rho_dim = ds_t.sizes.get("siglay", 45)
        s_rho = np.linspace(-1, 0, s_rho_dim)

        # Preallocate 3D arrays
        # shape: (s_rho, eta_rho, xi_rho)  - matching our target schema conventions
        ny, nx = len(lat_rho), len(lon_rho)
        nz = s_rho_dim

        temp_out = np.zeros((nz, ny, nx), dtype=np.float32)
        salt_out = np.zeros((nz, ny, nx), dtype=np.float32)
        u_out = np.zeros((nz, ny, nx), dtype=np.float32)
        v_out = np.zeros((nz, ny, nx), dtype=np.float32)

        for k in range(nz):
            # node based
            t_k = interpolate_nodes(temp[k, :])
            s_k = interpolate_nodes(salt[k, :])
            # elem based
            u_k = interpolate_elems(u[k, :], land_mask)
            v_k = interpolate_elems(v[k, :], land_mask)

            temp_out[k, :, :] = t_k
            salt_out[k, :, :] = s_k
            u_out[k, :, :] = u_k
            v_out[k, :, :] = v_k

        # Assemble Output Dataset matching general ROMS/DOPPIO output layout
        # (Variables: u, v, temp, salt, zeta, s_rho, lon_rho, lat_rho)

        ds_out = xr.Dataset(
            data_vars={
                "temp": (("s_rho", "eta_rho", "xi_rho"), temp_out),
                "salt": (("s_rho", "eta_rho", "xi_rho"), salt_out),
                "u": (("s_rho", "eta_rho", "xi_rho"), u_out),
                "v": (("s_rho", "eta_rho", "xi_rho"), v_out),
                "zeta": (("eta_rho", "xi_rho"), zeta_interp),
            },
            coords={
                "s_rho": s_rho,
                "eta_rho": eta_rho,
                "xi_rho": xi_rho,
                "lon_rho": (("eta_rho", "xi_rho"), lon_grid),
                "lat_rho": (("eta_rho", "xi_rho"), lat_grid),
            },
            attrs={"type": "NECOFS/FVCOM GOM3", "source": "UMass Dartmouth SMAST"},
        )

        logger.info(
            f"Successfully processed NECOFS data. Target grid dims: {dict(ds_out.sizes)}"
        )
        return ds_out

    except Exception as e:
        logger.error(f"Failed to fetch or process NECOFS data: {str(e)}")
        return None


# Parent-ocean delivery schema. Bump when the layout or meaning of the boundary store changes, so
# a cached store built under an older schema can never be served for a newer request.
OBC_SCHEMA = "z-v2"


def necofs_archive_file_date(t: pd.Timestamp) -> pd.Timestamp:
    """Date stamp of the GOM7 daily history file that holds the record at time `t`.

    File D holds (D-1 01:00, D 00:00], so the record at exactly midnight lives in the file stamped
    with that same day, and every other hour in the file stamped the following day.
    """
    return (t - pd.Timedelta(hours=1)).normalize() + pd.Timedelta(days=1)


def z_levels_for_depth(
    max_depth_m: float, spacing_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """Fixed z levels covering 0 to -max_depth_m: (centres, faces), both ordered bottom to top.

    The bottom face sits at or below -max_depth_m, so every wet donor column is spanned.
    """
    if spacing_m <= 0:
        raise ValueError(f"vertical spacing must be positive, got {spacing_m}")
    nz = max(1, int(np.ceil(max_depth_m / spacing_m)))
    faces = -spacing_m * np.arange(nz, -1, -1, dtype=np.float64)
    centres = 0.5 * (faces[:-1] + faces[1:])
    return centres, faces


def sigma_to_z(
    values: np.ndarray,
    z_layers: np.ndarray,
    z_bottom: np.ndarray,
    z_targets: np.ndarray,
) -> np.ndarray:
    """Interpolate sigma-layer columns onto fixed z levels.

    values, z_layers: (nsig, ny, nx), layer index 0 at the surface (FVCOM convention), z negative.
    z_bottom: (ny, nx), the sea floor (-h); NaN marks land.
    z_targets: (nz,) target level centres, any order.

    Returns (nz, ny, nx). Between the surface and the shallowest layer centre the shallowest value
    is used, and between the deepest layer centre and the sea floor the deepest value; levels below
    the sea floor, and land columns, are NaN.
    """
    nsig = values.shape[0]
    out = np.full((len(z_targets),) + values.shape[1:], np.nan, dtype=np.float64)
    land = ~np.isfinite(z_bottom)
    for n, zt in enumerate(z_targets):
        # Number of layer centres at or above this level; layers are ordered surface first.
        above = np.sum(z_layers >= zt, axis=0)
        upper = np.clip(above - 1, 0, nsig - 1)
        lower = np.clip(above, 0, nsig - 1)
        zu = np.take_along_axis(z_layers, upper[None], axis=0)[0]
        zl = np.take_along_axis(z_layers, lower[None], axis=0)[0]
        vu = np.take_along_axis(values, upper[None], axis=0)[0]
        vl = np.take_along_axis(values, lower[None], axis=0)[0]
        span = zu - zl
        w = np.where(span > 0, (zu - zt) / np.where(span > 0, span, 1.0), 0.0)
        w = np.clip(w, 0.0, 1.0)
        level = (1.0 - w) * vu + w * vl
        level[zt < z_bottom] = np.nan
        level[land] = np.nan
        out[n] = level
    return out


def fetch_necofs_boundary_conditions(
    start_date: str,
    duration_hours: int,
    bbox: list[float],
    pad_cells: int = 3,
    vertical_spacing_m: float = 2.0,
) -> Optional[xr.Dataset]:
    """NECOFS GOM7 as a parent ocean for a nested regional model.

    Returns u, v, temp, salt on (time, z, lat, lon) and zeta on (time, lat, lon), on a regular
    0.002 degree grid covering `bbox` plus `pad_cells` donor cells on every side, so the parent
    brackets the child. The vertical axis is true depth: each FVCOM sigma layer is placed at
    z = siglay * (h + zeta) + zeta for its column and interpolated onto fixed levels every
    `vertical_spacing_m` metres, ordered bottom to top. Land, points outside the mesh and levels
    below the sea floor are NaN. The first record is at `start_date` itself.
    """
    min_lon, min_lat, max_lon, max_lat = bbox

    if max_lat < 35.0 or min_lat > 46.0 or max_lon < -77.0 or min_lon > -65.0:
        logger.info(f"Bounding box {bbox} outside NECOFS domain.")
        return None

    target_dt = pd.to_datetime(start_date)
    if target_dt.tzinfo is not None:
        target_dt = target_dt.tz_convert("UTC").tz_localize(None)

    import requests
    import concurrent.futures

    d_spacing = 0.002
    pad = pad_cells * d_spacing
    lon_axis = np.arange(min_lon - pad, max_lon + pad + d_spacing / 2, d_spacing)
    lat_axis = np.arange(min_lat - pad, max_lat + pad + d_spacing / 2, d_spacing)
    lon_grid, lat_grid = np.meshgrid(lon_axis, lat_axis)
    target_pts = np.column_stack((lon_grid.ravel(), lat_grid.ravel()))
    ny, nx = lon_grid.shape

    tri_node = tri_elem = None
    h_grid = siglay_grid = None
    z_centres = z_faces = None

    collected_times: list = []
    collected: dict[str, list[np.ndarray]] = {
        k: [] for k in ("u", "v", "temp", "salt", "zeta")
    }

    current_dt = target_dt
    hours_fetched = 0

    while hours_fetched < duration_hours:
        # get_necofs_url(t) opens the file stamped t.normalize() + 1 day.
        file_date = necofs_archive_file_date(current_dt)
        dap_url = get_necofs_url(file_date - pd.Timedelta(days=1))
        try:
            requests.get(dap_url + ".dds", timeout=10).raise_for_status()
        except requests.RequestException as e:
            logger.warning(f"NECOFS server unreachable for {current_dt}: {e}")
            break

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ds_raw = xr.open_dataset(dap_url, engine="pydap", decode_times=False)
            drop_vars = [v for v in ["Itime", "Itime2"] if v in ds_raw.variables]
            ds = xr.decode_cf(ds_raw.drop_vars(drop_vars))

            ds_times = pd.DatetimeIndex(ds.time.values)
            if ds_times.tz is not None:
                ds_times = ds_times.tz_convert("UTC").tz_localize(None)

            if tri_node is None:
                logger.info("Building NECOFS parent interpolators and vertical grid...")
                tri_node = Delaunay(
                    np.column_stack((ds["lon"].values, ds["lat"].values))
                )
                tri_elem = Delaunay(
                    np.column_stack((ds["lonc"].values, ds["latc"].values))
                )

                h_grid = LinearNDInterpolator(tri_node, ds["h"].values)(
                    target_pts
                ).reshape(ny, nx)
                siglay = np.asarray(ds["siglay"].values)  # (nsig, node), surface first
                siglay_grid = np.stack(
                    [
                        LinearNDInterpolator(tri_node, siglay[k])(target_pts).reshape(
                            ny, nx
                        )
                        for k in range(siglay.shape[0])
                    ]
                )
                if not np.any(np.isfinite(h_grid)):
                    raise ValueError(f"No NECOFS ocean inside {bbox}")
                z_centres, z_faces = z_levels_for_depth(
                    float(np.nanmax(h_grid)), vertical_spacing_m
                )

            valid_mask = ds_times >= current_dt
            if not valid_mask.any():
                logger.warning(f"No valid times >= {current_dt} found in {dap_url}")
                current_dt += pd.Timedelta(days=1)
                continue

            start_idx = int(np.argmax(valid_mask))
            take_steps = min(duration_hours - hours_fetched, len(ds_times) - start_idx)
            ds_t = ds.isel(time=slice(start_idx, start_idx + take_steps))
            nt = int(take_steps)
            logger.info(f"Extracting {nt} time steps from {dap_url} (parallelized)...")

            def process_time_step(t_idx):
                zeta_t = LinearNDInterpolator(
                    tri_node, ds_t["zeta"].isel(time=t_idx).values
                )(target_pts).reshape(ny, nx)
                z_bottom = np.where(np.isfinite(zeta_t), -h_grid, np.nan)
                z_layers = siglay_grid * (h_grid + zeta_t)[None] + zeta_t[None]

                def layers(name, tri):
                    raw = ds_t[name].isel(time=t_idx).values
                    return np.stack(
                        [
                            LinearNDInterpolator(tri, raw[k])(target_pts).reshape(
                                ny, nx
                            )
                            for k in range(raw.shape[0])
                        ]
                    )

                fields = {
                    "u": layers("u", tri_elem),
                    "v": layers("v", tri_elem),
                    "temp": layers("temp", tri_node),
                    "salt": layers("salinity", tri_node),
                }
                out = {
                    k: sigma_to_z(v, z_layers, z_bottom, z_centres).astype(np.float32)
                    for k, v in fields.items()
                }
                out["zeta"] = zeta_t.astype(np.float32)
                return t_idx, out

            results: list[Optional[dict]] = [None] * nt
            max_workers = int(os.environ.get("ECODATA_CACHE_MAX_WORKERS", 4))
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers
            ) as executor:
                for future in concurrent.futures.as_completed(
                    [executor.submit(process_time_step, t) for t in range(nt)]
                ):
                    t_idx, out = future.result()
                    results[t_idx] = out
            for k in collected:
                collected[k].extend(
                    _require_all([r[k] if r else None for r in results], k)
                )

            hours_fetched += nt
            collected_times.extend(ds_t.time.values)
            last = start_idx + take_steps - 1
            current_dt = ds_times[last] + pd.Timedelta(hours=1)

        except Exception as e:
            logger.error(f"Failed to process NECOFS OBC chunk: {e}")
            break

    if hours_fetched == 0 or h_grid is None:
        return None

    mask = np.isfinite(h_grid).astype(np.int8)
    ds_out = xr.Dataset(
        data_vars={
            "u": (("time", "z", "lat", "lon"), np.stack(collected["u"])),
            "v": (("time", "z", "lat", "lon"), np.stack(collected["v"])),
            "temp": (("time", "z", "lat", "lon"), np.stack(collected["temp"])),
            "salt": (("time", "z", "lat", "lon"), np.stack(collected["salt"])),
            "zeta": (("time", "lat", "lon"), np.stack(collected["zeta"])),
            "h": (("lat", "lon"), h_grid.astype(np.float32)),
            "mask": (("lat", "lon"), mask),
            "z_face": (("z_face",), z_faces),
        },
        coords={
            "time": collected_times,
            "z": z_centres,
            "lat": lat_axis,
            "lon": lon_axis,
        },
        attrs={
            "type": "NECOFS/FVCOM GOM7 parent ocean",
            "source": "UMass Dartmouth SMAST",
            "schema": OBC_SCHEMA,
            "pad_cells": pad_cells,
            "vertical_spacing_m": vertical_spacing_m,
            "requested_bbox": list(bbox),
        },
    )
    units = {
        "u": ("m s-1", "eastward_sea_water_velocity"),
        "v": ("m s-1", "northward_sea_water_velocity"),
        "temp": ("degree_Celsius", "sea_water_temperature"),
        "salt": ("1e-3", "sea_water_practical_salinity"),
        "zeta": ("m", "sea_surface_height_above_geoid"),
        "h": ("m", "sea_floor_depth_below_geoid"),
    }
    for name, (unit, standard_name) in units.items():
        ds_out[name].attrs.update(units=unit, standard_name=standard_name)
    ds_out["mask"].attrs.update(
        long_name="1 where NECOFS has ocean, 0 on land or outside the mesh"
    )
    ds_out["z"].attrs.update(
        units="m", positive="up", axis="Z", long_name="level centre depth"
    )
    ds_out["z_face"].attrs.update(
        units="m", positive="up", long_name="level faces, bottom to top"
    )
    ds_out["lat"].attrs.update(
        units="degrees_north", standard_name="latitude", axis="Y"
    )
    ds_out["lon"].attrs.update(
        units="degrees_east", standard_name="longitude", axis="X"
    )
    return ds_out
