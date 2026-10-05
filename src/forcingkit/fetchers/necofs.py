import warnings
import logging
import pandas as pd
import numpy as np
import xarray as xr
from typing import Optional
from scipy.spatial import Delaunay
from forcingkit import settings


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


class Barycentric:
    """Linear interpolation from scattered points onto fixed targets, with the simplex search and
    barycentric weights computed once.

    Equivalent to `LinearNDInterpolator(tri, values)(targets)`, which repeats the search for every
    call: per parent hour that is one search per layer per variable (181 for LIS), each over every
    target point. Values may carry leading dimensions: (..., npoints) -> (..., *shape). Targets
    outside the triangulation are NaN.
    """

    def __init__(self, tri: Delaunay, targets: np.ndarray, shape: tuple[int, ...]):
        simplex = tri.find_simplex(targets)
        self.inside = simplex >= 0
        s = np.where(self.inside, simplex, 0)
        transform = tri.transform[s]
        b = np.einsum("ijk,ik->ij", transform[:, :2], targets - transform[:, 2])
        self.weights = np.column_stack([b, 1.0 - b.sum(axis=1)])
        self.vertices = tri.simplices[s]
        self.shape = shape

    def __call__(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        out = np.einsum("...pk,pk->...p", values[..., self.vertices], self.weights)
        out[..., ~self.inside] = np.nan
        return out.reshape(values.shape[:-1] + self.shape)


PARENT_RECORD_DIMS = {
    "u": ("z", "lat", "lon"),
    "v": ("z", "lat", "lon"),
    "temp": ("z", "lat", "lon"),
    "salt": ("z", "lat", "lon"),
    "zeta": ("lat", "lon"),
}

PARENT_UNITS = {
    "u": ("m s-1", "eastward_sea_water_velocity"),
    "v": ("m s-1", "northward_sea_water_velocity"),
    "temp": ("degree_Celsius", "sea_water_temperature"),
    "salt": ("1e-3", "sea_water_practical_salinity"),
    "zeta": ("m", "sea_surface_height_above_geoid"),
    "h": ("m", "sea_floor_depth_below_geoid"),
}


def _parent_static(lon_axis, lat_axis, z_centres, z_faces, h_grid, attrs) -> xr.Dataset:
    mask = np.isfinite(h_grid).astype(np.int8)
    ds = xr.Dataset(
        data_vars={
            "h": (("lat", "lon"), h_grid.astype(np.float32)),
            "mask": (("lat", "lon"), mask),
            "z_face": (("z_face",), z_faces),
        },
        coords={"z": z_centres, "lat": lat_axis, "lon": lon_axis},
        attrs=attrs,
    )
    ds["h"].attrs.update(units="m", standard_name="sea_floor_depth_below_geoid")
    ds["mask"].attrs.update(
        long_name="1 where NECOFS has ocean, 0 on land or outside the mesh"
    )
    ds["z"].attrs.update(
        units="m", positive="up", axis="Z", long_name="level centre depth"
    )
    ds["z_face"].attrs.update(
        units="m", positive="up", long_name="level faces, bottom to top"
    )
    ds["lat"].attrs.update(units="degrees_north", standard_name="latitude", axis="Y")
    ds["lon"].attrs.update(units="degrees_east", standard_name="longitude", axis="X")
    return ds


def iter_parent(
    start_date: str,
    duration_hours: int,
    bbox: list[float],
    pad_cells: int = 3,
    vertical_spacing_m: float = 2.0,
):
    """NECOFS GOM7 as a parent ocean, one hourly record at a time.

    Yields `("static", Dataset)` once (lon, lat, z, z_face, h, mask and the store attributes),
    then `("record", time, {u, v, temp, salt, zeta})` for each of `duration_hours` hours starting
    at `start_date`, in order. u, v, temp and salt are (z, lat, lon) and zeta (lat, lon) on a
    regular 0.002 degree grid covering `bbox` plus `pad_cells` donor cells on every side, so the
    parent brackets the child. The vertical axis is true depth: each FVCOM sigma layer is placed at
    z = siglay * (h + zeta) + zeta for its column and interpolated onto fixed levels every
    `vertical_spacing_m` metres, ordered bottom to top. Land, points outside the mesh and levels
    below the sea floor are NaN.

    Raises rather than returning a short record: an unreachable archive file, a missing hour or a
    failed interpolation ends the delivery with an error.
    """
    import concurrent.futures

    min_lon, min_lat, max_lon, max_lat = bbox
    if max_lat < 35.0 or min_lat > 46.0 or max_lon < -77.0 or min_lon > -65.0:
        raise ValueError(f"Bounding box {bbox} outside NECOFS domain.")

    target_dt = pd.to_datetime(start_date)
    if target_dt.tzinfo is not None:
        target_dt = target_dt.tz_convert("UTC").tz_localize(None)

    d_spacing = 0.002
    pad = pad_cells * d_spacing
    lon_axis = np.arange(min_lon - pad, max_lon + pad + d_spacing / 2, d_spacing)
    lat_axis = np.arange(min_lat - pad, max_lat + pad + d_spacing / 2, d_spacing)
    lon_grid, lat_grid = np.meshgrid(lon_axis, lat_axis)
    target_pts = np.column_stack((lon_grid.ravel(), lat_grid.ravel()))
    shape = lon_grid.shape

    at_node = at_elem = None
    h_grid = siglay_grid = None
    z_centres = None
    max_workers = settings.max_workers()

    current_dt = target_dt
    end_dt = target_dt + pd.Timedelta(hours=duration_hours - 1)
    while current_dt <= end_dt:
        # get_necofs_url(t) opens the file stamped t.normalize() + 1 day.
        file_date = necofs_archive_file_date(current_dt)
        dap_url = get_necofs_url(file_date - pd.Timedelta(days=1))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ds_raw = xr.open_dataset(dap_url, engine="pydap", decode_times=False)
        drop_vars = [v for v in ["Itime", "Itime2"] if v in ds_raw.variables]
        ds = xr.decode_cf(ds_raw.drop_vars(drop_vars))
        ds_times = pd.DatetimeIndex(ds.time.values)
        if ds_times.tz is not None:
            ds_times = ds_times.tz_convert("UTC").tz_localize(None)

        if at_node is None:
            logger.info(
                "Building NECOFS parent interpolation weights and vertical grid..."
            )
            at_node = Barycentric(
                Delaunay(np.column_stack((ds["lon"].values, ds["lat"].values))),
                target_pts,
                shape,
            )
            at_elem = Barycentric(
                Delaunay(np.column_stack((ds["lonc"].values, ds["latc"].values))),
                target_pts,
                shape,
            )
            h_grid = at_node(ds["h"].values)
            siglay_grid = at_node(np.asarray(ds["siglay"].values))  # surface first
            if not np.any(np.isfinite(h_grid)):
                raise ValueError(f"No NECOFS ocean inside {bbox}")
            z_centres, z_faces = z_levels_for_depth(
                float(np.nanmax(h_grid)), vertical_spacing_m
            )
            yield (
                "static",
                _parent_static(
                    lon_axis,
                    lat_axis,
                    z_centres,
                    z_faces,
                    h_grid,
                    {
                        "type": "NECOFS/FVCOM GOM7 parent ocean",
                        "source": "UMass Dartmouth SMAST",
                        "schema": OBC_SCHEMA,
                        "pad_cells": pad_cells,
                        "vertical_spacing_m": vertical_spacing_m,
                        "requested_bbox": list(bbox),
                    },
                ),
            )

        wanted = pd.date_range(current_dt, min(end_dt, ds_times[-1]), freq="1h")
        idx = ds_times.get_indexer(wanted)
        if len(wanted) == 0 or (idx < 0).any():
            raise RuntimeError(
                f"{dap_url} lacks hourly records from {current_dt} "
                f"(holds {ds_times[0]} to {ds_times[-1]})"
            )
        logger.info(f"Extracting {len(wanted)} hours from {dap_url}...")

        def process(t_idx):
            snap = ds.isel(time=int(t_idx))
            zeta_t = at_node(snap["zeta"].values)
            z_bottom = np.where(np.isfinite(zeta_t), -h_grid, np.nan)
            z_layers = siglay_grid * (h_grid + zeta_t)[None] + zeta_t[None]
            fields = {
                "u": at_elem(snap["u"].values),
                "v": at_elem(snap["v"].values),
                "temp": at_node(snap["temp"].values),
                "salt": at_node(snap["salinity"].values),
            }
            out = {
                k: sigma_to_z(v, z_layers, z_bottom, z_centres).astype(np.float32)
                for k, v in fields.items()
            }
            out["zeta"] = zeta_t.astype(np.float32)
            return out

        # In batches of max_workers, so at most that many records are held at once.
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            for b in range(0, len(idx), max_workers):
                batch = idx[b : b + max_workers]
                for t, out in zip(
                    wanted[b : b + len(batch)], executor.map(process, batch)
                ):
                    yield ("record", t, out)

        current_dt = wanted[-1] + pd.Timedelta(hours=1)


def fetch_necofs_boundary_conditions(
    start_date: str,
    duration_hours: int,
    bbox: list[float],
    pad_cells: int = 3,
    vertical_spacing_m: float = 2.0,
) -> Optional[xr.Dataset]:
    """`iter_parent` assembled into one in-memory Dataset, for callers that want it whole. The
    service streams the records to disk instead (`dispatcher.dispatch_obc_request`)."""
    static = None
    times: list = []
    collected: dict[str, list[np.ndarray]] = {k: [] for k in PARENT_RECORD_DIMS}
    for item in iter_parent(
        start_date, duration_hours, bbox, pad_cells, vertical_spacing_m
    ):
        if item[0] == "static":
            static = item[1]
            continue
        _, t, rec = item
        times.append(t)
        for k in collected:
            collected[k].append(rec[k])
    if static is None:
        return None
    ds_out = static.assign(
        {
            k: (("time",) + dims, np.stack(collected[k]))
            for k, dims in PARENT_RECORD_DIMS.items()
        }
    ).assign_coords(time=times)
    for name, (unit, standard_name) in PARENT_UNITS.items():
        ds_out[name].attrs.update(units=unit, standard_name=standard_name)
    return ds_out
