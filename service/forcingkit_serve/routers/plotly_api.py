from fastapi import APIRouter, Query, HTTPException, Response
import os
from pathlib import Path
import logging
import json
import numpy as np
import xarray as xr
from forcingkit import settings

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Cache View Data"])


@router.get("/api/v1/cache/data")
async def get_dataset_data(
    dataset_id: str = Query(..., description="ID of the dataset"),
    ext: str = Query(..., description="Extension"),
    var_name: str = Query(..., description="Variable query"),
    time_idx: str = Query("0", description="Time step index OR 'all'"),
    lod: int = Query(0, description="Level of Detail stride (0=auto)"),
):
    cache_dir = Path(settings.cache_dir()).expanduser()
    full_path = os.path.join(
        cache_dir, f"{dataset_id}.zarr" if ext == "zarr" else f"{dataset_id}.{ext}"
    )

    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="Dataset not found")

    try:
        ds = (
            xr.open_zarr(full_path)
            if ext == "zarr"
            else xr.open_dataset(
                full_path, engine="cfgrib" if ext == "grib" else "netcdf4"
            )
        )
        if var_name not in ds.variables:
            raise HTTPException(status_code=404, detail="Variable not found")

        data_var = ds[var_name]

        # Dimensions
        slice_opts = {}
        fetch_all_time = time_idx == "all"

        for time_cand in ["time", "valid_time", "step", "ocean_time"]:
            if time_cand in data_var.dims:
                if not fetch_all_time:
                    idx = int(time_idx)
                    slice_opts[time_cand] = min(idx, ds.sizes[time_cand] - 1)

        for z_name in ["depth", "zC", "level", "s_rho", "surface", "siglay", "siglev"]:
            if z_name in data_var.dims and z_name not in slice_opts:
                slice_opts[z_name] = (
                    -1 if z_name in ("s_rho", "siglay", "siglev") else 0
                )

        valid_opts = {k: v for k, v in slice_opts.items() if k in data_var.dims}
        da = data_var.isel(valid_opts)

        # Smart sub-sampling
        y_name = next(
            (
                str(d)
                for d in da.dims
                if "lat" in str(d) or "eta" in str(d) or str(d) == "y"
            ),
            None,
        )
        x_name = next(
            (
                str(d)
                for d in da.dims
                if "lon" in str(d) or "xi" in str(d) or str(d) == "x"
            ),
            None,
        )

        actual_stride = 1
        if lod > 0:
            actual_stride = lod
        else:
            # Auto compute stride based on spatial coords
            if y_name and x_name:
                max_dim = max(da.sizes[y_name], da.sizes[x_name])
            else:
                max_dim = max(da.shape)
            if max_dim > 200:
                actual_stride = int(np.ceil(max_dim / 200.0))

        if actual_stride > 1:
            slice_dict = {}
            if y_name:
                slice_dict[y_name] = slice(None, None, actual_stride)
            if x_name:
                slice_dict[x_name] = slice(None, None, actual_stride)
            if slice_dict:
                da = da.isel(slice_dict)
            else:
                # Fallback if no specific x/y dimension detected
                spatial_dims = [
                    d
                    for d in da.dims
                    if d
                    not in [
                        "time",
                        "valid_time",
                        "step",
                        "ocean_time",
                        "depth",
                        "zC",
                        "level",
                        "s_rho",
                        "surface",
                        "siglay",
                        "siglev",
                    ]
                ]
                da = da.isel(
                    {d: slice(None, None, actual_stride) for d in spatial_dims}
                )

        array = da.values

        # Replace NaNs with None for JSON
        arr = np.asarray(array)
        masked = np.where(np.isnan(arr), None, arr).tolist()  # type: ignore

        y_vals = (
            ds[y_name].isel({y_name: slice(None, None, actual_stride)}).values.tolist()
            if y_name and y_name in ds.coords
            else None
        )
        x_vals = (
            ds[x_name].isel({x_name: slice(None, None, actual_stride)}).values.tolist()
            if x_name and x_name in ds.coords
            else None
        )

        return Response(
            content=json.dumps(
                {"z": masked, "x": x_vals, "y": y_vals, "cube": fetch_all_time}
            ),
            media_type="application/json",
        )

    except Exception as e:
        logger.error(f"Plotly data prep failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
