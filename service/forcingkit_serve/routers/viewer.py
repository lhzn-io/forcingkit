import os
from pathlib import Path
from fastapi import APIRouter, HTTPException, Query, Response as FastAPIResponse
from fastapi.responses import Response
from pydantic import BaseModel
from typing import List
import io
import logging
from forcingkit import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/cache", tags=["Cache Viewer"])


class CacheItem(BaseModel):
    id: str
    type: str  # 'zarr', 'grib', 'nc'
    size_mb: float
    modified_time: float
    path: str
    # Enriched metadata (populated for zarr datasets)
    label: str | None = None  # Human-readable label
    variables: list[str] | None = None  # Data variable names
    grid_shape: str | None = None  # e.g. "3×4" or "point"
    time_steps: int | None = None  # Number of time steps
    time_range: str | None = None  # e.g. "Mar 2 00:00 → Mar 3 23:00"
    source: str | None = None  # e.g. "ERA5", "HRRR", "NOAA Water Level"
    donor_model: str | None = None  # e.g. "HYCOM", "NERACOOS", "NECOFS"
    resolution: str | None = None  # e.g. "~7km", "~200m", "~31km"
    spatial_extent: str | None = None  # e.g. "-71.25, 42.75 → -70.25, 43.50"
    target_date: str | None = None  # ISO timestamp from sidecar metadata (e.g. ICs)


def _enrich_zarr_metadata(zarr_path: Path) -> dict:
    label = None
    label = None
    try:
        import xarray as xr
        import numpy as np

        # Open with decode_times=True for proper datetime parsing
        try:
            ds = xr.open_zarr(str(zarr_path), consolidated=False, decode_times=True)
        except Exception:
            ds = xr.open_zarr(str(zarr_path), consolidated=False, decode_times=False)
        # Filter out scalar (0-d) variables - these are GRIB metadata artifacts
        variables = [v for v in ds.data_vars if ds[v].ndim > 0]

        # Grid shape, spatial extent, and resolution from coordinate arrays
        lat_dim = next(
            (
                d
                for d in ["latitude", "lat", "y", "eta", "eta_rho", "eta_u", "eta_v"]
                if d in ds.sizes
            ),
            None,
        )
        lon_dim = next(
            (
                d
                for d in ["longitude", "lon", "x", "xi", "xi_rho", "xi_u", "xi_v"]
                if d in ds.sizes
            ),
            None,
        )
        # Separate dimensions from physical coordinates
        lat_coord = next(
            (
                c
                for c in ["lat_rho", "latitude", "lat", "lat_u", "lat_v"]
                if c in ds.coords
            ),
            None,
        )
        lon_coord = next(
            (
                c
                for c in ["lon_rho", "longitude", "lon", "lon_u", "lon_v"]
                if c in ds.coords
            ),
            None,
        )

        if not lat_coord and lat_dim in ds.coords:
            lat_coord = lat_dim
        if not lon_coord and lon_dim in ds.coords:
            lon_coord = lon_dim

        spatial_extent = None
        resolution = None
        if lat_dim and lon_dim:
            nlat, nlon = ds.sizes[lat_dim], ds.sizes[lon_dim]
            grid_shape = f"{nlat}\u00d7{nlon}" if nlat > 0 and nlon > 0 else "point"
        elif lat_coord and lon_coord and lat_coord in ds.coords:
            # Curvilinear: infer shape from coord arrays
            lat_arr = ds[lat_coord].values
            nlat = lat_arr.shape[0] if lat_arr.ndim >= 1 else 0
            nlon = (
                lat_arr.shape[1]
                if lat_arr.ndim >= 2
                else (lat_arr.shape[0] if lat_arr.ndim == 1 else 0)
            )
            grid_shape = f"{nlat}\u00d7{nlon}" if nlat > 0 and nlon > 0 else "point"
        else:
            nlat = nlon = 0
            grid_shape = "point"

        # Extract coordinate bounds for spatial extent and resolution
        if (
            lat_coord
            and lon_coord
            and lat_coord in ds.coords
            and lon_coord in ds.coords
        ):
            lat_vals = ds[lat_coord].values
            lon_vals = ds[lon_coord].values
            if lat_vals.size > 0 and lon_vals.size > 0:
                spatial_extent = (
                    f"{float(np.min(lon_vals)):.2f}, {float(np.min(lat_vals)):.2f}"
                    f" \u2192 {float(np.max(lon_vals)):.2f}, {float(np.max(lat_vals)):.2f}"
                )
                # Compute median grid spacing in meters
                try:
                    if lat_vals.ndim == 1 and len(lat_vals) > 1:
                        dlat = float(np.median(np.abs(np.diff(lat_vals))))
                        dlon = float(np.median(np.abs(np.diff(lon_vals))))
                    elif (
                        lat_vals.ndim == 2
                        and lat_vals.shape[0] > 1
                        and lat_vals.shape[1] > 1
                    ):
                        dlat = float(np.median(np.abs(np.diff(lat_vals, axis=0))))
                        dlon = float(np.median(np.abs(np.diff(lon_vals, axis=1))))
                    else:
                        dlat = dlon = 0
                    if dlat > 0 or dlon > 0:
                        mid_lat = float(np.mean(lat_vals))
                        dx_m = dlon * 111_320 * np.cos(np.radians(mid_lat))
                        dy_m = dlat * 111_320
                        avg_m = (dx_m + dy_m) / 2
                        if avg_m >= 1000:
                            resolution = f"~{avg_m / 1000:.0f}km"
                        else:
                            resolution = f"~{avg_m:.0f}m"
                except Exception:
                    pass

        # Time info - use decoded timestamps when available
        time_dim = next(
            (d for d in ["time", "t", "step", "nt", "timeseries"] if d in ds.sizes),
            None,
        )
        time_steps = ds.sizes[time_dim] if time_dim else None
        source = ds.attrs.get("type", None) or ds.attrs.get("source", None)
        if source and "NECOFS" in source:
            donor_model = "UMASS"
            source = "NECOFS ~193m"
        elif source and "HRRR" in source:
            donor_model = "NOAA"
        elif source and "ERA" in source:
            donor_model = "ECMWF"
        else:
            donor_model = ds.attrs.get("donor_id", None)

        time_range = None
        delta_t_str = ""

        if "target_date" in ds.attrs:
            try:
                from datetime import datetime as dt
                import pandas as pd

                d0 = pd.to_datetime(ds.attrs["target_date"])
                time_range = f"{d0.strftime('%b %d %H:%M')} UTC"
            except Exception:
                time_range = f"{ds.attrs['target_date']} UTC"
            if not time_steps:
                time_steps = 1
        elif time_dim and time_steps and time_steps > 0:
            try:
                from datetime import datetime as dt
                import pandas as pd

                t_vals = ds[time_dim].values
                t0 = np.datetime64(t_vals[0], "ns")
                t1 = np.datetime64(t_vals[-1], "ns")
                d0 = t0.astype("datetime64[s]").astype(dt)
                d1 = t1.astype("datetime64[s]").astype(dt)

                if len(t_vals) > 1:
                    dt_secs = int(
                        np.timedelta64(np.datetime64(t_vals[1], "ns") - t0, "s").astype(
                            int
                        )
                    )
                    if dt_secs > 0:
                        if dt_secs >= 3600:
                            delta_t_str = f" (Δt={dt_secs // 3600}hr)"
                        else:
                            delta_t_str = f" (Δt={dt_secs // 60}m)"
                    time_range = f"{d0.strftime('%b %d %H:%M')} → {d1.strftime('%b %d %H:%M')} UTC{delta_t_str}"
                else:
                    time_range = f"{d0.strftime('%b %d %H:%M')} UTC"
            except Exception:
                pass

        label = None
        return {
            "label": locals().get("label", None),
            "variables": locals().get("variables", []),
            "grid_shape": locals().get("grid_shape", None),
            "time_steps": locals().get("time_steps", None),
            "time_range": locals().get("time_range", None),
            "source": locals().get("source", None),
            "donor_model": locals().get("donor_model", None),
            "resolution": locals().get("resolution", None),
            "spatial_extent": locals().get("spatial_extent", None),
            "target_date": locals().get("target_date", None),
        }
    except Exception as e:
        logger.warning(f"Failed to enrich metadata for {zarr_path.name}: {e}")
        label = None
        return {}


@router.get("/inventory", response_model=List[CacheItem])
async def get_cache_inventory(response: FastAPIResponse):
    """Crawls the local data cache and returns an inventory of all processed forcing datasets."""
    response.headers["Cache-Control"] = "no-store"
    cache_dir = Path(Path(settings.cache_dir()).expanduser())
    if not cache_dir.exists():
        return []

    inventory = []

    # 1. Find Zarr datasets (directories)
    for zarr_path in cache_dir.rglob("*.zarr"):
        if zarr_path.is_dir():
            size_bytes = sum(
                f.stat().st_size for f in zarr_path.rglob("*") if f.is_file()
            )
            meta = _enrich_zarr_metadata(zarr_path)
            inventory.append(
                CacheItem(
                    id=zarr_path.stem,
                    type="zarr",
                    size_mb=round(size_bytes / (1024 * 1024), 2),
                    modified_time=zarr_path.stat().st_mtime,
                    path=str(zarr_path),
                    **meta,
                )
            )

    # 2. Find GRIB/NC datasets (files)
    for ext_glob in ["*.grib", "*.grib2", "*.nc"]:
        for file_path in cache_dir.rglob(ext_glob):
            if file_path.is_file():
                stem = file_path.stem.lower()
                label = None
                source = None
                donor_model = None
                variables = None
                grid_shape = None
                spatial_extent = None
                resolution = None

                if "hrrr" in stem:
                    if "_f" in stem:
                        continue
                    source = "HRRR ~3km"
                    donor_model = "NOAA"
                    resolution = "~3km"
                    label = f"{source} Boundary Conditions"
                elif "era5" in stem:
                    source = "ERA5T ~31km" if "era5t" in stem else "ERA5 ~31km"
                    donor_model = "ECMWF"
                    resolution = "~31km"
                    label = f"{source} Boundary Conditions"
                elif "necofs" in stem:
                    source = "NECOFS ~193m"
                    donor_model = "UMASS"
                    resolution = "~193m"
                    label = "NECOFS Raw Validated"

                # Enrich GRIB/NC with grid and variable metadata
                try:
                    import xarray as xr
                    import numpy as np

                    # Suppress noisy expected grib warnings
                    import logging

                    logging.getLogger("cfgrib.messages").setLevel(logging.ERROR)

                    gds = None
                    for _inv_filter in [
                        {"typeOfLevel": "heightAboveGround", "stepType": "instant"},
                        {"typeOfLevel": "surface", "stepType": "instant"},
                        {"typeOfLevel": "heightAboveGround"},
                        {"typeOfLevel": "surface"},
                    ]:
                        try:
                            _candidate = xr.open_dataset(
                                str(file_path),
                                decode_times=True,
                                engine="cfgrib",
                                backend_kwargs={"filter_by_keys": _inv_filter},
                            )
                            if gds is None or len(list(_candidate.data_vars)) > len(
                                list(gds.data_vars)
                            ):
                                gds = _candidate
                            else:
                                _candidate.close()
                        except Exception:
                            continue
                    if gds is None:
                        try:
                            gds = xr.open_dataset(str(file_path), decode_times=True)
                        except Exception:
                            gds = None

                    if gds is not None:
                        variables = [str(v) for v in gds.data_vars if gds[v].ndim > 0]
                        _lat = next(
                            (c for c in gds.coords if "lat" in str(c).lower()), None
                        )
                        _lon = next(
                            (c for c in gds.coords if "lon" in str(c).lower()), None
                        )
                        time_dim = next(
                            (
                                d
                                for d in ["time", "t", "step", "valid_time"]
                                if d in gds.sizes or d in gds.coords
                            ),
                            None,
                        )
                        time_steps = 1
                        time_range = None
                        if time_dim:
                            try:
                                from datetime import datetime as dt
                                import numpy as np

                                t_vals = gds[time_dim].values
                                if getattr(t_vals, "ndim", 0) == 0:
                                    t_vals = np.array([t_vals])
                                time_steps = len(t_vals)
                                t0 = np.datetime64(t_vals[0], "ns")
                                t1 = np.datetime64(t_vals[-1], "ns")
                                d0 = t0.astype("datetime64[s]").astype(dt)
                                d1 = t1.astype("datetime64[s]").astype(dt)
                                delta_t_str = ""
                                if len(t_vals) > 1:
                                    dt_secs = int(
                                        np.timedelta64(
                                            np.datetime64(t_vals[1], "ns") - t0, "s"
                                        ).astype(int)
                                    )
                                    if dt_secs > 0:
                                        delta_t_str = (
                                            f" (Δt={dt_secs // 3600}hr)"
                                            if dt_secs >= 3600
                                            else f" (Δt={dt_secs // 60}m)"
                                        )
                                if d0 == d1:
                                    time_range = f"{d0.strftime('%b %d %H:%M')} UTC"
                                else:
                                    time_range = f"{d0.strftime('%b %d %H:%M')} → {d1.strftime('%b %d %H:%M')} UTC{delta_t_str}"
                            except Exception:
                                pass

                        if _lat and _lon:
                            if _lat in gds.coords and _lon in gds.coords:
                                lat_v = gds[_lat].values
                                lon_v = gds[_lon].values
                                if getattr(lat_v, "ndim", 0) == 2:
                                    grid_shape = (
                                        f"{lat_v.shape[0]}\u00d7{lat_v.shape[1]}"
                                    )
                                else:
                                    nlat = (
                                        lat_v.shape[0]
                                        if getattr(lat_v, "ndim", 0) > 0
                                        else 1
                                    )
                                    nlon = (
                                        lon_v.shape[0]
                                        if getattr(lon_v, "ndim", 0) > 0
                                        else 1
                                    )
                                    grid_shape = f"{nlat}\u00d7{nlon}"

                                if lat_v.size > 0 and lon_v.size > 0:
                                    spatial_extent = (
                                        f"{float(np.min(lon_v)):.2f}, {float(np.min(lat_v)):.2f}"
                                        f" \u2192 {float(np.max(lon_v)):.2f}, {float(np.max(lat_v)):.2f}"
                                    )
                                    if (
                                        getattr(lat_v, "ndim", 0) == 1
                                        and len(lat_v) > 1
                                        and not resolution
                                    ):
                                        dlat = float(np.median(np.abs(np.diff(lat_v))))
                                        dlon = float(np.median(np.abs(np.diff(lon_v))))
                                        mid_lat = float(np.mean(lat_v))
                                        avg_m = (
                                            (
                                                dlon
                                                * 111320
                                                * np.cos(np.radians(mid_lat))
                                            )
                                            + (dlat * 111320)
                                        ) / 2
                                        resolution = (
                                            f"~{avg_m / 1000:.0f}km"
                                            if avg_m >= 1000
                                            else f"~{avg_m:.0f}m"
                                        )
                        gds.close()
                except Exception as enrich_err:
                    logger.debug(
                        f"GRIB enrichment failed for {file_path.name}: {enrich_err}"
                    )

                inventory.append(
                    CacheItem(
                        id=file_path.stem,
                        type=file_path.suffix.lstrip("."),
                        size_mb=round(file_path.stat().st_size / (1024 * 1024), 2),
                        modified_time=file_path.stat().st_mtime,
                        path=str(file_path),
                        label=label,
                        source=source,
                        donor_model=donor_model,
                        variables=variables,
                        grid_shape=grid_shape,
                        spatial_extent=spatial_extent,
                        resolution=resolution,
                        time_steps=time_steps,
                        time_range=time_range,
                    )
                )

    # Sort by descending modification time (newest first)
    inventory.sort(key=lambda x: x.modified_time, reverse=True)
    return inventory


@router.delete("/dataset/{dataset_id}")
async def delete_dataset(dataset_id: str):
    """Deletes a single dataset from the cache by ID."""
    import shutil

    cache_dir = Path(settings.cache_dir()).expanduser()

    # Use rglob to find the dataset ID anywhere in the cache
    matches = list(cache_dir.rglob(f"{dataset_id}.zarr"))
    if matches:
        zarr_path = matches[0]
        if zarr_path.is_dir():
            shutil.rmtree(zarr_path)
            # Also remove sidecar metadata if it exists (JSON companion may be named nicely)
            sidecar = zarr_path.with_suffix(".json")
            if sidecar.exists():
                sidecar.unlink()
            # Also remove old metadata pattern
            old_sidecar = zarr_path.parent / f"{dataset_id}_metadata.json"
            if old_sidecar.exists():
                old_sidecar.unlink()
            logger.info(f"Deleted Zarr dataset: {zarr_path}")
            return {"status": "success", "message": f"Deleted {dataset_id}"}

    for ext in ["grib", "grib2", "nc"]:
        matches = list(cache_dir.rglob(f"{dataset_id}.{ext}"))
        if matches:
            file_path = matches[0]
            if file_path.is_file():
                file_path.unlink()
                # Remove sidecar if exists
                sidecar = file_path.with_suffix(".json")
                if sidecar.exists():
                    sidecar.unlink()
                logger.info(f"Deleted dataset: {file_path}")
                return {"status": "success", "message": f"Deleted {dataset_id}"}

    raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found.")


@router.get("/preview")
async def get_dataset_preview(
    dataset_id: str = Query(..., description="ID of the dataset (without extension)"),
    ext: str = Query(..., description="Extension of dataset (zarr, grib, nc)"),
    var_name: str = Query(..., description="Variable to plot (e.g., u10, v10, tp)"),
    time_idx: int = Query(0, description="Time step index relative to data slice"),
):
    """
    Renders a dynamic Matplotlib base64 preview of a cached multidimensional dataset.
    """

    # Deferred plotting imports for memory savings on edge devices
    import xarray as xr
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np

    cache_dir = Path(settings.cache_dir()).expanduser()
    full_path = ""

    if ext == "zarr":
        full_path = os.path.join(cache_dir, f"{dataset_id}.zarr")
    else:
        full_path = os.path.join(cache_dir, f"{dataset_id}.{ext}")

    if not os.path.exists(full_path):
        raise HTTPException(
            status_code=404, detail=f"Dataset file {full_path} not found."
        )

    try:
        ds = None
        if ext == "zarr":
            ds = xr.open_zarr(full_path, consolidated=False, decode_times=True)
        else:
            # Handle ambiguous GRIB levels by attempting common filters
            try:
                ds = xr.open_dataset(full_path, decode_times=True)
            except Exception as e:
                # Try common level types in priority order for surface forcing
                for filter_obj in [
                    {"typeOfLevel": "surface", "stepType": "instant"},
                    {"typeOfLevel": "heightAboveGround", "stepType": "instant"},
                    {"typeOfLevel": "surface"},
                    {"typeOfLevel": "heightAboveGround"},
                    {"typeOfLevel": "isobaricInhPa"},
                    {"typeOfLevel": "meanSea"},
                    {"typeOfLevel": "atmosphere"},
                ]:
                    try:
                        candidate = xr.open_dataset(
                            full_path,
                            decode_times=True,
                            engine="cfgrib",
                            backend_kwargs={"filter_by_keys": filter_obj},
                        )
                        # Accept this filter if it contains the requested variable
                        if var_name in candidate.data_vars or ds is None:
                            ds = candidate
                            if var_name in candidate.data_vars:
                                break
                    except Exception:
                        continue
                if ds is None:
                    raise HTTPException(
                        status_code=500, detail=f"Failed to parse GRIB: {e}"
                    )

        # Map common variable names across naming conventions
        if var_name not in ds.variables:
            var_mapping = {
                # Ocean IC aliases
                "u": ["water_u", "u_current", "uo", "u10"],
                "v": ["water_v", "v_current", "vo", "v10"],
                "temp": ["water_temp", "temperature", "thetao"],
                "salt": ["salinity", "so"],
                "zeta": ["surf_el", "ssh", "zos"],
                # Atmospheric aliases (ERA5 u10/v10 ↔ HRRR u/v at heightAboveGround)
                "u10": ["u", "10u", "u_10m"],
                "v10": ["v", "10v", "v_10m"],
                "t2m": ["t", "2t", "t_2m"],
            }
            if var_name in var_mapping:
                for alt_name in var_mapping[var_name]:
                    if alt_name in ds.data_vars:
                        var_name = alt_name
                        break

        if var_name not in ds.variables:
            available = list(ds.data_vars.keys())
            raise HTTPException(
                status_code=400,
                detail=f"Variable '{var_name}' not found. Try: {available}",
            )

        data_var = ds[var_name]

        if data_var.ndim == 0:
            raise HTTPException(
                status_code=400,
                detail=f"Variable '{var_name}' is scalar (0-d) and cannot be rendered.",
            )

        # Detect time dimension
        time_dim = next(
            (
                d
                for d in ["time", "t", "step", "nt", "timeseries"]
                if d in data_var.dims
            ),
            None,
        )
        if not time_dim and len(data_var.dims) == 1:
            time_dim = str(data_var.dims[0])

        # Detect spatial dimensions
        lat_dim = next(
            (
                d
                for d in ["latitude", "lat", "y", "eta", "eta_rho", "eta_u", "eta_v"]
                if d in data_var.dims
            ),
            None,
        )
        lon_dim = next(
            (
                d
                for d in ["longitude", "lon", "x", "xi", "xi_rho", "xi_u", "xi_v"]
                if d in data_var.dims
            ),
            None,
        )
        has_spatial = (
            lat_dim
            and lon_dim
            and ds.sizes.get(lat_dim, 0) > 0
            and ds.sizes.get(lon_dim, 0) > 0
        )

        # Determine rendering mode:
        # - "timeseries" for 1D time-only data or point grids (0×0 spatial)
        # - "timeseries_grid" for very small grids (<=3×3) where overlaid timeseries are useful
        # - "heatmap_small" for small grids (<= 20 cells per dim) with annotated values
        # - "heatmap" for larger spatial grids
        nlat = ds.sizes.get(lat_dim, 0) if lat_dim else 0
        nlon = ds.sizes.get(lon_dim, 0) if lon_dim else 0

        if not has_spatial and time_dim:
            render_mode = "timeseries"
        elif has_spatial and nlat <= 3 and nlon <= 3 and time_dim:
            render_mode = "timeseries_grid"
        elif has_spatial and nlat <= 20 and nlon <= 20:
            render_mode = "heatmap_small"
        else:
            render_mode = "heatmap"

        # Decode timestamp for the selected time index
        time_label = f"t={time_idx}"
        if time_dim and time_dim in ds.variables:
            try:
                t_vals = ds[time_dim].values
                idx = min(time_idx, len(t_vals) - 1)
                t_val = t_vals[idx]

                # Robust decoding to numpy datetime64
                if isinstance(t_val, np.datetime64):
                    d64 = t_val.astype("datetime64[us]")
                else:
                    try:
                        d64 = np.datetime64(t_val, "us")
                    except Exception:
                        # Fallback for floats: assume seconds from epoch
                        d64 = np.datetime64(int(float(t_val)), "s").astype(
                            "datetime64[us]"
                        )

                from datetime import datetime as dt

                d = d64.astype(dt)
                time_label = d.strftime("%Y%m%d-%H:%M:%S.%f")[:-3]
            except Exception:
                pass

        fig, ax = plt.subplots(figsize=(7, 4.5))
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#161b22")

        # Subtitle with decoded timestamp
        if time_label:
            fig.text(
                0.5,
                0.97,
                time_label,
                ha="center",
                va="top",
                fontsize=8.5,
                color="#8b949e",
                fontstyle="italic",
            )

        # Style for dark theme
        for spine in ax.spines.values():
            spine.set_color("#30363d")
        ax.tick_params(colors="#8b949e", labelsize=8)
        ax.xaxis.label.set_color("#c9d1d9")
        ax.yaxis.label.set_color("#c9d1d9")
        ax.title.set_color("#f0f6fc")

        if render_mode == "timeseries":
            # Plot all time steps as a line chart
            values = data_var.values.flatten()
            valid = (
                ~np.isnan(values)
                if values.dtype.kind == "f"
                else np.ones_like(values, dtype=bool)
            )
            ax.plot(
                np.arange(len(values)),
                values,
                color="#58a6ff",
                linewidth=1.5,
                marker="o" if len(values) < 50 else None,
                markersize=3,
            )
            if time_idx < len(values):
                ax.axvline(
                    time_idx, color="#f85149", linewidth=1, linestyle="--", alpha=0.7
                )
                if valid[time_idx]:
                    ax.annotate(
                        f"{values[time_idx]:.3f}",
                        xy=(time_idx, values[time_idx]),
                        xytext=(5, 10),
                        textcoords="offset points",
                        color="#f0f6fc",
                        fontsize=9,
                        fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.3", fc="#21262d", ec="#30363d"),
                    )
            ax.set_xlabel("Time Step", fontsize=9)
            ax.set_ylabel(var_name, fontsize=9)
            ax.set_title(f"{dataset_id} - {var_name} ({time_label})", fontsize=10)
            ax.grid(True, alpha=0.15, color="#8b949e")

        elif render_mode == "timeseries_grid":
            # Small grid: plot timeseries for each cell as overlaid lines
            if time_dim:
                for ilat in range(nlat):
                    for ilon in range(nlon):
                        slice_opts = {}
                        if lat_dim:
                            slice_opts[lat_dim] = ilat
                        if lon_dim:
                            slice_opts[lon_dim] = ilon
                        ts = data_var.isel(slice_opts).values.flatten()
                        label = f"({ilat},{ilon})" if nlat * nlon <= 16 else None
                        ax.plot(ts, linewidth=1, alpha=0.7, label=label)
                if time_idx is not None:
                    ax.axvline(
                        time_idx,
                        color="#f85149",
                        linewidth=1,
                        linestyle="--",
                        alpha=0.7,
                    )
                ax.set_xlabel("Time Step", fontsize=9)
                ax.set_ylabel(var_name, fontsize=9)
                ax.set_title(
                    f"{dataset_id} \u2014 {var_name} ({nlat}\u00d7{nlon} grid)",
                    fontsize=10,
                )
                ax.grid(True, alpha=0.15, color="#8b949e")
                if nlat * nlon <= 16:
                    ax.legend(fontsize=7, loc="upper right", ncol=2, framealpha=0.3)

        elif render_mode == "heatmap_small":
            # Small spatial grid with value annotations
            slice_opts = {}
            if time_dim:
                max_t = ds.sizes[time_dim] - 1
                slice_opts[time_dim] = min(time_idx, max_t)
            for z_name in [
                "depth",
                "zC",
                "level",
                "s_rho",
                "s_w",
                "isobaricInhPa",
                "number",
                "surface",
                "step",
            ]:
                if z_name in data_var.dims:
                    slice_opts[z_name] = -1 if z_name in ("s_rho", "s_w") else 0
            valid_opts = {k: v for k, v in slice_opts.items() if k in data_var.dims}
            array = data_var.isel(valid_opts).values

            vmin = np.nanmin(array) if not np.all(np.isnan(array)) else 0.0
            vmax = np.nanmax(array) if not np.all(np.isnan(array)) else 1.0
            im = ax.imshow(
                array, origin="lower", cmap="turbo", aspect="auto", vmin=vmin, vmax=vmax
            )
            # Annotate cell values
            for yi in range(array.shape[0]):
                for xi in range(array.shape[1]):
                    v = array[yi, xi]
                    if not np.isnan(v):
                        ax.text(
                            xi,
                            yi,
                            f"{v:.2f}",
                            ha="center",
                            va="center",
                            fontsize=7,
                            color="white",
                            fontweight="bold",
                            bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.4),
                        )
            cbar = plt.colorbar(im, ax=ax, label=var_name)
            cbar.ax.yaxis.set_tick_params(color="#8b949e")
            cbar.ax.yaxis.label.set_color("#c9d1d9")
            plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#8b949e")
            ax.set_title(f"{dataset_id} \u2014 {var_name} ({time_label})", fontsize=10)

        else:
            # Standard heatmap for larger grids
            slice_opts = {}
            if time_dim:
                max_t = ds.sizes[time_dim] - 1
                slice_opts[time_dim] = min(time_idx, max_t)
            for z_name in [
                "depth",
                "zC",
                "level",
                "s_rho",
                "s_w",
                "isobaricInhPa",
                "number",
                "surface",
                "step",
            ]:
                if z_name in data_var.dims:
                    slice_opts[z_name] = -1 if z_name in ("s_rho", "s_w") else 0
            valid_opts = {k: v for k, v in slice_opts.items() if k in data_var.dims}
            array = data_var.isel(valid_opts).values

            vmin = np.nanmin(array) if not np.all(np.isnan(array)) else 0.0
            vmax = np.nanmax(array) if not np.all(np.isnan(array)) else 1.0
            im = ax.imshow(
                array, origin="lower", cmap="turbo", aspect="auto", vmin=vmin, vmax=vmax
            )
            cbar = plt.colorbar(im, ax=ax, label=var_name)
            cbar.ax.yaxis.set_tick_params(color="#8b949e")
            cbar.ax.yaxis.label.set_color("#c9d1d9")
            plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#8b949e")
            ax.set_title(f"{dataset_id} \u2014 {var_name} ({time_label})", fontsize=10)

        buf = io.BytesIO()
        plt.tight_layout(rect=(0, 0, 1, 0.95) if time_label else (0, 0, 1, 1))
        plt.savefig(buf, format="png", dpi=100, facecolor=fig.get_facecolor())
        buf.seek(0)
        plt.close(fig)

        return Response(content=buf.getvalue(), media_type="image/png")

    except Exception as e:
        logger.error(f"Failed to generate preview for {dataset_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/preview3d")
async def get_dataset_preview_3d(
    dataset_id: str = Query(..., description="ID of the dataset (without extension)"),
    time_idx: int = Query(0, description="Time step index"),
):
    """
    Returns downsampled 3D vector field JSON for Three.js rendering.
    Designed for IC/BC datasets with u,v velocity fields.
    """
    import xarray as xr
    import numpy as np
    import json

    cache_dir = Path(settings.cache_dir()).expanduser()
    full_path = os.path.join(cache_dir, f"{dataset_id}.zarr")

    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found.")

    try:
        ds = xr.open_zarr(full_path, consolidated=False, decode_times=True)
        vars_list = list(ds.data_vars)

        # Detect u/v variable pairs
        u_var = v_var = None
        for pair in [("u", "v"), ("water_u", "water_v"), ("u10", "v10")]:
            if pair[0] in vars_list and pair[1] in vars_list:
                u_var, v_var = pair
                break

        if u_var is None:
            raise HTTPException(
                status_code=400,
                detail=f"No u,v vector pair found. Variables: {vars_list}",
            )

        u_da = ds[u_var]
        v_da = ds[v_var]

        # Detect depth coordinate (s_rho, depth, z)
        depth_name = next(
            (
                n
                for n in ["s_rho", "depth", "z", "level"]
                if n in ds.coords or n in ds.dims
            ),
            None,
        )

        # Select time slice
        time_name = next((n for n in ["time", "ocean_time", "t"] if n in ds.dims), None)
        if time_name and time_name in u_da.dims:
            t_idx = min(time_idx, u_da.sizes[time_name] - 1)
            u_da = u_da.isel({time_name: t_idx})
            v_da = v_da.isel({time_name: t_idx})

        # Find the best lon/lat coordinate for each variable.
        # Staggered ROMS grids have per-variable coords (lon_u/lat_u, lon_v/lat_v).
        # We require the coord's dims to be a subset of the variable's dims so we
        # don't accidentally pick up a rho-grid coord for a u-grid variable.
        def _find_coord(da, candidates):
            for n in candidates:
                if n in ds.variables and set(ds[n].dims).issubset(set(da.dims)):
                    return n
            return None

        u_lon_name = _find_coord(
            u_da, [f"lon_{u_var}", "longitude", "lon", "lon_rho", "x", "xi"]
        )
        u_lat_name = _find_coord(
            u_da, [f"lat_{u_var}", "latitude", "lat", "lat_rho", "y", "eta"]
        )

        # Squeeze out broadcast/extra dims: keep only depth + the spatial dims that
        # belong to the variable's own coordinate (handles the rho-dim bleed-in from
        # ROMS curvilinear datasets).
        def _squeeze_broadcast(da, lon_cname, lat_cname):
            if lon_cname is None:
                return da
            spatial = set(ds[lon_cname].dims)
            if lat_cname and lat_cname in ds.variables:
                spatial |= set(ds[lat_cname].dims)
            keep = spatial | ({depth_name} if depth_name else set())
            extra = [d for d in da.dims if d not in keep]
            # Mean across broadcast dims (skipna so masked corners don't pollute)
            return da.mean(dim=extra, skipna=True) if extra else da

        u_da = _squeeze_broadcast(u_da, u_lon_name, u_lat_name)
        v_lon_name = _find_coord(
            v_da, [f"lon_{v_var}", "longitude", "lon", "lon_rho", "x", "xi"]
        )
        v_lat_name = _find_coord(
            v_da, [f"lat_{v_var}", "latitude", "lat", "lat_rho", "y", "eta"]
        )
        v_da = _squeeze_broadcast(v_da, v_lon_name, v_lat_name)

        # Get coordinate arrays (2D for curvilinear grids)
        lons = (
            ds[u_lon_name].values
            if u_lon_name
            else np.arange(u_da.shape[-1], dtype=float)
        )
        lats = (
            ds[u_lat_name].values
            if u_lat_name
            else np.arange(u_da.shape[-2] if u_da.ndim >= 2 else 1, dtype=float)
        )
        depths = ds[depth_name].values if depth_name else np.array([0.0])

        lons_2d = lons.ndim == 2
        lats_2d = lats.ndim == 2

        u_arr = u_da.values
        v_arr = v_da.values

        # Replace NaN/missing with 0
        u_arr = np.nan_to_num(u_arr, nan=0.0)
        v_arr = np.nan_to_num(v_arr, nan=0.0)

        def _lon(iy: int, ix: int) -> float:
            if lons_2d:
                return float(lons[iy, ix])
            return float(lons[ix]) if ix < len(lons) else float(ix)

        def _lat(iy: int, ix: int) -> float:
            if lats_2d:
                return float(lats[iy, ix])
            return float(lats[iy]) if iy < len(lats) else float(iy)

        # u and v may have different spatial shapes on staggered grids - clamp v indices
        v_shape = v_arr.shape

        def _v(iz_or_none, iy: int, ix: int) -> float:
            if v_arr.ndim == 2:
                return float(v_arr[min(iy, v_shape[0] - 1), min(ix, v_shape[1] - 1)])
            if v_arr.ndim == 3:
                iz = iz_or_none if iz_or_none is not None else 0
                return float(
                    v_arr[
                        min(iz, v_shape[0] - 1),
                        min(iy, v_shape[1] - 1),
                        min(ix, v_shape[2] - 1),
                    ]
                )
            return 0.0

        vectors = []
        max_vectors = 5000  # cap for UI performance

        if u_arr.ndim == 1:
            # 1D time-series (already time-sliced → scalar)
            vectors.append(
                {
                    "lon": float(lons[0]) if len(lons) > 0 else 0,
                    "lat": float(lats[0]) if len(lats) > 0 else 0,
                    "depth": 0,
                    "u": float(u_arr),
                    "v": float(v_arr),
                }
            )
        elif u_arr.ndim == 2:
            # 2D: (lat, lon) or (eta, xi) - curvilinear or rectilinear
            ny, nx = u_arr.shape
            step = max(1, int(np.sqrt(ny * nx / max_vectors)))
            for iy in range(0, ny, step):
                for ix in range(0, nx, step):
                    u_val = float(u_arr[iy, ix])
                    v_val = _v(None, iy, ix)
                    if np.isnan(u_val) or np.isnan(v_val):
                        continue
                    if abs(u_val) < 1e-10 and abs(v_val) < 1e-10:
                        continue
                    vectors.append(
                        {
                            "lon": _lon(iy, ix),
                            "lat": _lat(iy, ix),
                            "depth": 0,
                            "u": u_val,
                            "v": v_val,
                        }
                    )
        elif u_arr.ndim == 3:
            # 3D: (depth, lat, lon) or (s_rho, eta, xi)
            nz, ny, nx = u_arr.shape
            z_step = max(1, nz // 8)  # ~8 depth slices like hydro viewer
            xy_step = max(
                1, int(np.sqrt(ny * nx / (max_vectors // max(1, nz // z_step))))
            )
            for iz in range(0, nz, z_step):
                d_raw = float(depths[iz]) if iz < len(depths) else float(iz)
                # Map s_rho (-1 to 0) to physical depth in meters
                if depth_name == "s_rho" and -1.5 < d_raw < 0.5:
                    depth_frac = float(np.clip(-d_raw, 0, 1))
                    d = d_raw * 50.0  # approximate 50m water column
                else:
                    # For regular depth coords, compute frac from min/max
                    d = d_raw
                    d_min = float(np.min(depths))
                    d_max = float(np.max(depths))
                    d_range = abs(d_max - d_min) if d_max != d_min else 1.0

                    # Surface is usually closest to 0.
                    # If coords are negative (-50 to 0), max is surface.
                    if d_max <= 0:
                        depth_frac = float(np.clip(abs(d - d_max) / d_range, 0, 1))
                    else:
                        depth_frac = float(np.clip(abs(d - d_min) / d_range, 0, 1))

                    # Ensure physical plotting depth negative downwards
                    if d_max > 0 and d_min >= 0:
                        d = -d

                for iy in range(0, ny, xy_step):
                    for ix in range(0, nx, xy_step):
                        u_val = float(u_arr[iz, iy, ix])
                        v_val = _v(iz, iy, ix)
                        if np.isnan(u_val) or np.isnan(v_val):
                            continue
                        if abs(u_val) < 1e-10 and abs(v_val) < 1e-10:
                            continue
                        vectors.append(
                            {
                                "lon": _lon(iy, ix),
                                "lat": _lat(iy, ix),
                                "depth": d,
                                "depth_frac": depth_frac,
                                "u": u_val,
                                "v": v_val,
                            }
                        )

        # Compute bounds for Three.js scene setup
        all_lons = [v["lon"] for v in vectors] if vectors else [0]
        all_lats = [v["lat"] for v in vectors] if vectors else [0]

        result = {
            "vectors": vectors,
            "u_var": u_var,
            "v_var": v_var,
            "bounds": [min(all_lons), min(all_lats), max(all_lons), max(all_lats)],
            "depth_levels": sorted(set(v["depth"] for v in vectors)),
            "count": len(vectors),
        }

        return Response(
            content=json.dumps(result),
            media_type="application/json",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate 3D preview for {dataset_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/point_value")
async def get_point_value(
    dataset_id: str = Query(..., description="ID of the dataset (without extension)"),
    ext: str = Query(..., description="Extension of dataset (zarr, grib, nc)"),
    var_name: str = Query(..., description="Variable to query"),
    time_idx: int = Query(0, description="Time step index relative to data slice"),
    nx: float = Query(
        ..., ge=0.0, le=1.0, description="Normalized X coordinate (0.0 to 1.0)"
    ),
    ny: float = Query(
        ..., ge=0.0, le=1.0, description="Normalized Y coordinate (0.0 to 1.0)"
    ),
):
    """Returns a specific value for a dataset given a normalized X/Y hover coordinate."""
    import xarray as xr
    import numpy as np

    cache_dir = Path(settings.cache_dir()).expanduser()
    full_path = os.path.join(
        cache_dir, f"{dataset_id}.zarr" if ext == "zarr" else f"{dataset_id}.{ext}"
    )

    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="Dataset not found.")

    try:
        ds = (
            xr.open_zarr(full_path)
            if ext == "zarr"
            else xr.open_dataset(
                full_path, engine="cfgrib" if ext == "grib" else "netcdf4"
            )
        )
        if var_name not in ds.variables:
            return {"value": None}

        data_var = ds[var_name]

        slice_opts = {}
        for time_cand in ["time", "valid_time", "step", "ocean_time"]:
            if time_cand in data_var.dims:
                max_t = ds.sizes[time_cand] - 1
                slice_opts[time_cand] = min(time_idx, max_t)

        for z_name in [
            "depth",
            "zC",
            "level",
            "s_rho",
            "s_w",
            "isobaricInhPa",
            "number",
            "surface",
            "step",
        ]:
            if z_name in data_var.dims and z_name not in slice_opts:
                slice_opts[z_name] = -1 if z_name in ("s_rho", "s_w") else 0

        valid_opts = {k: v for k, v in slice_opts.items() if k in data_var.dims}
        array = data_var.isel(valid_opts).values

        # Image is rendered origin="lower", so Y goes from bottom (0) to top (1) in plotting space.
        # nx, ny from frontend will correspond to image dimensions.
        # Make sure we don't go out of bounds:
        h, w = array.shape
        xi = int(np.clip(nx * w, 0, w - 1))
        # Note: In matplotlib with origin="lower", ny=0 means the lowest array index
        yi = int(np.clip(ny * h, 0, h - 1))

        val = array[yi, xi]

        return {
            "value": float(val) if not np.isnan(val) else None,
            "xi": xi,
            "yi": yi,
            "shape": [h, w],
        }
    except Exception as e:
        logger.error(f"Failed to query point value: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
