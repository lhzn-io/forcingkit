from pathlib import Path
from fastapi import APIRouter, HTTPException, Response as FastAPIResponse
from pydantic import BaseModel
from typing import List
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
    # Structured fields for maps and coverage timelines (see store_footprint)
    bbox: list[float] | None = None  # [west, south, east, north], degrees, west < east
    time_start: str | None = None  # ISO 8601 UTC, first record (or the target date)
    time_end: str | None = None  # ISO 8601 UTC, last record
    family: str | None = (
        None  # source model or provider, e.g. "NECOFS", "Doppio", "HRRR"
    )
    kind: str | None = None  # e.g. "ocean boundary", "ocean volume", "atmosphere"


# Longitude/latitude variable pairs, most specific first. Coordinates or data variables.
_LONLAT_PAIRS = (
    ("lon_rho", "lat_rho"),
    ("longitude", "latitude"),
    ("lon", "lat"),
    ("node_lon", "node_lat"),
    ("lonc", "latc"),
    ("mouth_lon", "mouth_lat"),
    ("xi", "eta"),
)
_BBOX_ATTRS = ("bbox", "requested_bbox", "domain_bbox")
# Family keywords searched in the store's path and its type/source/donor attributes.
_FAMILIES = (
    ("doppio", "Doppio"),
    ("necofs", "NECOFS"),
    ("fvcom", "NECOFS"),
    ("nyofs", "NYOFS"),
    ("dbofs", "DBOFS"),
    ("hycom", "HYCOM"),
    ("hrdps", "HRDPS"),
    ("hrrr", "HRRR"),
    ("era5", "ERA5"),
    ("nwps", "NWPS"),
    ("hfradar", "HF radar"),
    ("usgs", "USGS rivers"),
    ("regridded_ic", "Regridded IC"),
)
_KINDS = (
    ("volume", "ocean volume"),
    ("obc", "ocean boundary"),
    ("ic_regrid", "initial condition"),
    ("ic", "initial condition"),
    ("lev", "atmosphere levels"),
    ("atm", "atmosphere"),
    ("bc", "atmosphere"),
    ("wav", "waves"),
    ("nwps", "waves"),
    ("riv", "rivers"),
    ("hfradar", "surface currents (radar)"),
)


def _bbox_from(lon, lat) -> list[float] | None:
    import numpy as np

    x = np.asarray(lon, dtype=np.float64).ravel()
    y = np.asarray(lat, dtype=np.float64).ravel()
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return None
    # Index axes (0, 1, 2, ...) are not positions.
    if x.size > 2 and np.array_equal(x, np.arange(x.size)):
        return None
    if np.abs(y).max() > 90 or x.min() < -180 or x.max() > 360:
        return None
    x = np.where(x > 180, x - 360, x)
    return [round(float(v), 4) for v in (x.min(), y.min(), x.max(), y.max())]


def store_footprint(ds, path: Path) -> dict:
    """Numeric bbox, ISO time span, source family and kind of a cached dataset, for the
    inventory's map and coverage timeline. Every field is None when it cannot be told."""
    import json

    import numpy as np
    import pandas as pd

    out: dict = {"bbox": None, "time_start": None, "time_end": None}
    for lo, la in _LONLAT_PAIRS:
        if lo in ds.variables and la in ds.variables:
            try:
                out["bbox"] = _bbox_from(ds[lo].values, ds[la].values)
            except Exception:
                out["bbox"] = None
            if out["bbox"]:
                break
    if not out["bbox"]:
        for key in _BBOX_ATTRS:
            try:
                b = ds.attrs.get(key)
                b = json.loads(b) if isinstance(b, str) else b
                if b is not None and len(b) == 4:
                    out["bbox"] = _bbox_from([b[0], b[2]], [b[1], b[3]])
                    break
            except Exception:
                continue

    def iso(t) -> str | None:
        try:
            ts = pd.Timestamp(t)
            if ts.tzinfo is not None:
                ts = ts.tz_convert("UTC").tz_localize(None)
            return None if pd.isna(ts) else ts.strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            return None

    tname = next((n for n in ("time", "valid_time", "t") if n in ds.variables), None)
    if tname is not None:
        tv = np.atleast_1d(ds[tname].values)
        if tv.size and np.issubdtype(tv.dtype, np.datetime64):
            out["time_start"], out["time_end"] = iso(tv.min()), iso(tv.max())
    if out["time_start"] is None and ds.attrs.get("target_date"):
        out["time_start"] = out["time_end"] = iso(ds.attrs["target_date"])
    if out["time_start"] is None and ds.attrs.get("start_time"):
        out["time_start"] = out["time_end"] = iso(ds.attrs["start_time"])

    text = " ".join(
        [str(path).lower()]
        + [
            str(ds.attrs.get(k, "")).lower()
            for k in (
                "type",
                "source",
                "donor_id",
                "donor",
                "forcingkit:donor",
                "schema",
            )
        ]
    )
    out["family"] = next((name for key, name in _FAMILIES if key in text), None)
    stem = path.name.lower()
    out["kind"] = next(
        (name for key, name in _KINDS if stem.startswith(key) or f"_{key}_" in stem),
        None,
    )
    if out["kind"] is None and path.suffix.lower() in (".grib", ".grib2", ".nc"):
        out["kind"] = "raw donor file"
    return out


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
        footprint = store_footprint(ds, zarr_path)
        return {
            **footprint,
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


# Inventory entries by path, with the key they were built under: a store is reopened only when
# its metadata files change, so the page's two-second poll stays cheap.
_INVENTORY_CACHE: dict[str, tuple[tuple, CacheItem]] = {}
_ZARR_KEY_FILES = (
    ".zmetadata",
    ".zattrs",
    ".zgroup",
    "zarr.json",
    "time/.zarray",
    "time/zarr.json",
)


def _zarr_key(path: Path) -> tuple:
    key = [("dir", path.stat().st_mtime_ns)]
    for name in _ZARR_KEY_FILES:
        try:
            key.append((name, (path / name).stat().st_mtime_ns))
        except OSError:
            pass
    return tuple(key)


@router.get("/inventory", response_model=List[CacheItem])
async def get_cache_inventory(response: FastAPIResponse):
    """Crawls the local data cache and returns an inventory of all processed forcing datasets."""
    from starlette.concurrency import run_in_threadpool

    response.headers["Cache-Control"] = "no-store"
    return await run_in_threadpool(_crawl_inventory)


def _crawl_inventory() -> list[CacheItem]:
    cache_dir = Path(Path(settings.cache_dir()).expanduser())
    if not cache_dir.exists():
        return []

    inventory = []

    # 1. Find Zarr datasets (directories)
    seen: set[str] = set()
    for zarr_path in cache_dir.rglob("*.zarr"):
        if zarr_path.is_dir():
            seen.add(str(zarr_path))
            try:
                ckey = _zarr_key(zarr_path)
            except OSError:
                continue
            hit = _INVENTORY_CACHE.get(str(zarr_path))
            if hit and hit[0] == ckey:
                inventory.append(hit[1])
                continue
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
            _INVENTORY_CACHE[str(zarr_path)] = (ckey, inventory[-1])

    # 2. Find GRIB/NC datasets (files)
    for ext_glob in ["*.grib", "*.grib2", "*.nc"]:
        for file_path in cache_dir.rglob(ext_glob):
            if file_path.is_file():
                seen.add(str(file_path))
                st = file_path.stat()
                ckey = (st.st_mtime_ns, st.st_size)
                hit = _INVENTORY_CACHE.get(str(file_path))
                if hit and hit[0] == ckey:
                    inventory.append(hit[1])
                    continue
                stem = file_path.stem.lower()
                label = None
                source = None
                donor_model = None
                variables = None
                grid_shape = None
                spatial_extent = None
                resolution = None
                footprint: dict = {}

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
                    # NetCDF files are not GRIB: open them directly rather than failing in
                    # cfgrib once per filter first.
                    _grib_filters = (
                        []
                        if file_path.suffix.lower() == ".nc"
                        else [
                            {"typeOfLevel": "heightAboveGround", "stepType": "instant"},
                            {"typeOfLevel": "surface", "stepType": "instant"},
                            {"typeOfLevel": "heightAboveGround"},
                            {"typeOfLevel": "surface"},
                        ]
                    )
                    for _inv_filter in _grib_filters:
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
                        footprint = store_footprint(gds, file_path)
                        gds.close()
                except Exception as enrich_err:
                    logger.debug(
                        f"GRIB enrichment failed for {file_path.name}: {enrich_err}"
                    )

                if not footprint:
                    import xarray as xr

                    footprint = store_footprint(xr.Dataset(), file_path)
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
                        **footprint,
                    )
                )
                _INVENTORY_CACHE[str(file_path)] = (ckey, inventory[-1])

    for gone in set(_INVENTORY_CACHE) - seen:
        _INVENTORY_CACHE.pop(gone, None)
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
            # A finished job for this store would otherwise still report it as built.
            from forcingkit_serve.jobs import registry

            registry.forget(dataset_id)
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


# Relief basemap for the coverage map: PMTiles archives in FORCINGKIT_BASEMAP_DIR (raster WebP or
# PNG tiles, optional <name>.coast.geojson shoreline), read by the page in byte ranges. Without
# the setting the page uses the fallback vector style below.
BASEMAP_FALLBACK_STYLE = "https://tiles.openfreemap.org/styles/dark"


def _basemap_dir() -> Path | None:
    d = settings.env("FORCINGKIT_BASEMAP_DIR")
    if not d:
        return None
    path = Path(d).expanduser()
    return path if path.is_dir() else None


def _basemap_files() -> dict[str, Path]:
    d = _basemap_dir()
    if d is None:
        return {}
    return {
        f.name: f
        for f in sorted(d.iterdir())
        if f.is_file() and (f.suffix == ".pmtiles" or f.name.endswith(".coast.geojson"))
    }


@router.get("/basemap")
async def get_basemaps():
    """The relief archives the coverage map draws, and the style it falls back to."""
    files = _basemap_files()
    basemaps = [
        {
            "name": f.stem,
            "file": name,
            "size_mb": round(f.stat().st_size / 2**20, 1),
            "shoreline": f"{f.stem}.coast.geojson"
            if f"{f.stem}.coast.geojson" in files
            else None,
        }
        for name, f in files.items()
        if f.suffix == ".pmtiles"
    ]
    return {
        "basemaps": basemaps,
        "attribution": settings.env("FORCINGKIT_BASEMAP_ATTRIBUTION") or "",
        "fallback_style": BASEMAP_FALLBACK_STYLE,
    }


@router.get("/basemap/files/{name}")
async def get_basemap_file(name: str):
    """One basemap file; byte ranges are honoured, which the PMTiles reader relies on."""
    from fastapi.responses import FileResponse

    f = _basemap_files().get(name)
    if f is None:
        raise HTTPException(status_code=404, detail="no such basemap file")
    media = (
        "application/geo+json"
        if name.endswith(".geojson")
        else "application/octet-stream"
    )
    return FileResponse(
        f, media_type=media, headers={"Cache-Control": "public, max-age=3600"}
    )
