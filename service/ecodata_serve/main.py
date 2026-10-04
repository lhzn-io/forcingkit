import logging
import os
import sys
import time
import shutil
from pathlib import Path
from typing import Dict, Any, Callable, Awaitable, Optional

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException, Request, Response  # noqa: E402
from fastapi.responses import FileResponse, RedirectResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from ecodata_cache.dispatcher import (  # noqa: E402
    dispatch_forcing_request,
    dispatch_station_profiles_request,
    dispatch_bounding_box_profiles_request,
)
from ecodata_serve.routers import viewer, bathymetry, plotly_api  # noqa: E402
from ecodata_cache.fetchers.noaa import fetch_noaa_tide_data  # noqa: E402
from ecodata_cache.fetchers.tides_tmd import HarmonicFetcher  # noqa: E402

# Configure Logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

log_level = logging.INFO
root_logger = logging.getLogger()
logging.getLogger("cfgrib").setLevel(logging.ERROR)
logging.getLogger("cfgrib.messages").setLevel(logging.ERROR)
root_logger.setLevel(log_level)

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

# File Handler
file_handler = logging.FileHandler(log_dir / "service.log")
file_handler.setFormatter(formatter)
root_logger.addHandler(file_handler)

# Stream Handler
has_console = False
for h in root_logger.handlers:
    if isinstance(h, logging.StreamHandler) and h.stream == sys.stdout:
        h.setFormatter(formatter)
        h.setLevel(log_level)
        has_console = True
        break

if not has_console:
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(log_level)
    root_logger.addHandler(stream_handler)

logging.getLogger("ecodata_serve").setLevel(log_level)
logger = logging.getLogger("ecodata_serve")

from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

# Initialize FastAPI application
app = FastAPI(
    title="ecodata-cache",
    description="Microservice for fetching, parsing, and regridding multi-domain environmental data.",
    version="1.0.0",
    openapi_tags=[
        {
            "name": "Cache Viewer",
            "description": "Endpoints to navigate and preview static cached data.",
        }
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(viewer.router)
app.include_router(bathymetry.router)
app.include_router(plotly_api.router)

# Ensure static UI directory exists
ui_dir = Path("static")
ui_dir.mkdir(exist_ok=True)
app.mount("/ui", StaticFiles(directory="static", html=True), name="static_ui")


@app.get("/", include_in_schema=False)
def ui_root() -> RedirectResponse:
    return RedirectResponse("/ui/")


@app.middleware("http")
async def log_requests(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(
        f"Request: {request.method} {request.url} | "
        f"Status: {response.status_code} | "
        f"Latency: {process_time:.4f}s"
    )
    return response


class HarmonicsRequest(BaseModel):
    lons: list[float]
    lats: list[float]
    model_name: Optional[str] = "GOT4.10c"


@app.post("/api/v1/harmonics")
async def get_harmonics(req: HarmonicsRequest):
    import numpy as np

    fetcher = HarmonicFetcher(model_name=req.model_name)
    try:
        amplitudes, phases, constituents = fetcher.fetch_constituents(
            np.array(req.lons, dtype=np.float32), np.array(req.lats, dtype=np.float32)
        )
        return {
            "constituents": constituents,
            "amplitudes": amplitudes.tolist(),
            "phases": phases.tolist(),
        }
    except Exception as e:
        logger.error(f"Error fetching harmonics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class BoundingBox(BaseModel):
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float


class ForcingRequest(BaseModel):
    bbox: BoundingBox
    start_time: str  # ISO8601 string
    end_time: str  # ISO8601 string
    station_id: Optional[str] = None
    cache_bust: bool = False
    forcing_type: str = "surface"  # "surface" or "volumetric"


class OBCRequest(BaseModel):
    bbox: BoundingBox
    start_date: str  # ISO8601 string
    duration_hours: int
    cache_bust: bool = False
    allow_donor_fallback: bool = True
    sponge_cells: int = 0
    include_tides: bool = True
    tidal_model: str = "GOT4.10c"
    # Parent-ocean options (schema z-v2): donor cells of padding beyond the bbox, so the parent
    # brackets the child, and the spacing of the fixed z levels.
    pad_cells: int = 3
    vertical_spacing_m: float = 2.0


class ICRequest(BaseModel):
    bbox: BoundingBox
    target_date: str  # ISO8601 string
    cache_bust: bool = False


class ICRegridRequest(BaseModel):
    zarr_id: str  # raw IC zarr ID (e.g., "ic_b6584ecacf33")
    bbox: BoundingBox  # simulation domain
    resolution: float  # target grid resolution in meters
    cache_bust: bool = False


class TideRequest(BaseModel):
    station_id: str
    start_time: str  # ISO8601 string
    end_time: str  # ISO8601 string
    cache_bust: bool = False


class TelemetryRequest(BaseModel):
    station_id: str
    start_time: str  # ISO8601 string
    end_time: str  # ISO8601 string
    cache_bust: bool = False


class HoTRequest(BaseModel):
    lat: float

    lon: float

    radius_km: float = 20.0


@app.post("/api/v1/hot_discovery")
async def hot_discovery(req: HoTRequest):
    """Discover head of tide limits using OSM API."""

    from ecodata_cache.fetchers.hydrography import find_head_of_tide

    try:
        results = find_head_of_tide(req.lat, req.lon, req.radius_km)

        return {"status": "success", "results": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Basic health check endpoint."""
    return {"status": "healthy", "service": "ecodata-cache"}


@app.api_route("/api/v1/cache/purge", methods=["GET", "POST"])
async def purge_cache() -> Dict[str, str]:
    """Purges the forcing and IC data cache. Supports both GET (manual) and POST (UI)."""
    cache_dir = Path(
        os.environ.get("COASTAL_SIM_DATA_CACHE_DIR", "~/.cache/ecodata-cache")
    ).expanduser()
    if os.path.exists(cache_dir):
        logger.info(f"Purging cache directory: {cache_dir}")
        try:
            shutil.rmtree(cache_dir)
            os.makedirs(cache_dir, exist_ok=True)
            return {"status": "success", "message": "Cache purged successfully."}
        except Exception as e:
            logger.error(f"Failed to purge cache: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to purge cache: {e}")
    else:
        return {"status": "success", "message": "Cache directory does not exist."}


@app.post("/api/v1/tide")
async def get_tide_data(request: TideRequest) -> Dict[str, Any]:
    """Fetch and cache NOAA tide data for a given station and time window."""
    try:
        data = fetch_noaa_tide_data(
            station_id=request.station_id,
            start_time=request.start_time,
            end_time=request.end_time,
            cache_bust=request.cache_bust,
        )
        return {"status": "success", "data": data}
    except Exception as e:
        logger.error(f"Tide fetch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class TelemetryBBoxRequest(BaseModel):
    bbox: BoundingBox
    start_time: str  # ISO8601 string
    end_time: str  # ISO8601 string
    cache_bust: bool = False


@app.post("/api/v1/telemetry/station")
async def get_station_telemetry(request: TelemetryRequest) -> Dict[str, Any]:
    """Fetch 3-depth telemetry profile data for structural nudging."""
    logger.info(
        f"Received telemetry request for station {request.station_id} from {request.start_time} to {request.end_time}"
    )
    try:
        data = dispatch_station_profiles_request(
            station_id=request.station_id,
            start_time=request.start_time,
            end_time=request.end_time,
            cache_bust=request.cache_bust,
        )
        return {"status": "success", "data": data}
    except Exception as e:
        logger.error(f"Telemetry fetch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/telemetry/bbox")
async def get_bbox_telemetry(request: TelemetryBBoxRequest) -> Dict[str, Any]:
    """Fetch structured nudge telemetry for all stations in bounding box."""
    logger.info(
        f"Received bbox telemetry request for {request.bbox} from {request.start_time} to {request.end_time}"
    )
    try:
        # Convert Request bbox to list [min_lon, min_lat, max_lon, max_lat]
        bbox_list = [
            request.bbox.min_lon,
            request.bbox.min_lat,
            request.bbox.max_lon,
            request.bbox.max_lat,
        ]
        data = dispatch_bounding_box_profiles_request(
            bbox=bbox_list,
            start_time=request.start_time,
            end_time=request.end_time,
            cache_bust=request.cache_bust,
        )
        return {"status": "success", "data": data}
    except Exception as e:
        logger.error(f"BBox telemetry fetch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/bc/generate")
async def generate_bc(request: ForcingRequest) -> Dict[str, Any]:
    """
    Generate boundary conditions (u_wind, v_wind, patm, water_level) for a given bounding box and time window.
    """
    logger.info(
        f"Received {request.forcing_type} BC request for bbox {request.bbox} from {request.start_time} to {request.end_time}"
    )

    import hashlib
    import json

    # Generate a deterministic hash ID based on the exact spatial and temporal bounds
    hash_str = f"{request.bbox.min_lon}_{request.bbox.max_lon}_{request.bbox.min_lat}_{request.bbox.max_lat}_{request.start_time}_{request.end_time}_{request.station_id}_{request.forcing_type}"
    md5_hash = hashlib.md5(hash_str.encode()).hexdigest()[:12]
    zarr_id = f"bc_{md5_hash}"

    base_dir = Path(
        os.environ.get("COASTAL_SIM_DATA_CACHE_DIR", "~/.cache/ecodata-cache")
    ).expanduser()
    zarr_path = os.path.join(base_dir, f"{zarr_id}.zarr")
    meta_path = os.path.join(base_dir, f"{zarr_id}_metadata.json")

    # Native cache hit evaluation
    if not request.cache_bust and os.path.exists(zarr_path):
        logger.info(f"Deterministic Hash Cache Hit for BC: {zarr_id}")
        return {
            "status": "success",
            "message": "Tiered BC fetch successful. Loaded from hash cache.",
            "zarr_file": zarr_path,
            "zarr_id": zarr_id,
            "request": request.model_dump(),
        }

    try:
        # 1. Dispatch Request: evaluate Tier (I-IV) and fetch native GRIB
        bbox_list = [
            request.bbox.max_lat,  # North
            request.bbox.min_lon,  # West
            request.bbox.min_lat,  # South
            request.bbox.max_lon,  # East
        ]

        # We only pass the start date string for now: "YYYY-MM-DD"
        start_date_str = request.start_time.split("T")[0]

        # 2. Fetch NOAA tides if requested
        tide_data = None
        if request.station_id:
            try:
                tide_data = fetch_noaa_tide_data(
                    request.station_id,
                    request.start_time,
                    request.end_time,
                    cache_bust=request.cache_bust,
                )
            except Exception as e:
                logger.warning(
                    f"NOAA tide fetch failed (proceeding without tides): {e}"
                )

        from datetime import datetime

        start_dt = datetime.strptime(request.start_time, "%Y-%m-%dT%H:%M:%SZ")
        end_dt = datetime.strptime(request.end_time, "%Y-%m-%dT%H:%M:%SZ")
        duration_hours = int((end_dt - start_dt).total_seconds() / 3600) + 1

        grib_paths = dispatch_forcing_request(
            target_date=start_date_str,
            duration_hours=duration_hours,
            bbox=bbox_list,
            forcing_type=request.forcing_type,
            cache_bust=request.cache_bust,
        )

        from ecodata_cache.regridder import process_and_regrid_grib

        zarr_path = process_and_regrid_grib(
            grib_paths, bbox_list, zarr_path, tide_data=tide_data
        )

        # Write metadata manifest for traceability
        with open(meta_path, "w") as f:
            json.dump(request.model_dump(), f, indent=4)

        return {
            "status": "success",
            "message": "Tiered BC fetch and regridding successful. Ready for Oceananigans.",
            "zarr_file": zarr_path,
            "zarr_id": zarr_id,
            "request": request.model_dump(),
        }
    except Exception as e:
        logger.error(f"BC generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/bc/download/{zarr_id}")
async def download_bc(zarr_id: str):
    """
    Downloads the processed CF-compliant Zarr array as a ZIP file.
    """
    base_dir = Path(
        os.environ.get("COASTAL_SIM_DATA_CACHE_DIR", "~/.cache/ecodata-cache")
    ).expanduser()
    matches = list(base_dir.rglob(f"{zarr_id}.zarr"))
    if not matches:
        raise HTTPException(
            status_code=404,
            detail=f"Zarr dataset {zarr_id} not found in cache. Did you generate it?",
        )
    zarr_dir = str(matches[0])

    zip_path = os.path.join(base_dir, f"{zarr_id}.zip")

    # We create the zip if it doesn't exist, or if the zarr directory is newer than the zip
    if not os.path.exists(zip_path) or os.path.getmtime(zarr_dir) > os.path.getmtime(
        zip_path
    ):
        logger.info(f"Packing Zarr directory into archive: {zip_path}")
        shutil.make_archive(zip_path.replace(".zip", ""), "zip", zarr_dir)

    return FileResponse(
        zip_path, media_type="application/zip", filename=f"{zarr_id}.zip"
    )


class TCPredictRequest(BaseModel):
    target_date: str


@app.post("/api/v1/bc/cache")
async def cache_bc(request: ForcingRequest) -> Dict[str, Any]:
    import hashlib

    hash_str = f"{request.bbox.min_lon}_{request.bbox.max_lon}_{request.bbox.min_lat}_{request.bbox.max_lat}_{request.start_time}_{request.end_time}_{request.station_id}_{request.forcing_type}"
    md5_hash = hashlib.md5(hash_str.encode()).hexdigest()[:12]
    zarr_id = f"bc_{md5_hash}"
    return {"status": "success", "zarr_id": zarr_id}


@app.post("/api/v1/bc/predict-donor")
async def predict_bc_donor_endpoint(request: TCPredictRequest) -> Dict[str, Any]:
    """Predicts which boundary condition donor will be used for a given target date."""
    try:
        from ecodata_cache.dispatcher import predict_bc_donor

        donor_meta = predict_bc_donor(request.target_date)
        if donor_meta:
            return {"status": "success", "donor": donor_meta}
        else:
            return {
                "status": "error",
                "message": "No donor model supports this target date.",
            }
    except Exception as e:
        logger.error(f"Failed to predict BC donor: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/ic/cache")
async def cache_ic(request: ICRequest) -> Dict[str, Any]:
    import hashlib

    from ecodata_cache.dispatcher import predict_ic_donor

    bbox_list = [
        request.bbox.min_lon,
        request.bbox.min_lat,
        request.bbox.max_lon,
        request.bbox.max_lat,
    ]
    donor_meta = predict_ic_donor(bbox_list)
    donor_id = donor_meta["id"] if donor_meta else "unknown"
    hash_str = f"{request.bbox.min_lon}_{request.bbox.max_lon}_{request.bbox.min_lat}_{request.bbox.max_lat}_{request.target_date}_{donor_id}"
    md5_hash = hashlib.md5(hash_str.encode()).hexdigest()[:12]
    zarr_id = f"ic_{md5_hash}"
    return {"status": "success", "zarr_id": zarr_id}


@app.post("/api/v1/ic/predict-donor")
async def predict_ic_donor_endpoint(request: ICRequest) -> Dict[str, Any]:
    """Predicts which donor model will be used for a given bounding box."""
    try:
        from ecodata_cache.dispatcher import predict_ic_donor

        bbox_list = [
            request.bbox.min_lon,
            request.bbox.min_lat,
            request.bbox.max_lon,
            request.bbox.max_lat,
        ]
        donor_meta = predict_ic_donor(bbox_list)
        if donor_meta:
            return {"status": "success", "donor": donor_meta}
        else:
            return {
                "status": "error",
                "message": "No donor model supports this bounding box.",
            }
    except Exception as e:
        logger.error(f"Failed to predict IC donor: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/ic/generate")
async def generate_ic(request: ICRequest) -> Dict[str, Any]:
    """
    Generate initial conditions (u, v, temp, salt, etc.) for a given bounding box and target date.
    Evaluates MARACOOS, NERACOOS, and HYCOM in tiered priority.
    """
    logger.info(f"Received IC request for bbox {request.bbox} at {request.target_date}")

    import hashlib
    import json
    from ecodata_cache.dispatcher import predict_ic_donor

    # Include predicted donor in hash so different models produce different cache keys
    bbox_list = [
        request.bbox.min_lon,
        request.bbox.min_lat,
        request.bbox.max_lon,
        request.bbox.max_lat,
    ]
    donor_meta = predict_ic_donor(bbox_list)
    donor_id = donor_meta["id"] if donor_meta else "unknown"

    hash_str = f"{request.bbox.min_lon}_{request.bbox.max_lon}_{request.bbox.min_lat}_{request.bbox.max_lat}_{request.target_date}_{donor_id}"
    md5_hash = hashlib.md5(hash_str.encode()).hexdigest()[:12]
    zarr_id = f"ic_{md5_hash}"

    base_dir = Path(
        os.environ.get("COASTAL_SIM_DATA_CACHE_DIR", "~/.cache/ecodata-cache")
    ).expanduser()
    zarr_path = os.path.join(base_dir, f"{zarr_id}.zarr")
    meta_path = os.path.join(base_dir, f"{zarr_id}_metadata.json")

    # Native cache hit evaluation
    if not request.cache_bust and os.path.exists(zarr_path):
        logger.info(f"Deterministic Hash Cache Hit for IC: {zarr_id}")
        return {
            "status": "success",
            "message": "Hierarchical initial conditions fetch successful. Loaded from hash cache.",
            "zarr_file": zarr_path,
            "zarr_id": zarr_id,
            "request": request.model_dump(),
        }

    try:
        from ecodata_cache.dispatcher import dispatch_ic_request

        _ = dispatch_ic_request(
            target_date=request.target_date,
            bbox=bbox_list,
            cache_bust=request.cache_bust,
            zarr_path=zarr_path,
        )

        with open(meta_path, "w") as f:
            json.dump(request.model_dump(), f, indent=4)

        return {
            "status": "success",
            "message": "Hierarchical initial conditions fetch successful. Ready for Oceananigans.",
            "zarr_file": zarr_path,
            "zarr_id": zarr_id,
            "request": request.model_dump(),
        }
    except Exception as e:
        logger.error(f"IC generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/ic/download/{zarr_id}")
async def download_ic(zarr_id: str):
    """
    Downloads the processed CF-compliant IC Zarr array as a ZIP file.
    """
    base_dir = Path(
        os.environ.get("COASTAL_SIM_DATA_CACHE_DIR", "~/.cache/ecodata-cache")
    ).expanduser()
    matches = list(base_dir.rglob(f"{zarr_id}.zarr"))
    if not matches:
        raise HTTPException(
            status_code=404,
            detail=f"Zarr dataset {zarr_id} not found in cache. Did you generate it?",
        )
    zarr_dir = str(matches[0])

    zip_path = os.path.join(base_dir, f"{zarr_id}.zip")

    if not os.path.exists(zip_path) or os.path.getmtime(zarr_dir) > os.path.getmtime(
        zip_path
    ):
        import shutil

        logger.info(f"Packing IC Zarr directory into archive: {zip_path}")
        shutil.make_archive(
            zip_path.replace(".zip", ""), "zip", root_dir=zarr_dir, base_dir="."
        )

    return FileResponse(
        zip_path, media_type="application/zip", filename=f"{zarr_id}.zip"
    )


@app.post("/api/v1/ic/regrid")
async def regrid_ic(request: ICRegridRequest) -> Dict[str, Any]:
    """
    Regrid a raw IC zarr (curvilinear sigma coords) onto a regular grid at the
    target simulation resolution. Performs horizontal interpolation only (Stages 1+2).
    The vertical sigma→z remapping is left to the physics engine which has the actual
    grid bathymetry.

    The output zarr retains sigma levels and is keyed by (zarr_id, bbox, resolution).
    """
    import hashlib
    import math

    logger.info(
        f"IC regrid request: {request.zarr_id} at {request.resolution}m for bbox {request.bbox}"
    )

    # Deterministic cache key: raw zarr + target grid params
    hash_str = f"{request.zarr_id}_{request.bbox.min_lon}_{request.bbox.max_lon}_{request.bbox.min_lat}_{request.bbox.max_lat}_{request.resolution}"
    md5_hash = hashlib.md5(hash_str.encode()).hexdigest()[:12]
    regrid_id = f"ic_regrid_{md5_hash}"

    base_dir = Path(
        os.environ.get("COASTAL_SIM_DATA_CACHE_DIR", "~/.cache/ecodata-cache")
    ).expanduser()

    # Source: raw IC zarr (in data service cache or engine cache)
    src_zarr = os.path.join(base_dir, f"{request.zarr_id}.zarr")
    if not os.path.exists(src_zarr):
        # Try engine cache as fallback
        engine_cache = (
            Path(
                os.environ.get("COASTAL_SIM_CACHE_DIR", "~/.cache/coastal-sim")
            ).expanduser()
            / "ic"
        )
        src_zarr = os.path.join(engine_cache, f"{request.zarr_id}.zarr")
        if not os.path.exists(src_zarr):
            raise HTTPException(
                status_code=404,
                detail=f"Source IC zarr not found: {request.zarr_id}",
            )

    out_zarr = os.path.join(base_dir, f"{regrid_id}.zarr")

    if not request.cache_bust and os.path.exists(out_zarr):
        try:
            import zarr

            sz = zarr.open(out_zarr, mode="r")
            if sz.attrs.get("type") == "regridded_ic":
                logger.info(f"Regrid cache hit: {regrid_id}")
                return {
                    "status": "success",
                    "message": "IC regrid loaded from cache.",
                    "zarr_id": regrid_id,
                    "zarr_file": out_zarr,
                    "source_zarr_id": request.zarr_id,
                }
        except Exception:
            pass

    try:
        import numpy as np
        import zarr
        from scipy.spatial import cKDTree

        src: zarr.Group = zarr.open(src_zarr, mode="r")  # type: ignore[assignment]

        _lon = src["lon_rho"]
        _lat = src["lat_rho"]
        _s = src["s_rho"]
        assert (
            isinstance(_lon, zarr.Array)
            and isinstance(_lat, zarr.Array)
            and isinstance(_s, zarr.Array)
        )
        lon_rho = np.array(_lon[:], dtype=np.float64)
        lat_rho = np.array(_lat[:], dtype=np.float64)
        s_rho = np.array(_s[:], dtype=np.float64)
        n_sigma = len(s_rho)

        # Detect axis layout: match u shape against coordinate shape
        _u = src["u"]
        assert isinstance(_u, zarr.Array)
        u_shape = _u.shape
        coord_shape = lon_rho.shape
        if len(u_shape) == 3 and u_shape[0] == n_sigma and u_shape[1:] == coord_shape:
            depth_first = True
            nxi, neta = coord_shape
            logger.info(f"Axis layout: depth-first (s_rho={n_sigma}, {nxi}×{neta})")
        elif len(u_shape) == 3 and u_shape[2] == n_sigma and u_shape[:2] == coord_shape:
            depth_first = False
            nxi, neta = coord_shape
            logger.info(f"Axis layout: depth-last ({nxi}×{neta}, s_rho={n_sigma})")
        else:
            raise ValueError(
                f"Unrecognized IC shape: u={u_shape}, coords={coord_shape}"
            )

        # Physical coordinate projection
        bbox = request.bbox
        mid_lat = (bbox.min_lat + bbox.max_lat) / 2.0
        deg2m_lon = 111111.0 * math.cos(math.radians(mid_lat))
        deg2m_lat = 111111.0
        Lx = abs((bbox.max_lon - bbox.min_lon) * deg2m_lon)
        Ly = abs((bbox.max_lat - bbox.min_lat) * deg2m_lat)

        x_donor = (lon_rho - bbox.min_lon) * deg2m_lon
        y_donor = (lat_rho - bbox.min_lat) * deg2m_lat

        # Build KDTree from valid donor points within domain buffer
        valid = ~(np.isnan(lon_rho) | np.isnan(lat_rho))
        valid &= (x_donor > -0.5 * Lx) & (x_donor < 1.5 * Lx)
        valid &= (y_donor > -0.5 * Ly) & (y_donor < 1.5 * Ly)
        valid_idx = np.argwhere(valid)
        coords = np.column_stack([x_donor[valid], y_donor[valid]])
        tree = cKDTree(coords)
        logger.info(f"KDTree: {len(valid_idx)} valid donor points")

        # Target regular grid
        Nx = max(2, round(Lx / request.resolution))
        Ny = max(2, round(Ly / request.resolution))
        x_target = np.linspace(0, Lx, Nx)
        y_target = np.linspace(0, Ly, Ny)
        xx, yy = np.meshgrid(x_target, y_target, indexing="ij")
        query_pts = np.column_stack([xx.ravel(), yy.ravel()])

        # KDTree IDW query (all target points at once)
        K = 4
        dists, idxs = tree.query(query_pts, k=K, workers=-1)
        dists = np.maximum(dists, 1e-10)
        weights = 1.0 / dists**2

        # Read all fields and regrid per sigma level
        if os.path.exists(out_zarr):
            import shutil

            shutil.rmtree(out_zarr)
        store: zarr.Group = zarr.open(out_zarr, mode="w", zarr_format=2)  # type: ignore[assignment]

        for var_name in ["u", "v", "temp", "salt"]:
            if var_name not in src:
                continue
            _var = src[var_name]
            assert isinstance(_var, zarr.Array)
            raw = np.array(_var[:], dtype=np.float64)
            result = np.zeros((n_sigma, Ny, Nx), dtype=np.float32)

            for k in range(n_sigma):
                slab = raw[k] if depth_first else raw[:, :, k]
                # Extract values at valid points, handle fill values
                vals = slab[valid]
                bad = np.isnan(vals) | (vals < -9000)

                # Per-query IDW with level validity
                vals_at_neighbors = vals[idxs]  # (n_query, K)
                bad_at_neighbors = bad[idxs]

                # Optimized weights calculation without full array copy for every layer
                w = np.where(bad_at_neighbors, 0.0, weights)
                ws = w.sum(axis=1)

                # Filter out pure-zero denominators
                valid_ws = ws > 0

                interpolated = np.zeros(query_pts.shape[0], dtype=np.float64)
                if np.any(valid_ws):
                    interpolated[valid_ws] = (
                        w[valid_ws] * vals_at_neighbors[valid_ws]
                    ).sum(axis=1) / ws[valid_ws]

                result[k, :, :] = interpolated.reshape(Nx, Ny).T

            z = store.create_array(
                var_name,
                shape=(n_sigma, Ny, Nx),
                chunks=(n_sigma, min(Ny, 256), min(Nx, 256)),
                dtype="f4",
                fill_value=np.nan,
            )
            z[:] = result
            z.attrs["_ARRAY_DIMENSIONS"] = ["s_rho", "y", "x"]
            logger.info(f"Regridded {var_name}: ({n_sigma}, {Ny}, {Nx})")

        # Copy coords and write metadata
        s_arr = store.create_array(
            "s_rho", shape=(n_sigma,), dtype="f4", fill_value=np.nan
        )
        s_arr[:] = s_rho.astype(np.float32)
        s_arr.attrs["_ARRAY_DIMENSIONS"] = ["s_rho"]

        x_arr = store.create_array("x", shape=(Nx,), dtype="f4", fill_value=np.nan)
        x_arr[:] = x_target.astype(np.float32)
        x_arr.attrs["_ARRAY_DIMENSIONS"] = ["x"]

        y_arr = store.create_array("y", shape=(Ny,), dtype="f4", fill_value=np.nan)
        y_arr[:] = y_target.astype(np.float32)
        y_arr.attrs["_ARRAY_DIMENSIONS"] = ["y"]

        # Write projected lat/lon back for UI preview mapping
        lon_arr = store.create_array(
            "lon_rho", shape=(Ny, Nx), dtype="f4", fill_value=np.nan
        )
        lat_arr = store.create_array(
            "lat_rho", shape=(Ny, Nx), dtype="f4", fill_value=np.nan
        )
        _lon_m = xx / deg2m_lon + bbox.min_lon
        _lat_m = yy / deg2m_lat + bbox.min_lat
        lon_arr[:] = _lon_m.T.astype(np.float32)
        lat_arr[:] = _lat_m.T.astype(np.float32)
        lon_arr.attrs["_ARRAY_DIMENSIONS"] = ["y", "x"]
        lat_arr.attrs["_ARRAY_DIMENSIONS"] = ["y", "x"]

        store.attrs["bbox"] = [bbox.min_lon, bbox.min_lat, bbox.max_lon, bbox.max_lat]
        store.attrs["Nx"] = int(Nx)
        store.attrs["Ny"] = int(Ny)
        store.attrs["n_sigma"] = int(n_sigma)
        store.attrs["resolution"] = float(request.resolution)
        store.attrs["source_zarr_id"] = request.zarr_id
        store.attrs["type"] = "regridded_ic"
        store.attrs["schema_version"] = 2

        logger.info(f"IC regrid complete: {regrid_id} ({Nx}×{Ny}×{n_sigma})")

        return {
            "status": "success",
            "message": f"IC regridded to {Nx}×{Ny} at {request.resolution}m ({n_sigma} sigma levels).",
            "zarr_id": regrid_id,
            "zarr_file": out_zarr,
            "source_zarr_id": request.zarr_id,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"IC regrid failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def _obc_hash_source(request: "OBCRequest", bbox_list: list, donor_id: str) -> str:
    """The string an OBC store id is hashed from. Shared by /obc/cache and /obc so the two can
    never disagree; it includes the store schema, so a store built under an older schema is never
    served for a newer request."""
    from ecodata_cache.fetchers.necofs import OBC_SCHEMA

    return (
        f"{bbox_list}_{request.start_date}_{request.duration_hours}_{donor_id}_"
        f"{request.sponge_cells}_{request.include_tides}_{request.tidal_model}_obc_"
        f"{OBC_SCHEMA}_p{request.pad_cells}_dz{request.vertical_spacing_m}"
    )


def _zip_is_stale(zip_path: str, store_path: str) -> bool:
    """True when the zip is missing or older than the newest file in the store."""
    if not os.path.exists(zip_path):
        return True
    newest = max(
        (
            os.path.getmtime(os.path.join(d, f))
            for d, _, files in os.walk(store_path)
            for f in files
        ),
        default=os.path.getmtime(store_path),
    )
    return os.path.getmtime(zip_path) < newest


@app.post("/api/v1/obc/cache")
async def cache_obc(request: OBCRequest) -> Dict[str, Any]:
    import hashlib

    from ecodata_cache.dispatcher import predict_obc_donor

    bbox_list = [
        request.bbox.min_lon,
        request.bbox.min_lat,
        request.bbox.max_lon,
        request.bbox.max_lat,
    ]
    donor_meta = predict_obc_donor(bbox_list)
    donor_id = donor_meta.get("id", "unknown")
    hash_str = _obc_hash_source(request, bbox_list, donor_id)
    raw_id = hashlib.md5(hash_str.encode()).hexdigest()[:12]
    return {"status": "success", "zarr_id": f"obc_{raw_id}"}


@app.post("/api/v1/obc/predict-donor")
async def predict_obc_donor_endpoint(request: OBCRequest) -> Dict[str, Any]:
    try:
        from ecodata_cache.dispatcher import predict_obc_donor

        bbox_list = [
            request.bbox.min_lon,
            request.bbox.min_lat,
            request.bbox.max_lon,
            request.bbox.max_lat,
        ]
        donor_meta = predict_obc_donor(bbox_list)
        if not donor_meta:
            return {
                "status": "error",
                "message": "No donor found for OBC",
                "donor": None,
            }
        return {"status": "success", "donor": donor_meta}
    except Exception as e:
        logger.error(f"Failed to predict OBC donor: {e}")
        return {"status": "error", "message": str(e), "donor": None}


# Plain `def` for the long handlers: FastAPI runs them in its thread pool, so an hour-long parent
# fetch no longer blocks every other request on the event loop.
@app.post("/api/v1/obc")
def generate_obc(request: OBCRequest) -> Dict[str, Any]:
    bbox_list = [
        request.bbox.min_lon,
        request.bbox.min_lat,
        request.bbox.max_lon,
        request.bbox.max_lat,
    ]
    from ecodata_cache.dispatcher import predict_obc_donor, dispatch_obc_request

    meta = predict_obc_donor(bbox_list)
    donor_id = meta.get("id", "unknown")

    import hashlib

    # Hash unique configuration plus donor
    hash_str = _obc_hash_source(request, bbox_list, donor_id)
    raw_id = hashlib.md5(hash_str.encode()).hexdigest()[:12]
    zarr_id = f"obc_{raw_id}"
    zarr_name = f"{zarr_id}.zarr"
    cache_dir = os.environ.get(
        "COASTAL_SIM_DATA_CACHE_DIR", os.path.expanduser("~/.cache/ecodata-cache")
    )
    zarr_path = os.path.join(cache_dir, zarr_name)

    from ecodata_cache.fetchers.necofs import OBC_SCHEMA
    from ecodata_cache.zarr_stream import store_is_complete

    if not request.cache_bust and store_is_complete(zarr_path, (OBC_SCHEMA,)):
        return {
            "status": "cached",
            "zarr_id": zarr_id,
            "zarr_path": zarr_path,
            "download_url": f"/api/v1/obc/download/{zarr_id}",
            "donor": donor_id,
        }

    try:
        final_path = dispatch_obc_request(
            start_date=request.start_date,
            duration_hours=request.duration_hours,
            bbox=bbox_list,
            cache_bust=request.cache_bust,
            zarr_path=zarr_path,
            allow_donor_fallback=request.allow_donor_fallback,
            include_tides=request.include_tides,
            tidal_model=request.tidal_model,
            sponge_cells=request.sponge_cells,
            pad_cells=request.pad_cells,
            vertical_spacing_m=request.vertical_spacing_m,
        )
        return {
            "status": "success",
            "zarr_id": zarr_id,
            "zarr_path": final_path,
            "download_url": f"/api/v1/obc/download/{zarr_id}",
            "donor": donor_id,
        }
    except Exception as e:
        logger.error(f"OBC generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/obc/download/{zarr_id}")
def download_obc(zarr_id: str):
    cache_dir = Path(
        os.environ.get(
            "COASTAL_SIM_DATA_CACHE_DIR",
            os.path.expanduser("~/.cache/ecodata-cache"),
        )
    )

    # Handle both raw hash and obc_ prefixed hashes gracefully
    search_id = zarr_id if zarr_id.startswith("obc_") else f"obc_{zarr_id}"

    matches = list(cache_dir.rglob(f"{search_id}.zarr"))
    if not matches:
        raise HTTPException(status_code=404, detail="OBC Zarr archive not found.")
    zarr_path = str(matches[0])

    # Compress the folder on the fly, and rebuild a zip older than its store: after a cache_bust
    # the store is rewritten under the same id, and a stale zip would otherwise be served.
    zip_path = os.path.join(cache_dir, f"{search_id}.zip")
    if _zip_is_stale(zip_path, zarr_path):
        shutil.make_archive(zip_path.replace(".zip", ""), "zip", zarr_path)

    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=f"{search_id}.zip",
    )


@app.get("/api/v2/coupled-boundaries/{region}")
async def get_coupled_boundaries(
    region: str,
    start: str,
    end: str,
    sponge_cells: int = 10,
    include_tides: bool = True,
    cache_bust: bool = False,
):
    """
    V2 endpoint for fetching coupled ocean + tidal boundaries.
    Example: GET /api/v2/coupled-boundaries/mab?start=2021-08-01&end=2021-09-30&sponge_cells=10
    """
    # Map region name to bbox (simplified for demo)
    region_map = {
        "mab": {"min_lon": -76.0, "max_lon": -69.0, "min_lat": 35.0, "max_lat": 42.0},
        "harbor": {
            "min_lon": -74.2,
            "max_lon": -73.6,
            "min_lat": 40.4,
            "max_lat": 40.9,
        },
    }

    if region not in region_map:
        raise HTTPException(status_code=404, detail=f"Region {region} not supported.")

    bbox = region_map[region]
    bbox_list = [bbox["min_lon"], bbox["min_lat"], bbox["max_lon"], bbox["max_lat"]]

    # Calculate duration
    from datetime import datetime

    try:
        # Simple ISO parser
        s = start.replace("Z", "")
        e = end.replace("Z", "")
        start_dt = datetime.fromisoformat(s)
        end_dt = datetime.fromisoformat(e)
        duration_hours = int((end_dt - start_dt).total_seconds() / 3600)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {exc}")

    # Use existing generate_obc logic via the internal dispatcher
    from ecodata_cache.dispatcher import dispatch_obc_request

    try:
        zarr_path = dispatch_obc_request(
            start_date=start,
            duration_hours=duration_hours,
            bbox=bbox_list,
            cache_bust=cache_bust,
            include_tides=include_tides,
            sponge_cells=sponge_cells,
        )

        # Determine zarr_id from path
        import os

        zarr_id = os.path.basename(zarr_path).replace(".zarr", "")

        # Strip 'obc_' prefix for the download URL if it exists
        dl_id = zarr_id.replace("obc_", "")

        return {
            "status": "success",
            "region": region,
            "zarr_id": zarr_id,
            "zarr_path": zarr_path,
            "download_url": f"/api/v1/obc/download/{dl_id}",
        }
    except Exception as exc2:
        logger.error(f"Coupled boundaries fetch failed: {exc2}")
        raise HTTPException(status_code=500, detail=str(exc2))


if __name__ == "__main__":
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9598)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)
