import logging
import os
import sys
import time
import shutil
from pathlib import Path
from typing import Dict, Any, Callable, Awaitable, List

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException, Request, Response  # noqa: E402
from fastapi.responses import FileResponse, RedirectResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from ecodata_cache.dispatcher import (  # noqa: E402
    dispatch_station_profiles_request,
    dispatch_bounding_box_profiles_request,
)
from ecodata_serve.routers import viewer, bathymetry, plotly_api, removed  # noqa: E402
from ecodata_cache.fetchers.noaa import fetch_noaa_tide_data  # noqa: E402

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
# 410 Gone stubs for the routes removed on 2026-10-05; drop at the next release.
app.include_router(removed.router)

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


class BoundingBox(BaseModel):
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float


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


class NDBCRequest(BaseModel):
    station_id: str
    start_time: str  # ISO 8601, UTC
    end_time: str
    products: List[str] = ["stdmet", "adcp"]


@app.post("/api/v1/ndbc")
def get_ndbc(request: NDBCRequest) -> Dict[str, Any]:
    """Observations at an NDBC buoy over a window, for validation: stdmet (sea and air
    temperature, pressure, wind, waves) and ADCP near-surface current as east and north
    components. Missing values are null."""
    from ecodata_cache.fetchers.ndbc import fetch_ndbc

    try:
        data = fetch_ndbc(
            request.station_id,
            request.start_time,
            request.end_time,
            tuple(request.products),
            cache_dir=_atmosphere_cache_dir(),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"NDBC fetch failed: {e}")
        raise HTTPException(status_code=502, detail=str(e))
    return {"status": "success", "data": data}


class AtmosphereRequest(BaseModel):
    bbox: BoundingBox
    start_time: str  # ISO 8601, UTC
    hours: int
    source: str = "hrrr"
    resolution_deg: float = 0.03
    margin_deg: float = 0.25
    cache_bust: bool = False


def _atmosphere_cache_dir() -> str:
    return os.environ.get(
        "COASTAL_SIM_DATA_CACHE_DIR", os.path.expanduser("~/.cache/ecodata-cache")
    )


@app.post("/api/v1/atmosphere")
def generate_atmosphere(request: AtmosphereRequest) -> Dict[str, Any]:
    """A prescribed atmosphere for an ocean model: HRRR hourly fields on a regular lon/lat grid,
    from one hour before `start_time` to one hour after the end, streamed to a Zarr store."""
    from ecodata_cache.dispatcher import atmosphere_key, dispatch_atmosphere_request

    if request.source != "hrrr":
        raise HTTPException(
            status_code=400, detail=f"unsupported source {request.source}"
        )
    if request.hours < 1:
        raise HTTPException(status_code=400, detail="hours must be at least 1")
    bbox_list = [
        request.bbox.min_lon,
        request.bbox.min_lat,
        request.bbox.max_lon,
        request.bbox.max_lat,
    ]
    zarr_id = atmosphere_key(
        bbox_list,
        request.start_time,
        request.hours,
        request.resolution_deg,
        request.margin_deg,
        request.source,
    )
    zarr_path = os.path.join(_atmosphere_cache_dir(), f"{zarr_id}.zarr")
    try:
        final_path = dispatch_atmosphere_request(
            bbox_list,
            request.start_time,
            request.hours,
            zarr_path,
            resolution_deg=request.resolution_deg,
            margin_deg=request.margin_deg,
            cache_bust=request.cache_bust,
        )
    except Exception as e:
        logger.error(f"Atmosphere generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    return {
        "status": "success",
        "zarr_id": zarr_id,
        "zarr_path": final_path,
        "download_url": f"/api/v1/atmosphere/download/{zarr_id}",
        "source": request.source,
    }


@app.get("/api/v1/atmosphere/download/{zarr_id}")
def download_atmosphere(zarr_id: str):
    cache_dir = _atmosphere_cache_dir()
    search_id = zarr_id if zarr_id.startswith("atm_") else f"atm_{zarr_id}"
    zarr_path = os.path.join(cache_dir, f"{search_id}.zarr")
    if not os.path.isdir(zarr_path):
        raise HTTPException(status_code=404, detail="Atmosphere Zarr store not found.")
    zip_path = os.path.join(cache_dir, f"{search_id}.zip")
    if _zip_is_stale(zip_path, zarr_path):
        shutil.make_archive(zip_path.replace(".zip", ""), "zip", zarr_path)
    return FileResponse(
        zip_path, media_type="application/zip", filename=f"{search_id}.zip"
    )


if __name__ == "__main__":
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9598)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)
