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
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from forcingkit.dispatcher import (  # noqa: E402
    dispatch_station_profiles_request,
    dispatch_bounding_box_profiles_request,
)
from forcingkit_serve.jobs import Job  # noqa: E402
from forcingkit_serve.routers import (  # noqa: E402
    viewer,
    bathymetry,
    plotly_api,
    removed,
    stores,
)
from forcingkit.fetchers.noaa import fetch_noaa_tide_data  # noqa: E402

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

logging.getLogger("forcingkit_serve").setLevel(log_level)
logger = logging.getLogger("forcingkit_serve")

from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from forcingkit import settings  # noqa: E402

# Initialize FastAPI application
app = FastAPI(
    title="forcingkit",
    description="Spatiotemporal forcing for computational Earth-system models: selects, regrids and serves model-ready time series with provenance.",
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
app.include_router(stores.router)
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
    # Parent-ocean options (schema z-v3): donor cells of padding beyond the bbox, so the parent
    # brackets the child, and the spacing of the fixed z levels.
    pad_cells: int = 3
    vertical_spacing_m: float = 2.0


class TideRequest(BaseModel):
    station_id: str
    start_time: str  # ISO8601 string
    end_time: str  # ISO8601 string
    cache_bust: bool = False


class CurrentPredictionsRequest(BaseModel):
    station_id: str
    bin: int
    start_time: str  # ISO8601 string
    end_time: str  # ISO8601 string
    interval_minutes: int = 30
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

    from forcingkit.fetchers.hydrography import find_head_of_tide

    try:
        results = find_head_of_tide(req.lat, req.lon, req.radius_km)

        return {"status": "success", "results": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Basic health check endpoint."""
    return {"status": "healthy", "service": "forcingkit"}


@app.api_route("/api/v1/cache/purge", methods=["GET", "POST"])
async def purge_cache() -> Dict[str, str]:
    """Purges the forcing and IC data cache. Supports both GET (manual) and POST (UI)."""
    cache_dir = Path(settings.cache_dir()).expanduser()
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


# Plain `def` for the tide and telemetry handlers too: their remote reads block, and FastAPI runs a
# plain handler in its thread pool rather than on the event loop.
@app.post("/api/v1/tide")
def get_tide_data(request: TideRequest) -> Dict[str, Any]:
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


class CurrentStationsRequest(BaseModel):
    bbox: BoundingBox
    cache_bust: bool = False


@app.post("/api/v1/currents/stations")
def get_current_stations(request: CurrentStationsRequest) -> Dict[str, Any]:
    """Harmonic CO-OPS current-prediction stations and bins inside a bounding box."""
    from forcingkit.fetchers.noaa import fetch_noaa_current_stations

    b = request.bbox
    try:
        stations = fetch_noaa_current_stations(
            [b.min_lon, b.min_lat, b.max_lon, b.max_lat], cache_bust=request.cache_bust
        )
        return {"status": "success", "data": stations}
    except Exception as e:
        logger.error(f"Current station listing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/currents")
def get_current_predictions(request: CurrentPredictionsRequest) -> Dict[str, Any]:
    """Harmonic current predictions for a CO-OPS station and depth bin (along the flood axis)."""
    from forcingkit.fetchers.noaa import fetch_noaa_current_predictions

    try:
        data = fetch_noaa_current_predictions(
            station_id=request.station_id,
            bin_number=request.bin,
            start_time=request.start_time,
            end_time=request.end_time,
            interval_minutes=request.interval_minutes,
            cache_bust=request.cache_bust,
        )
        return {"status": "success", "data": data}
    except Exception as e:
        logger.error(f"Current predictions fetch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class TelemetryBBoxRequest(BaseModel):
    bbox: BoundingBox
    start_time: str  # ISO8601 string
    end_time: str  # ISO8601 string
    cache_bust: bool = False
    # Substitute proxy stations, marked within_bbox false, when the box contains none.
    allow_proxy_stations: bool = False


@app.post("/api/v1/telemetry/station")
def get_station_telemetry(request: TelemetryRequest) -> Dict[str, Any]:
    """Fetch 3-depth telemetry profile data for structural nudging. `skipped` lists the datasets
    left out because they, or their host, failed now or recently."""
    logger.info(
        f"Received telemetry request for station {request.station_id} from {request.start_time} to {request.end_time}"
    )
    try:
        skipped: List[Dict[str, Any]] = []
        data = dispatch_station_profiles_request(
            station_id=request.station_id,
            start_time=request.start_time,
            end_time=request.end_time,
            cache_bust=request.cache_bust,
            skipped=skipped,
        )
        return {"status": "success", "data": data, "skipped": skipped}
    except Exception as e:
        logger.error(f"Telemetry fetch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/telemetry/bbox")
def get_bbox_telemetry(request: TelemetryBBoxRequest) -> Dict[str, Any]:
    """Fetch structured nudge telemetry for all stations in bounding box.

    `data` is empty when the box contains no known station, unless `allow_proxy_stations` is
    set. Each station carries `within_bbox` and `distance_km`; `fallback` is true when any
    returned station lies outside the box. `skipped` lists the datasets left out because they,
    or their host, failed now or recently."""
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
        skipped: List[Dict[str, Any]] = []
        data = dispatch_bounding_box_profiles_request(
            bbox=bbox_list,
            start_time=request.start_time,
            end_time=request.end_time,
            cache_bust=request.cache_bust,
            allow_proxy_stations=request.allow_proxy_stations,
            skipped=skipped,
        )
        fallback = any(not st.get("within_bbox", True) for st in data.values())
        return {
            "status": "success",
            "data": data,
            "fallback": fallback,
            "skipped": skipped,
        }
    except Exception as e:
        logger.error(f"BBox telemetry fetch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _bbox_list(bbox: BoundingBox) -> list[float]:
    return [bbox.min_lon, bbox.min_lat, bbox.max_lon, bbox.max_lat]


def _obc_hash_source(request: "OBCRequest", bbox_list: list, donor_id: str) -> str:
    """The string an OBC store id is hashed from. Shared by /obc/cache and /obc so the two can
    never disagree; it includes the store schema, so a store built under an older schema is never
    served for a newer request."""
    from forcingkit.fetchers.necofs import OBC_SCHEMA

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

    from forcingkit.dispatcher import predict_obc_donor

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
        from forcingkit.dispatcher import predict_obc_donor

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
def _obc_store(request: OBCRequest) -> tuple[str, str, str]:
    """(zarr_id, zarr_path, predicted donor id) of the parent store `request` names."""
    import hashlib

    from forcingkit.dispatcher import predict_obc_donor

    donor_id = predict_obc_donor(_bbox_list(request.bbox)).get("id", "unknown")
    # Hash unique configuration plus donor
    hash_str = _obc_hash_source(request, _bbox_list(request.bbox), donor_id)
    zarr_id = f"obc_{hashlib.md5(hash_str.encode()).hexdigest()[:12]}"
    return zarr_id, os.path.join(settings.cache_dir(), f"{zarr_id}.zarr"), donor_id


def _build_obc(request: OBCRequest) -> Dict[str, Any]:
    """Build (or find cached) the parent store for `request`; raises on failure. Shared by the
    synchronous route and the job queue."""
    from forcingkit.dispatcher import delivered_obc_donor, dispatch_obc_request
    from forcingkit.fetchers.necofs import OBC_SCHEMA
    from forcingkit.zarr_stream import store_is_complete

    zarr_id, zarr_path, donor_id = _obc_store(request)
    status = "cached"
    if request.cache_bust or not store_is_complete(zarr_path, (OBC_SCHEMA,)):
        zarr_path = dispatch_obc_request(
            start_date=request.start_date,
            duration_hours=request.duration_hours,
            bbox=_bbox_list(request.bbox),
            cache_bust=request.cache_bust,
            zarr_path=zarr_path,
            allow_donor_fallback=request.allow_donor_fallback,
            include_tides=request.include_tides,
            tidal_model=request.tidal_model,
            sponge_cells=request.sponge_cells,
            pad_cells=request.pad_cells,
            vertical_spacing_m=request.vertical_spacing_m,
        )
        status = "success"
    return {
        "status": status,
        "zarr_id": zarr_id,
        "zarr_path": zarr_path,
        "download_url": f"/api/v1/obc/download/{zarr_id}",
        # The donor that delivered, which differs from the predicted one after a fallback.
        "donor": delivered_obc_donor(zarr_path) or donor_id,
        "predicted_donor": donor_id,
    }


@app.post("/api/v1/obc")
def generate_obc(request: OBCRequest) -> Dict[str, Any]:
    try:
        return _build_obc(request)
    except Exception as e:
        logger.error(f"OBC generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/obc/download/{zarr_id}")
def download_obc(zarr_id: str):
    cache_dir = Path(settings.cache_dir())

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
    from forcingkit.fetchers.ndbc import fetch_ndbc

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
    return settings.cache_dir()


def _atmosphere_store(request: AtmosphereRequest) -> tuple[str, str]:
    """(zarr_id, zarr_path) of the atmosphere store `request` names; HTTPException 400 for a
    request that cannot be served."""
    from forcingkit.dispatcher import atmosphere_key

    if request.source != "hrrr":
        raise HTTPException(
            status_code=400, detail=f"unsupported source {request.source}"
        )
    if request.hours < 1:
        raise HTTPException(status_code=400, detail="hours must be at least 1")
    zarr_id = atmosphere_key(
        _bbox_list(request.bbox),
        request.start_time,
        request.hours,
        request.resolution_deg,
        request.margin_deg,
        request.source,
    )
    return zarr_id, os.path.join(_atmosphere_cache_dir(), f"{zarr_id}.zarr")


def _build_atmosphere(request: AtmosphereRequest) -> Dict[str, Any]:
    """Build (or find cached) the atmosphere store for `request`; raises on failure. Shared by
    the synchronous route and the job queue."""
    from forcingkit.dispatcher import dispatch_atmosphere_request

    zarr_id, zarr_path = _atmosphere_store(request)
    final_path = dispatch_atmosphere_request(
        _bbox_list(request.bbox),
        request.start_time,
        request.hours,
        zarr_path,
        resolution_deg=request.resolution_deg,
        margin_deg=request.margin_deg,
        cache_bust=request.cache_bust,
    )
    return {
        "status": "success",
        "zarr_id": zarr_id,
        "zarr_path": final_path,
        "download_url": f"/api/v1/atmosphere/download/{zarr_id}",
        "source": request.source,
    }


@app.post("/api/v1/atmosphere")
def generate_atmosphere(request: AtmosphereRequest) -> Dict[str, Any]:
    """A prescribed atmosphere for an ocean model: HRRR hourly fields on a regular lon/lat grid,
    from one hour before `start_time` to one hour after the end, streamed to a Zarr store."""
    _atmosphere_store(request)  # rejects a bad request with 400 before any work
    try:
        return _build_atmosphere(request)
    except Exception as e:
        logger.error(f"Atmosphere generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class WavesRequest(BaseModel):
    bbox: BoundingBox
    start_time: str  # ISO 8601, UTC, on the hour
    hours: int
    source: str = "nwps"
    wfo: str = "okx"  # NWS Weather Forecast Office
    grid: str = "CG1"  # the office's computational grid
    cache_bust: bool = False


def _waves_store(request: WavesRequest) -> tuple[str, str, Any]:
    """(zarr_id, zarr_path, newest cycle) of the waves store `request` names; HTTPException 400
    for a request that cannot be served. The newest cycle is fixed here, once, so a queued job
    builds the store its id names."""
    from forcingkit.dispatcher import waves_key
    from forcingkit.fetchers import nwps

    if request.source != "nwps":
        raise HTTPException(
            status_code=400, detail=f"unsupported source {request.source}"
        )
    if request.hours < 1:
        raise HTTPException(status_code=400, detail="hours must be at least 1")
    try:
        wfo, grid = nwps.check_wfo_grid(request.wfo, request.grid)
        times = nwps.record_times(request.start_time, request.hours)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    latest = nwps.newest_cycle(times[-1])
    zarr_id = waves_key(
        _bbox_list(request.bbox),
        request.start_time,
        request.hours,
        wfo,
        grid,
        latest,
        request.source,
    )
    return zarr_id, os.path.join(settings.cache_dir(), f"{zarr_id}.zarr"), latest


def _build_waves(request: WavesRequest, store: tuple[str, str, Any]) -> Dict[str, Any]:
    """Build (or find cached) the waves store `store` names for `request`; raises on failure.
    Shared by the synchronous route and the job queue."""
    from forcingkit.dispatcher import dispatch_waves_request

    zarr_id, zarr_path, latest = store
    final_path = dispatch_waves_request(
        _bbox_list(request.bbox),
        request.start_time,
        request.hours,
        zarr_path,
        wfo=request.wfo,
        grid=request.grid,
        latest_cycle=latest,
        cache_bust=request.cache_bust,
    )
    return {
        "status": "success",
        "zarr_id": zarr_id,
        "zarr_path": final_path,
        "download_url": f"/api/v1/waves/download/{zarr_id}",
        "source": request.source,
    }


@app.post("/api/v1/waves")
def generate_waves(request: WavesRequest) -> Dict[str, Any]:
    """Surface waves from NOAA's Nearshore Wave Prediction System on the office's own grid
    (significant height, peak period, direction of travel, swell height, forcing wind), hourly
    from one hour before `start_time` to one hour after the end, streamed to a Zarr store. NOMADS
    keeps about five days; older windows are served only from a configured archive, and a window
    neither holds is a 404."""
    store = _waves_store(request)  # rejects a bad request with 400 before any work
    try:
        return _build_waves(request, store)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Waves generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/waves/download/{zarr_id}")
def download_waves(zarr_id: str):
    cache_dir = settings.cache_dir()
    search_id = zarr_id if zarr_id.startswith("wav_") else f"wav_{zarr_id}"
    zarr_path = os.path.join(cache_dir, f"{search_id}.zarr")
    if not os.path.isdir(zarr_path):
        raise HTTPException(status_code=404, detail="Waves Zarr store not found.")
    zip_path = os.path.join(cache_dir, f"{search_id}.zip")
    if _zip_is_stale(zip_path, zarr_path):
        shutil.make_archive(zip_path.replace(".zip", ""), "zip", zarr_path)
    return FileResponse(
        zip_path, media_type="application/zip", filename=f"{search_id}.zip"
    )


# Jobs: the same builds as /obc, /atmosphere and /waves, queued, so a caller polls for the result
# instead of holding a request open through a cold build. The job id is the store id.

JOB_KINDS = {"obc": "ocean", "atm": "atmosphere", "wav": "waves"}

# Donor reported for a finished store found on disk after a restart, by kind (ocean reads it).
STORE_DONORS = {"atmosphere": "hrrr", "waves": "nwps"}


def _job_view(job: Job, http: Request) -> Dict[str, Any]:
    from forcingkit_serve.stac import store_item

    base = str(http.base_url).rstrip("/")
    view: Dict[str, Any] = {
        "job_id": job.id,
        "kind": job.kind,
        "status": job.status,
        "submitted": job.submitted,
        "started": job.started,
        "finished": job.finished,
        "error": job.error,
        "status_url": f"{base}/api/v1/jobs/{job.id}",
        "result": None,
    }
    if job.status == "succeeded" and job.result is not None:
        donor = job.result.get("donor") or job.result.get("source")
        view["result"] = store_item(
            job.id, job.result["zarr_path"], base, job.kind, donor
        )
    return view


def _submit(
    http: Request,
    zarr_id: str,
    zarr_path: str,
    cache_bust: bool,
    build: Callable[[], Dict[str, Any]],
    cached: Callable[[], Dict[str, Any]],
) -> JSONResponse:
    from forcingkit_serve.jobs import registry
    from forcingkit_serve.routers.stores import store_complete

    kind = JOB_KINDS[zarr_id.split("_", 1)[0]]
    if not cache_bust and not registry.is_active(zarr_id) and store_complete(zarr_path):
        job = registry.finished(zarr_id, kind, cached())
    else:
        job = registry.submit(zarr_id, kind, build, rebuild=cache_bust)
    return JSONResponse(status_code=202, content=_job_view(job, http))


@app.post("/api/v1/jobs/obc", status_code=202)
def submit_obc_job(request: OBCRequest, http: Request) -> JSONResponse:
    """Queue the parent-ocean build that /api/v1/obc runs; poll `status_url` for the result, a
    STAC Item whose `zarr` asset is the store served over HTTP."""
    from forcingkit.dispatcher import delivered_obc_donor

    zarr_id, zarr_path, donor_id = _obc_store(request)
    return _submit(
        http,
        zarr_id,
        zarr_path,
        request.cache_bust,
        lambda: _build_obc(request),
        lambda: {
            "zarr_path": zarr_path,
            "donor": delivered_obc_donor(zarr_path) or donor_id,
        },
    )


@app.post("/api/v1/jobs/atmosphere", status_code=202)
def submit_atmosphere_job(request: AtmosphereRequest, http: Request) -> JSONResponse:
    """Queue the atmosphere build that /api/v1/atmosphere runs; poll `status_url`."""
    zarr_id, zarr_path = _atmosphere_store(request)
    return _submit(
        http,
        zarr_id,
        zarr_path,
        request.cache_bust,
        lambda: _build_atmosphere(request),
        lambda: {"zarr_path": zarr_path, "source": request.source},
    )


@app.post("/api/v1/jobs/waves", status_code=202)
def submit_waves_job(request: WavesRequest, http: Request) -> JSONResponse:
    """Queue the waves build that /api/v1/waves runs; poll `status_url`."""
    store = _waves_store(request)
    zarr_id, zarr_path, _ = store
    return _submit(
        http,
        zarr_id,
        zarr_path,
        request.cache_bust,
        lambda: _build_waves(request, store),
        lambda: {"zarr_path": zarr_path, "source": request.source},
    )


@app.get("/api/v1/jobs/{job_id}")
def get_job(job_id: str, http: Request) -> Dict[str, Any]:
    """A job's status, and its STAC Item once it has succeeded. A store that finished before the
    service restarted is reported as a succeeded job."""
    from forcingkit.dispatcher import delivered_obc_donor
    from forcingkit_serve.jobs import registry
    from forcingkit_serve.routers.stores import STORE_ID, store_complete

    if not STORE_ID.match(job_id):
        raise HTTPException(status_code=400, detail="invalid job id")
    job = registry.get(job_id)
    if job is None:
        zarr_path = os.path.join(settings.cache_dir(), f"{job_id}.zarr")
        if not store_complete(zarr_path):
            raise HTTPException(status_code=404, detail=f"no job or store {job_id}")
        kind = JOB_KINDS[job_id.split("_", 1)[0]]
        donor = (
            delivered_obc_donor(zarr_path) if kind == "ocean" else STORE_DONORS[kind]
        )
        job = Job(
            id=job_id,
            kind=kind,
            status="succeeded",
            result={"zarr_path": zarr_path, "donor": donor},
        )
    return _job_view(job, http)


class AtmosphereLevelsRequest(BaseModel):
    bbox: BoundingBox
    start_time: str  # ISO 8601, UTC
    hours: int
    n_levels: int = 14
    resolution_deg: float = 0.03
    margin_deg: float = 0.25
    cache_bust: bool = False


@app.post("/api/v1/atmosphere/levels")
def generate_atmosphere_levels(request: AtmosphereLevelsRequest) -> Dict[str, Any]:
    """The lower atmosphere above the surface delivery: HRRR wind, vertical velocity and
    temperature on its lowest `n_levels` native levels (14 reach about 2.9 km over water), hourly
    over the same window and grid as /api/v1/atmosphere, streamed to a Zarr store. About 63 MB of
    HRRR is read per hour at 14 levels."""
    from forcingkit.dispatcher import dispatch_levels_request, levels_key
    from forcingkit.fetchers.hrrr_levels import MAX_LEVELS

    if request.hours < 1:
        raise HTTPException(status_code=400, detail="hours must be at least 1")
    if not 1 <= request.n_levels <= MAX_LEVELS:
        raise HTTPException(
            status_code=400, detail=f"n_levels must be 1 to {MAX_LEVELS}"
        )
    bbox_list = [
        request.bbox.min_lon,
        request.bbox.min_lat,
        request.bbox.max_lon,
        request.bbox.max_lat,
    ]
    zarr_id = levels_key(
        bbox_list,
        request.start_time,
        request.hours,
        request.n_levels,
        request.resolution_deg,
        request.margin_deg,
    )
    zarr_path = os.path.join(_atmosphere_cache_dir(), f"{zarr_id}.zarr")
    try:
        final_path = dispatch_levels_request(
            bbox_list,
            request.start_time,
            request.hours,
            zarr_path,
            n_levels=request.n_levels,
            resolution_deg=request.resolution_deg,
            margin_deg=request.margin_deg,
            cache_bust=request.cache_bust,
        )
    except Exception as e:
        logger.error(f"Atmosphere levels generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    return {"status": "success", "zarr_id": zarr_id, "zarr_path": final_path}


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


class RiverRequest(BaseModel):
    bbox: BoundingBox
    start_time: str  # ISO 8601, UTC
    hours: int
    max_gap_hours: int = 6
    allow_climatology: bool = False
    # Mouths to serve instead of the listed ones in the bbox: each {name, lon, lat} with
    # optional comid or trace_from (the outlet), include / exclude (USGS site numbers) and
    # lag_hours.
    mouths: List[Dict[str, Any]] | None = None
    # Gauges draining less than this share of the mouth's area are left out.
    min_share: float = 0.01
    # Leave out, rather than fail on, a river with no gauge record over the window.
    skip_ungauged: bool = False
    cache_bust: bool = False


def _river_cache_dir() -> str:
    return settings.cache_dir("rivers")


@app.post("/api/v1/rivers")
def generate_rivers(request: RiverRequest) -> Dict[str, Any]:
    """Hourly discharge at each river mouth in the bbox (or each of `mouths`), for point sources
    in an ocean model: the lowest tide-free USGS gauge on each branch, derived per request from
    USGS NLDI and site metadata, summed and scaled by drainage area, from `start_time` to
    `start_time + hours`, streamed to a Zarr store."""
    import json

    import zarr

    from forcingkit.dispatcher import dispatch_river_request, river_key, river_mouths

    if request.hours < 1:
        raise HTTPException(status_code=400, detail="hours must be at least 1")
    if request.max_gap_hours < 0:
        raise HTTPException(status_code=400, detail="max_gap_hours must be at least 0")
    if not 0 <= request.min_share < 1:
        raise HTTPException(status_code=400, detail="min_share must be in [0, 1)")
    bbox_list = [
        request.bbox.min_lon,
        request.bbox.min_lat,
        request.bbox.max_lon,
        request.bbox.max_lat,
    ]
    try:
        mouths = river_mouths(bbox_list, request.mouths)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not mouths:
        raise HTTPException(
            status_code=400, detail=f"no listed river mouth lies in bbox {bbox_list}"
        )
    zarr_id = river_key(
        bbox_list,
        request.start_time,
        request.hours,
        request.max_gap_hours,
        request.allow_climatology,
        mouths,
        request.min_share,
        request.skip_ungauged,
    )
    os.makedirs(_river_cache_dir(), exist_ok=True)
    zarr_path = os.path.join(_river_cache_dir(), f"{zarr_id}.zarr")
    try:
        final_path = dispatch_river_request(
            bbox_list,
            request.start_time,
            request.hours,
            zarr_path,
            max_gap_hours=request.max_gap_hours,
            allow_climatology=request.allow_climatology,
            cache_bust=request.cache_bust,
            mouths=mouths,
            min_share=request.min_share,
            skip_ungauged=request.skip_ungauged,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"River delivery failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    attrs = dict(zarr.open_group(final_path, mode="r", zarr_format=2).attrs)
    return {
        "status": "success",
        "zarr_id": zarr_id,
        "zarr_path": final_path,
        "download_url": f"/api/v1/rivers/download/{zarr_id}",
        "rivers": json.loads(str(attrs.get("provenance", "[]"))),
        "provisional": bool(attrs.get("provisional", False)),
        "skipped_rivers": json.loads(str(attrs.get("skipped_rivers", "[]"))),
    }


@app.get("/api/v1/rivers/download/{zarr_id}")
def download_rivers(zarr_id: str):
    search_id = zarr_id if zarr_id.startswith("riv_") else f"riv_{zarr_id}"
    if not search_id[4:].isalnum():
        raise HTTPException(status_code=400, detail="invalid store id")
    cache_dir = _river_cache_dir()
    zarr_path = os.path.join(cache_dir, f"{search_id}.zarr")
    if not os.path.isdir(zarr_path):
        raise HTTPException(status_code=404, detail="River Zarr store not found.")
    zip_path = os.path.join(cache_dir, f"{search_id}.zip")
    if _zip_is_stale(zip_path, zarr_path):
        shutil.make_archive(zip_path.replace(".zip", ""), "zip", zarr_path)
    return FileResponse(
        zip_path, media_type="application/zip", filename=f"{search_id}.zip"
    )


class SatelliteRequest(BaseModel):
    bbox: BoundingBox
    quantity: (
        str  # "sst", "chlor_a", "kd_490" or "spm" (see fetchers.coastwatch.QUANTITIES)
    )
    end_time: str | None = None  # ISO 8601, UTC; default now
    days: int = 7
    coast_pixels: int = 1
    cache_bust: bool = False


def _satellite_cache_dir() -> str:
    return settings.cache_dir("satellite")


def _satellite_retention_days() -> float:
    return float(settings.env("FORCINGKIT_SATELLITE_RETENTION_DAYS", "30") or 30)


@app.post("/api/v1/satellite")
def get_satellite(request: SatelliteRequest) -> Dict[str, Any]:
    """A satellite surface field over the bbox: each pixel's most recent valid value across the quantity's sensors
    within `days` of `end_time`, with its age, a band of `coast_pixels` masked along the shore, as a Zarr store.

    Stores older than FORCINGKIT_SATELLITE_RETENTION_DAYS (default 30) are pruned on every request, so the cache
    stays bounded.
    """
    import json
    from datetime import datetime, timezone

    import pandas as pd

    from forcingkit.fetchers import coastwatch

    products = coastwatch.QUANTITIES.get(request.quantity)
    if products is None:
        raise HTTPException(
            status_code=400,
            detail=f"quantity must be one of {sorted(coastwatch.QUANTITIES)}",
        )
    if not 1 <= request.days <= 31:
        raise HTTPException(status_code=400, detail="days must be 1 to 31")
    end = (
        pd.Timestamp(request.end_time).to_pydatetime()
        if request.end_time
        else datetime.now(timezone.utc)
    )
    bbox_list = [
        request.bbox.min_lon,
        request.bbox.min_lat,
        request.bbox.max_lon,
        request.bbox.max_lat,
    ]
    cache_dir = _satellite_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)
    coastwatch.prune_cache(cache_dir, _satellite_retention_days())
    zarr_id = coastwatch.cache_key(
        products, bbox_list, end, request.days, request.coast_pixels
    )
    zarr_path = os.path.join(cache_dir, f"{zarr_id}.zarr")
    if request.cache_bust or not os.path.isdir(zarr_path):
        try:
            ds = coastwatch.mask_coast(
                coastwatch.composite(products, bbox_list, end, request.days),
                request.coast_pixels,
            )
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            logger.error(f"Satellite delivery failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        shutil.rmtree(zarr_path, ignore_errors=True)
        ds.to_zarr(zarr_path, mode="w", zarr_format=2)
    import xarray as xr

    ds = xr.open_zarr(zarr_path)
    valid = ds["value"].notnull()
    return {
        "status": "success",
        "zarr_id": zarr_id,
        "zarr_path": zarr_path,
        "download_url": f"/api/v1/satellite/download/{zarr_id}",
        "products": str(ds.attrs.get("products", "")).split(","),
        "attribution": ds.attrs.get("attribution", ""),
        "valid_share": float(valid.mean()),
        "age_days_median": float(ds["age_days"].where(valid).median())
        if bool(valid.any())
        else None,
        "attrs": json.loads(json.dumps(dict(ds.attrs), default=str)),
    }


@app.get("/api/v1/satellite/download/{zarr_id}")
def download_satellite(zarr_id: str):
    if not (zarr_id.startswith("sat_") and zarr_id[4:].isalnum()):
        raise HTTPException(status_code=400, detail="invalid store id")
    cache_dir = _satellite_cache_dir()
    zarr_path = os.path.join(cache_dir, f"{zarr_id}.zarr")
    if not os.path.isdir(zarr_path):
        raise HTTPException(status_code=404, detail="Satellite Zarr store not found.")
    zip_path = os.path.join(cache_dir, f"{zarr_id}.zip")
    if _zip_is_stale(zip_path, zarr_path):
        shutil.make_archive(zip_path.replace(".zip", ""), "zip", zarr_path)
    return FileResponse(
        zip_path, media_type="application/zip", filename=f"{zarr_id}.zip"
    )


if __name__ == "__main__":
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9598)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)
