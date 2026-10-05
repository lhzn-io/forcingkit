import os
import hashlib
import httpx
import logging
from pathlib import Path
from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from starlette.background import BackgroundTask
from pydantic import BaseModel
from forcingkit import settings

router = APIRouter(prefix="/api/v1/bathymetry", tags=["bathymetry"])
logger = logging.getLogger(__name__)


async def stream_remote_file(url: str):
    """
    Passthrough proxy stream for local TopoBathySim setups to prevent local disk double-caching.
    Large timeout applied because fusing geospatial data can take 5+ minutes.
    """
    client = httpx.AsyncClient()
    request = client.build_request("GET", url, timeout=3600.0)
    response = await client.send(request, stream=True)

    if response.status_code != 200:
        await response.aread()
        await response.aclose()
        await client.aclose()
        raise HTTPException(
            status_code=response.status_code,
            detail=f"TopoBathySim Error: {response.text}",
        )

    async def cleanup():
        await response.aclose()
        await client.aclose()

    return StreamingResponse(
        response.aiter_raw(),
        media_type="application/zip",
        background=BackgroundTask(cleanup),
    )


@router.get("/fuse")
async def fuse_bathymetry(
    west: float = Query(..., description="Bounding box west (lon)"),
    south: float = Query(..., description="Bounding box south (lat)"),
    east: float = Query(..., description="Bounding box east (lon)"),
    north: float = Query(..., description="Bounding box north (lat)"),
    resolution: float = Query(10.0, description="Output resolution in meters"),
    format: str = Query("zarr", description="Output format: geotiff or zarr"),
):
    """
    Smart proxy for TopoBathySim that retrieves fused topography.
    If TopoBathySim is on a remote cluster, this caches the zipped output locally
    in forcingkit so future identical runs bypass the network.
    """
    topobathy_url = os.environ.get("TOPOBATHYSIM_URL", "http://localhost:9595").rstrip(
        "/"
    )
    target_url = f"{topobathy_url}/fuse?west={west}&south={south}&east={east}&north={north}&resolution={resolution}&format={format}"

    is_local = (
        "localhost" in topobathy_url
        or "127.0.0.1" in topobathy_url
        or "0.0.0.0" in topobathy_url
    )

    if is_local:
        logger.info(
            f"TopoBathySim is local ({topobathy_url}). Streaming directly to avoid disk duplication."
        )
        try:
            return await stream_remote_file(target_url)
        except httpx.ConnectError:
            raise HTTPException(
                status_code=502,
                detail=f"TopoBathySim is unreachable at {topobathy_url}. Please ensure the service is running on port 9595.",
            )
        except httpx.ReadTimeout:
            raise HTTPException(
                status_code=504,
                detail=f"TopoBathySim at {topobathy_url} timed out during fusion.",
            )

    # --- IF REMOTE, Cache Strategy is Engaged ---
    logger.info(
        f"TopoBathySim is remote ({topobathy_url}). Entering proxy-cache routine."
    )
    cache_dir = Path(settings.cache_dir()).expanduser() / "bathymetry"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Predictable cache key
    key_str = f"{west},{south},{east},{north}_{resolution}_{format}"
    cache_key = hashlib.md5(key_str.encode()).hexdigest()
    cache_path = cache_dir / f"fused_proxy_{cache_key}.zip"

    if cache_path.exists():
        logger.info(f"Serving topography from local cache: {cache_path}")
        return FileResponse(
            path=cache_path,
            media_type="application/zip",
            filename=f"fused_{cache_key}.zip",
        )

    # Download and cache it
    logger.info(
        f"Cache miss. Fetching from remote TopoBathySim and writing to {cache_path}..."
    )
    try:
        async with httpx.AsyncClient(timeout=3600.0) as client:
            async with client.stream("GET", target_url) as response:
                if response.status_code != 200:
                    text = await response.aread()
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"TopoBathySim Error: {text.decode('utf-8')}",
                    )

                # Stream directly to disk
                with open(cache_path, "wb") as f_out:
                    async for chunk in response.aiter_bytes(chunk_size=8192):
                        f_out.write(chunk)

    except Exception as e:
        if cache_path.exists():
            cache_path.unlink()
        raise HTTPException(status_code=500, detail=f"Proxy stream failed: {str(e)}")

    logger.info(f"Successfully cached and serving: {cache_path}")
    return FileResponse(
        path=cache_path, media_type="application/zip", filename=f"fused_{cache_key}.zip"
    )


class FusionRequest(BaseModel):
    bbox: list[float]
    resolution: float = 30.0
    format: str = "zarr"
    policy_name: str | None = None
    policy_yaml: str | None = None


@router.post("/fuse")
async def fuse_bathymetry_post(request: FusionRequest):
    topobathy_url = os.environ.get("TOPOBATHYSIM_URL", "http://localhost:9595").rstrip(
        "/"
    )
    target_url = f"{topobathy_url}/fuse"
    is_local = (
        "localhost" in topobathy_url
        or "127.0.0.1" in topobathy_url
        or "0.0.0.0" in topobathy_url
    )
    if is_local:
        client = httpx.AsyncClient()
        req = client.build_request(
            "POST", target_url, timeout=3600.0, json=request.model_dump()
        )
        try:
            response = await client.send(req, stream=True)
        except httpx.ConnectError:
            raise HTTPException(
                status_code=502,
                detail=f"TopoBathySim is unreachable at {topobathy_url}. Please ensure the service is running on port 9595.",
            )
        except httpx.ReadTimeout:
            raise HTTPException(
                status_code=504,
                detail=f"TopoBathySim at {topobathy_url} timed out during fusion.",
            )

        if response.status_code != 200:
            text = await response.aread()
            await response.aclose()
            await client.aclose()
            raise HTTPException(
                status_code=response.status_code,
                detail=f"TopoBathySim Error: {text.decode('utf-8')}",
            )

        async def cleanup():
            await response.aclose()
            await client.aclose()

        return StreamingResponse(
            response.aiter_raw(),
            media_type="application/zip",
            background=BackgroundTask(cleanup),
        )

    logger.info(
        f"TopoBathySim is remote ({topobathy_url}). Entering proxy-cache routine."
    )
    cache_dir = Path(settings.cache_dir()).expanduser() / "bathymetry"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Include policy in cache key
    policy_hash = "default"
    if request.policy_yaml:
        policy_hash = hashlib.md5(request.policy_yaml.encode()).hexdigest()
    elif request.policy_name:
        policy_hash = request.policy_name

    key_str = f"{request.bbox}_{request.resolution}_{request.format}_{policy_hash}"
    cache_key = hashlib.md5(key_str.encode()).hexdigest()
    cache_path = cache_dir / f"fused_proxy_{cache_key}.zip"

    if cache_path.exists():
        logger.info(f"Serving topography from local cache: {cache_path}")
        return FileResponse(
            path=cache_path,
            media_type="application/zip",
            filename=f"fused_{cache_key}.zip",
        )

    logger.info(
        f"Cache miss. Fetching from remote TopoBathySim and writing to {cache_path}..."
    )
    try:
        async with httpx.AsyncClient(timeout=3600.0) as client:
            async with client.stream(
                "POST", target_url, json=request.model_dump()
            ) as response:
                if response.status_code != 200:
                    text = await response.aread()
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"TopoBathySim Error: {text.decode('utf-8')}",
                    )
                with open(cache_path, "wb") as f_out:
                    async for chunk in response.aiter_bytes(chunk_size=8192):
                        f_out.write(chunk)
    except Exception as e:
        if cache_path.exists():
            cache_path.unlink()
        raise HTTPException(status_code=500, detail=f"Proxy stream failed: {str(e)}")

    return FileResponse(
        path=cache_path, media_type="application/zip", filename=f"fused_{cache_key}.zip"
    )
