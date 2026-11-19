"""The cache viewer's drawing API: any cached store on its own cells (see `view_store`).

Every route takes the store id from the inventory and `detail` (``2d`` or ``3d``, the cell
budget for large grids; element numbering is fixed for a store and detail). Binary bodies are
little-endian.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse, Response
from starlette.concurrency import run_in_threadpool

from forcingkit import settings
from forcingkit_serve import view_store
from forcingkit_serve.routers.viewer import _crawl_inventory

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/cache/view", tags=["Cache Viewer"])

Detail = Query("2d", pattern="^(2d|3d)$", description="Cell budget: 2d or 3d")
BINARY = "application/octet-stream"


# Crawling the whole cache takes about 0.4 s on a cache of a few hundred stores, far more than
# drawing a cached record, so the store ids the crawl found are reused for a few seconds.
# Only ids a crawl listed resolve, and a store whose path is gone is looked up again.
RESOLVE_TTL_S = 10.0
_resolved: dict[str, Any] = {"dir": None, "at": 0.0, "stores": {}}
_resolve_lock = threading.Lock()


def _fresh(cache_dir: str) -> bool:
    return (
        _resolved["dir"] == cache_dir
        and time.monotonic() - _resolved["at"] < RESOLVE_TTL_S
    )


def _locate(dataset_id: str) -> tuple[Path, Any] | None:
    """The store's path and change key (from the inventory), or None."""
    cache_dir = str(settings.cache_dir())
    if _fresh(cache_dir):
        hit = _resolved["stores"].get(dataset_id)
        if hit is not None and hit[0].exists():
            return hit
    with _resolve_lock:
        # Another request may have crawled while this one waited.
        hit = _resolved["stores"].get(dataset_id) if _fresh(cache_dir) else None
        if hit is None or not hit[0].exists():
            items = _crawl_inventory()
            _resolved["stores"] = {
                i.id: (Path(i.path), (i.modified_time, i.size_mb)) for i in items
            }
            _resolved["dir"], _resolved["at"] = cache_dir, time.monotonic()
        return _resolved["stores"].get(dataset_id)


def forget_stores() -> None:
    """Drop the remembered store ids, so the next request crawls the cache."""
    with _resolve_lock:
        _resolved.update(dir=None, at=0.0, stores={})


def _view(dataset_id: str, detail: str) -> view_store.StoreView:
    found = _locate(dataset_id)
    if found is None:
        raise HTTPException(
            status_code=404, detail=f"No store {dataset_id} in the cache."
        )
    path, key = found
    try:
        return view_store.get_view(dataset_id, path, detail, key)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


async def _run(fn: Callable[[], Any]) -> Any:
    try:
        return await run_in_threadpool(fn)
    except HTTPException:
        raise
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e).strip("'\""))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


def _cells_only(view: view_store.StoreView) -> None:
    if view.store == "points":
        raise HTTPException(status_code=422, detail="this store holds sites, not cells")


def _field(view: view_store.StoreView, name: str) -> None:
    if name not in view.fields:
        raise HTTPException(
            status_code=404,
            detail=f"no field {name!r}; fields: {', '.join(view.fields)}",
        )


@router.get("/{dataset_id}/manifest")
async def manifest(dataset_id: str, detail: str = Detail):
    """Fields (label, units, fixed scale, ticks, 256-entry RGBA ramp by byte code), layers
    surface first with mean heights, record times and cadence, cell counts and stride."""
    return JSONResponse(await _run(lambda: _view(dataset_id, detail).manifest()))


@router.get("/{dataset_id}/mesh")
async def mesh(dataset_id: str, detail: str = Detail):
    """Node longitudes, node latitudes (Float64 each) and triangles (Uint32, three per
    element). Sizes are in the manifest (n_node, n_elem)."""

    def body():
        view = _view(dataset_id, detail)
        _cells_only(view)
        return view.mesh_bytes()

    return Response(await _run(body), media_type=BINARY)


@router.get("/{dataset_id}/frame")
async def frame(
    dataset_id: str,
    field: str = Query(..., description="Field name from the manifest"),
    layer: int = Query(0, ge=0, description="Layer, 0 at the surface"),
    t: int = Query(0, ge=0, description="Record index"),
    detail: str = Detail,
):
    """One byte per element on the field's fixed scale (0: no value): for sites, one per site."""

    def body():
        view = _view(dataset_id, detail)
        _field(view, field)
        return view.codes(field, t, layer)

    return Response(await _run(body), media_type=BINARY)


@router.get("/{dataset_id}/heights")
async def heights(
    dataset_id: str,
    layer: int = Query(0, ge=0),
    t: int = Query(0, ge=0),
    detail: str = Detail,
):
    """Float32 per element: the layer center's height (m, positive up); 404 when the store
    holds no heights."""

    def body():
        view = _view(dataset_id, detail)
        _cells_only(view)
        z = view.heights(t, layer)
        if z is None:
            raise HTTPException(
                status_code=404, detail="this store holds no layer heights"
            )
        return np.asarray(z, "<f4").tobytes()

    return Response(await _run(body), media_type=BINARY)


@router.get("/{dataset_id}/bed")
async def bed(
    dataset_id: str,
    at: str = Query("element", pattern="^(element|node)$"),
    detail: str = Detail,
):
    """Float32 per element (or, with at=node, per mesh node, where the store holds the depth at
    its nodes): the bed's height (m, positive up), or the ground's for an atmosphere; 404 when
    the store holds none."""

    def body():
        view = _view(dataset_id, detail)
        _cells_only(view)
        b = view.bed_nodes() if at == "node" else view.bed()
        if b is None:
            raise HTTPException(status_code=404, detail="this store holds no bed")
        return np.asarray(b, "<f4").tobytes()

    return Response(await _run(body), media_type=BINARY)


@router.get("/{dataset_id}/point")
async def point(
    dataset_id: str,
    lon: float = Query(..., ge=-180, le=180),
    lat: float = Query(..., ge=-90, le=90),
    field: str = Query(...),
    layer: int = Query(0, ge=0),
    t: int = Query(0, ge=0),
    detail: str = Detail,
):
    """The cell under a point: value, layer height, bed, source cell index, record time (for a
    store of sites, the nearest site)."""

    def body():
        view = _view(dataset_id, detail)
        _field(view, field)
        return view.point(lon, lat, field, t, layer)

    return JSONResponse(await _run(body))


@router.get("/{dataset_id}/section")
async def section(
    dataset_id: str,
    pts: str = Query(..., description="lon,lat;lon,lat;... (two or more points)"),
    field: str = Query(...),
    t: int = Query(0, ge=0),
    n: int = Query(200, ge=2, le=1000),
    detail: str = Detail,
):
    """Values in every layer at n points along a polyline, each from its host cell."""

    def body():
        try:
            points = [[float(x) for x in p.split(",")] for p in pts.split(";") if p]
        except ValueError:
            raise HTTPException(
                status_code=422, detail="pts must be lon,lat;lon,lat;..."
            )
        if len(points) > 50:
            raise HTTPException(status_code=422, detail="at most 50 points")
        view = _view(dataset_id, detail)
        _cells_only(view)
        _field(view, field)
        return view.section(points, field, t, n)

    return JSONResponse(await _run(body))


@router.get("/{dataset_id}/tracers")
async def tracers(
    dataset_id: str,
    pair: str = Query(..., description="speed or wind_speed"),
    t: int = Query(0, ge=0),
    levels: int = Query(6, ge=1, le=12),
    limit: int = Query(3000, ge=10, le=20000),
    detail: str = Detail,
):
    """East and north components for tracers kept inside their own cells: up to `limit`
    columns (thinned evenly, the same at every level), each column's element, and up to
    `levels` layers from the surface down with their heights."""

    def body():
        view = _view(dataset_id, detail)
        _cells_only(view)
        return view.tracers(pair, t, levels, limit)

    return JSONResponse(await _run(body))
