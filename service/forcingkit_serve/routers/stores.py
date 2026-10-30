"""Finished Zarr stores served read-only over HTTP, so a client opens a store remotely and reads
only the variables and chunks it needs, instead of downloading a zip of the whole store."""

import os
import re

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from forcingkit import settings

router = APIRouter(prefix="/api/v1/stores", tags=["stores"])

STORE_ID = re.compile(r"^(obc|atm|wav)_[0-9a-f]{12}$")


def store_complete(zarr_path: str) -> bool:
    from forcingkit.fetchers.hrrr_atmosphere import HRRR_ATM_SCHEMA
    from forcingkit.fetchers.necofs import OBC_SCHEMA
    from forcingkit.fetchers.nwps import NWPS_SCHEMA
    from forcingkit.zarr_stream import store_is_complete

    return store_is_complete(zarr_path, (OBC_SCHEMA, HRRR_ATM_SCHEMA, NWPS_SCHEMA))


@router.api_route("/{store}/{path:path}", methods=["GET", "HEAD"])
def store_file(store: str, path: str) -> FileResponse:
    from forcingkit_serve.jobs import registry

    zarr_id = store.removesuffix(".zarr")
    if not store.endswith(".zarr") or not STORE_ID.match(zarr_id):
        raise HTTPException(status_code=400, detail="invalid store id")
    root = os.path.realpath(os.path.join(settings.cache_dir(), store))
    target = os.path.realpath(os.path.join(root, path))
    if not target.startswith(root + os.sep):
        raise HTTPException(status_code=400, detail="path outside the store")
    # A store still being built (by a job, or written in one shot by a legacy donor) is not
    # served, so no reader ever sees half a store.
    if registry.is_active(zarr_id) or not store_complete(root):
        raise HTTPException(status_code=404, detail=f"store {zarr_id} not available")
    if not os.path.isfile(target):
        raise HTTPException(status_code=404, detail=f"{path} not in store {zarr_id}")
    return FileResponse(target, media_type="application/octet-stream")
