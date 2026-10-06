"""Stubs for the routes removed on 2026-10-05.

Each answers 410 Gone with the endpoint that replaces it, so a client still calling one learns
where to go instead of meeting a bare 404. The response carries `Deprecation` (RFC 9745) and
`Sunset` (RFC 8594) headers, both at the removal date, and a `Link` header to the documentation
at forcingkit.docs.lhzn.io. The next release drops this module and its `include_router`.
"""

import logging
from datetime import datetime, timezone
from email.utils import format_datetime

from fastapi import APIRouter
from fastapi.responses import JSONResponse

logger = logging.getLogger("forcingkit_serve")

REMOVED_ON = datetime(2026, 10, 5, tzinfo=timezone.utc)
DOCS_URL = "https://forcingkit.docs.lhzn.io/"

_PARENT = (
    "POST /api/v1/obc: the parent ocean on true z (schema z-v3); its first record is the "
    "initial condition"
)
_ATMOSPHERE = (
    "POST /api/v1/atmosphere: the HRRR surface atmosphere (schema hrrr-atm-v1), including "
    "10 m wind"
)

# (method, path) -> replacement
REMOVED_ROUTES: dict[tuple[str, str], str] = {
    ("POST", "/api/v1/ic/generate"): _PARENT,
    ("POST", "/api/v1/ic/regrid"): _PARENT,
    ("POST", "/api/v1/ic/cache"): "POST /api/v1/obc/cache",
    ("POST", "/api/v1/ic/predict-donor"): "POST /api/v1/obc/predict-donor",
    ("GET", "/api/v1/ic/download/{zarr_id}"): "GET /api/v1/obc/download/{zarr_id}",
    ("POST", "/api/v1/bc/generate"): _ATMOSPHERE,
    ("POST", "/api/v1/bc/cache"): _ATMOSPHERE + " (its response carries the store id)",
    ("POST", "/api/v1/bc/predict-donor"): (
        "none: the atmosphere is HRRR from 2014-07-30; earlier runs use ERA5 through "
        "NumericalEarth in the model"
    ),
    (
        "GET",
        "/api/v1/bc/download/{zarr_id}",
    ): "GET /api/v1/atmosphere/download/{zarr_id}",
    ("POST", "/api/v1/harmonics"): (
        "POST /api/v1/tide for observed water level, or the `zeta` variable of the "
        "/api/v1/obc parent store"
    ),
    ("GET", "/api/v2/coupled-boundaries/{region}"): "POST /api/v1/obc",
}

router = APIRouter(tags=["Removed"])


def _gone(method: str, path: str, replacement: str):
    async def handler() -> JSONResponse:
        logger.warning(f"Removed endpoint called: {method} {path}; use {replacement}")
        return JSONResponse(
            status_code=410,
            content={
                "status": "gone",
                "detail": f"{method} {path} was removed on {REMOVED_ON.date().isoformat()}.",
                "replacement": replacement,
                "docs": DOCS_URL,
            },
            headers={
                "Deprecation": f"@{int(REMOVED_ON.timestamp())}",
                "Sunset": format_datetime(REMOVED_ON, usegmt=True),
                "Link": f'<{DOCS_URL}>; rel="deprecation", <{DOCS_URL}>; rel="sunset"',
            },
        )

    return handler


for (_method, _path), _replacement in REMOVED_ROUTES.items():
    router.add_api_route(
        _path,
        _gone(_method, _path, _replacement),
        methods=[_method],
        status_code=410,
        deprecated=True,
        summary=f"Removed on {REMOVED_ON.date().isoformat()}; use {_replacement.split(':')[0]}",
    )
