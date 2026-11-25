"""Clients for a forcingkit service, synchronous (`Client`) and asynchronous (`AsyncClient`).

Store builds go through the service's job queue: the client submits a job, polls its status URL
and, once it has succeeded, opens the store named by the job's STAC Item over HTTP through a local
chunk cache. A second request for the same store is answered from the cache.
"""

import asyncio
import hashlib
import io
import logging
import os
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, TypeAlias
from urllib.parse import urlsplit

import httpx
import numpy as np
import pandas as pd
import xarray as xr

from .schema import normalise
from .store import open_remote

log = logging.getLogger(__name__)

BBox: TypeAlias = Sequence[float]  # (west, south, east, north), degrees
TimeLike: TypeAlias = str | datetime | pd.Timestamp | np.datetime64

DEFAULT_CACHE_DIR = "~/.cache/forcingkit-client"
ProgressCallback = Callable[[Dict[str, Any]], None]


class ForcingkitError(RuntimeError):
    """A request the service refused or a job that failed; `detail` carries the server's text."""

    def __init__(self, message: str, status: Optional[int] = None, detail: Any = None):
        super().__init__(message)
        self.status = status
        self.detail = detail


def iso_utc(t: TimeLike) -> str:
    """`t` as an ISO 8601 UTC string without offset; naive times are taken as UTC."""
    ts = pd.Timestamp(t)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.strftime("%Y-%m-%dT%H:%M:%S")


def _bbox(bbox: BBox) -> Dict[str, float]:
    west, south, east, north = (float(v) for v in bbox)
    if not (south < north and west < east):
        raise ValueError(f"bbox must be (west, south, east, north), got {tuple(bbox)}")
    return {"min_lon": west, "min_lat": south, "max_lon": east, "max_lat": north}


def _ocean_request(
    bbox,
    start,
    hours,
    vertical_spacing_m,
    pad_cells,
    sponge_cells,
    donor: Optional[str] = None,
    **options,
):
    request = {
        "bbox": _bbox(bbox),
        "start_date": iso_utc(start),
        "duration_hours": int(hours),
        "vertical_spacing_m": float(vertical_spacing_m),
        # Model-boundary options a non-model caller rarely wants: one donor cell of padding so
        # the store brackets the box, and no sponge.
        "pad_cells": int(pad_cells),
        "sponge_cells": int(sponge_cells),
        **options,
    }
    if donor:
        request["donor"] = str(donor)
    return request


def _check_donor(item: Dict[str, Any], donor: Optional[str]) -> Dict[str, Any]:
    """`item`, unless a named `donor` is not the one that delivered it: a service older than the
    `donor` option ignores it and ranks the donors, and its store must not pass for the named one."""
    delivered = item["properties"].get("forcingkit:donor")
    if donor and delivered != donor:
        raise ForcingkitError(
            f"asked for donor {donor!r} but store {item['id']} is from {delivered!r}; "
            "the service may predate the donor option",
            detail={"requested": donor, "delivered": delivered},
        )
    return item


def _atmosphere_request(bbox, start, hours, resolution_deg, margin_deg, cache_bust):
    return {
        "bbox": _bbox(bbox),
        "start_time": iso_utc(start),
        "hours": int(hours),
        "resolution_deg": float(resolution_deg),
        "margin_deg": float(margin_deg),
        "cache_bust": bool(cache_bust),
    }


def _waves_request(bbox, start, hours, wfo, grid, cache_bust):
    return {
        "bbox": _bbox(bbox),
        "start_time": iso_utc(start),
        "hours": int(hours),
        "source": "nwps",
        "wfo": str(wfo),
        "grid": str(grid),
        "cache_bust": bool(cache_bust),
    }


def _ndbc_dataset(data: Dict[str, Any]) -> xr.Dataset:
    """The /api/v1/ndbc payload as one Dataset on a shared time axis (outer join)."""
    parts = []
    products = data.get("products", {})
    if "stdmet" in products:
        p = products["stdmet"]
        times = pd.to_datetime(p["times"]).tz_localize(None)
        ds = xr.Dataset(coords={"time": times})
        for name, units in p.get("units", {}).items():
            ds[name] = ("time", np.asarray(p[name], dtype=np.float64))
            ds[name].attrs["units"] = units
        parts.append(ds)
    if "adcp" in products:
        p = products["adcp"]
        times = pd.to_datetime(p["times"]).tz_localize(None)
        ds = xr.Dataset(coords={"time": times})
        for name, key in (
            ("current_u", "u"),
            ("current_v", "v"),
            ("current_depth", "depth_m"),
        ):
            if len(p.get(key, [])) == len(times):
                ds[name] = ("time", np.asarray(p[key], dtype=np.float64))
        if "current_u" in ds:
            ds["current_u"].attrs.update(
                units="m s-1", standard_name="eastward_sea_water_velocity"
            )
            ds["current_v"].attrs.update(
                units="m s-1", standard_name="northward_sea_water_velocity"
            )
        if "current_depth" in ds:
            ds["current_depth"].attrs.update(units="m", positive="down")
        parts.append(ds)
    out = xr.merge(parts, join="outer") if parts else xr.Dataset()
    out.attrs.update(
        station_id=data.get("station_id"),
        lat=data.get("lat"),
        lon=data.get("lon"),
        sources=list(data.get("sources", [])),
    )
    return out


def _bathymetry_key(bbox: BBox, resolution_m: float) -> str:
    return hashlib.sha256(
        f"{tuple(map(float, bbox))}_{float(resolution_m)}".encode()
    ).hexdigest()[:16]


def _unpack_bathymetry(content: bytes, target: Path) -> xr.DataArray:
    partial = target.with_suffix(".part")
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        z.extractall(partial)
    os.replace(partial, target)
    return _open_bathymetry(target)


def _open_bathymetry(target: Path) -> xr.DataArray:
    # The archive holds either the store's contents or the store directory itself.
    markers = (".zgroup", ".zmetadata", "zarr.json")
    root = next(
        (
            p
            for p in [target, *sorted(target.rglob("*"))]
            if p.is_dir() and any((p / m).exists() for m in markers)
        ),
        None,
    )
    if root is None:
        raise ForcingkitError(f"bathymetry archive at {target} holds no Zarr store")
    ds = xr.open_zarr(root)
    name = next(iter(ds.data_vars))
    return ds[name]


class _Base:
    def __init__(
        self,
        base_url: str,
        cache_dir: str | Path = DEFAULT_CACHE_DIR,
        poll_s: float = 5.0,
        timeout_s: float = 3 * 3600.0,
        on_progress: Optional[ProgressCallback] = None,
        attempts: int = 4,
        retry_wait_s: float = 2.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.cache_dir = Path(cache_dir).expanduser()
        self.poll_s = poll_s
        self.timeout_s = timeout_s
        self.on_progress = on_progress
        self.attempts = max(1, attempts)
        self.retry_wait_s = retry_wait_s

    def _retry_wait(
        self, method: str, url: str, error: Exception, attempt: int
    ) -> Optional[float]:
        """Seconds to wait before retrying a job request that failed in transport (a failed name
        lookup, a refused or reset connection, a timeout), or None when its attempts are spent.
        Submitting a job again is safe: the service keys a job by its request."""
        if attempt >= self.attempts:
            return None
        log.warning(
            "%s %s failed (%s: %s); retry %d of %d",
            method,
            url,
            type(error).__name__,
            error,
            attempt,
            self.attempts - 1,
        )
        return self.retry_wait_s * attempt

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _local(self, href: str) -> str:
        """`href` re-rooted on this client's base URL: the service names itself as it saw the
        request, which need not be an address this host can reach."""
        return self._url(urlsplit(href).path)

    @staticmethod
    def _check(response: httpx.Response) -> Any:
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail")
            except Exception:
                detail = response.text
            raise ForcingkitError(
                f"{response.request.method} {response.request.url.path} failed "
                f"({response.status_code}): {detail}",
                status=response.status_code,
                detail=detail,
            )
        return response.json()

    def _settled(self, job: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """The job's STAC Item once it has succeeded, None while it runs; raises on failure."""
        if self.on_progress is not None:
            self.on_progress(job)
        if job["status"] == "succeeded":
            return job["result"]
        if job["status"] == "failed":
            raise ForcingkitError(
                f"{job['kind']} job {job['job_id']} failed: {job['error']}",
                detail=job["error"],
            )
        return None

    def open_item(self, item: Dict[str, Any]) -> xr.Dataset:
        """Open the store a job's STAC Item describes, through the local cache, normalised."""
        props = item["properties"]
        ds = open_remote(
            self._local(item["assets"]["zarr"]["href"]),
            self.cache_dir,
            item["id"],
            str(props["forcingkit:store_version"]),
        )
        ds = normalise(ds)
        ds.attrs.update(
            {
                "forcingkit_store_id": item["id"],
                "forcingkit_store_version": str(props["forcingkit:store_version"]),
                "forcingkit_donor": props.get("forcingkit:donor") or "",
                "forcingkit_schema": props.get("forcingkit:schema") or "",
            }
        )
        # Ocean stores say where their vertical velocity comes from ("model", "continuity" or
        # "none"); a service older than the property, and other kinds of store, leave it out.
        if props.get("forcingkit:w_source"):
            ds.attrs["forcingkit_w_source"] = props["forcingkit:w_source"]
        return ds


class Client(_Base):
    """Synchronous client.

    >>> fk = Client("http://localhost:9598")
    >>> ocean = fk.ocean((-70.9, 41.3, -70.6, 41.6), "2025-07-01", 24)
    """

    def __init__(
        self,
        base_url: str,
        *args: Any,
        http: Optional[httpx.Client] = None,
        **kwargs: Any,
    ):
        super().__init__(base_url, *args, **kwargs)
        self._http = http or httpx.Client(timeout=httpx.Timeout(120.0, connect=10.0))

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "Client":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def _send(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        attempt = 1
        while True:
            try:
                return self._http.request(method, url, **kwargs)
            except httpx.TransportError as e:
                wait = self._retry_wait(method, url, e, attempt)
                if wait is None:
                    raise
                time.sleep(wait)
                attempt += 1

    def _wait(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        job = self._check(self._send("POST", self._url(path), json=payload))
        deadline = time.monotonic() + self.timeout_s
        while (item := self._settled(job)) is None:
            if time.monotonic() > deadline:
                raise ForcingkitError(
                    f"job {job['job_id']} still {job['status']} after {self.timeout_s:.0f} s"
                )
            time.sleep(self.poll_s)
            job = self._check(self._send("GET", self._local(job["status_url"])))
        return item

    def ocean(
        self,
        bbox: BBox,
        start: TimeLike,
        hours: int,
        vertical_spacing_m: float = 2.0,
        pad_cells: int = 1,
        sponge_cells: int = 0,
        donor: Optional[str] = None,
        **options: Any,
    ) -> xr.Dataset:
        """The parent ocean over `bbox` for `hours` from `start`: `u, v` (m/s), `temp` (deg C),
        `salt`, `zeta` (m) on `(time, z, lat, lon)`, z in meters positive up, plus `w` (m/s,
        upward; its origin in the `forcingkit_w_source` attribute).

        `donor` names one parent model to use alone ("necofs", "hycom", "doppio", ...): the
        service neither ranks nor falls back, and refuses an unknown id or a box outside that
        model's domain. A store from any other donor (a service too old to know the option) is
        refused here with ForcingkitError. Without it the service picks the donor. Further
        `options` (`include_tides`, `allow_donor_fallback`, `cache_bust`, ...) pass through to
        the service."""
        payload = _ocean_request(
            bbox,
            start,
            hours,
            vertical_spacing_m,
            pad_cells,
            sponge_cells,
            donor=donor,
            **options,
        )
        item = _check_donor(self._wait("/api/v1/jobs/obc", payload), donor)
        return self.open_item(item)

    def atmosphere(
        self,
        bbox: BBox,
        start: TimeLike,
        hours: int,
        resolution_deg: float = 0.03,
        margin_deg: float = 0.25,
        cache_bust: bool = False,
    ) -> xr.Dataset:
        """HRRR surface atmosphere (10 m wind, 2 m temperature and humidity, pressure, radiation,
        precipitation) hourly on a regular longitude-latitude grid."""
        payload = _atmosphere_request(
            bbox, start, hours, resolution_deg, margin_deg, cache_bust
        )
        return self.open_item(self._wait("/api/v1/jobs/atmosphere", payload))

    def waves(
        self,
        bbox: BBox,
        start: TimeLike,
        hours: int,
        wfo: str = "okx",
        grid: str = "CG1",
        cache_bust: bool = False,
    ) -> xr.Dataset:
        """NWPS surface waves (`hs`, `tp`, `dir` as direction of travel, `shts`, and the forcing
        wind `ws`, `wdir`) hourly on the office's own grid, land as NaN. `start` must be on the
        hour; NOMADS reaches back about five days unless the service has an archive."""
        payload = _waves_request(bbox, start, hours, wfo, grid, cache_bust)
        return self.open_item(self._wait("/api/v1/jobs/waves", payload))

    def ndbc(
        self,
        station_id: str,
        start: TimeLike,
        end: TimeLike,
        products: Sequence[str] = ("stdmet",),
    ) -> xr.Dataset:
        """Observations at an NDBC station: `stdmet` columns as named by NDBC, `adcp` as
        `current_u`, `current_v` (m/s) and `current_depth` (m)."""
        body = self._check(
            self._http.post(
                self._url("/api/v1/ndbc"),
                json={
                    "station_id": station_id,
                    "start_time": iso_utc(start),
                    "end_time": iso_utc(end),
                    "products": list(products),
                },
            )
        )
        return _ndbc_dataset(body["data"])

    def bathymetry(self, bbox: BBox, resolution_m: float = 10.0) -> xr.DataArray:
        """Fused topography and bathymetry (m, positive up) over `bbox`, cached locally."""
        target = self.cache_dir / "bathymetry" / _bathymetry_key(bbox, resolution_m)
        if target.exists():
            return _open_bathymetry(target)
        west, south, east, north = (float(v) for v in bbox)
        response = self._http.get(
            self._url("/api/v1/bathymetry/fuse"),
            params={
                "west": west,
                "south": south,
                "east": east,
                "north": north,
                "resolution": resolution_m,
                "format": "zarr",
            },
            timeout=httpx.Timeout(self.timeout_s, connect=10.0),
        )
        if response.status_code >= 400:
            self._check(response)
        target.parent.mkdir(parents=True, exist_ok=True)
        return _unpack_bathymetry(response.content, target)

    def interpolate_trajectory(
        self, lon, lat, time, z=None, **kwargs: Any
    ) -> xr.Dataset:
        """Forcing interpolated to each point of a trajectory; see
        `forcingkit_client.trajectory.interpolate_trajectory`."""
        from .trajectory import interpolate_trajectory

        return interpolate_trajectory(self, lon, lat, time, z, **kwargs)


class AsyncClient(_Base):
    """Asynchronous client with the same methods as `Client`, for callers that must not block
    their event loop (an application's main thread, for example)."""

    def __init__(
        self,
        base_url: str,
        *args: Any,
        http: Optional[httpx.AsyncClient] = None,
        **kwargs: Any,
    ):
        super().__init__(base_url, *args, **kwargs)
        self._http = http or httpx.AsyncClient(
            timeout=httpx.Timeout(120.0, connect=10.0)
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> "AsyncClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    async def _send(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        attempt = 1
        while True:
            try:
                return await self._http.request(method, url, **kwargs)
            except httpx.TransportError as e:
                wait = self._retry_wait(method, url, e, attempt)
                if wait is None:
                    raise
                await asyncio.sleep(wait)
                attempt += 1

    async def _wait(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        job = self._check(await self._send("POST", self._url(path), json=payload))
        deadline = time.monotonic() + self.timeout_s
        while (item := self._settled(job)) is None:
            if time.monotonic() > deadline:
                raise ForcingkitError(
                    f"job {job['job_id']} still {job['status']} after {self.timeout_s:.0f} s"
                )
            await asyncio.sleep(self.poll_s)
            job = self._check(await self._send("GET", self._local(job["status_url"])))
        return item

    async def ocean(
        self,
        bbox: BBox,
        start: TimeLike,
        hours: int,
        vertical_spacing_m: float = 2.0,
        pad_cells: int = 1,
        sponge_cells: int = 0,
        donor: Optional[str] = None,
        **options: Any,
    ) -> xr.Dataset:
        """`Client.ocean` (including the `donor` option) for an asynchronous client."""
        payload = _ocean_request(
            bbox,
            start,
            hours,
            vertical_spacing_m,
            pad_cells,
            sponge_cells,
            donor=donor,
            **options,
        )
        item = _check_donor(await self._wait("/api/v1/jobs/obc", payload), donor)
        return await asyncio.to_thread(self.open_item, item)

    async def atmosphere(
        self,
        bbox: BBox,
        start: TimeLike,
        hours: int,
        resolution_deg: float = 0.03,
        margin_deg: float = 0.25,
        cache_bust: bool = False,
    ) -> xr.Dataset:
        payload = _atmosphere_request(
            bbox, start, hours, resolution_deg, margin_deg, cache_bust
        )
        item = await self._wait("/api/v1/jobs/atmosphere", payload)
        return await asyncio.to_thread(self.open_item, item)

    async def waves(
        self,
        bbox: BBox,
        start: TimeLike,
        hours: int,
        wfo: str = "okx",
        grid: str = "CG1",
        cache_bust: bool = False,
    ) -> xr.Dataset:
        payload = _waves_request(bbox, start, hours, wfo, grid, cache_bust)
        item = await self._wait("/api/v1/jobs/waves", payload)
        return await asyncio.to_thread(self.open_item, item)

    async def ndbc(
        self,
        station_id: str,
        start: TimeLike,
        end: TimeLike,
        products: Sequence[str] = ("stdmet",),
    ) -> xr.Dataset:
        response = await self._http.post(
            self._url("/api/v1/ndbc"),
            json={
                "station_id": station_id,
                "start_time": iso_utc(start),
                "end_time": iso_utc(end),
                "products": list(products),
            },
        )
        return _ndbc_dataset(self._check(response)["data"])

    async def bathymetry(self, bbox: BBox, resolution_m: float = 10.0) -> xr.DataArray:
        target = self.cache_dir / "bathymetry" / _bathymetry_key(bbox, resolution_m)
        if target.exists():
            return await asyncio.to_thread(_open_bathymetry, target)
        west, south, east, north = (float(v) for v in bbox)
        response = await self._http.get(
            self._url("/api/v1/bathymetry/fuse"),
            params={
                "west": west,
                "south": south,
                "east": east,
                "north": north,
                "resolution": resolution_m,
                "format": "zarr",
            },
            timeout=httpx.Timeout(self.timeout_s, connect=10.0),
        )
        if response.status_code >= 400:
            self._check(response)
        target.parent.mkdir(parents=True, exist_ok=True)
        return await asyncio.to_thread(_unpack_bathymetry, response.content, target)

    async def interpolate_trajectory(
        self, lon, lat, time, z=None, **kwargs: Any
    ) -> xr.Dataset:
        from .trajectory import interpolate_trajectory_async

        return await interpolate_trajectory_async(self, lon, lat, time, z, **kwargs)
