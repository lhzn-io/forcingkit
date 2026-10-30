"""Client job flow against a mocked job API, with stores read from a real local HTTP server."""

import io
import json
import zipfile

import httpx
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from conftest import T0, hycom_legacy_store, linear_field, z_v3_store

from forcingkit_client import AsyncClient, Client, ForcingkitError

STORE_ID = "obc_0123456789ab"
BBOX = (-70.9, 41.3, -70.6, 41.6)


def _item(base: str, store_id: str = STORE_ID, schema: str = "z-v3") -> dict:
    return {
        "type": "Feature",
        "id": store_id,
        "properties": {
            "forcingkit:schema": schema,
            "forcingkit:donor": "necofs",
            "forcingkit:store_version": "1",
        },
        # The service names itself as it saw the request; the client re-roots on its own URL.
        "assets": {
            "zarr": {"href": f"http://elsewhere:9598/api/v1/stores/{store_id}.zarr"}
        },
    }


class FakeService:
    """/api/v1/jobs: queued, then running, then `final` ('succeeded' or 'failed')."""

    def __init__(self, base: str, final: str = "succeeded", schema: str = "z-v3"):
        self.base, self.final, self.schema = base, final, schema
        self.posts: list[dict] = []
        self.polls = 0

    def _job(self, status: str) -> dict:
        return {
            "job_id": STORE_ID,
            "kind": "ocean",
            "status": status,
            "error": "All OBC donors failed" if status == "failed" else None,
            "status_url": f"http://elsewhere:9598/api/v1/jobs/{STORE_ID}",
            "result": _item(self.base, schema=self.schema)
            if status == "succeeded"
            else None,
        }

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path.startswith("/api/v1/jobs/"):
            self.posts.append(json.loads(request.content))
            return httpx.Response(202, json=self._job("queued"))
        if request.method == "GET" and request.url.path == f"/api/v1/jobs/{STORE_ID}":
            assert request.url.host == "127.0.0.1", "status URL must be re-rooted"
            self.polls += 1
            return httpx.Response(
                200, json=self._job("running" if self.polls < 2 else self.final)
            )
        if request.url.path == "/api/v1/ndbc":
            return httpx.Response(200, json={"status": "success", "data": NDBC})
        if request.url.path == "/api/v1/bathymetry/fuse":
            return httpx.Response(200, content=_bathymetry_zip())
        return httpx.Response(
            400, json={"detail": f"unsupported source {request.url.path}"}
        )


NDBC = {
    "station_id": "44013",
    "lat": 42.35,
    "lon": -70.65,
    "products": {
        "stdmet": {
            "times": ["2026-06-01T00:00:00Z", "2026-06-01T01:00:00Z"],
            "units": {"WSPD": "m s-1"},
            "WSPD": [5.0, None],
        },
        "adcp": {
            "times": ["2026-06-01T01:00:00Z"],
            "depth_m": [2.0],
            "u": [0.1],
            "v": [-0.2],
            "units": {},
        },
    },
    "sources": ["https://www.ndbc.noaa.gov/"],
}


def _bathymetry_zip() -> bytes:
    buffer = io.BytesIO()
    store = xr.Dataset({"elevation": (("y", "x"), np.full((2, 2), -12.0))})
    with zipfile.ZipFile(buffer, "w") as z:
        mapping: dict = {}
        store.to_zarr(mapping, zarr_format=2, consolidated=True)
        for key, value in mapping.items():
            z.writestr(
                key, value.to_bytes() if hasattr(value, "to_bytes") else bytes(value)
            )
    return buffer.getvalue()


def _publish(root, ds: xr.Dataset) -> None:
    ds.to_zarr(
        root / "api/v1/stores" / f"{STORE_ID}.zarr",
        mode="w",
        consolidated=True,
        zarr_format=2,
    )


def _client(base, service, tmp_path, **kwargs) -> Client:
    return Client(
        base,
        cache_dir=tmp_path / "cache",
        poll_s=0,
        http=httpx.Client(transport=httpx.MockTransport(service)),
        **kwargs,
    )


def test_ocean_job_flow_and_cache(store_server, tmp_path):
    root, base, requests = store_server
    _publish(root, z_v3_store())
    service = FakeService(base)
    seen = []
    fk = _client(
        base, service, tmp_path, on_progress=lambda job: seen.append(job["status"])
    )

    ds = fk.ocean(BBOX, "2026-06-01T00:00:00Z", 6)
    assert seen == ["queued", "running", "succeeded"]
    assert service.posts[0]["pad_cells"] == 1 and service.posts[0]["sponge_cells"] == 0
    assert service.posts[0]["start_date"] == "2026-06-01T00:00:00"
    assert (
        ds.attrs["forcingkit_store_id"] == STORE_ID
        and ds.attrs["forcingkit_donor"] == "necofs"
    )
    np.testing.assert_allclose(
        ds["u"].isel(time=2).values, z_v3_store()["u"].isel(time=2).values
    )
    assert any("/u/" in path for path in requests), "chunks are read from the server"

    requests.clear()
    service.polls = 0
    again = fk.ocean(BBOX, "2026-06-01T00:00:00Z", 6)
    float(again["u"].isel(time=2).sum())
    assert requests == [], "a second read is served from the local cache"


def test_legacy_store_is_normalised(store_server, tmp_path):
    root, base, _ = store_server
    _publish(root, hycom_legacy_store())
    ds = _client(base, FakeService(base, schema="legacy"), tmp_path).ocean(BBOX, T0, 6)
    assert "z" in ds.dims and float(ds["lon"].min()) < 0


def test_failed_job_raises_with_detail(store_server, tmp_path):
    _, base, _ = store_server
    fk = _client(base, FakeService(base, final="failed"), tmp_path)
    with pytest.raises(ForcingkitError, match="All OBC donors failed") as excinfo:
        fk.ocean(BBOX, T0, 6)
    assert excinfo.value.detail == "All OBC donors failed"


def test_refused_request_raises(store_server, tmp_path):
    _, base, _ = store_server
    fk = _client(base, FakeService(base), tmp_path)
    with pytest.raises(ForcingkitError, match="400"):
        fk._check(fk._http.get(f"{base}/api/v1/unknown"))
    with pytest.raises(ValueError, match="west, south"):
        fk.ocean((-70.6, 41.3, -70.9, 41.6), T0, 6)


def test_ndbc_payload_to_dataset(store_server, tmp_path):
    _, base, _ = store_server
    ds = _client(base, FakeService(base), tmp_path).ndbc(
        "44013", T0, T0 + pd.Timedelta(hours=1), ("stdmet", "adcp")
    )
    assert ds.sizes["time"] == 2 and ds["WSPD"].attrs["units"] == "m s-1"
    assert np.isnan(ds["WSPD"].values[1]) and ds["current_u"].values[
        1
    ] == pytest.approx(0.1)
    assert ds.attrs["station_id"] == "44013"


def test_bathymetry_unpacked_and_cached(store_server, tmp_path):
    _, base, _ = store_server
    service = FakeService(base)
    fk = _client(base, service, tmp_path)
    elevation = fk.bathymetry(BBOX, 30.0)
    assert float(elevation.mean()) == -12.0
    fk._http = httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(500))
    )
    assert float(fk.bathymetry(BBOX, 30.0).mean()) == -12.0


def test_interpolate_trajectory_through_client(store_server, tmp_path):
    root, base, _ = store_server
    _publish(root, z_v3_store())
    fk = _client(base, FakeService(base), tmp_path)
    lon, lat = np.array([-70.85, -70.7]), np.array([41.35, 41.5])
    time = T0 + pd.to_timedelta([1.5, 4.25], unit="h")
    z = np.array([-6.0, -2.5])
    ds = fk.interpolate_trajectory(
        lon, lat, time, z, variables=("u",), trajectory_id="tag-01"
    )
    hours = (time - T0) / pd.Timedelta(hours=1)
    np.testing.assert_allclose(
        ds["u"].values, linear_field(lon, lat, z, hours), atol=1e-4
    )
    assert ds.attrs["forcingkit_ocean_store_id"] == STORE_ID
    assert ds.attrs["featureType"] == "trajectory"


def _waves_store() -> xr.Dataset:
    """A small store in the nwps-waves-v1 layout, with one land cell."""
    lon = np.linspace(-72.5, -72.4, 3)
    lat = np.linspace(41.0, 41.1, 2)
    time = pd.date_range(T0 - pd.Timedelta(hours=1), periods=5, freq="1h")
    hs = np.ones((5, 2, 3), dtype=np.float32)
    hs[:, 0, 0] = np.nan
    return xr.Dataset(
        {
            "hs": (("time", "lat", "lon"), hs, {"units": "m"}),
            "dir": (("time", "lat", "lon"), 90.0 * hs, {"units": "degree"}),
            "forecast_hour": ("time", np.arange(5, dtype=np.float32)),
            "wet": (("lat", "lon"), np.array([[0, 1, 1], [1, 1, 1]], dtype=np.uint8)),
        },
        coords={"time": time, "lat": lat, "lon": lon},
        attrs={"schema": "nwps-waves-v1", "complete": True, "wfo": "okx"},
    )


def test_waves_job_flow(store_server, tmp_path):
    root, base, _ = store_server
    _publish(root, _waves_store())
    service = FakeService(base, schema="nwps-waves-v1")
    ds = _client(base, service, tmp_path).waves(BBOX, T0, 2, wfo="box", grid="CG2")
    post = service.posts[0]
    assert (post["source"], post["wfo"], post["grid"], post["hours"]) == (
        "nwps",
        "box",
        "CG2",
        2,
    )
    assert post["start_time"] == "2026-06-01T00:00:00"
    assert ds.attrs["forcingkit_schema"] == "nwps-waves-v1"
    assert np.isnan(ds["hs"].values[0, 0, 0]) and ds["wet"].values[0, 0] == 0


async def test_async_client(store_server, tmp_path):
    root, base, _ = store_server
    _publish(root, z_v3_store())
    service = FakeService(base)
    async with AsyncClient(
        base,
        cache_dir=tmp_path / "cache",
        poll_s=0,
        http=httpx.AsyncClient(transport=httpx.MockTransport(service)),
    ) as fk:
        ds = await fk.ocean(BBOX, T0, 6)
        assert ds.attrs["forcingkit_store_id"] == STORE_ID
        service.polls = 0
        traj = await fk.interpolate_trajectory(
            [-70.8], [41.4], [T0 + pd.Timedelta(hours=2)], [-3.0]
        )
        assert traj["u"].values[0] == pytest.approx(
            linear_field(-70.8, 41.4, -3.0, 2.0), abs=1e-4
        )


async def test_async_waves(store_server, tmp_path):
    root, base, _ = store_server
    _publish(root, _waves_store())
    service = FakeService(base, schema="nwps-waves-v1")
    async with AsyncClient(
        base,
        cache_dir=tmp_path / "cache",
        poll_s=0,
        http=httpx.AsyncClient(transport=httpx.MockTransport(service)),
    ) as fk:
        ds = await fk.waves(BBOX, T0, 2)
    assert service.posts[0]["wfo"] == "okx" and service.posts[0]["grid"] == "CG1"
    assert float(ds["dir"].values[0, 1, 1]) == 90.0
