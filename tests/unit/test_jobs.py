"""The job queue (/api/v1/jobs), STAC Item results and stores served over HTTP (/api/v1/stores)."""

import threading
import time

import numpy as np
import pandas as pd
import pystac
import pytest
import xarray as xr
from fastapi.testclient import TestClient

from forcingkit import dispatcher, settings
from forcingkit_serve import jobs, main

BBOX = {"min_lon": -70.9, "min_lat": 41.3, "max_lon": -70.6, "max_lat": 41.6}
OBC = {"bbox": BBOX, "start_date": "2026-06-01T00:00:00", "duration_hours": 6}


def _write_store(
    path: str, complete: bool = True, source: str = "forcingkit.fetchers.necofs"
):
    lon = np.linspace(-70.95, -70.55, 5)
    lat = np.linspace(41.25, 41.65, 4)
    time_ = pd.date_range("2026-06-01", periods=7, freq="1h")
    ds = xr.Dataset(
        {"u": (("time", "z", "lat", "lon"), np.zeros((7, 2, 4, 5), dtype=np.float32))},
        coords={"time": time_, "z": [-3.0, -1.0], "lat": lat, "lon": lon},
        attrs={"schema": "z-v3", "complete": complete, "source": source},
    )
    ds.to_zarr(path, mode="w", consolidated=True, zarr_format=2)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(
        settings, "cache_dir", lambda *parts: str(tmp_path.joinpath(*parts))
    )
    monkeypatch.setattr(jobs, "registry", jobs.JobRegistry(workers=2))
    return TestClient(main.app)


def _wait(client, url, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(url).json()
        if body["status"] not in ("queued", "running"):
            return body
        time.sleep(0.02)
    raise AssertionError(f"job at {url} still {body['status']}")


def test_job_builds_store_and_returns_stac_item(client, monkeypatch):
    calls = []

    def fake_dispatch(**kwargs):
        calls.append(kwargs)
        _write_store(kwargs["zarr_path"])
        return kwargs["zarr_path"]

    monkeypatch.setattr(dispatcher, "dispatch_obc_request", fake_dispatch)
    submitted = client.post("/api/v1/jobs/obc", json=OBC)
    assert submitted.status_code == 202
    body = submitted.json()
    assert body["job_id"].startswith("obc_") and body["kind"] == "ocean"

    done = _wait(client, body["status_url"])
    assert done["status"] == "succeeded" and len(calls) == 1
    item = pystac.Item.from_dict(done["result"])
    assert item.id == body["job_id"]
    assert item.bbox == pytest.approx([-70.95, 41.25, -70.55, 41.65])
    assert item.properties["forcingkit:schema"] == "z-v3"
    assert item.properties["forcingkit:donor"] == "necofs"
    assert item.properties["start_datetime"] == "2026-06-01T00:00:00Z"
    assert item.properties["end_datetime"] == "2026-06-01T06:00:00Z"
    asset = item.assets["zarr"]
    assert asset.href.endswith(f"/api/v1/stores/{item.id}.zarr")
    assert asset.media_type == "application/vnd+zarr"


def test_identical_submissions_share_one_build(client, monkeypatch):
    release = threading.Event()
    calls = []

    def slow_dispatch(**kwargs):
        calls.append(kwargs)
        release.wait(5)
        _write_store(kwargs["zarr_path"])
        return kwargs["zarr_path"]

    monkeypatch.setattr(dispatcher, "dispatch_obc_request", slow_dispatch)
    first = client.post("/api/v1/jobs/obc", json=OBC).json()
    second = client.post("/api/v1/jobs/obc", json=OBC).json()
    assert first["job_id"] == second["job_id"]
    # The store is not served while its build runs.
    assert (
        client.get(f"/api/v1/stores/{first['job_id']}.zarr/.zmetadata").status_code
        == 404
    )
    release.set()
    assert _wait(client, first["status_url"])["status"] == "succeeded"
    assert len(calls) == 1


def test_complete_store_succeeds_without_a_build(client, tmp_path, monkeypatch):
    def unexpected(**kwargs):
        raise AssertionError("dispatch should not run for a cached store")

    monkeypatch.setattr(dispatcher, "dispatch_obc_request", unexpected)
    zarr_id, zarr_path, _ = main._obc_store(main.OBCRequest(**OBC))
    _write_store(zarr_path)
    body = client.post("/api/v1/jobs/obc", json=OBC).json()
    assert body["job_id"] == zarr_id and body["status"] == "succeeded"
    assert body["result"]["properties"]["forcingkit:store_version"]


def test_failed_job_reports_error_and_retries(client, monkeypatch):
    attempts = []

    def failing(**kwargs):
        attempts.append(1)
        raise RuntimeError("All OBC donors failed")

    monkeypatch.setattr(dispatcher, "dispatch_obc_request", failing)
    body = client.post("/api/v1/jobs/obc", json=OBC).json()
    done = _wait(client, body["status_url"])
    assert done["status"] == "failed" and "All OBC donors failed" in done["error"]
    assert done["result"] is None
    _wait(client, client.post("/api/v1/jobs/obc", json=OBC).json()["status_url"])
    assert len(attempts) == 2


def test_store_finished_before_restart_is_a_succeeded_job(client):
    zarr_id, zarr_path, _ = main._obc_store(main.OBCRequest(**OBC))
    _write_store(zarr_path)
    body = client.get(f"/api/v1/jobs/{zarr_id}").json()
    assert body["status"] == "succeeded" and body["result"]["id"] == zarr_id


def test_unknown_and_invalid_job_ids(client):
    assert client.get("/api/v1/jobs/obc_000000000000").status_code == 404
    assert client.get("/api/v1/jobs/not-a-store").status_code == 400


def test_atmosphere_job_rejects_unsupported_source(client):
    request = {
        "bbox": BBOX,
        "start_time": "2026-06-01T00:00:00",
        "hours": 3,
        "source": "era5",
    }
    assert client.post("/api/v1/jobs/atmosphere", json=request).status_code == 400


WAVES = {"bbox": BBOX, "start_time": "2026-06-01T00:00:00", "hours": 3}


def _write_waves_store(path: str):
    time_ = pd.date_range("2026-05-31T23:00", periods=6, freq="1h")
    ds = xr.Dataset(
        {"hs": (("time", "lat", "lon"), np.ones((6, 2, 3), dtype=np.float32))},
        coords={"time": time_, "lat": [41.3, 41.4], "lon": [-70.9, -70.8, -70.7]},
        attrs={"schema": "nwps-waves-v1", "complete": True},
    )
    ds.to_zarr(path, mode="w", consolidated=True, zarr_format=2)


@pytest.mark.parametrize(
    "change, message",
    [
        ({"source": "ww3"}, "unsupported source"),
        ({"wfo": "xyz"}, "unknown NWPS office"),
        ({"grid": "CG0"}, "CG1 to CG9"),
        ({"start_time": "2026-06-01T00:30:00"}, "on the hour"),
        ({"hours": 0}, "at least 1"),
    ],
)
def test_waves_rejects_bad_requests(client, change, message):
    for route in ("/api/v1/waves", "/api/v1/jobs/waves"):
        response = client.post(route, json={**WAVES, **change})
        assert response.status_code == 400 and message in response.json()["detail"]


def test_waves_job_returns_stac_item_and_survives_restart(client, monkeypatch):
    calls = []

    def fake_dispatch(bbox, start_time, hours, zarr_path, **kwargs):
        calls.append(kwargs)
        _write_waves_store(zarr_path)
        return zarr_path

    monkeypatch.setattr(dispatcher, "dispatch_waves_request", fake_dispatch)
    body = client.post("/api/v1/jobs/waves", json=WAVES).json()
    assert body["job_id"].startswith("wav_") and body["kind"] == "waves"
    done = _wait(client, body["status_url"])
    assert done["status"] == "succeeded"
    assert calls[0]["wfo"] == "okx" and calls[0]["latest_cycle"] is not None
    item = pystac.Item.from_dict(done["result"])
    assert item.properties["forcingkit:schema"] == "nwps-waves-v1"
    assert item.properties["forcingkit:donor"] == "nwps"
    assert client.get(f"/api/v1/stores/{item.id}.zarr/.zattrs").status_code == 200

    monkeypatch.setattr(jobs, "registry", jobs.JobRegistry(workers=1))
    again = client.get(f"/api/v1/jobs/{item.id}").json()
    assert again["status"] == "succeeded"
    assert again["result"]["properties"]["forcingkit:donor"] == "nwps"


def test_waves_out_of_reach_is_404(client, monkeypatch):
    def out_of_reach(*args, **kwargs):
        raise LookupError("no NWPS cycle holds 2026-06-01T00:00Z")

    monkeypatch.setattr(dispatcher, "dispatch_waves_request", out_of_reach)
    response = client.post("/api/v1/waves", json=WAVES)
    assert response.status_code == 404 and "no NWPS cycle" in response.json()["detail"]


def test_store_files_are_served(client, tmp_path):
    _write_store(str(tmp_path / "obc_0123456789ab.zarr"))
    base = "/api/v1/stores/obc_0123456789ab.zarr"
    meta = client.get(f"{base}/.zmetadata")
    assert meta.status_code == 200 and "metadata" in meta.json()
    head = client.head(f"{base}/.zmetadata")
    assert head.status_code == 200 and int(head.headers["content-length"]) > 0
    assert client.get(f"{base}/u/.zarray").status_code == 200
    assert client.get(f"{base}/absent").status_code == 404


def test_store_guards(client, tmp_path):
    _write_store(str(tmp_path / "obc_0123456789ab.zarr"), complete=False)
    (tmp_path / "secret.txt").write_text("x")
    assert (
        client.get("/api/v1/stores/obc_0123456789ab.zarr/.zmetadata").status_code == 404
    ), "an incomplete store must not be served"
    assert client.get("/api/v1/stores/elsewhere.zarr/.zmetadata").status_code == 400
    traversal = client.get("/api/v1/stores/obc_0123456789ab.zarr/..%2Fsecret.txt")
    assert traversal.status_code in (400, 404)
