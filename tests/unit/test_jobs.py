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


NECOFS_FIELDS = ("u", "v", "w", "temp", "salt", "zeta")


def _write_store(
    path: str,
    complete: bool = True,
    source: str = "forcingkit.fetchers.necofs",
    fields: tuple[str, ...] = NECOFS_FIELDS,
    attrs: dict | None = None,
):
    lon = np.linspace(-70.95, -70.55, 5)
    lat = np.linspace(41.25, 41.65, 4)
    time_ = pd.date_range("2026-06-01", periods=7, freq="1h")
    ds = xr.Dataset(
        {
            name: (
                ("time", "z", "lat", "lon"),
                np.zeros((7, 2, 4, 5), dtype=np.float32),
            )
            for name in fields
        },
        coords={"time": time_, "z": [-3.0, -1.0], "lat": lat, "lon": lon},
        attrs={
            "schema": "z-v3",
            "complete": complete,
            "source": source,
            **(attrs or {}),
        },
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
    assert item.properties["forcingkit:w_source"] == "model"
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


def test_cached_necofs_store_without_w_is_rebuilt(client, monkeypatch):
    calls = []

    def fake_dispatch(**kwargs):
        calls.append(kwargs)
        _write_store(kwargs["zarr_path"])
        return kwargs["zarr_path"]

    monkeypatch.setattr(dispatcher, "dispatch_obc_request", fake_dispatch)
    zarr_id, zarr_path, _ = main._obc_store(main.OBCRequest(**OBC))
    _write_store(zarr_path, fields=("u", "v", "temp", "salt", "zeta"))
    body = client.post("/api/v1/jobs/obc", json=OBC).json()
    assert body["job_id"] == zarr_id
    assert _wait(client, body["status_url"])["status"] == "succeeded"
    assert len(calls) == 1


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


def test_registry_rebuilds_succeeded_job_on_request_and_forgets_finished_ones():
    registry = jobs.JobRegistry(workers=1)
    release = threading.Event()
    first = registry.submit("obc_0123456789ab", "ocean", lambda: {"n": 1})
    deadline = time.monotonic() + 5
    while first.status != "succeeded" and time.monotonic() < deadline:
        time.sleep(0.01)
    assert first.status == "succeeded"
    assert registry.submit("obc_0123456789ab", "ocean", lambda: {"n": 2}) is first

    def slow_build():
        release.wait(5)
        return {"n": 3}

    rebuilt = registry.submit("obc_0123456789ab", "ocean", slow_build, rebuild=True)
    assert rebuilt is not first

    # An active job survives forget; a finished one does not.
    registry.forget("obc_0123456789ab")
    registry.forget()
    assert registry.get("obc_0123456789ab") is rebuilt
    release.set()
    while rebuilt.status != "succeeded" and time.monotonic() < deadline:
        time.sleep(0.01)
    registry.forget("obc_0123456789ab")
    assert registry.get("obc_0123456789ab") is None


def _built_obc_job(client, monkeypatch, calls):
    def fake_dispatch(**kwargs):
        calls.append(kwargs)
        _write_store(kwargs["zarr_path"])
        return kwargs["zarr_path"]

    monkeypatch.setattr(dispatcher, "dispatch_obc_request", fake_dispatch)
    body = client.post("/api/v1/jobs/obc", json=OBC).json()
    assert _wait(client, body["status_url"])["status"] == "succeeded"
    return body


def test_succeeded_job_whose_store_vanished_is_rebuilt(client, monkeypatch):
    import shutil

    calls = []
    body = _built_obc_job(client, monkeypatch, calls)
    _, zarr_path, _ = main._obc_store(main.OBCRequest(**OBC))
    shutil.rmtree(zarr_path)  # removed behind the service's back, registry untouched

    assert client.get(body["status_url"]).status_code == 404
    again = client.post("/api/v1/jobs/obc", json=OBC)
    assert again.status_code == 202
    done = _wait(client, again.json()["status_url"])
    assert done["status"] == "succeeded" and len(calls) == 2
    assert done["result"]["id"] == body["job_id"]


def test_deleting_a_store_drops_its_job(client, monkeypatch):
    calls = []
    body = _built_obc_job(client, monkeypatch, calls)
    deleted = client.delete(f"/api/v1/cache/dataset/{body['job_id']}")
    assert deleted.status_code == 200
    assert jobs.registry.get(body["job_id"]) is None

    again = client.post("/api/v1/jobs/obc", json=OBC)
    assert again.status_code == 202
    assert _wait(client, again.json()["status_url"])["status"] == "succeeded"
    assert len(calls) == 2


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


@pytest.mark.parametrize("prefix", ["atm_", "lev_"])
def test_atmosphere_download_serves_surface_and_levels(client, tmp_path, prefix):
    import io
    import zipfile

    zarr_id = prefix + "0123456789ab"
    _write_store(str(tmp_path / f"{zarr_id}.zarr"))
    response = client.get(f"/api/v1/atmosphere/download/{zarr_id}")
    assert response.status_code == 200
    assert f'filename="{zarr_id}.zip"' in response.headers["content-disposition"]
    assert ".zattrs" in zipfile.ZipFile(io.BytesIO(response.content)).namelist()
    if prefix == "atm_":  # a bare hash still means the surface store
        assert client.get("/api/v1/atmosphere/download/0123456789ab").status_code == 200


def test_atmosphere_download_refuses_other_ids(client):
    for bad in (
        "obc_0123456789ab",
        "lev_0123",
        "..%2Fatm_0123456789ab",
        "wav_0123456789ab",
    ):
        assert client.get(f"/api/v1/atmosphere/download/{bad}").status_code in (
            400,
            404,
        )
    assert client.get("/api/v1/atmosphere/download/obc_0123456789ab").status_code == 400
    assert client.get("/api/v1/atmosphere/download/lev_000000000000").status_code == 404


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


def test_stac_item_names_where_w_comes_from(tmp_path):
    from forcingkit_serve.stac import store_item

    def w_source(name, **kwargs):
        path = str(tmp_path / f"{name}.zarr")
        _write_store(path, **kwargs)
        props = store_item(name, path, "http://svc", "ocean", "x")["properties"]
        return props.get("forcingkit:w_source")

    hycom = "forcingkit.fetchers.hycom"
    assert w_source("a", attrs={"w_source": "model"}) == "model"
    assert w_source("b", source=hycom, attrs={"w_source": "continuity"}) == "continuity"
    # A NECOFS store from before the attribute already holds the model's w.
    assert w_source("c") == "model"
    assert w_source("d", fields=("u", "v"), attrs={"w_source": "none"}) == "none"
    # No w and no attribute (an atmosphere or wave store): no property.
    assert w_source("e", source="elsewhere", fields=("u10",)) is None


def test_a_named_donor_gets_its_own_store_and_is_passed_on(client, monkeypatch):
    calls = []

    def fake_dispatch(**kwargs):
        calls.append(kwargs)
        _write_store(kwargs["zarr_path"], source="forcingkit.fetchers.doppio")
        return kwargs["zarr_path"]

    monkeypatch.setattr(dispatcher, "dispatch_obc_request", fake_dispatch)
    ranked = client.post("/api/v1/obc/cache", json=OBC).json()["zarr_id"]
    named = client.post("/api/v1/obc/cache", json={**OBC, "donor": "doppio"}).json()
    necofs = client.post("/api/v1/obc/cache", json={**OBC, "donor": "necofs"}).json()
    assert len({ranked, named["zarr_id"], necofs["zarr_id"]}) == 3

    body = client.post("/api/v1/jobs/obc", json={**OBC, "donor": "doppio"}).json()
    assert body["job_id"] == named["zarr_id"]
    done = _wait(client, body["status_url"])
    assert done["status"] == "succeeded" and calls[0]["donor"] == "doppio"
    props = done["result"]["properties"]
    assert props["forcingkit:donor"] == "doppio"
    assert props["forcingkit:w_source"] == "model"
    predicted = client.post(
        "/api/v1/obc/predict-donor", json={**OBC, "donor": "doppio"}
    ).json()
    assert predicted["donor"]["id"] == "doppio"


@pytest.mark.parametrize(
    "donor, bbox, message",
    [
        ("roms", BBOX, "Unknown parent-ocean donor 'roms'"),
        (
            "nyofs",
            {"min_lon": -70.0, "min_lat": 41.0, "max_lon": -69.0, "max_lat": 42.0},
            "outside",
        ),
    ],
)
def test_a_bad_donor_is_refused_before_queueing(
    client, monkeypatch, donor, bbox, message
):
    def never(**kwargs):
        raise AssertionError("dispatched a refused request")

    monkeypatch.setattr(dispatcher, "dispatch_obc_request", never)
    request = {**OBC, "bbox": bbox, "donor": donor}
    for url in ("/api/v1/jobs/obc", "/api/v1/obc", "/api/v1/obc/cache"):
        response = client.post(url, json=request)
        assert response.status_code == 400, url
        assert message in response.json()["detail"]


# ---- the volume job: a model's own mesh, every sigma layer -------------------------------------

VOLUME = {"bbox": BBOX, "start_time": "2026-06-01T00:00:00", "hours": 3}


def _volume_dataset(hours: int = 3, layers: int = 4) -> xr.Dataset:
    """A small dataset in the volume-v1 layout: 3 nodes, 1 element, `layers` sigma layers."""
    time_ = pd.date_range("2026-06-01", periods=hours, freq="1h")
    siglay = np.repeat(-(np.arange(layers) + 0.5)[:, None] / layers, 3, axis=1)
    return xr.Dataset(
        {
            "u": (
                ("time", "siglay", "element"),
                np.full((hours, layers, 1), 0.3, np.float32),
            ),
            "v": (
                ("time", "siglay", "element"),
                np.full((hours, layers, 1), -0.1, np.float32),
            ),
            "zeta": (("time", "node"), np.zeros((hours, 3), np.float32)),
            "h": (("node",), np.array([10.0, 12.0, 11.0], np.float32)),
            "siglay": (("siglay", "node"), siglay.astype(np.float32)),
            "node_lon": (("node",), np.array([-70.9, -70.7, -70.8], np.float32)),
            "node_lat": (("node",), np.array([41.3, 41.3, 41.5], np.float32)),
            "triangles": (("element", "vertex"), np.array([[0, 1, 2]], np.int32)),
        },
        coords={"time": time_},
        attrs={"schema": "volume-v1", "source": "UMass Dartmouth SMAST"},
    )


@pytest.mark.parametrize(
    "change, message",
    [
        ({"source": "fvcom-other"}, "unsupported source"),
        ({"hours": 0}, "hours must be 1 to"),
        ({"hours": 241}, "hours must be 1 to"),
        ({"start_time": "2026-06-01T00:30:00"}, "on the hour"),
        ({"variables": ["u", "speed"]}, "variables must be among"),
        ({"layers": [-1]}, "negative layers"),
    ],
)
def test_volume_rejects_bad_requests(client, change, message):
    response = client.post("/api/v1/jobs/volume", json={**VOLUME, **change})
    assert response.status_code == 400 and message in response.json()["detail"]


def test_volume_job_returns_stac_item_and_survives_restart(client, monkeypatch):
    from forcingkit.fetchers import necofs

    calls = []

    def fake_volume(
        bbox, start, end, cache=True, variables=None, layers=None, **kwargs
    ):
        calls.append(
            {
                "start": start,
                "end": end,
                "cache": cache,
                "variables": variables,
                "layers": layers,
            }
        )
        return _volume_dataset()

    monkeypatch.setattr(necofs, "volume", fake_volume)
    body = client.post("/api/v1/jobs/volume", json=VOLUME).json()
    assert body["job_id"].startswith("vol_") and body["kind"] == "volume"
    done = _wait(client, body["status_url"])
    assert done["status"] == "succeeded", done
    call = calls[0]
    assert (
        call["variables"] == ("u", "v", "zeta")
        and call["layers"] is None
        and call["cache"] is False
    )
    assert call["start"] == pd.Timestamp("2026-06-01T00:00") and call[
        "end"
    ] == pd.Timestamp("2026-06-01T02:00")
    item = pystac.Item.from_dict(done["result"])
    assert item.properties["forcingkit:schema"] == "volume-v1"
    assert item.properties["forcingkit:donor"] == "necofs"
    assert item.bbox == pytest.approx(
        [-70.9, 41.3, -70.7, 41.5], abs=1e-5
    )  # from the mesh's nodes
    assert client.get(f"/api/v1/stores/{item.id}.zarr/.zattrs").status_code == 200
    store = xr.open_zarr(
        main.settings.cache_dir(f"{item.id}.zarr"), consolidated=True, zarr_format=2
    )
    assert (
        store.attrs["complete"] is True and store["u"].encoding["chunks"][0] == 1
    )  # one hour a chunk

    monkeypatch.setattr(jobs, "registry", jobs.JobRegistry(workers=1))
    again = client.get(f"/api/v1/jobs/{item.id}").json()
    assert again["status"] == "succeeded"
    assert again["result"]["properties"]["forcingkit:donor"] == "necofs"
    assert len(calls) == 1


def test_volume_selection_is_part_of_the_store_id(client, monkeypatch):
    from forcingkit.fetchers import necofs

    monkeypatch.setattr(necofs, "volume", lambda *a, **k: _volume_dataset())
    bodies = [
        client.post("/api/v1/jobs/volume", json={**VOLUME, **extra}).json()
        for extra in ({}, {"layers": [0, 1]}, {"variables": ["u", "v"]}, {"hours": 4})
    ]
    assert len({b["job_id"] for b in bodies}) == 4
    for b in bodies:  # let each build finish inside the test
        assert _wait(client, b["status_url"])["status"] == "succeeded"


def test_volume_key_carries_the_day_only_near_the_forecast():
    key = dispatcher.volume_key
    sel = ("u", "v", "zeta")
    box = [-70.9, 41.3, -70.6, 41.6]
    old = "2026-06-01T00:00:00"
    assert key(box, old, 24, sel, None, now="2026-11-01T06:00") == key(
        box, old, 24, sel, None, now="2026-11-02T06:00"
    )  # an archived window: one store whenever asked
    recent = "2026-11-01T00:00:00"
    assert key(box, recent, 48, sel, None, now="2026-11-01T06:00") != key(
        box, recent, 48, sel, None, now="2026-11-02T06:00"
    )  # may read the forecast: rebuilt on a later day
    assert key(box, recent, 48, sel, None, now="2026-11-01T06:00") == key(
        box, recent, 48, sel, None, now="2026-11-01T22:00"
    )
