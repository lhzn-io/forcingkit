"""Structured fields of the cache inventory: bbox, time span, family and kind."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from httpx import ASGITransport, AsyncClient

sys.path.insert(0, "service")
from forcingkit_serve.routers.viewer import store_footprint  # noqa: E402

TIMES = pd.date_range("2022-08-11", periods=3, freq="h")


def test_regular_grid_in_0_to_360_longitudes():
    ds = xr.Dataset(
        {"u": (("time", "lat", "lon"), np.zeros((3, 2, 2)))},
        coords={"time": TIMES, "lat": [39.5, 40.0], "lon": [290.0, 290.5]},
        attrs={"type": "HYCOM GLBy0.08"},
    )
    fp = store_footprint(ds, Path("obc_0123456789ab.zarr"))
    assert fp["bbox"] == [-70.0, 39.5, -69.5, 40.0]
    assert (fp["time_start"], fp["time_end"]) == (
        "2022-08-11T00:00:00Z",
        "2022-08-11T02:00:00Z",
    )
    assert (fp["family"], fp["kind"]) == ("HYCOM", "ocean boundary")


def test_unstructured_mesh_positions_in_data_variables():
    ds = xr.Dataset(
        {
            "node_lon": ("node", [-69.9, -69.2, -69.5]),
            "node_lat": ("node", [39.6, 39.7, 40.1]),
        },
        coords={"time": TIMES},
        attrs={"type": "NECOFS/FVCOM GOM7 full-depth subset on the model mesh"},
    )
    fp = store_footprint(ds, Path("necofs/volume/necofs_volume_0123456789ab.zarr"))
    assert fp["bbox"] == [-69.9, 39.6, -69.2, 40.1]
    assert (fp["family"], fp["kind"]) == ("NECOFS", "ocean volume")


def test_index_axes_fall_back_to_the_requested_bbox():
    ds = xr.Dataset(
        {"T": (("eta", "xi"), np.zeros((4, 5)))},
        coords={"eta": np.arange(4), "xi": np.arange(5)},
        attrs={
            "requested_bbox": "[-74.6, 40.0, -69.8, 42.3]",
            "target_date": "2026-03-02T15:00:00Z",
        },
    )
    fp = store_footprint(ds, Path("ic_0123456789ab.zarr"))
    assert fp["bbox"] == [-74.6, 40.0, -69.8, 42.3]
    assert fp["time_start"] == fp["time_end"] == "2026-03-02T15:00:00Z"
    assert fp["kind"] == "initial condition"


def test_nothing_known_gives_nones():
    fp = store_footprint(xr.Dataset(), Path("mystery.grib2"))
    assert fp["bbox"] is None and fp["time_start"] is None and fp["family"] is None
    assert fp["kind"] == "raw donor file"


@pytest.mark.asyncio
async def test_inventory_route_returns_the_structured_fields(tmp_path, monkeypatch):
    monkeypatch.setenv("FORCINGKIT_CACHE_DIR", str(tmp_path))
    store = tmp_path / "doppio" / "volume" / "doppio_volume_0123456789ab.zarr"
    xr.Dataset(
        {"temp": (("time", "eta_rho", "xi_rho"), np.zeros((3, 2, 2)))},
        coords={
            "time": TIMES,
            "lon": (("eta_rho", "xi_rho"), [[-69.9, -69.8], [-69.85, -69.75]]),
            "lat": (("eta_rho", "xi_rho"), [[39.7, 39.72], [39.8, 39.82]]),
        },
        attrs={"type": "Doppio ROMS full-depth subset on the model grid"},
    ).to_zarr(store)
    from forcingkit_serve.main import app

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        items = (await ac.get("/api/v1/cache/inventory")).json()
    (item,) = [i for i in items if i["id"] == store.stem]
    assert item["bbox"] == [-69.9, 39.7, -69.75, 39.82]
    assert item["time_end"] == "2022-08-11T02:00:00Z"
    assert (item["family"], item["kind"]) == ("Doppio", "ocean volume")


@pytest.mark.asyncio
async def test_inventory_etag_answers_304_until_the_cache_changes(
    tmp_path, monkeypatch
):
    """A poll that sends back the ETag downloads nothing while the list is unchanged, and the
    full list again once a store appears (here in a subfolder)."""
    monkeypatch.setenv("FORCINGKIT_CACHE_DIR", str(tmp_path))

    def store(path: Path) -> None:
        xr.Dataset({"temp": (("time",), np.zeros(3))}, coords={"time": TIMES}).to_zarr(
            path
        )

    store(tmp_path / "obc_000000000001.zarr")
    from forcingkit_serve.main import app

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        first = await ac.get("/api/v1/cache/inventory")
        etag = first.headers["etag"]
        assert [i["id"] for i in first.json()] == ["obc_000000000001"]
        same = await ac.get("/api/v1/cache/inventory", headers={"If-None-Match": etag})
        assert same.status_code == 304 and same.content == b""
        store(tmp_path / "sub" / "obc_000000000002.zarr")
        changed = await ac.get(
            "/api/v1/cache/inventory", headers={"If-None-Match": etag}
        )
        assert changed.status_code == 200 and changed.headers["etag"] != etag
        assert sorted(i["id"] for i in changed.json()) == [
            "obc_000000000001",
            "obc_000000000002",
        ]
