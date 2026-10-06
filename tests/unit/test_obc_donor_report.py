"""The /api/v1/obc response names the donor that delivered, not only the predicted one."""

import numpy as np
import xarray as xr
from fastapi.testclient import TestClient

from forcingkit import dispatcher
from forcingkit_serve import main

# Inside the NYOFS grid, so NYOFS is the predicted donor.
UPPER_BAY = {"min_lon": -74.10, "min_lat": 40.60, "max_lon": -74.00, "max_lat": 40.70}


def _write_store(path: str, source: str) -> None:
    ds = xr.Dataset({"u": (("time",), np.zeros(2, dtype=np.float32))})
    ds.attrs["source"] = source
    ds.to_zarr(path, mode="w", consolidated=True, zarr_format=2)


def test_delivered_obc_donor_reads_source(tmp_path):
    path = str(tmp_path / "obc_x.zarr")
    _write_store(path, "forcingkit.fetchers.necofs")
    assert dispatcher.delivered_obc_donor(path) == "necofs"


def test_delivered_obc_donor_missing_store(tmp_path):
    assert dispatcher.delivered_obc_donor(str(tmp_path / "absent.zarr")) is None


def test_obc_response_reports_fallback_donor(tmp_path, monkeypatch):
    monkeypatch.setattr(main.settings, "cache_dir", lambda: str(tmp_path))

    def fake_dispatch(**kwargs):
        # NYOFS was predicted but NECOFS delivered after a fallback.
        _write_store(kwargs["zarr_path"], "forcingkit.fetchers.necofs")
        return kwargs["zarr_path"]

    monkeypatch.setattr(dispatcher, "dispatch_obc_request", fake_dispatch)

    response = TestClient(main.app).post(
        "/api/v1/obc",
        json={
            "start_date": "2023-12-15T06:00:00Z",
            "duration_hours": 3,
            "bbox": UPPER_BAY,
            "cache_bust": True,
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["predicted_donor"] == "nyofs"
    assert body["donor"] == "necofs"
