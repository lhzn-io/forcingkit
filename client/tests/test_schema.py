import numpy as np
import pytest
import xarray as xr
from conftest import hycom_legacy_store, z_v3_store

from forcingkit_client import SchemaError, normalise


def test_z_v3_passes_through():
    ds = z_v3_store()
    assert normalise(ds) is ds


def test_legacy_hycom_becomes_z_v3_conventions():
    ds = normalise(hycom_legacy_store())
    ref = z_v3_store()
    assert "depth" not in ds.dims and ds["z"].attrs["positive"] == "up"
    np.testing.assert_allclose(ds["z"], ref["z"])
    np.testing.assert_allclose(ds["lon"], ref["lon"])
    np.testing.assert_allclose(ds["u"], ref["u"])


def test_legacy_latitude_longitude_names():
    ds = z_v3_store().rename(lat="latitude", lon="longitude")
    ds.attrs = {"schema": "legacy"}
    assert {"lat", "lon"} <= set(normalise(ds).dims)


def test_legacy_curvilinear_is_refused():
    ds = xr.Dataset(
        {"u": (("y", "x"), np.zeros((2, 2)))},
        coords={
            "lat": (("y", "x"), np.zeros((2, 2))),
            "lon": (("y", "x"), np.zeros((2, 2))),
        },
        attrs={"schema": "legacy"},
    )
    with pytest.raises(SchemaError, match="curvilinear"):
        normalise(ds)


def test_unknown_schema_is_refused():
    with pytest.raises(SchemaError, match="z-v9"):
        normalise(xr.Dataset(attrs={"schema": "z-v9"}))
