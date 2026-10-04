"""Streaming Zarr delivery: records appended one hour at a time, published only when complete,
and never served while partial. No network."""

import types

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr

from ecodata_cache.dispatcher import _stream_parent
from ecodata_cache.zarr_stream import (
    PARTIAL_SUFFIX,
    StreamingZarrWriter,
    store_is_complete,
)

SCHEMA = "test-v1"
DIMS = {"temp": ("z", "lat", "lon"), "zeta": ("lat", "lon")}


def _static(nz=2, ny=3, nx=4):
    return xr.Dataset(
        {"h": (("lat", "lon"), np.full((ny, nx), 10.0, dtype=np.float32))},
        coords={
            "z": np.linspace(-9.0, -1.0, nz),
            "lat": np.linspace(40.0, 40.1, ny),
            "lon": np.linspace(-74.0, -73.9, nx),
        },
    )


def _record(n, nz=2, ny=3, nx=4):
    return {
        "temp": np.full((nz, ny, nx), 5.0 + n),
        "zeta": np.full((ny, nx), 0.1 * n),
    }


def _writer(path, expected=3):
    return StreamingZarrWriter(
        str(path), _static(), DIMS, expected_records=expected, attrs={"schema": SCHEMA}
    )


T0 = pd.Timestamp("2026-04-02T00:00")


def test_complete_store_is_published_with_one_record_per_chunk(tmp_path):
    path = tmp_path / "obc_x.zarr"
    w = _writer(path)
    for n in range(3):
        w.append(T0 + pd.Timedelta(hours=n), _record(n))
        # Nothing is visible at the final path until close.
        assert not path.exists()
        assert not store_is_complete(str(path), (SCHEMA,))
    w.close()
    assert store_is_complete(str(path), (SCHEMA,))
    assert not (tmp_path / ("obc_x.zarr" + PARTIAL_SUFFIX)).exists()

    ds = xr.open_zarr(str(path), decode_times=False)
    assert ds["temp"].shape == (3, 2, 3, 4)
    assert ds["temp"].dtype == np.float32
    assert zarr.open_group(str(path), mode="r", zarr_format=2)["temp"].chunks == (
        1,
        2,
        3,
        4,
    )
    np.testing.assert_array_equal(ds["time"].values, [0.0, 3600.0, 7200.0])
    assert ds["time"].attrs["units"].startswith("seconds since 2026-04-02")
    assert ds.attrs["schema"] == SCHEMA
    np.testing.assert_allclose(ds["temp"].values[2, 0, 0, 0], 7.0)
    assert ds["h"].shape == (3, 4)
    assert ds.attrs["complete"] is True and ds.attrs["records"] == 3


def test_short_record_raises_and_leaves_nothing(tmp_path):
    path = tmp_path / "obc_short.zarr"
    w = _writer(path, expected=3)
    w.append(T0, _record(0))
    with pytest.raises(RuntimeError, match="1 records written, 3 expected"):
        w.close()
    assert not path.exists()
    assert not (tmp_path / ("obc_short.zarr" + PARTIAL_SUFFIX)).exists()


def test_gap_in_hours_raises(tmp_path):
    path = tmp_path / "obc_gap.zarr"
    w = _writer(path, expected=2)
    w.append(T0, _record(0))
    w.append(T0 + pd.Timedelta(hours=2), _record(1))
    with pytest.raises(RuntimeError, match="not hourly"):
        w.close()
    assert not path.exists()


def test_records_must_advance(tmp_path):
    w = _writer(tmp_path / "obc_order.zarr")
    w.append(T0, _record(0))
    with pytest.raises(ValueError, match="not after"):
        w.append(T0, _record(1))


def test_store_of_the_schema_without_complete_is_not_served(tmp_path):
    path = tmp_path / "obc_old.zarr"
    _static().assign_attrs(schema=SCHEMA).to_zarr(str(path), zarr_format=2)
    assert not store_is_complete(str(path), (SCHEMA,))
    # A store of another schema (written in one shot) is served as before.
    other = tmp_path / "obc_legacy.zarr"
    _static().assign_attrs(schema="legacy").to_zarr(str(other), zarr_format=2)
    assert store_is_complete(str(other), (SCHEMA,))
    assert not store_is_complete(str(tmp_path / "missing.zarr"), (SCHEMA,))


def _fake_donor(hours_available):
    """A donor module whose iter_parent yields `hours_available` records."""

    def iter_parent(start_date, duration_hours, bbox, pad_cells, vertical_spacing_m):
        yield ("static", _static().assign_attrs(schema=SCHEMA))
        for n in range(min(duration_hours, hours_available)):
            yield (
                "record",
                pd.Timestamp(start_date) + pd.Timedelta(hours=n),
                _record(n),
            )

    return types.SimpleNamespace(
        __name__="fake_donor",
        iter_parent=iter_parent,
        PARENT_RECORD_DIMS=DIMS,
        PARENT_UNITS={"temp": ("degree_Celsius", "sea_water_temperature")},
    )


def test_stream_parent_publishes_a_complete_store(tmp_path):
    path = str(tmp_path / "obc_stream.zarr")
    out = _stream_parent(
        _fake_donor(5),
        path,
        "2026-04-02T00:00:00",
        4,
        [0, 0, 1, 1],
        pad_cells=3,
        vertical_spacing_m=2.0,
        sponge_cells=0,
    )
    assert out == path
    assert store_is_complete(path, (SCHEMA,))
    ds = xr.open_zarr(path, decode_times=False)
    assert ds.sizes["time"] == 4
    assert ds.attrs["source"] == "fake_donor"
    assert ds["temp"].attrs["units"] == "degree_Celsius"


def test_stream_parent_fails_on_a_short_donor(tmp_path):
    path = str(tmp_path / "obc_stream_short.zarr")
    with pytest.raises(RuntimeError, match="records written"):
        _stream_parent(
            _fake_donor(2),
            path,
            "2026-04-02T00:00:00",
            4,
            [0, 0, 1, 1],
            pad_cells=3,
            vertical_spacing_m=2.0,
            sponge_cells=0,
        )
    assert not store_is_complete(path, (SCHEMA,))
