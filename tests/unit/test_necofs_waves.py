"""necofs.read_waves on a synthetic wave file shaped like NECOFS_WAVE_FORECAST.nc (no network)."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from forcingkit.fetchers import necofs

# Six nodes: four inside the box (a square of two triangles), two outside to the east.
LON = (
    np.array([-72.0, -71.9, -72.0, -71.9, -71.0, -70.9]) + 360.0
)  # the file may store 0..360
LAT = np.array([41.0, 41.0, 41.1, 41.1, 41.0, 41.1])
NV = np.array(
    [[1, 2, 3], [2, 4, 3], [2, 5, 4], [5, 6, 4]]
).T  # one-based, (three, nele)
BOX = [-72.05, 40.95, -71.85, 41.15]


def wave_file(n_times=30, start_mjd=61365.0, zero_coords=False, itime=True):
    t = (start_mjd + np.arange(n_times) / 24.0).astype(
        np.float32
    )  # float32, as in the file: minutes off
    hs = (np.arange(n_times)[:, None] * 10 + np.arange(6)[None, :]).astype(np.float32)
    hours = np.arange(n_times)
    itimes = (
        {
            "Itime": (("time",), int(start_mjd) + hours // 24),
            "Itime2": (("time",), (hours % 24) * 3_600_000),
        }
        if itime
        else {}
    )
    return xr.Dataset(
        {
            **itimes,
            "lon": (("node",), np.zeros(6) if zero_coords else LON),
            "lat": (("node",), np.zeros(6) if zero_coords else LAT),
            "nv": (("three", "nele"), NV),
            "hs": (("time", "node"), hs, {"units": "meters s-1"}),
            "wdir": (("time", "node"), hs + 100),
            "tpeak": (("time", "node"), hs / 10),
        },
        coords={"time": ("time", t, {"units": "days since 1858-11-17 00:00:00"})},
        attrs={"title": "NECOFS WW3 Global Forecast UPDATE"},
    )


def test_the_box_keeps_triangles_wholly_inside_and_their_nodes():
    out = necofs._subset_waves(wave_file(), BOX, "2026-11-21T00:00", 3, ("hs",))
    assert out["node_id"].values.tolist() == [0, 1, 2, 3]
    assert out["triangles"].values.tolist() == [[0, 1, 2], [1, 3, 2]]
    np.testing.assert_allclose(
        out["node_lon"].values, [-72.0, -71.9, -72.0, -71.9], atol=1e-5
    )


def test_hours_are_matched_by_time_and_values_by_node():
    # 2026-11-21 00:00 is MJD 61365; ask from 02:00 for 4 hours.
    out = necofs._subset_waves(
        wave_file(), BOX, pd.Timestamp("2026-11-21T02:00", tz="UTC"), 4, ("hs", "tpeak")
    )
    assert list(out.time.values) == list(
        pd.date_range("2026-11-21T02:00", periods=4, freq="1h").values
    )
    np.testing.assert_array_equal(out["hs"].values[0], [20, 21, 22, 23])
    np.testing.assert_array_equal(out["hs"].values[-1], [50, 51, 52, 53])
    np.testing.assert_allclose(out["tpeak"].values[1], [3.0, 3.1, 3.2, 3.3], rtol=1e-6)


def test_units_and_names_are_corrected():
    out = necofs._subset_waves(wave_file(), BOX, "2026-11-21T00:00", 1, ("hs", "tpeak"))
    assert out["hs"].attrs["units"] == "m"
    assert "mean wave period" in out["tpeak"].attrs["long_name"]
    assert out.attrs["title_in_file"].startswith("NECOFS WW3")


def test_reads_in_pieces_give_the_same_values(monkeypatch):
    whole = necofs._subset_waves(wave_file(), BOX, "2026-11-21T00:00", 30, ("hs",))
    monkeypatch.setattr(necofs, "WAVE_READ_HOURS", 7)
    pieces = necofs._subset_waves(wave_file(), BOX, "2026-11-21T00:00", 30, ("hs",))
    np.testing.assert_array_equal(whole["hs"].values, pieces["hs"].values)


def test_hours_outside_the_file_are_dropped_and_none_raises():
    out = necofs._subset_waves(
        wave_file(n_times=5), BOX, "2026-11-21T03:00", 10, ("hs",)
    )
    assert out.sizes["time"] == 2
    with pytest.raises(RuntimeError, match="none of"):
        necofs._subset_waves(wave_file(n_times=5), BOX, "2026-12-01T00:00", 3, ("hs",))


def test_bad_requests_raise():
    with pytest.raises(ValueError, match="unknown wave variables"):
        necofs._subset_waves(wave_file(), BOX, "2026-11-21T00:00", 1, ("u",))
    with pytest.raises(ValueError, match="all zero"):
        necofs._subset_waves(
            wave_file(zero_coords=True), BOX, "2026-11-21T00:00", 1, ("hs",)
        )
    with pytest.raises(ValueError, match="No NECOFS wave triangles"):
        necofs._subset_waves(
            wave_file(), [-60, 30, -59, 31], "2026-11-21T00:00", 1, ("hs",)
        )


def test_times_come_from_itime_and_the_float_time_is_only_a_rounded_fallback():
    exact = necofs._wave_times(wave_file(n_times=48))
    assert list(exact) == list(pd.date_range("2026-11-21T00:00", periods=48, freq="1h"))
    rough = necofs._wave_times(wave_file(n_times=48, itime=False))
    assert (abs(rough - exact) <= pd.Timedelta(minutes=3)).all()
