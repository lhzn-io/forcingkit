import xarray as xr
import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch

from forcingkit.fetchers.hycom import (
    _SEGMENTS,
    _fetch_hycom_data,
    _get_hycom_segment,
    _split_window,
    fetch_hycom_boundary_conditions,
)

TDS = "https://tds.hycom.org/thredds/dodsC"
BBOX = [-74.0, 40.0, -73.0, 41.0]
ESPC_VARS = ("u3z", "v3z", "t3z", "s3z", "ssh")


def _ds(start, periods, freq="3h", lat=None, var="u", depth=True):
    """A subset as `_fetch_hycom_data` returns it: one variable on a small grid."""
    lat = np.linspace(40.0, 41.0, 5) if lat is None else lat
    lon = np.linspace(286.0, 287.0, 5)
    time = pd.date_range(start, periods=periods, freq=freq)
    if depth:
        dims, shape = ("time", "depth", "lat", "lon"), (periods, 2, lat.size, lon.size)
        coords = {"time": time, "depth": [0.0, 2.0], "lat": lat, "lon": lon}
    else:
        dims, shape = ("time", "lat", "lon"), (periods, lat.size, lon.size)
        coords = {"time": time, "lat": lat, "lon": lon}
    return xr.Dataset({var: (dims, np.ones(shape))}, coords=coords)


def _glb(start, periods, lat=None):
    """A GLBv/GLBy subset: all variables in one 3-hourly dataset."""
    return xr.merge(
        [_ds(start, periods, lat=lat, var=v) for v in ("u", "v", "temp", "salt")]
        + [_ds(start, periods, lat=lat, var="zeta", depth=False)]
    )


def _espc(start, hours):
    """ESPC-D-V02 subsets in fetch order: u, v, temp, salt 3-hourly, zeta hourly."""
    parts = [_ds(start, hours // 3 + 1, var=v) for v in ("u", "v", "temp", "salt")]
    parts.append(_ds(start, hours + 1, freq="h", var="zeta", depth=False))
    return parts


@pytest.mark.parametrize(
    "when, name",
    [
        ("1994-01-01", "GLBv0.08/expt_53.X"),
        ("2010-06-01", "GLBv0.08/expt_53.X"),
        ("2016-02-01", "GLBv0.08/expt_56.3"),
        ("2017-12-01", "GLBv0.08/expt_92.9"),
        ("2018-12-04 09:00", "GLBv0.08/expt_93.0"),
        ("2018-12-04 12:00", "GLBy0.08/expt_93.0"),
        ("2024-09-04 21:00", "GLBy0.08/expt_93.0"),
        ("2024-09-05", "ESPC-D-V02"),
        ("2026-10-06", "ESPC-D-V02"),
    ],
)
def test_segment_resolution(when, name):
    assert _get_hycom_segment(pd.Timestamp(when)).name == name


def test_segment_urls():
    """expt_53.X is served under GLBv0.08 (the GLBy0.08 path is empty); ESPC-D-V02 is one
    dataset per variable."""
    assert _get_hycom_segment(pd.Timestamp("2010-01-01")).urls == (
        f"{TDS}/GLBv0.08/expt_53.X",
    )
    assert _get_hycom_segment(pd.Timestamp("2025-01-01")).urls == tuple(
        f"{TDS}/ESPC-D-V02/{v}" for v in ESPC_VARS
    )


def test_segments_ordered():
    starts = [seg.start for seg in _SEGMENTS]
    assert starts == sorted(starts)


def test_before_first_segment():
    assert _get_hycom_segment(pd.Timestamp("1993-12-31")) is None
    assert _split_window(pd.Timestamp("1993-12-31"), pd.Timestamp("1994-01-02")) == []
    with patch("forcingkit.fetchers.hycom._fetch_hycom_data") as mock_fetch:
        assert fetch_hycom_boundary_conditions("1993-12-31", 48, BBOX) is None
        mock_fetch.assert_not_called()


def test_split_window():
    t = pd.Timestamp
    pieces = _split_window(t("2025-03-01"), t("2025-03-03"))
    assert [(s.name, a, b) for s, a, b in pieces] == [
        ("ESPC-D-V02", t("2025-03-01"), t("2025-03-03"))
    ]
    # A window ending exactly on a switch stays in one segment.
    pieces = _split_window(t("2024-09-03"), t("2024-09-05"))
    assert [s.name for s, _, _ in pieces] == ["GLBy0.08/expt_93.0"]
    pieces = _split_window(t("2017-01-30"), t("2017-06-03"))
    assert [(s.name, a, b) for s, a, b in pieces] == [
        ("GLBv0.08/expt_57.2", t("2017-01-30"), t("2017-02-01 12:00")),
        ("GLBv0.08/expt_92.8", t("2017-02-01 12:00"), t("2017-06-01 12:00")),
        ("GLBv0.08/expt_57.7", t("2017-06-01 12:00"), t("2017-06-03")),
    ]


def test_hycom_historical_stitch():
    """Crossing 2018-12-04 12:00 stitches GLBv0.08 and GLBy0.08 expt_93.0 on the first grid."""
    start_date = "2018-12-03"
    glbv = _glb(start_date, 13, lat=np.linspace(40.0, 41.0, 5))
    glby = _glb("2018-12-04 12:00", 5, lat=np.linspace(40.0, 41.0, 11))

    with patch("forcingkit.fetchers.hycom._fetch_hycom_data") as mock_fetch:
        mock_fetch.side_effect = [glbv, glby]
        ds = fetch_hycom_boundary_conditions(start_date, 48, BBOX)

    assert [c.args[3] for c in mock_fetch.call_args_list] == [
        f"{TDS}/GLBv0.08/expt_93.0",
        f"{TDS}/GLBy0.08/expt_93.0",
    ]
    # Both pieces include 2018-12-04 12:00; it appears once in the result.
    assert ds.indexes["time"].equals(pd.date_range(start_date, "2018-12-05", freq="3h"))
    assert ds.sizes["lat"] == 5
    assert not np.isnan(ds["u"].values).any()


def test_espc_merges_per_variable_datasets():
    """ESPC-D-V02 fetches each variable separately and keeps the 3-hourly common times."""
    with patch("forcingkit.fetchers.hycom._fetch_hycom_data") as mock_fetch:
        mock_fetch.side_effect = _espc("2025-03-01", 48)
        ds = fetch_hycom_boundary_conditions("2025-03-01", 48, BBOX)

    assert [c.args[3] for c in mock_fetch.call_args_list] == [
        f"{TDS}/ESPC-D-V02/{v}" for v in ESPC_VARS
    ]
    assert set(ds.data_vars) == {"u", "v", "temp", "salt", "zeta"}
    assert ds.indexes["time"].equals(
        pd.date_range("2025-03-01", "2025-03-03", freq="3h")
    )
    assert not np.isnan(ds["zeta"].values).any()


def test_stitch_across_espc_switch():
    """A window across 2024-09-05 stitches GLBy0.08/expt_93.0 with merged ESPC-D-V02."""
    start_date = "2024-09-04"
    with patch("forcingkit.fetchers.hycom._fetch_hycom_data") as mock_fetch:
        mock_fetch.side_effect = [_glb(start_date, 9), *_espc("2024-09-05", 24)]
        ds = fetch_hycom_boundary_conditions(start_date, 48, BBOX)

    assert mock_fetch.call_count == 6
    assert set(ds.data_vars) == {"u", "v", "temp", "salt", "zeta"}
    assert ds.indexes["time"].equals(pd.date_range(start_date, "2024-09-06", freq="3h"))
    assert not np.isnan(ds["zeta"].values).any()


def test_missing_piece_drops_window():
    """A stitched window with a failed piece returns None rather than a shortened dataset."""
    with patch("forcingkit.fetchers.hycom._fetch_hycom_data") as mock_fetch:
        mock_fetch.side_effect = [_glb("2018-12-03", 9), None]
        assert fetch_hycom_boundary_conditions("2018-12-03", 48, BBOX) is None


def _raw(lon, time=None):
    """A raw HYCOM dataset as opened (undecoded hours since 2000), on a 1 degree grid, with
    the `tau` and `*_bottom` fields the GLB experiments also carry."""
    lat = np.arange(-80.0, 91.0)
    time = np.arange(219180.0, 219190.0, 3.0) if time is None else np.asarray(time)
    data = np.zeros((time.size, lat.size, lon.size), dtype=np.float32)
    return xr.Dataset(
        {
            "water_u": (("time", "lat", "lon"), data),
            "water_u_bottom": (("time", "lat", "lon"), data),
            "tau": (("time",), np.zeros(time.size)),
        },
        coords={"time": time, "lat": lat, "lon": lon},
    )


@pytest.mark.parametrize("lon", [np.arange(0.0, 360.0), np.arange(-180.0, 180.0)])
@pytest.mark.parametrize(
    "bbox, expected",
    [
        ([-74.0, 40.0, -73.0, 41.0], [286.0, 287.0]),
        ([-2.0, 40.0, 2.0, 41.0], [358.0, 359.0, 0.0, 1.0, 2.0]),
    ],
)
def test_fetch_lon_conventions(lon, bbox, expected):
    """Both 0..360 (9x-series, ESPC-D-V02) and -180..180 (5x-series GLBv0.08) datasets are
    sliced in their own convention and returned on 0..360, with latitude always sliced."""
    with patch("forcingkit.fetchers.hycom.xr.open_dataset", return_value=_raw(lon)):
        ds = _fetch_hycom_data(
            pd.Timestamp("2025-01-01 12:00"),
            pd.Timestamp("2025-01-01 18:00"),
            bbox,
            f"{TDS}/GLBy0.08/expt_93.0",
        )
    assert ds["lon"].values.tolist() == expected
    assert ds["lat"].values.tolist() == [40.0, 41.0]
    assert ds.sizes["time"] == 3
    # Only the canonical fields are downloaded; `tau` and `*_bottom` are dropped.
    assert set(ds.data_vars) == {"u"}


def test_fetch_non_monotonic_time():
    """GLBv0.08/expt_93.0 steps back once (2018-06-21 09:00 to 06:00); the subset comes back
    strictly increasing, without the duplicate, and limited to the requested range."""
    t0 = (
        pd.Timestamp("2018-06-21") - pd.Timestamp("2000-01-01")
    ).total_seconds() / 3600
    time = t0 + np.array([0.0, 3.0, 6.0, 9.0, 6.0, 12.0, 15.0, 18.0])
    raw = _raw(np.arange(0.0, 360.0), time=time)
    with patch("forcingkit.fetchers.hycom.xr.open_dataset", return_value=raw):
        ds = _fetch_hycom_data(
            pd.Timestamp("2018-06-21 03:00"),
            pd.Timestamp("2018-06-21 15:00"),
            BBOX,
            f"{TDS}/GLBv0.08/expt_93.0",
        )
    expected = pd.date_range("2018-06-21 03:00", "2018-06-21 15:00", freq="3h")
    assert ds.indexes["time"].equals(expected)
