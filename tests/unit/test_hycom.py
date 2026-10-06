import logging

import xarray as xr
import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch

from forcingkit.dispatcher import _resample_hourly
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


def _three_hourly(ds, var):
    """Whether `var` has data exactly at the 3-hourly steps of `ds` and NaN in between."""
    on_step = ds.indexes["time"].hour % 3 == 0
    has_data = ds[var].notnull().any([d for d in ds[var].dims if d != "time"]).values
    return bool((has_data == on_step).all())


def test_espc_merges_per_variable_datasets():
    """ESPC-D-V02 fetches each variable separately; zeta keeps its hourly steps and the
    3-hourly fields are NaN between theirs."""
    with patch("forcingkit.fetchers.hycom._fetch_hycom_data") as mock_fetch:
        mock_fetch.side_effect = _espc("2025-03-01", 48)
        ds = fetch_hycom_boundary_conditions("2025-03-01", 48, BBOX)

    assert [c.args[3] for c in mock_fetch.call_args_list] == [
        f"{TDS}/ESPC-D-V02/{v}" for v in ESPC_VARS
    ]
    assert set(ds.data_vars) == {"u", "v", "temp", "salt", "zeta"}
    assert ds.indexes["time"].equals(
        pd.date_range("2025-03-01", "2025-03-03", freq="h")
    )
    assert not np.isnan(ds["zeta"].values).any()
    for var in ("u", "v", "temp", "salt"):
        assert _three_hourly(ds, var)
    assert ds.attrs["tides"] == "included"


def test_espc_trims_to_shared_span():
    """Hourly zeta past the last 3-hourly step is dropped, so no field ends on a step it has
    no data for."""
    parts = _espc("2025-03-01", 48)
    parts[-1] = _ds("2025-02-28 23:00", 51, freq="h", var="zeta", depth=False)
    with patch("forcingkit.fetchers.hycom._fetch_hycom_data") as mock_fetch:
        mock_fetch.side_effect = parts
        ds = fetch_hycom_boundary_conditions("2025-03-01", 48, BBOX)
    assert ds.indexes["time"][0] == pd.Timestamp("2025-03-01")
    assert ds.indexes["time"][-1] == pd.Timestamp("2025-03-03")


def test_glb_window_is_not_tidal(caplog):
    with (
        patch("forcingkit.fetchers.hycom._fetch_hycom_data") as mock_fetch,
        caplog.at_level(logging.WARNING, logger="forcingkit.fetchers.hycom"),
    ):
        mock_fetch.side_effect = [_glb("2024-09-01", 9)]
        ds = fetch_hycom_boundary_conditions("2024-09-01", 24, BBOX)
    assert ds.attrs["tides"] == "none"
    assert not caplog.records


def test_stitch_across_espc_switch(caplog):
    """A window across 2024-09-05 stitches GLBy0.08/expt_93.0 with merged ESPC-D-V02, is
    marked as mixed and logs a warning."""
    start_date = "2024-09-04"
    with (
        patch("forcingkit.fetchers.hycom._fetch_hycom_data") as mock_fetch,
        caplog.at_level(logging.WARNING, logger="forcingkit.fetchers.hycom"),
    ):
        mock_fetch.side_effect = [_glb(start_date, 9), *_espc("2024-09-05", 24)]
        ds = fetch_hycom_boundary_conditions(start_date, 48, BBOX)

    assert mock_fetch.call_count == 6
    assert set(ds.data_vars) == {"u", "v", "temp", "salt", "zeta"}
    # 3-hourly before the switch, hourly after it.
    expected = pd.date_range(start_date, "2024-09-05", freq="3h").append(
        pd.date_range("2024-09-05 01:00", "2024-09-06", freq="h")
    )
    assert ds.indexes["time"].equals(expected)
    assert not np.isnan(ds["zeta"].values).any()
    assert _three_hourly(ds, "u")
    assert ds.attrs["tides"] == "mixed"
    [record] = caplog.records
    assert record.levelno == logging.WARNING
    assert "2024-09-05" in record.getMessage() and "tidal" in record.getMessage()


def test_resample_keeps_hourly_zeta():
    """The dispatcher's hourly resample keeps hourly zeta at its native values and
    interpolates the 3-hourly fields between their own steps. Treating the gaps as data
    would leave the 3-hourly fields NaN at two hours in three."""
    hours = np.arange(25)
    time = pd.date_range("2025-03-01", periods=hours.size, freq="h")
    # An M2 tide, which 3-hourly sampling and linear interpolation would flatten.
    tide = 0.43 * np.sin(2 * np.pi * hours / 12.42)
    u = np.where(hours % 3 == 0, hours.astype(float), np.nan)
    land = np.full(hours.size, np.nan)
    ds = xr.Dataset(
        {
            "zeta": (("time", "lat"), np.stack([tide, land], axis=1)),
            "u": (("time", "lat"), np.stack([u, land], axis=1)),
            "h": (("lat",), [10.0, 0.0]),
        },
        coords={"time": time, "lat": [40.0, 40.1]},
        attrs={"tides": "included"},
    )
    out = _resample_hourly(ds)

    assert out.indexes["time"].equals(time)
    np.testing.assert_allclose(out["zeta"].values[:, 0], tide)
    np.testing.assert_allclose(out["u"].values[:, 0], hours)
    assert np.isnan(out["u"].values[:, 1]).all()
    assert out["h"].values.tolist() == [10.0, 0.0]
    assert out.attrs["tides"] == "included"


def test_resample_regular_input_unchanged():
    """A dataset with one cadence resamples as `resample().interpolate()` does."""
    ds = _glb("2025-03-01", 9)
    ds["u"][:] = np.arange(9.0)[:, None, None, None]
    out = _resample_hourly(ds)
    ref = ds.resample(time="1h").interpolate("linear")
    xr.testing.assert_allclose(out, ref)


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


def test_fetch_rounds_times_to_the_second():
    """Float hours that decode a few hundred nanoseconds off the hour (as ESPC-D-V02 `ssh`
    does) come back on whole seconds, so they align with the other fields in the merge."""
    t0 = (pd.Timestamp("2025-01-01 12:00") - pd.Timestamp("2000-01-01")).total_seconds()
    time = t0 / 3600 + np.arange(4.0) - np.array([0.0, 2e-13, 0.0, -2e-13]) * 3600
    raw = _raw(np.arange(0.0, 360.0), time=time)
    with patch("forcingkit.fetchers.hycom.xr.open_dataset", return_value=raw):
        ds = _fetch_hycom_data(
            pd.Timestamp("2025-01-01 12:00"),
            pd.Timestamp("2025-01-01 15:00"),
            BBOX,
            f"{TDS}/ESPC-D-V02/ssh",
        )
    expected = pd.date_range("2025-01-01 12:00", periods=4, freq="h")
    assert ds.indexes["time"].equals(expected)


def _fetch_2025(open_dataset):
    """`_fetch_hycom_data` for 2025-01-01 12:00 to 18:00 with `xr.open_dataset` and
    `time.sleep` patched; returns (result, open_dataset mock, sleep mock)."""
    with (
        patch("forcingkit.fetchers.hycom.xr.open_dataset", **open_dataset) as mock_open,
        patch("forcingkit.fetchers.hycom.time.sleep") as mock_sleep,
    ):
        ds = _fetch_hycom_data(
            pd.Timestamp("2025-01-01 12:00"),
            pd.Timestamp("2025-01-01 18:00"),
            BBOX,
            f"{TDS}/ESPC-D-V02/u3z",
        )
    return ds, mock_open, mock_sleep


def test_fetch_retries_after_failed_read():
    """A read that times out is repeated once, after a pause, and the retry's data is used."""
    raw = _raw(np.arange(0.0, 360.0))
    ds, mock_open, mock_sleep = _fetch_2025(
        {"side_effect": [TimeoutError("read"), raw]}
    )
    assert ds is not None and ds.sizes["time"] == 3
    assert mock_open.call_count == 2
    mock_sleep.assert_called_once()


def test_fetch_gives_up_after_second_failure():
    ds, mock_open, _ = _fetch_2025({"side_effect": TimeoutError("read")})
    assert ds is None
    assert mock_open.call_count == 2


def test_fetch_does_not_retry_out_of_range():
    """A window outside the dataset's time axis is not a transient failure."""
    raw = _raw(np.arange(0.0, 360.0), time=[100.0, 103.0])
    ds, mock_open, mock_sleep = _fetch_2025({"return_value": raw})
    assert ds is None
    assert mock_open.call_count == 1
    mock_sleep.assert_not_called()
