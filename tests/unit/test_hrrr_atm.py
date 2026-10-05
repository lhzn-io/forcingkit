"""HRRR prescribed atmosphere (schema hrrr-atm-v1): wind rotation, cycle chaining, centred
precipitation, regridding, fail-loud fetching and the cache key. No network."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pyproj import Proj

from forcingkit.dispatcher import atmosphere_key
from forcingkit.fetchers import hrrr_atmosphere as ha

CONE = float(np.sin(np.deg2rad(38.5)))
LOV = 262.5


def test_rotation_is_identity_on_the_central_meridian():
    u, v = ha.earth_relative_winds(
        np.array([3.0]), np.array([4.0]), np.array([-97.5]), CONE, LOV
    )
    np.testing.assert_allclose([u[0], v[0]], [3.0, 4.0], atol=1e-12)


@pytest.mark.parametrize("lon", [-122.0, -97.5, -80.0, -73.9, -70.6])
def test_rotation_matches_the_projection(lon):
    # An eastward wind of 1 m/s has grid components along the projected direction of east;
    # rotating them back must give (1, 0). The direction comes from pyproj, independently.
    lat = 40.0
    p = Proj(proj="lcc", lat_1=38.5, lat_2=38.5, lat_0=38.5, lon_0=-97.5, R=6371229)
    x0, y0 = p(lon, lat)
    x1, y1 = p(lon + 1e-4, lat)
    theta = np.arctan2(y1 - y0, x1 - x0)
    u, v = ha.earth_relative_winds(
        np.array([np.cos(theta)]),
        np.array([np.sin(theta)]),
        np.array([lon + 360.0]),
        CONE,
        LOV,
    )
    np.testing.assert_allclose([u[0], v[0]], [1.0, 0.0], atol=1e-6)


def test_rotation_angle_off_new_york():
    # About 14.7 degrees at 73.9 W.
    alpha = np.rad2deg(CONE * np.deg2rad(-73.9 + 97.5))
    assert alpha == pytest.approx(14.69, abs=0.01)


def test_cycle_chaining_crosses_midnight():
    t = pd.Timestamp("2026-04-02T00:00")
    assert ha.cycle_for_valid_time(t) == pd.Timestamp("2026-04-01T23:00")
    assert ha.s3_key(ha.cycle_for_valid_time(t)).endswith(
        "hrrr.20260401/conus/hrrr.t23z.wrfsfcf01.grib2"
    )


def test_target_axes_cover_the_bbox_with_margin():
    lon, lat = ha.target_axes([-74.0, 40.0, -73.0, 41.0], 0.03, 0.25)
    assert lon[0] == pytest.approx(-74.25) and lon[-1] >= -72.75 - 1e-9
    assert lat[0] == pytest.approx(39.75) and lat[-1] >= 41.25 - 1e-9
    np.testing.assert_allclose(np.diff(lon), 0.03)


def test_regridding_a_linear_field_is_exact():
    # A curvilinear native grid in 0..360 longitudes, like HRRR's.
    j, i = np.meshgrid(np.arange(40), np.arange(50), indexing="ij")
    native_lon = 285.5 + 0.03 * i + 0.004 * j
    native_lat = 39.5 + 0.03 * j - 0.003 * i
    lon_axis = np.arange(-74.0, -73.4, 0.05)
    lat_axis = np.arange(39.7, 40.3, 0.05)
    r = ha.HRRRRegridder(native_lon, native_lat, lon_axis, lat_axis)
    field = 2.0 + 3.0 * (native_lon - 360.0) - 5.0 * native_lat
    out = r(field)
    gx, gy = np.meshgrid(lon_axis, lat_axis)
    np.testing.assert_allclose(out, 2.0 + 3.0 * gx - 5.0 * gy, atol=1e-9)


def _fake_native(t, value):
    """The eight fields at `t` on a small curvilinear grid, each a constant `value + offset`."""
    j, i = np.meshgrid(np.arange(60), np.arange(60), indexing="ij")
    lon = 285.0 + 0.05 * i
    lat = 39.0 + 0.05 * j
    out = {}
    for n, name in enumerate(ha.ATM_IDX_PATTERNS):
        da = xr.DataArray(
            np.full(lon.shape, value + n, dtype=np.float64),
            dims=("y", "x"),
            coords={
                "longitude": (("y", "x"), lon),
                "latitude": (("y", "x"), lat),
                "valid_time": np.datetime64(t),
            },
            attrs={
                "GRIB_gridType": "lambert",
                "GRIB_Latin1InDegrees": 38.5,
                "GRIB_Latin2InDegrees": 38.5,
                "GRIB_LoVInDegrees": LOV,
                "GRIB_uvRelativeToGrid": 0,
            },
        )
        out[name] = da
    return out


def test_records_bracket_the_run_and_precipitation_is_centred(monkeypatch):
    # apcp at hour t is (hours since 2026-04-01T22) + 5, so the centred rate at t is the mean
    # of the accumulations at t and t + 1 h, divided by 3600 s.
    t_base = pd.Timestamp("2026-04-01T22:00")

    def fake_fetch(t, workdir):
        h = (t - t_base) / pd.Timedelta(hours=1)
        f = _fake_native(t, 0.0)
        f["apcp"] = f["apcp"] * 0 + h + 5.0
        return f

    monkeypatch.setattr(ha, "fetch_hour", fake_fetch)
    items = list(
        ha.iter_atmosphere("2026-04-02T00:00:00Z", 2, [-74.0, 40.0, -73.5, 40.3])
    )
    assert items[0][0] == "static"
    records = items[1:]
    times = [r[1] for r in records]
    # From start - 1 h to end + 1 h: hours + 3 records.
    assert times[0] == pd.Timestamp("2026-04-01T23:00")
    assert times[-1] == pd.Timestamp("2026-04-02T03:00")
    assert len(records) == 2 + 3
    for t, (_, _, fields) in zip(times, records):
        h = (t - t_base) / pd.Timedelta(hours=1)
        np.testing.assert_allclose(fields["prate"], ((h + 5) + (h + 6)) / 7200.0)
        assert set(fields) == set(ha.RECORD_DIMS)
    assert items[0][1].attrs["schema"] == ha.HRRR_ATM_SCHEMA


def test_a_missing_hour_raises(monkeypatch):
    def fake_fetch(t, workdir):
        if t == pd.Timestamp("2026-04-02T01:00"):
            raise FileNotFoundError("HRRR index missing")
        return _fake_native(t, 1.0)

    monkeypatch.setattr(ha, "fetch_hour", fake_fetch)
    with pytest.raises(FileNotFoundError):
        list(ha.iter_atmosphere("2026-04-02T00:00:00Z", 3, [-74.0, 40.0, -73.5, 40.3]))


def test_the_key_carries_the_schema_and_the_request():
    a = atmosphere_key([-74, 40, -73, 41], "2026-04-02T00:00:00Z", 48, 0.03, 0.25)
    assert a.startswith("atm_")
    assert a != atmosphere_key(
        [-74, 40, -73, 41], "2026-04-02T00:00:00Z", 168, 0.03, 0.25
    )
    assert a != atmosphere_key(
        [-74, 40, -73, 41], "2026-04-02T00:00:00Z", 48, 0.05, 0.25
    )
