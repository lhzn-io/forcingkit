"""HRDPS at points: Datamart URLs, run discovery, the rotated-pole transform, wind rotation and the bitemporal
rows. No network."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from forcingkit.fetchers import hrdps as hd

# The HRDPS continental grid's southern pole of rotation (GRIB2 headers, 2026-11-09).
SP_LAT, SP_LON = -36.08852, 245.305142


def test_urls_follow_the_datamart_nomenclature():
    # ECCC's example file name, under the run's dated directory.
    assert hd.file_url(pd.Timestamp("2020-11-23T00:00"), 12, "GUST", "AGL-10m") == (
        "https://dd.weather.gc.ca/20201123/WXO-DD/model_hrdps/continental/2.5km/00/012/"
        "20201123T00Z_MSC_HRDPS_GUST_AGL-10m_RLatLon0.0225_PT012H.grib2"
    )
    assert hd.field_url("2026-11-09T18:00Z", 0, "v10", base_url=hd.HPFX_URL).endswith(
        "/20261109/WXO-DD/model_hrdps/continental/2.5km/18/000/"
        "20261109T18Z_MSC_HRDPS_VGRD_AGL-10m_RLatLon0.0225_PT000H.grib2"
    )


def test_runs_are_every_six_hours():
    assert hd.is_cycle("2026-11-09T12:00")
    assert not hd.is_cycle("2026-11-09T13:00")
    assert not hd.is_cycle("2026-11-09T12:30")


def test_newest_cycle_is_the_newest_with_every_field_at_the_last_lead():
    published = {hd.field_url("2026-11-09T06:00", 48, f) for f in hd.DEFAULT_FIELDS} | {
        hd.field_url("2026-11-09T12:00", 48, "u10")
    }  # 12Z is still being posted
    got = hd.newest_cycle("2026-11-09T15:30", exists=published.__contains__)
    assert got == pd.Timestamp("2026-11-09T06:00")
    assert hd.newest_cycle("2026-11-09T15:30", exists=lambda url: False) is None


def test_rotated_pole_round_trips_and_matches_the_grid_origin():
    lon = np.array([-73.73, -68.0, -134.0, -60.0])
    lat = np.array([40.88, 43.5, 39.0, 60.0])
    rlon, rlat = hd.geographic_to_rotated(lon, lat, SP_LAT, SP_LON)
    back = hd.rotated_to_geographic(rlon, rlat, SP_LAT, SP_LON)
    np.testing.assert_allclose(back, (lon, lat), atol=1e-9)
    # The grid's first point (rotated 345.17878 E, 12.302501 S): ECCC documents about 39 N, 134 W, and eccodes
    # places it at 39.626032 N, 133.62952 W.
    glon, glat = hd.rotated_to_geographic(345.17878, -12.302501, SP_LAT, SP_LON)
    assert glon == pytest.approx(-133.62952, abs=1e-4)
    assert glat == pytest.approx(39.626032, abs=1e-4)


def test_an_unrotated_pole_is_the_identity():
    lon, lat = np.array([-73.0, 10.0]), np.array([41.0, -20.0])
    np.testing.assert_allclose(
        hd.geographic_to_rotated(lon, lat, -90.0, 0.0), (lon, lat), atol=1e-9
    )
    np.testing.assert_allclose(hd.grid_east_angle(lon, lat, -90.0, 0.0), 0.0, atol=1e-6)


def test_grid_axes_stay_orthogonal_and_rotation_keeps_speed():
    lon, lat = np.array([-73.73, -68.0]), np.array([40.88, 43.5])
    a = hd.grid_east_angle(lon, lat, SP_LAT, SP_LON)
    # The rotated grid's y axis, by the same finite difference, is 90 degrees counterclockwise of its x axis.
    rlon, rlat = hd.geographic_to_rotated(lon, lat, SP_LAT, SP_LON)
    lon1, lat1 = hd.rotated_to_geographic(rlon, rlat, SP_LAT, SP_LON)
    lon2, lat2 = hd.rotated_to_geographic(rlon, rlat + 1e-3, SP_LAT, SP_LON)
    b = np.arctan2(lat2 - lat1, (lon2 - lon1) * np.cos(np.deg2rad(lat1)))
    np.testing.assert_allclose(np.rad2deg(b - a), 90.0, atol=0.01)
    ue, vn = hd.earth_relative_winds(np.array([3.0, -4.0]), np.array([4.0, 1.0]), a)
    np.testing.assert_allclose(np.hypot(ue, vn), [5.0, np.hypot(4.0, 1.0)])


def _grid():
    j, i = np.meshgrid(np.arange(30), np.arange(30), indexing="ij")
    lon = -74.0 + 0.03 * i
    lat = 40.5 + 0.03 * j
    land = (i < 10).astype(float)  # the western third is land
    return lon, lat, land


def _fake_lead(cycle, lead, fields, workdir, client, base_url, sp=(SP_LAT, SP_LON)):
    lon, lat, _ = _grid()
    return {
        name: xr.DataArray(
            np.full(lon.shape, float(n + 1 + lead)),
            dims=("y", "x"),
            coords={"longitude": (("y", "x"), lon), "latitude": (("y", "x"), lat)},
            attrs={
                "GRIB_latitudeOfSouthernPoleInDegrees": sp[0],
                "GRIB_longitudeOfSouthernPoleInDegrees": sp[1],
                "GRIB_angleOfRotationInDegrees": 0.0,
                "GRIB_uvRelativeToGrid": 1,
            },
        )
        for n, name in enumerate(fields)
    }


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setattr(hd, "_fetch_lead", _fake_lead)
    monkeypatch.setattr(
        hd, "_fetch_land", lambda *a: xr.DataArray(_grid()[2], dims=("y", "x"))
    )
    hd._weights_cache.clear()
    hd._angle_cache.clear()


def test_rows_are_bitemporal_and_skip_what_hrdps_does_not_carry(fake):
    df = hd.fetch_points(
        ["2026-11-09T12:00", "2026-11-09T13:00"],  # 13Z is not a run
        [0, 1, 48, 49],  # 49 is past the last lead
        {"buoy": (-73.4, 40.8)},
        fields=("gust",),
    )
    assert list(df.issue_time.unique()) == [pd.Timestamp("2026-11-09T12:00")]
    assert sorted(df.lead_h.unique()) == [0, 1, 48]
    row = df[df.lead_h == 48].iloc[0]
    assert row.valid_time == pd.Timestamp("2026-11-11T12:00")
    assert row.value == pytest.approx(1 + 48)


def test_an_unpublished_lead_is_skipped(fake, monkeypatch):
    monkeypatch.setattr(
        hd,
        "_fetch_lead",
        lambda c, lead, *a: None if lead == 2 else _fake_lead(c, lead, *a),
    )
    df = hd.fetch_points(
        ["2026-11-09T12:00"], [1, 2], {"buoy": (-73.4, 40.8)}, fields=("gust",)
    )
    assert list(df.lead_h.unique()) == [1]


def test_grid_relative_winds_are_rotated_to_east_and_north(fake):
    point = (-74.0 + 0.03 * 20, 40.5 + 0.03 * 12)  # on a water cell, so its weight is 1
    df = hd.fetch_points(
        ["2026-11-09T12:00"], [1], {"buoy": point}, fields=("u10", "v10")
    )
    u = df[df.field == "u10"].value.iloc[0]
    v = df[df.field == "v10"].value.iloc[0]
    a = hd.grid_east_angle(np.array([point[0]]), np.array([point[1]]), SP_LAT, SP_LON)[
        0
    ]
    # The fake gives grid u = 1 + 1 and grid v = 2 + 1 (field order plus lead).
    assert u == pytest.approx(np.cos(a) * 2.0 - np.sin(a) * 3.0, abs=1e-5)
    assert v == pytest.approx(np.sin(a) * 2.0 + np.cos(a) * 3.0, abs=1e-5)
    # Near Long Island Sound the grid's x axis points about 32 degrees south of east.
    assert np.rad2deg(a) == pytest.approx(-32.3, abs=0.1)


def test_one_wind_component_alone_and_unknown_fields_are_refused():
    with pytest.raises(ValueError):
        hd.fetch_points(
            ["2026-11-09T12:00"], [1], {"b": (-73.4, 40.8)}, fields=("u10",)
        )
    with pytest.raises(ValueError):
        hd.fetch_points(
            ["2026-11-09T12:00"], [1], {"b": (-73.4, 40.8)}, fields=("cape",)
        )
