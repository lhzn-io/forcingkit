"""HRRR lower atmosphere on native levels (schema hrrr-levels-v1): omega to w, the records'
window, static level geometry, earth-relative winds per level, fail-loud fetching and the cache
key. No network."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from forcingkit.dispatcher import levels_key
from forcingkit.fetchers import hrrr_atmosphere as ha
from forcingkit.fetchers import hrrr_levels as hl

LOV = 262.5
CONE = float(np.sin(np.deg2rad(38.5)))
BBOX = [-74.0, 40.0, -73.5, 40.3]


def test_omega_to_w():
    # Sinking air (omega > 0) moves down: at 1000 hPa and 288 K, rho = 1.2097 kg/m3, so
    # 1 Pa/s is -0.0843 m/s.
    w = hl.omega_to_w(1.0, 288.0, 100000.0)
    assert w == pytest.approx(-1.0 / (100000.0 / (hl.R_DRY * 288.0) * hl.G))
    assert w == pytest.approx(-0.0843, abs=1e-4)


def test_patterns_and_key():
    assert hl.level_pattern("UGRD", 3) == ":UGRD:3 hybrid level:1 hour fcst:"
    assert hl.s3_key(pd.Timestamp("2026-04-01T23:00")).endswith(
        "hrrr.20260401/conus/hrrr.t23z.wrfnatf01.grib2"
    )
    a = levels_key(BBOX, "2026-04-02T00:00:00Z", 48, 14, 0.03, 0.25)
    assert a.startswith("lev_")
    assert a != levels_key(BBOX, "2026-04-02T00:00:00Z", 48, 20, 0.03, 0.25)
    assert a != levels_key(BBOX, "2026-04-02T00:00:00Z", 24, 14, 0.03, 0.25)


def _fake_hour(t, n_levels, names, surface, *, grid_relative=0, hour_offset=0.0):
    """Every (name, level) on a small curvilinear grid: a constant set by the name and level,
    plus hour_offset for the hourly fields."""
    j, i = np.meshgrid(np.arange(60), np.arange(60), indexing="ij")
    lon = 285.0 + 0.05 * i
    lat = 39.0 + 0.05 * j
    base = {
        "UGRD": 3.0,
        "VGRD": 4.0,
        "TMP": 280.0,
        "VVEL": 1.0,
        "HGT": 0.0,
        "PRES": 1e5,
    }
    out = {}
    keys = [(n, k) for n in names for k in range(1, n_levels + 1)]
    if surface:
        keys.append(("HGT", 0))
    for name, k in keys:
        if name == "HGT":
            value = 2.0 if k == 0 else 40.0 * k
        elif name == "PRES":
            value = 1e5 - 500.0 * k
        else:
            value = base[name] + (0.0 if name == "VVEL" else hour_offset) + 0.1 * k
        out[(name, k)] = xr.DataArray(
            np.full(lon.shape, value),
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
                "GRIB_uvRelativeToGrid": grid_relative,
            },
        )
    return out


def test_records_static_geometry_and_w(monkeypatch):
    calls = []

    def fake(t, workdir, n_levels, names, surface=False, max_workers=8):
        calls.append((t, tuple(names), surface))
        return _fake_hour(t, n_levels, names, surface, hour_offset=(t.hour % 24) * 0.01)

    monkeypatch.setattr(hl, "fetch_hour", fake)
    items = list(hl.iter_levels("2026-04-02T00:00:00Z", 2, BBOX, n_levels=3))
    assert items[0][0] == "static"
    static = items[0][1]
    assert static.attrs["schema"] == hl.HRRR_LEVELS_SCHEMA
    np.testing.assert_allclose(static["z"].values[:, 0, 0], [40.0, 80.0, 120.0])
    np.testing.assert_allclose(static["zs"].values, 2.0)
    np.testing.assert_allclose(static["p"].values[:, 0, 0], [99500.0, 99000.0, 98500.0])
    # Geometry only in the first hour's request, and the ground with it.
    assert calls[0][2] and set(hl.STATIC) <= set(calls[0][1])
    assert all(not s and not set(hl.STATIC) & set(n) for _, n, s in calls[1:])

    records = items[1:]
    times = [r[1] for r in records]
    assert times[0] == pd.Timestamp("2026-04-01T23:00")
    assert times[-1] == pd.Timestamp("2026-04-02T03:00")
    assert len(records) == 2 + 3
    for t, (_, _, f) in zip(times, records):
        assert set(f) == set(hl.RECORD_DIMS)
        assert f["u"].shape == (3,) + static["z"].shape[1:]
        np.testing.assert_allclose(f["u"][1], 3.2 + (t.hour % 24) * 0.01)
        np.testing.assert_allclose(f["t"][2], 280.3 + (t.hour % 24) * 0.01)
        # w from this hour's temperature and the first hour's pressure.
        np.testing.assert_allclose(
            f["w"][0], hl.omega_to_w(1.1, 280.1 + (t.hour % 24) * 0.01, 99500.0)
        )


def test_winds_are_rotated_per_level(monkeypatch):
    def fake(t, workdir, n_levels, names, surface=False, max_workers=8):
        return _fake_hour(t, n_levels, names, surface, grid_relative=1)

    monkeypatch.setattr(hl, "fetch_hour", fake)
    items = list(hl.iter_levels("2026-04-02T00:00:00Z", 1, BBOX, n_levels=2))
    _, _, f = items[1]
    lon = items[0][1]["lon"].values[3]
    ue, ve = ha.earth_relative_winds(
        np.array([3.2]), np.array([4.2]), np.array([lon]), CONE, LOV
    )
    np.testing.assert_allclose(f["u"][1, :, 3], ue[0], atol=1e-5)
    np.testing.assert_allclose(f["v"][1, :, 3], ve[0], atol=1e-5)
    # A rotation keeps the speed.
    np.testing.assert_allclose(np.hypot(f["u"][0], f["v"][0]), np.hypot(3.1, 4.1))


def test_a_missing_hour_raises(monkeypatch):
    def fake(t, workdir, n_levels, names, surface=False, max_workers=8):
        if t == pd.Timestamp("2026-04-02T01:00"):
            raise FileNotFoundError("HRRR index missing")
        return _fake_hour(t, n_levels, names, surface)

    monkeypatch.setattr(hl, "fetch_hour", fake)
    with pytest.raises(FileNotFoundError):
        list(hl.iter_levels("2026-04-02T00:00:00Z", 3, BBOX, n_levels=2))


def test_level_count_is_checked():
    with pytest.raises(ValueError):
        list(hl.iter_levels("2026-04-02T00:00:00Z", 1, BBOX, n_levels=0))
    with pytest.raises(ValueError):
        list(hl.iter_levels("2026-04-02T00:00:00Z", 1, BBOX, n_levels=51))
