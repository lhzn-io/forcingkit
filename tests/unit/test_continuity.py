"""Vertical velocity from continuity (forcingkit.continuity) and its use for HYCOM."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from forcingkit.continuity import (
    EARTH_RADIUS_M,
    continuity_w,
    faces_from_levels,
    horizontal_divergence,
)
from forcingkit.fetchers.hycom import add_continuity_w

LON = np.linspace(-71.0, -70.0, 13)
LAT = np.linspace(40.0, 41.0, 11)
FACES = np.arange(-10.0, 0.1, 2.0)  # flat bed at -10 m, five 2 m levels
Z = 0.5 * (FACES[:-1] + FACES[1:])
A = 1e-5  # s-1


def _stretch(a: float = A) -> np.ndarray:
    """u = a * x, x eastward distance from LON[0]: a uniform divergence `a`. (z, lat, lon)."""
    x = (
        EARTH_RADIUS_M
        * np.cos(np.deg2rad(LAT))[:, None]
        * np.deg2rad(LON - LON[0])[None, :]
    )
    return np.broadcast_to(a * x, (Z.size, LAT.size, LON.size)).copy()


def test_uniform_divergence_gives_linear_w_from_the_bed():
    u = _stretch()
    w = continuity_w(u, np.zeros_like(u), LON, LAT, Z, FACES)
    expected = -A * (Z - FACES[0])  # w = 0 at the bed, -a * height above it
    np.testing.assert_allclose(
        w, np.broadcast_to(expected[:, None, None], w.shape), rtol=1e-9
    )


def test_divergence_includes_the_meridional_metric():
    """v = c / cos(lat) carries the same transport across every latitude: no divergence."""
    v = np.broadcast_to(
        0.3 / np.cos(np.deg2rad(LAT))[:, None], (LAT.size, LON.size)
    ).copy()
    div = horizontal_divergence(np.zeros_like(v), v, LON, LAT)
    np.testing.assert_allclose(div, 0.0, atol=1e-15)


def test_leading_dimensions_pass_through():
    u = np.stack([_stretch(A), _stretch(2 * A)])
    w = continuity_w(u, np.zeros_like(u), LON, LAT, Z, FACES)
    assert w.shape == u.shape
    np.testing.assert_allclose(w[1], 2 * w[0], rtol=1e-9)


def test_land_and_levels_below_the_bed_are_nan_and_the_bed_is_where_w_is_zero():
    u = _stretch()
    v = np.zeros_like(u)
    u[:, 4, 6] = v[:, 4, 6] = np.nan  # a land column
    u[:2, 7, 3] = np.nan  # a shallower column: the bed is at FACES[2] = -6 m
    w = continuity_w(u, v, LON, LAT, Z, FACES)
    assert np.isnan(w[:, 4, 6]).all()
    assert np.isnan(w[:2, 7, 3]).all()
    np.testing.assert_allclose(w[2:, 7, 3], -A * (Z[2:] - FACES[2]), rtol=1e-9)
    # Cells beside the land column use one-sided differences, exact for a linear field.
    np.testing.assert_allclose(w[:, 4, 5], -A * (Z - FACES[0]), rtol=1e-9)
    np.testing.assert_allclose(w[:, 4, 7], -A * (Z - FACES[0]), rtol=1e-9)


def test_a_wet_cell_with_no_wet_neighbor_adds_no_divergence():
    u = np.full((1, 3, 3), np.nan)
    v = np.full((1, 3, 3), np.nan)
    u[0, 1, 1] = v[0, 1, 1] = 0.5
    div = horizontal_divergence(u, v, LON[:3], LAT[:3])
    assert div[0, 1, 1] == 0.0 and np.isnan(div[0, 0, 0])


def test_faces_from_point_levels():
    np.testing.assert_allclose(
        faces_from_levels(np.array([-4.0, -2.0, 0.0])), [-5.0, -3.0, -1.0, 0.0]
    )
    with pytest.raises(ValueError):
        faces_from_levels(np.array([0.0]))
    with pytest.raises(ValueError, match="faces"):
        continuity_w(np.zeros((2, 1, 1)), np.zeros((2, 1, 1)), [0.0], [0.0], Z[:2], Z)


def test_surface_correction_meets_the_target_at_the_top_and_zero_at_the_bed():
    """A uniformly divergent column has a net outflow the surface must supply; removing it in
    proportion to height (O'Brien 1970) leaves w linear from 0 at the bed to the target."""
    u = _stretch()
    v = np.zeros_like(u)
    u[:2, 7, 3] = np.nan  # bed at FACES[2] = -6 m
    u[:, 4, 6] = v[:, 4, 6] = np.nan  # land
    target = np.full((LAT.size, LON.size), 2e-5)
    target[0, 0] = np.nan  # taken as 0
    w = continuity_w(u, v, LON, LAT, Z, FACES, surface_w=target)
    height = Z - FACES[0]
    np.testing.assert_allclose(w[:, 5, 5], 2e-5 * height / 10.0, rtol=1e-9, atol=1e-18)
    np.testing.assert_allclose(w[:, 0, 0], 0.0, atol=1e-15)
    np.testing.assert_allclose(
        w[2:, 7, 3], 2e-5 * (Z[2:] - FACES[2]) / 6.0, rtol=1e-9, atol=1e-18
    )
    assert np.isnan(w[:2, 7, 3]).all() and np.isnan(w[:, 4, 6]).all()


def test_surface_correction_keeps_the_shape_of_a_non_divergent_column():
    """Divergence in the lower half and the opposite convergence in the upper half already
    integrate to zero at the surface; the correction changes nothing."""
    u = _stretch()
    u[3:] *= -1.5  # levels 0-1 (and half of 2) diverge, levels 3-4 converge
    u[2] = 0.0
    bare = continuity_w(u, np.zeros_like(u), LON, LAT, Z, FACES)
    top = -(A * 4.0 - 1.5 * A * 4.0)  # w at the surface face from the bare integral
    target = np.full((LAT.size, LON.size), top)
    corrected = continuity_w(u, np.zeros_like(u), LON, LAT, Z, FACES, surface_w=target)
    np.testing.assert_allclose(corrected, bare, rtol=1e-9, atol=1e-18)


def _hycom_ds(zeta_rate=None):
    """A HYCOM-like subset: depths surface first, positive down, a uniformly divergent u, and
    the bed above 6 m at the first cell. With `zeta_rate`, zeta rises at that rate (m s-1)."""
    depth = np.array([0.0, 2.0, 4.0, 6.0])
    x = (
        EARTH_RADIUS_M
        * np.cos(np.deg2rad(LAT))[:, None]
        * np.deg2rad(LON - LON[0])[None, :]
    )
    u = np.broadcast_to(A * x, (3, depth.size, LAT.size, LON.size)).astype(np.float32)
    u = u.copy()
    u[:, 3, 0, 0] = np.nan  # 6 m below the surface is under the bed here
    time = pd.date_range("2025-07-01", periods=3, freq="3h")
    ds = xr.Dataset(
        {
            "u": (("time", "depth", "lat", "lon"), u),
            "v": (("time", "depth", "lat", "lon"), np.zeros_like(u)),
        },
        coords={"time": time, "depth": depth, "lat": LAT, "lon": LON % 360},
    )
    if zeta_rate is not None:
        seconds = (time - time[0]).total_seconds().values
        ds["zeta"] = (
            ("time", "lat", "lon"),
            np.broadcast_to(
                zeta_rate * seconds[:, None, None], (3, LAT.size, LON.size)
            ),
        )
    return ds


def test_hycom_w_on_depths_positive_down_meets_the_rising_surface():
    """HYCOM's depths run surface first and positive down; w comes back on the same depths,
    from 0 at the bed (half a level below the deepest wet depth) to d(zeta)/dt at the top."""
    rate = 3e-5
    out = add_continuity_w(_hycom_ds(zeta_rate=rate))
    depth = out["depth"].values
    assert out.attrs["w_source"] == "continuity"
    assert out["w"].dims == out["u"].dims and out["w"].dtype == np.float32
    assert out["w"].attrs["standard_name"] == "upward_sea_water_velocity"
    assert "O'Brien" in out["w"].attrs["comment"]
    # Bed at -7 m (faces -7, -5, -3, -1, 0).
    w = out["w"].isel(time=1, lat=5, lon=5).values
    np.testing.assert_allclose(w, rate * (7.0 - depth) / 7.0, rtol=1e-5)
    # Where the bed is shallower the column runs from -5 m.
    col = out["w"].isel(time=0, lat=0, lon=0).values
    assert np.isnan(col[3])
    np.testing.assert_allclose(col[:3], rate * (5.0 - depth[:3]) / 5.0, rtol=1e-5)


def test_hycom_w_without_zeta_closes_at_the_surface():
    out = add_continuity_w(_hycom_ds())
    w = out["w"].values
    np.testing.assert_allclose(w[np.isfinite(w)], 0.0, atol=1e-10)
    assert np.isnan(w[:, 3, 0, 0]).all()
