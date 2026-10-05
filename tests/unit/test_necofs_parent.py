"""NECOFS parent-ocean delivery (schema z-v2): the vertical conversion, archive-file selection and
cache-key behaviour, on constructed data. No network.

The conversion matters because the previous store labelled FVCOM sigma layers with fake depths in
inverted order (surface labelled -50 m), so a test here pins the surface to the top."""

import numpy as np
import pandas as pd
import pytest

from forcingkit.dispatcher import _supported_kwargs
from forcingkit.fetchers.necofs import (
    OBC_SCHEMA,
    necofs_archive_file_date,
    sigma_to_z,
    z_levels_for_depth,
)


def _column(h=20.0, zeta=0.5, nsig=10, ny=2, nx=3):
    """Uniform sigma layers, surface first, on an (ny, nx) patch of identical columns."""
    siglay = -(np.arange(nsig) + 0.5) / nsig  # -0.05 ... -0.95, surface first
    z_layers = siglay[:, None, None] * (h + zeta) + zeta
    z_layers = np.broadcast_to(z_layers, (nsig, ny, nx)).copy()
    z_bottom = np.full((ny, nx), -h)
    return z_layers, z_bottom


def test_z_levels_cover_the_depth_bottom_to_top():
    centres, faces = z_levels_for_depth(29.4, 2.0)
    assert faces[0] == pytest.approx(
        -30.0
    )  # bottom face at or below the deepest column
    assert faces[-1] == 0.0
    assert np.all(np.diff(faces) > 0)  # ordered bottom to top
    assert len(centres) == len(faces) - 1
    assert centres[-1] == pytest.approx(-1.0)


def test_z_levels_reject_non_positive_spacing():
    with pytest.raises(ValueError):
        z_levels_for_depth(10.0, 0.0)


def test_linear_profile_is_reproduced_on_z():
    z_layers, z_bottom = _column()
    values = 10.0 + 0.1 * z_layers  # linear in z, warmer at the surface
    targets = np.array([-1.0, -5.0, -10.0, -15.0])
    out = sigma_to_z(values, z_layers, z_bottom, targets)
    assert out.shape == (4, 2, 3)
    np.testing.assert_allclose(out[:, 0, 0], 10.0 + 0.1 * targets, atol=1e-12)


def test_surface_is_on_top():
    # A profile that is fast at the surface and slow at the bed: the z output must keep the fast
    # water at the top, the opposite of the old inverted labelling.
    z_layers, z_bottom = _column()
    nsig = z_layers.shape[0]
    speed = np.linspace(0.2, 0.05, nsig)[:, None, None] * np.ones_like(z_layers)
    out = sigma_to_z(speed, z_layers, z_bottom, np.array([-18.0, -1.0]))
    assert out[1, 0, 0] > out[0, 0, 0]


def test_levels_outside_the_layers_take_the_nearest_layer_and_below_the_bed_are_nan():
    z_layers, z_bottom = _column(h=20.0, zeta=0.0)
    values = np.arange(z_layers.shape[0], dtype=float)[:, None, None] * np.ones_like(
        z_layers
    )
    out = sigma_to_z(values, z_layers, z_bottom, np.array([-0.2, -19.9, -25.0]))
    assert out[0, 0, 0] == pytest.approx(
        0.0
    )  # above the top centre: surface layer value
    assert out[1, 0, 0] == pytest.approx(
        values[-1, 0, 0]
    )  # above the bed: bottom layer value
    assert np.isnan(out[2, 0, 0])  # below the sea floor


def test_land_columns_are_nan_never_zero():
    z_layers, z_bottom = _column()
    z_bottom[0, 1] = np.nan
    out = sigma_to_z(np.ones_like(z_layers), z_layers, z_bottom, np.array([-5.0]))
    assert np.isnan(out[0, 0, 1])
    assert out[0, 0, 0] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "t, expected",
    [
        (
            "2026-04-02 00:00",
            "2026-04-02",
        ),  # midnight is the last record of the same-day file
        ("2026-04-02 01:00", "2026-04-03"),
        ("2026-04-02 23:00", "2026-04-03"),
        ("2026-04-03 00:00", "2026-04-03"),
    ],
)
def test_archive_file_holding_each_hour(t, expected):
    assert necofs_archive_file_date(pd.Timestamp(t)) == pd.Timestamp(expected)


def test_options_reach_only_fetchers_that_accept_them():
    def legacy(start, duration, bbox):
        return None

    def v2(start, duration, bbox, pad_cells=3, vertical_spacing_m=2.0):
        return None

    opts = {"pad_cells": 5, "vertical_spacing_m": 1.0}
    assert _supported_kwargs(legacy, **opts) == {}
    assert _supported_kwargs(v2, **opts) == opts


def test_schema_tag_is_set():
    assert OBC_SCHEMA == "z-v2"


def test_barycentric_weights_match_linear_nd_interpolation():
    # The precomputed weights must reproduce scipy's LinearNDInterpolator, which they replace,
    # including NaN outside the hull and leading (layer) dimensions.
    from scipy.interpolate import LinearNDInterpolator
    from scipy.spatial import Delaunay

    from forcingkit.fetchers.necofs import Barycentric

    rng = np.random.default_rng(0)
    pts = rng.uniform(0.0, 1.0, (200, 2))
    tri = Delaunay(pts)
    gx, gy = np.meshgrid(np.linspace(-0.1, 1.1, 13), np.linspace(-0.1, 1.1, 11))
    targets = np.column_stack((gx.ravel(), gy.ravel()))
    interp = Barycentric(tri, targets, gx.shape)

    values = rng.normal(size=(3, 200))
    got = interp(values)
    assert got.shape == (3,) + gx.shape
    for k in range(3):
        want = LinearNDInterpolator(tri, values[k])(targets).reshape(gx.shape)
        np.testing.assert_allclose(got[k], want, rtol=1e-12, atol=1e-12, equal_nan=True)
    assert np.isnan(got[0, 0, 0])  # (-0.1, -0.1) is outside the hull
