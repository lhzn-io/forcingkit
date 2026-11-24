"""Vertical velocity from continuity, for parent donors that do not provide w.

For an incompressible (Boussinesq) ocean, du/dx + dv/dy + dw/dz = 0. With w = 0 at the sea
floor, w at height z is minus the horizontal divergence integrated from the bed up to z:

    w(z) = -integral from the bed to z of (du/dx + dv/dy) dz'

on a regular longitude-latitude grid, with the spherical divergence
(1 / (R cos(lat))) * (du/dlon + d(v cos(lat))/dlat), angles in radians. The result is only as
good as the horizontal velocities and the bed it starts from: at 1/12 degree it is noisy and
dominated by the gradients the grid resolves, and an error in the deepest wet level carries
into every level above it. A column whose currents do not conserve volume (fields remapped from
another vertical coordinate, as HYCOM's are) reaches the surface with a large w that the free
surface cannot take up; `continuity_w` can remove that residual (O'Brien 1970) so w also meets
d(zeta)/dt at the top. Stores that hold it say so in ``w_source = "continuity"``.

O'Brien, J. J. (1970). Alternative solutions to the classical vertical velocity problem.
Journal of Applied Meteorology, 9(2), 197-203.
"""

from typing import Optional

import numpy as np

EARTH_RADIUS_M = 6_371_000.0


def _derivative(f: np.ndarray, coord: np.ndarray, axis: int) -> np.ndarray:
    """df/dcoord along `axis`, NaN-aware.

    Centered where both neighbors are finite, one-sided where only one is (a coast or the edge of
    the grid), and 0 where neither is (a single wet cell across that axis, whose gradient the grid
    cannot resolve). NaN where `f` itself is NaN.
    """
    f = np.moveaxis(np.asarray(f, dtype=np.float64), axis, -1)
    c = np.asarray(coord, dtype=np.float64)
    pad = [(0, 0)] * (f.ndim - 1) + [(1, 1)]
    fp = np.pad(f, pad, constant_values=np.nan)
    cp = np.pad(c, (1, 1), constant_values=np.nan)
    f_lo, f_hi = fp[..., :-2], fp[..., 2:]
    c_lo, c_hi = cp[:-2], cp[2:]
    has_lo, has_hi = np.isfinite(f_lo), np.isfinite(f_hi)
    with np.errstate(invalid="ignore", divide="ignore"):
        centered = (f_hi - f_lo) / (c_hi - c_lo)
        forward = (f_hi - f) / (c_hi - c)
        backward = (f - f_lo) / (c - c_lo)
    out = np.where(
        has_lo & has_hi,
        centered,
        np.where(has_hi, forward, np.where(has_lo, backward, 0.0)),
    )
    out[~np.isfinite(f)] = np.nan
    return np.moveaxis(out, -1, axis)


def horizontal_divergence(
    u: np.ndarray, v: np.ndarray, lon: np.ndarray, lat: np.ndarray
) -> np.ndarray:
    """du/dx + dv/dy (s-1) of fields shaped (..., lat, lon) on 1-D axes in degrees."""
    lam = np.deg2rad(np.asarray(lon, dtype=np.float64))
    phi = np.deg2rad(np.asarray(lat, dtype=np.float64))
    cos_phi = np.cos(phi)[:, None]
    du = _derivative(u, lam, axis=-1)
    dv = _derivative(np.asarray(v, dtype=np.float64) * cos_phi, phi, axis=-2)
    return (du + dv) / (EARTH_RADIUS_M * cos_phi)


def faces_from_levels(z: np.ndarray, surface: float = 0.0) -> np.ndarray:
    """Level faces (len(z) + 1, bottom to top) for point levels `z` (m, positive up, bottom to
    top), as for HYCOM's fixed depths: midway between neighboring levels, the top face at
    `surface` (or the top level, if that is higher) and the bottom face as far below the
    deepest level as the face above it is above."""
    z = np.asarray(z, dtype=np.float64)
    if z.size < 2:
        raise ValueError("continuity needs at least two levels")
    mid = 0.5 * (z[:-1] + z[1:])
    bottom = z[0] - (mid[0] - z[0])
    return np.concatenate([[bottom], mid, [max(z[-1], surface)]])


def continuity_w(
    u: np.ndarray,
    v: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    z: np.ndarray,
    z_face: np.ndarray,
    surface_w: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Upward velocity (m s-1) at the levels `z` from continuity, shaped like `u`.

    u, v: (..., z, lat, lon), eastward and northward velocity, NaN on land and below the bed.
    lon, lat: 1-D axes in degrees. z: (nz,) level heights, positive up, bottom to top; level k
    spans z_face[k] to z_face[k + 1] (nz + 1 faces). A cell is wet where both u and v are finite.
    The integral starts at the lower face of each column's deepest wet level, where w = 0, and
    w is NaN wherever the cell is not wet.

    surface_w: (..., lat, lon), the upward velocity the top of each column must have, usually
    d(zeta)/dt (NaN is taken as 0). When given, the column's residual (the integrated w at the
    upper face of its shallowest wet level minus `surface_w`) is removed in proportion to height
    above the bed (O'Brien 1970), which is the same as subtracting one constant from the
    divergence of every wet level in the column. w then meets both the bed (0) and the surface.
    Without it, w at the top carries whatever net divergence the column has.
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    z_face = np.asarray(z_face, dtype=np.float64)
    if z_face.shape != (z.size + 1,):
        raise ValueError(f"need {z.size + 1} level faces, got {z_face.shape}")
    wet = np.isfinite(u) & np.isfinite(v)
    div = np.where(wet, horizontal_divergence(u, v, lon, lat), 0.0)
    div = np.nan_to_num(div)  # a wet cell next to a NaN v (never expected) adds nothing
    shape = (-1, 1, 1)
    dz = np.diff(z_face).reshape(shape)
    rise = (z - z_face[:-1]).reshape(shape)  # from a level's lower face to the level
    flux = div * dz
    below = np.cumsum(flux, axis=-3) - flux  # through all levels under this one
    w = -(below + div * rise)
    if surface_w is not None:
        nz = z.size
        any_wet = wet.any(axis=-3)
        # The bed is the lower face of the deepest wet level, the top the upper face of the
        # shallowest; the column's own w at the top is minus its whole integrated divergence.
        z_bed = z_face[np.argmax(wet, axis=-3)]
        z_top = z_face[nz - np.argmax(wet[..., ::-1, :, :], axis=-3)]
        residual = -flux.sum(axis=-3) - np.nan_to_num(
            np.broadcast_to(np.asarray(surface_w, dtype=np.float64), any_wet.shape)
        )
        height = np.where(any_wet, z_top - z_bed, 1.0)
        fraction = (z.reshape(shape) - z_bed[..., None, :, :]) / height[..., None, :, :]
        w = w - fraction * residual[..., None, :, :]
    w[~wet] = np.nan
    return w
