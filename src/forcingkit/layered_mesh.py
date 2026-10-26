"""Model output on a layered triangle mesh: every layer of a 3D model on its own horizontal mesh.

`LayeredMesh` is the 3D companion of `forcingkit.mesh_frames.MeshFrames` (which stays the
single-layer, surface-current contract): node coordinates, triangle connectivity, a vertical
coordinate, and any number of named fields, each held either on elements (one value per triangle
and layer, drawn flat) or on nodes. It is the one format a viewer needs to draw a model's own
cells, cut sections through them and colour them on a scale shared with other models.

Geometry
--------
- ``node_lon``, ``node_lat`` (n_node,) degrees; ``tri`` (n_elem, 3) zero-based node indices.
- One vertical coordinate, either
  - sigma (terrain-following): ``h`` (n_node,) depth below the datum, positive down;
    ``siglay`` (n_layer, n_node) sigma at layer centres, surface first (0 at the surface, -1 at
    the bed); optional ``siglev`` (n_layer + 1, n_node) at the interfaces and ``zeta``
    (t, n_node) surface elevation. Height of a layer centre: ``siglay * (h + zeta) + zeta``.
  - fixed heights: ``layer_z`` (n_layer, n_elem) height of each layer centre (m, positive up),
    with an optional ``bed_z`` (n_elem,).
  - neither, for a single layer at the surface.

Fields
------
``fields[name]`` is (t, n_layer, n_elem) or (t, n_layer, n_node); ``meta[name]`` is a `FieldMeta`
whose ``kind`` says which. Its ``label, units, lo, hi, log, ticks`` are the entries a viewer's
manifest lists for each field; `shared_scales` fills ``lo`` and ``hi`` over several meshes.
``speed`` is derived from ``u`` and ``v`` when it is not a field of its own.

Adapters
--------
`necofs` reads NECOFS FVCOM GOM7 through `forcingkit.fetchers.necofs.volume` (sigma layers,
element u, v and w, node temperature and salinity). `structured` turns a regular or curvilinear
grid into two triangles per cell carrying the cell's own value, so flat cells are drawn as the
model holds them; it is the basis for the NYOFS, DBOFS, HYCOM and HRRR adapters.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EARTH_RADIUS = 6.371e6  # m, for local metric coordinates
ELEMENT, NODE = "element", "node"


def _utc(times) -> pd.DatetimeIndex:
    times = pd.DatetimeIndex(times)
    return times.tz_localize("UTC") if times.tz is None else times.tz_convert("UTC")


def to_local(lon, lat, lon0: float, lat0: float) -> tuple[np.ndarray, np.ndarray]:
    """Metres east and north of (lon0, lat0), equirectangular about lat0."""
    x = EARTH_RADIUS * math.cos(math.radians(lat0)) * np.deg2rad(np.asarray(lon) - lon0)
    y = EARTH_RADIUS * np.deg2rad(np.asarray(lat) - lat0)
    return x, y


def from_local(x, y, lon0: float, lat0: float) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of `to_local`."""
    lon = lon0 + np.rad2deg(
        np.asarray(x) / (EARTH_RADIUS * math.cos(math.radians(lat0)))
    )
    lat = lat0 + np.rad2deg(np.asarray(y) / EARTH_RADIUS)
    return lon, lat


def nice_ticks(lo: float, hi: float, n: int = 4, log: bool = False) -> list[float]:
    """Round values between lo and hi: about n of them in steps of 1, 2 or 5 times a power of 10,
    or on a log axis the 1, 2, 5 sequence."""
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        return []
    if log:
        if lo <= 0:
            raise ValueError(f"a log axis needs lo > 0, got {lo}")
        out = []
        for p in range(math.floor(math.log10(lo)), math.ceil(math.log10(hi)) + 1):
            out += [m * 10.0**p for m in (1, 2, 5) if lo <= m * 10.0**p <= hi]
        return [float(round(x, 12)) for x in out]
    raw = (hi - lo) / n
    p = 10 ** math.floor(math.log10(raw))
    step = min((m * p for m in (1, 2, 5, 10) if m * p >= raw), default=10 * p)
    t = np.arange(math.ceil(lo / step) * step, hi + 1e-9 * step, step)
    return [float(round(x, 10)) for x in t]


@dataclass
class FieldMeta:
    """What a field is and how to colour it.

    kind: ``"element"`` (one value per triangle and layer) or ``"node"``.
    lo, hi: the colour scale's ends in ``units`` (NaN until set, for example by `shared_scales`).
    log: the scale is logarithmic. ticks: legend values (None: `nice_ticks` of lo and hi).
    symmetric: a signed field whose scale is centred on zero (``-hi`` to ``hi``).
    """

    name: str
    label: str = ""
    units: str = ""
    kind: str = ELEMENT
    lo: float = math.nan
    hi: float = math.nan
    log: bool = False
    ticks: list[float] | None = None
    symmetric: bool = False

    def __post_init__(self) -> None:
        if self.kind not in (ELEMENT, NODE):
            raise ValueError(f"kind must be {ELEMENT!r} or {NODE!r}, got {self.kind!r}")

    def with_range(self, lo: float, hi: float) -> "FieldMeta":
        """A copy on the scale lo to hi, with ticks recomputed."""
        return replace(self, lo=float(lo), hi=float(hi), ticks=None)

    def tick_values(self) -> list[float]:
        if self.ticks is not None:
            return list(self.ticks)
        return nice_ticks(self.lo, self.hi, log=self.log)

    def manifest_entry(self) -> dict:
        """The field's entry in a viewer manifest: name, label, units, lo, hi, log, ticks."""
        return {
            "name": self.name,
            "label": self.label or self.name,
            "units": self.units,
            "lo": self.lo,
            "hi": self.hi,
            "log": self.log,
            "ticks": self.tick_values(),
        }

    def _unit(self, v: np.ndarray) -> np.ndarray:
        if not (np.isfinite(self.lo) and np.isfinite(self.hi)) or self.hi <= self.lo:
            raise ValueError(
                f"{self.name}: no colour scale (lo={self.lo}, hi={self.hi})"
            )
        if self.log:
            v = np.log(np.clip(v, self.lo, self.hi))
            return (v - math.log(self.lo)) / (math.log(self.hi) - math.log(self.lo))
        return np.clip((v - self.lo) / (self.hi - self.lo), 0.0, 1.0)

    def encode(self, values) -> np.ndarray:
        """Values to one byte on the scale: 1 at or below lo, 255 at or above hi, 0 for NaN
        (no water)."""
        v = np.asarray(values, dtype=np.float64)
        with np.errstate(invalid="ignore"):
            f = self._unit(v)
        return np.where(np.isfinite(v), 1 + np.rint(np.nan_to_num(f) * 254), 0).astype(
            np.uint8
        )

    def decode(self, codes) -> np.ndarray:
        """Codes back to values (code 0 to NaN)."""
        c = np.asarray(codes, dtype=np.float64)
        f = (c - 1) / 254
        if self.log:
            v = np.exp(math.log(self.lo) + f * (math.log(self.hi) - math.log(self.lo)))
        else:
            v = self.lo + f * (self.hi - self.lo)
        return np.where(c > 0, v, np.nan)


SPEED_META = FieldMeta("speed", "Current speed", "m/s", ELEMENT, log=True)


@dataclass
class Section:
    """Values along a polyline through a `LayeredMesh`, sampled at n points.

    distance (n,) m from the first point; lon, lat (n,); element (n,) the host element, -1
    outside the mesh. z (t, n_layer, n) height of each layer centre, z_interfaces
    (t, n_layer + 1, n) of the layer interfaces (sigma meshes with ``siglev`` only, else None),
    bed_z (n,) height of the bed; values[name] (t, n_layer, n). NaN outside the mesh.
    """

    times: pd.DatetimeIndex
    distance: np.ndarray
    lon: np.ndarray
    lat: np.ndarray
    element: np.ndarray
    z: np.ndarray
    z_interfaces: np.ndarray | None
    bed_z: np.ndarray
    values: dict[str, np.ndarray]


@dataclass
class LayeredMesh:
    """Fields of a layered model on its own triangle mesh (see the module docstring).

    times: (t,) tz-aware UTC; naive input is taken as UTC. NaN in a field marks a cell with no
    value (dry, below the bed, masked). source_index: optional (n_elem,) index of each element
    in the source's own numbering (an FVCOM element id, a flat grid-cell index).
    """

    times: pd.DatetimeIndex
    node_lon: np.ndarray
    node_lat: np.ndarray
    tri: np.ndarray
    fields: dict[str, np.ndarray]
    meta: dict[str, FieldMeta] = field(default_factory=dict)
    h: np.ndarray | None = None
    siglay: np.ndarray | None = None
    siglev: np.ndarray | None = None
    zeta: np.ndarray | None = None
    layer_z: np.ndarray | None = None
    bed_z: np.ndarray | None = None
    source_index: np.ndarray | None = None
    attrs: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.times = _utc(self.times)
        self.node_lon = np.asarray(self.node_lon, dtype=np.float64)
        self.node_lat = np.asarray(self.node_lat, dtype=np.float64)
        self.tri = np.asarray(self.tri, dtype=np.int64)
        n_node, n_t = self.node_lon.shape[0], len(self.times)
        if self.node_lon.ndim != 1 or self.node_lat.shape != (n_node,):
            raise ValueError("node_lon and node_lat must be 1-D and the same length")
        if self.tri.ndim != 2 or self.tri.shape[1] != 3:
            raise ValueError(f"tri must be (n_elem, 3), got {self.tri.shape}")
        if self.tri.size and (self.tri.min() < 0 or self.tri.max() >= n_node):
            raise ValueError("tri holds node indices outside node_lon/node_lat")
        n_elem = self.tri.shape[0]

        fields, meta, n_layer = {}, dict(self.meta), None
        for name, arr in self.fields.items():
            a = np.asarray(arr, dtype=np.float32)
            if a.ndim == 2:
                a = a[:, None, :]
            if a.ndim != 3 or a.shape[0] != n_t:
                raise ValueError(
                    f"field {name!r} must be (t, n_layer, n), got {a.shape}"
                )
            n_layer = a.shape[1] if n_layer is None else n_layer
            if a.shape[1] != n_layer:
                raise ValueError(
                    f"field {name!r} has {a.shape[1]} layers, others {n_layer}"
                )
            m = meta.get(name)
            if m is None:
                if n_elem == n_node:
                    raise ValueError(
                        f"field {name!r}: give its kind in meta (n_elem == n_node)"
                    )
                m = meta[name] = FieldMeta(
                    name, kind=ELEMENT if a.shape[2] == n_elem else NODE
                )
            want = n_elem if m.kind == ELEMENT else n_node
            if a.shape[2] != want:
                raise ValueError(
                    f"{m.kind} field {name!r} must have {want} columns, got {a.shape[2]}"
                )
            fields[name] = a
        self.fields, self.meta = fields, meta
        self._n_layer = n_layer

        if self.siglay is not None:
            if self.layer_z is not None:
                raise ValueError("give sigma (h, siglay) or layer_z, not both")
            if self.h is None:
                raise ValueError("a sigma mesh needs h")
            self.h = np.asarray(self.h, dtype=np.float64)
            if self.h.shape != (n_node,):
                raise ValueError(f"h must be (n_node,), got {self.h.shape}")
            self.siglay = self._per_node(self.siglay, "siglay")
            nl = self.siglay.shape[0]
            if self._n_layer is not None and nl != self._n_layer:
                raise ValueError(f"siglay has {nl} layers, fields {self._n_layer}")
            self._n_layer = nl
            if self.siglev is not None:
                self.siglev = self._per_node(self.siglev, "siglev")
                if self.siglev.shape[0] != nl + 1:
                    raise ValueError(
                        f"siglev must have {nl + 1} levels, got {self.siglev.shape[0]}"
                    )
            if self.zeta is not None:
                self.zeta = np.asarray(self.zeta, dtype=np.float64)
                if self.zeta.shape != (n_t, n_node):
                    raise ValueError(
                        f"zeta must be (t, n_node) = {(n_t, n_node)}, got {self.zeta.shape}"
                    )
        elif self.layer_z is not None:
            lz = np.asarray(self.layer_z, dtype=np.float64)
            if lz.ndim == 1:
                lz = np.broadcast_to(lz[:, None], (lz.size, n_elem)).copy()
            if lz.ndim != 2 or lz.shape[1] != n_elem:
                raise ValueError(
                    f"layer_z must be (n_layer,) or (n_layer, n_elem), got {lz.shape}"
                )
            if self._n_layer is not None and lz.shape[0] != self._n_layer:
                raise ValueError(
                    f"layer_z has {lz.shape[0]} layers, fields {self._n_layer}"
                )
            self.layer_z, self._n_layer = lz, lz.shape[0]
        elif (self._n_layer or 1) > 1:
            raise ValueError(
                "a mesh with several layers needs sigma (h, siglay) or layer_z"
            )
        if self.bed_z is not None:
            self.bed_z = np.asarray(self.bed_z, dtype=np.float64)
            if self.bed_z.shape != (n_elem,):
                raise ValueError(f"bed_z must be (n_elem,), got {self.bed_z.shape}")
        if self.source_index is not None:
            self.source_index = np.asarray(self.source_index, dtype=np.int64)
            if self.source_index.shape != (n_elem,):
                raise ValueError("source_index must be (n_elem,)")
        self._lonlat0 = (
            (
                float(0.5 * (self.node_lon.min() + self.node_lon.max())),
                float(0.5 * (self.node_lat.min() + self.node_lat.max())),
            )
            if n_node
            else (0.0, 0.0)
        )
        self._finder: Any = None
        self._area: np.ndarray | None = None

    def _per_node(self, s, name: str) -> np.ndarray:
        s = np.asarray(s, dtype=np.float64)
        if s.ndim == 1:
            s = np.broadcast_to(s[:, None], (s.size, self.n_node)).copy()
        if s.ndim != 2 or s.shape[1] != self.n_node:
            raise ValueError(f"{name} must be (n,) or (n, n_node), got {s.shape}")
        return s

    # Sizes ------------------------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.times)

    @property
    def n_node(self) -> int:
        return self.node_lon.shape[0]

    @property
    def n_elem(self) -> int:
        return self.tri.shape[0]

    @property
    def n_layer(self) -> int:
        return self._n_layer or 1

    @property
    def vertical(self) -> str:
        """``"sigma"``, ``"z"`` or ``"surface"``."""
        if self.siglay is not None:
            return "sigma"
        return "z" if self.layer_z is not None else "surface"

    # Values -----------------------------------------------------------------------------------

    def _has_speed(self) -> bool:
        return "speed" not in self.fields and {"u", "v"} <= set(self.fields)

    def field_meta(self, name: str) -> FieldMeta:
        if name in self.meta:
            return self.meta[name]
        if name == "speed" and self._has_speed():
            return SPEED_META
        raise KeyError(f"no field {name!r}; fields: {sorted(self.fields)}")

    def element_values(self, name: str) -> np.ndarray:
        """(t, n_layer, n_elem): an element field as held, a node field as the mean of each
        element's three nodes (one flat value per cell, as drawn)."""
        if name == "speed" and self._has_speed():
            return np.hypot(self.element_values("u"), self.element_values("v"))
        if name not in self.fields:
            raise KeyError(f"no field {name!r}; fields: {sorted(self.fields)}")
        a = self.fields[name]
        if self.meta[name].kind == ELEMENT:
            return a
        return a[:, :, self.tri].mean(axis=3)

    def layer_heights(self) -> np.ndarray:
        """(t, n_layer, n_elem) height of each layer centre at each element (m, positive up): the
        mean over its three nodes on a sigma mesh; 0 for a surface-only mesh."""
        n_t = len(self.times)
        if self.siglay is not None and self.h is not None:
            zeta = self.zeta if self.zeta is not None else np.zeros((n_t, self.n_node))
            z = self.siglay[None] * (self.h + zeta)[:, None, :] + zeta[:, None, :]
            return z[:, :, self.tri].mean(axis=3)
        if self.layer_z is not None:
            return np.broadcast_to(self.layer_z, (n_t,) + self.layer_z.shape)
        return np.zeros((n_t, 1, self.n_elem))

    def element_area(self) -> np.ndarray:
        """(n_elem,) horizontal area of each element in m^2, each in metres about its own
        latitude."""
        area = self._area
        if area is None:
            lon, lat = self.node_lon[self.tri], self.node_lat[self.tri]
            kx = EARTH_RADIUS * np.cos(np.deg2rad(lat.mean(axis=1)))
            x, y = np.deg2rad(lon) * kx[:, None], np.deg2rad(lat) * EARTH_RADIUS
            area = self._area = 0.5 * np.abs(
                (x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0])
                - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0])
            )
        return area

    def surface(self):
        """The top layer's u and v as a `MeshFrames` (the single-layer contract)."""
        from forcingkit.mesh_frames import MeshFrames

        u, v = self.element_values("u"), self.element_values("v")
        return MeshFrames(
            self.times,
            self.node_lon,
            self.node_lat,
            self.tri,
            u[:, 0],
            v[:, 0],
            attrs=dict(self.attrs),
        )

    # Point location and sections --------------------------------------------------------------

    def locate(self, lon, lat) -> tuple[np.ndarray, np.ndarray]:
        """Host element of each point ((n,), -1 outside the mesh) and its barycentric weights on
        the element's three nodes ((n, 3), NaN outside)."""
        lon = np.atleast_1d(np.asarray(lon, dtype=np.float64))
        lat = np.atleast_1d(np.asarray(lat, dtype=np.float64))
        nx, ny = to_local(self.node_lon, self.node_lat, *self._lonlat0)
        px, py = to_local(lon, lat, *self._lonlat0)
        if self._finder is None:
            try:
                from matplotlib.tri import Triangulation

                self._finder = Triangulation(nx, ny, self.tri).get_trifinder()
            except Exception as exc:  # an invalid triangulation: search every element
                logger.info(
                    f"layered mesh: no trapezoid map ({exc}); brute-force search"
                )
                self._finder = False
        if self._finder is not False:
            elem = np.asarray(self._finder(px, py), dtype=np.int64)
        else:
            elem = self._brute_locate(nx, ny, px, py)
        w = np.full((lon.size, 3), np.nan)
        ok = elem >= 0
        if ok.any():
            t = self.tri[elem[ok]]
            w[ok] = _barycentric(nx[t], ny[t], px[ok], py[ok])
        return elem, w

    def _brute_locate(self, nx, ny, px, py) -> np.ndarray:
        xt, yt = nx[self.tri], ny[self.tri]
        out = np.full(px.size, -1, dtype=np.int64)
        for i in range(px.size):
            b = _barycentric(
                xt, yt, np.full(self.n_elem, px[i]), np.full(self.n_elem, py[i])
            )
            hit = np.flatnonzero((b >= -1e-9).all(axis=1))
            if hit.size:
                out[i] = hit[0]
        return out

    def section(
        self,
        points: Sequence[Sequence[float]],
        n: int = 200,
        fields: Sequence[str] | None = None,
        node_values: str = "element",
    ) -> Section:
        """Values at n points spaced evenly by distance along the polyline `points` = [(lon, lat),
        ...] (both ends included).

        Element fields take the host element's value. Node fields take the element mean
        (``node_values="element"``, the flat value drawn for that cell) or are interpolated
        linearly within the element (``"linear"``). Heights interpolate linearly within the
        element on a sigma mesh and take the host element's on a fixed-height mesh. `fields`
        defaults to every field, and speed when it is derived from u and v.
        """
        if node_values not in ("element", "linear"):
            raise ValueError(
                f"node_values must be 'element' or 'linear', got {node_values!r}"
            )
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] != 2:
            raise ValueError("points must be at least two (lon, lat) pairs")
        if n < 2:
            raise ValueError("n must be at least 2")
        lon0, lat0 = float(pts[:, 0].mean()), float(pts[:, 1].mean())
        x, y = to_local(pts[:, 0], pts[:, 1], lon0, lat0)
        cum = np.r_[0.0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
        dist = np.linspace(0.0, cum[-1], n)
        lon, lat = from_local(
            np.interp(dist, cum, x), np.interp(dist, cum, y), lon0, lat0
        )

        elem, w = self.locate(lon, lat)
        ok = elem >= 0
        e_ok, w_ok = elem[ok], w[ok]
        nodes = self.tri[e_ok]  # (m, 3)
        n_t, n_l = len(self.times), self.n_layer

        def at_points(values, lead) -> np.ndarray:  # values (*lead, m) -> (*lead, n)
            out = np.full(lead + (n,), np.nan)
            out[..., ok] = values
            return out

        def node_interp(a):  # (..., n_node) -> (..., m)
            return (a[..., nodes] * w_ok).sum(axis=-1)

        if fields is None:
            fields = list(self.fields) + (["speed"] if self._has_speed() else [])
        values = {}
        for name in fields:
            if self.field_meta(name).kind == NODE and node_values == "linear":
                values[name] = at_points(node_interp(self.fields[name]), (n_t, n_l))
            else:
                values[name] = at_points(
                    self.element_values(name)[:, :, e_ok], (n_t, n_l)
                )

        z_if = None
        bed = np.full(n, np.nan)
        if self.vertical == "sigma":
            zeta = self.zeta if self.zeta is not None else np.zeros((n_t, self.n_node))
            hp, zp = node_interp(self.h), node_interp(zeta)  # (m,), (t, m)

            def heights(sigma):  # (k, n_node) -> (t, k, m)
                return node_interp(sigma)[None] * (hp + zp)[:, None] + zp[:, None]

            z = at_points(heights(self.siglay), (n_t, n_l))
            if self.siglev is not None:
                z_if = at_points(heights(self.siglev), (n_t, n_l + 1))
            bed = at_points(-hp, ())
        else:
            lz = (
                self.layer_z if self.layer_z is not None else np.zeros((1, self.n_elem))
            )
            z = at_points(
                np.broadcast_to(lz[:, e_ok], (n_t, n_l, e_ok.size)), (n_t, n_l)
            )
            if self.bed_z is not None:
                bed = at_points(self.bed_z[e_ok], ())
        return Section(self.times, dist, lon, lat, elem, z, z_if, bed, values)


def _barycentric(xt, yt, x0, y0) -> np.ndarray:
    """(n, 3) barycentric weights of points (x0, y0) in triangles with vertices (xt, yt) (n, 3)."""
    det = (yt[:, 1] - yt[:, 2]) * (xt[:, 0] - xt[:, 2]) + (xt[:, 2] - xt[:, 1]) * (
        yt[:, 0] - yt[:, 2]
    )
    dx, dy = x0 - xt[:, 2], y0 - yt[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        b0 = ((yt[:, 1] - yt[:, 2]) * dx + (xt[:, 2] - xt[:, 1]) * dy) / det
        b1 = ((yt[:, 2] - yt[:, 0]) * dx + (xt[:, 0] - xt[:, 2]) * dy) / det
    return np.column_stack([b0, b1, 1.0 - b0 - b1])


# Particle tracks ------------------------------------------------------------------------------


@dataclass
class Tracks:
    """Particle positions over time.

    times: (t,) UTC. xyz: (t, n_particle, 3) longitude, latitude (degrees) and height (m, positive
    up). alive: (t, n_particle) the particle is in the water at that time (default: its position
    is finite; a position that is not finite is never alive).
    """

    times: pd.DatetimeIndex
    xyz: np.ndarray
    alive: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.times = _utc(self.times)
        self.xyz = np.asarray(self.xyz, dtype=np.float64)
        if (
            self.xyz.ndim != 3
            or self.xyz.shape[0] != len(self.times)
            or self.xyz.shape[2] != 3
        ):
            raise ValueError(f"xyz must be (t, n_particle, 3), got {self.xyz.shape}")
        finite = np.isfinite(self.xyz).all(axis=-1)
        if self.alive is None:
            self.alive = finite
        else:
            self.alive = np.asarray(self.alive, dtype=bool)
            if self.alive.shape != self.xyz.shape[:2]:
                raise ValueError(
                    f"alive must be (t, n_particle) = {self.xyz.shape[:2]}"
                )
            self.alive = self.alive & finite

    @property
    def n_particle(self) -> int:
        return self.xyz.shape[1]

    def local_xyz(self, lon0: float, lat0: float) -> np.ndarray:
        """(t, n_particle, 3) metres east and north of (lon0, lat0), and height."""
        x, y = to_local(self.xyz[..., 0], self.xyz[..., 1], lon0, lat0)
        return np.stack([x, y, self.xyz[..., 2]], axis=-1)

    def quantize(self, lon0: float, lat0: float) -> tuple[np.ndarray, dict]:
        """Local positions as int16 per axis, value = offset + scale * code, the codes -32767 to
        32767 spanning each axis over the alive samples (so the error is at most scale / 2, about
        0.3 m across 20 km). Samples that are not alive get code 0. Returns (codes, {"offset":
        [3], "scale": [3]})."""
        xyz = self.local_xyz(lon0, lat0)
        ok = np.asarray(self.alive, dtype=bool)
        pts = xyz[ok] if ok.any() else np.zeros((1, 3))
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        offset = 0.5 * (lo + hi)
        scale = np.maximum(0.5 * (hi - lo) / 32767, 1e-6)
        codes = np.rint((np.nan_to_num(xyz) - offset) / scale)
        codes = np.where(ok[..., None], np.clip(codes, -32767, 32767), 0)
        return codes.astype(np.int16), {
            "offset": offset.tolist(),
            "scale": scale.tolist(),
        }


# Shared colour scales -------------------------------------------------------------------------

DEFAULT_PERCENTILES = (1.0, 99.0)
DEFAULT_ABS_PERCENTILE = 99.5


def weighted_percentile(x, w, q) -> list[float]:
    """Percentiles q (0 to 100) of x with weights w: for each, the smallest value whose cumulative
    weight reaches q percent of the total."""
    x, w = (
        np.asarray(x, dtype=np.float64).ravel(),
        np.asarray(w, dtype=np.float64).ravel(),
    )
    order = np.argsort(x)
    xs, cw = x[order], np.cumsum(w[order])
    return [
        float(xs[min(np.searchsorted(cw, p / 100 * cw[-1]), x.size - 1)]) for p in q
    ]


def shared_scale(
    meshes: Sequence[LayeredMesh],
    name: str,
    window: tuple | None = None,
    q: tuple[float, float] = DEFAULT_PERCENTILES,
    q_abs: float = DEFAULT_ABS_PERCENTILE,
    symmetric: bool | None = None,
) -> FieldMeta:
    """`name`'s metadata with lo and hi set from every element value of every mesh that holds it,
    weighted by element area, at the mesh times inside `window` = (start, end) inclusive (all
    times when None).

    lo and hi are the q percentiles (1st and 99th by default) or, for a symmetric field
    (``meta.symmetric`` unless `symmetric` says otherwise), -hi to hi with hi the q_abs percentile
    (99.5th) of the magnitude. Weighting by area keeps a model's many small river and harbour
    elements from setting the scale: counted per element, they stretch it to their own extremes.
    """
    vals, wts, meta = [], [], None
    for m in meshes:
        try:
            fm = m.field_meta(name)
        except KeyError:
            continue
        meta = meta or fm
        sel = np.ones(len(m.times), dtype=bool)
        if window is not None:
            t0, t1 = (_utc([t])[0] for t in window)
            sel = (m.times >= t0) & (m.times <= t1)
        f = m.element_values(name)[sel]
        wt = np.broadcast_to(m.element_area(), f.shape)
        ok = np.isfinite(f)
        vals.append(f[ok])
        wts.append(wt[ok])
    if meta is None:
        raise KeyError(f"no mesh holds {name!r}")
    v, w = np.concatenate(vals), np.concatenate(wts)
    if v.size == 0:
        raise ValueError(f"{name}: no finite values in the window")
    if meta.symmetric if symmetric is None else symmetric:
        hi = weighted_percentile(np.abs(v), w, [q_abs])[0]
        lo = -hi
    else:
        lo, hi = weighted_percentile(v, w, list(q))
    logger.info(f"shared scale {name}: {lo:.4g} to {hi:.4g} {meta.units}")
    return meta.with_range(lo, hi)


def shared_scales(
    meshes: Sequence[LayeredMesh],
    names: Sequence[str] | None = None,
    window: tuple | None = None,
    **kw,
) -> dict[str, FieldMeta]:
    """`shared_scale` for each of `names` (default: every field any mesh holds)."""
    if names is None:
        names = list(dict.fromkeys(n for m in meshes for n in m.fields))
    return {n: shared_scale(meshes, n, window, **kw) for n in names}


# Adapters -------------------------------------------------------------------------------------

NECOFS_META = {
    "u": FieldMeta("u", "Eastward velocity", "m/s", ELEMENT, symmetric=True),
    "v": FieldMeta("v", "Northward velocity", "m/s", ELEMENT, symmetric=True),
    "w": FieldMeta(
        "w", "Vertical velocity, up positive", "m/s", ELEMENT, symmetric=True
    ),
    "T": FieldMeta("T", "Temperature", "°C", NODE),
    "S": FieldMeta("S", "Salinity", "PSU", NODE),
}


def from_necofs_volume(ds) -> LayeredMesh:
    """A `LayeredMesh` from a `forcingkit.fetchers.necofs.volume` (or `read_volume`) dataset:
    sigma layers with zeta, element u, v and w (FVCOM ``ww``), node T and S."""
    return LayeredMesh(
        times=pd.DatetimeIndex(ds["time"].values),
        node_lon=ds["node_lon"].values,
        node_lat=ds["node_lat"].values,
        tri=ds["triangles"].values,
        fields={
            "u": ds["u"].values,
            "v": ds["v"].values,
            "w": ds["ww"].values,
            "T": ds["temp"].values,
            "S": ds["salinity"].values,
        },
        meta={k: replace(m) for k, m in NECOFS_META.items()},
        h=ds["h"].values,
        siglay=ds["siglay"].values,
        siglev=ds["siglev"].values,
        zeta=ds["zeta"].values,
        source_index=ds["element_id"].values,
        attrs={
            "model": "necofs",
            "source": "NECOFS FVCOM GOM7, UMass Dartmouth SMAST",
            "bbox": [float(b) for b in ds.attrs.get("requested_bbox", [])],
        },
    )


def necofs(bbox: Sequence[float], start, end, cache: bool = True) -> LayeredMesh:
    """NECOFS FVCOM GOM7, every sigma layer, for the elements whose centres lie in `bbox` =
    (W, S, E, N), hourly from `start` to `end` inclusive (UTC). Pad the box for a margin."""
    from forcingkit.fetchers import necofs as fetch

    return from_necofs_volume(fetch.volume(list(bbox), start, end, cache=cache))


def _edges(c: np.ndarray) -> np.ndarray:
    """Cell edges of a 1-D axis of centres: midpoints, the ends extended by half a cell."""
    if c.size < 2:
        raise ValueError("an axis needs at least two cells")
    mid = 0.5 * (c[1:] + c[:-1])
    return np.r_[2 * c[0] - mid[0], mid, 2 * c[-1] - mid[-1]]


def _corners(c: np.ndarray) -> np.ndarray:
    """(ny + 1, nx + 1) cell corners of a curvilinear grid of (ny, nx) centres: the mean of the
    four centres around each corner, the grid extended linearly by one cell on every side."""
    if min(c.shape) < 2:
        raise ValueError("a curvilinear grid needs at least 2 x 2 cells")
    p = np.vstack([2 * c[:1] - c[1:2], c, 2 * c[-1:] - c[-2:-1]])
    p = np.hstack([2 * p[:, :1] - p[:, 1:2], p, 2 * p[:, -1:] - p[:, -2:-1]])
    return 0.25 * (p[:-1, :-1] + p[1:, :-1] + p[:-1, 1:] + p[1:, 1:])


def structured(
    lon,
    lat,
    mask=None,
    z=None,
    *,
    times,
    fields: Mapping[str, np.ndarray],
    meta: Mapping[str, FieldMeta] | None = None,
    bed_z=None,
    attrs: dict | None = None,
) -> LayeredMesh:
    """A regular or curvilinear grid as a `LayeredMesh`: each cell where `mask` is true becomes two
    triangles that both carry the cell's own value, so cells are drawn flat, as the model holds
    them.

    lon, lat: cell centres, either 1-D axes (nx,) and (ny,) or 2-D (ny, nx). Corners are midway
    between centres (curvilinear: the mean of the four centres around each corner), the outer
    ones half a cell beyond the outermost centres.
    mask: (ny, nx) true for cells to keep (water); None keeps every cell.
    z: None for one layer at the surface; (n_layer,) fixed heights of layer centres (m, positive
    up, so depths are negative); or (n_layer, ny, nx).
    fields: name -> (t, n_layer, ny, nx), or (t, ny, nx) for one layer.
    bed_z: optional (ny, nx) height of the bed (negative depth).

    Elements 2k and 2k + 1 are the k-th kept cell in row-major order; `source_index` holds each
    element's flat cell index (``j * nx + i``).
    """
    lon, lat = np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64)
    if lon.ndim == 1 and lat.ndim == 1:
        clon, clat = np.meshgrid(_edges(lon), _edges(lat))
    elif lon.ndim == 2 and lon.shape == lat.shape:
        clon, clat = _corners(lon), _corners(lat)
    else:
        raise ValueError("lon and lat must be 1-D axes or 2-D arrays of one shape")
    ny, nx = clon.shape[0] - 1, clon.shape[1] - 1
    mask = (
        np.ones((ny, nx), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    )
    if mask.shape != (ny, nx):
        raise ValueError(f"mask must be (ny, nx) = {(ny, nx)}, got {mask.shape}")

    cells = np.flatnonzero(mask.ravel())
    j, i = np.divmod(cells, nx)
    sw = j * (nx + 1) + i
    se, nw = sw + 1, sw + nx + 1
    ne = nw + 1
    corner = np.stack(
        [np.stack([sw, se, ne], 1), np.stack([sw, ne, nw], 1)], 1
    ).reshape(-1, 3)
    nodes, inv = np.unique(corner, return_inverse=True)
    elem_cell = np.repeat(cells, 2)

    n_t = len(pd.DatetimeIndex(times))
    flat = {}
    for name, arr in fields.items():
        a = np.asarray(arr, dtype=np.float32)
        if a.ndim == 3:
            a = a[:, None]
        if a.ndim != 4 or a.shape[0] != n_t or a.shape[2:] != (ny, nx):
            raise ValueError(
                f"field {name!r} must be (t, [n_layer,] ny, nx), got {np.shape(arr)}"
            )
        flat[name] = a.reshape(n_t, a.shape[1], ny * nx)[:, :, elem_cell]

    layer_z = None
    if z is not None:
        z = np.asarray(z, dtype=np.float64)
        if z.ndim == 1:
            layer_z = z
        elif z.ndim == 3 and z.shape[1:] == (ny, nx):
            layer_z = z.reshape(z.shape[0], ny * nx)[:, elem_cell]
        else:
            raise ValueError(
                f"z must be (n_layer,) or (n_layer, ny, nx), got {z.shape}"
            )
    bed = None
    if bed_z is not None:
        bed = np.asarray(bed_z, dtype=np.float64)
        if bed.shape != (ny, nx):
            raise ValueError(f"bed_z must be (ny, nx), got {bed.shape}")
        bed = bed.ravel()[elem_cell]

    m = {k: replace(v, kind=ELEMENT) for k, v in (meta or {}).items() if k in flat}
    for k in flat:
        m.setdefault(k, FieldMeta(k, kind=ELEMENT))
    return LayeredMesh(
        times=times,
        node_lon=clon.ravel()[nodes],
        node_lat=clat.ravel()[nodes],
        tri=inv.reshape(-1, 3),
        fields=flat,
        meta=m,
        layer_z=layer_z,
        bed_z=bed,
        source_index=elem_cell,
        attrs={"grid_shape": [ny, nx], **(attrs or {})},
    )
