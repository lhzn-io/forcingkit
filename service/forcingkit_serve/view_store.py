"""Cached stores as the viewer draws them: every store on its own cells, at true heights.

A `StoreView` reads one store lazily and answers what the cache viewer asks for:

- the cells, as a triangle mesh (`forcingkit.layered_mesh`): an FVCOM store's own elements, or a
  regular or curvilinear grid's cells as two triangles each (`layered_mesh.structured`), so every
  cell is drawn flat with its own value and nothing is interpolated;
- the layers, surface first, with their heights in meters (positive up) where the store holds
  them (fixed levels, sigma with the water depth, or level heights of an atmosphere);
- one value per cell for a field, layer and record, and one byte per cell on the field's fixed
  scale (`FieldMeta.encode`), so a page draws a record without knowing the store's schema;
- values along a polyline (a vertical section) and at a point.

Grids larger than a cell budget are shown every `stride`-th cell in each direction (each drawn
cell then holds one source cell's value, never an average), and the manifest says so.
"""

from __future__ import annotations

import base64
import logging
import math
import threading
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from forcingkit.layered_mesh import (
    FieldMeta,
    LayeredMesh,
    from_local,
    nice_ticks,
    structured,
    to_local,
    weighted_percentile,
)
from forcingkit.render import PACKAGED_RAMPS, RAMPS, lut_color, packaged_ramp_rows

logger = logging.getLogger(__name__)

# Cells drawn per store at most, in 2D and 3D; larger grids are strided to fit.
MAX_CELLS = {"2d": 400_000, "3d": 120_000}
# Records and layers sampled (evenly through the store) to set a field's fixed color scale.
SCALE_RECORDS = 4
SCALE_LAYERS = 8
# Field volumes (one record, every layer) kept per open store.
VOLUME_CACHE_BYTES = 256 * 2**20
# Water speed is drawn on the packaged currents ramp at its own speeds, so maps of different
# stores compare directly.
WATER_SPEED_RAMP = "currents_tempo_dark"

_TIME_DIMS = ("time", "ocean_time", "valid_time")
_VERTICAL_DIMS = ("z", "depth", "level", "s_rho", "siglay", "sigma", "lev")
# Variables that describe the grid, not a field to draw.
_STRUCTURAL = {
    "triangles",
    "neighbours",
    "element_id",
    "node_id",
    "on_boundary",
    "x",
    "y",
    "xc",
    "yc",
    "lonc",
    "latc",
    "node_lon",
    "node_lat",
    "lon",
    "lat",
    "longitude",
    "latitude",
    "lon_rho",
    "lat_rho",
    "siglay",
    "siglev",
    "s_rho",
    "z_face",
    "mask",
    "wet",
    "in_bbox",
    "angle",
    "eta",
    "xi",
    "eta_rho",
    "xi_rho",
    "depth",
    "level",
    "time",
    "mouth_lon",
    "mouth_lat",
    "river",
    "scale_factor",
    "fill",
}
# (u, v, derived name, label): horizontal vector pairs, east and north components.
VECTOR_PAIRS = (
    ("u", "v", "speed", "Current speed"),
    ("water_u", "water_v", "speed", "Current speed"),
    ("u10", "v10", "wind_speed", "Wind speed at 10 m"),
)
_LABELS = {
    "u": "Eastward current",
    "v": "Northward current",
    "ww": "Vertical velocity",
    "w": "Vertical velocity",
    "temp": "Temperature",
    "temperature": "Temperature",
    "salt": "Salinity",
    "salinity": "Salinity",
    "zeta": "Surface elevation",
    "h": "Water depth",
    "u10": "Eastward wind at 10 m",
    "v10": "Northward wind at 10 m",
    "t2m": "Air temperature at 2 m",
    "q2m": "Specific humidity at 2 m",
    "sp": "Surface pressure",
    "dswrf": "Downward shortwave radiation",
    "dlwrf": "Downward longwave radiation",
    "prate": "Precipitation rate",
    "hs": "Significant wave height",
    "zs": "Ground height",
    "p": "Pressure",
    "t": "Air temperature",
    "discharge": "River discharge",
}
_UNITS = {
    "m s-1": "m/s",
    "m s**-1": "m/s",
    "degree_Celsius": "°C",
    "degC": "°C",
    "1e-3": "PSU",
    "W m-2": "W/m²",
    "kg m-2 s-1": "kg/m²/s",
    "kg kg-1": "kg/kg",
    "m3 s-1": "m³/s",
}
_SYMMETRIC = {"u", "v", "w", "ww", "water_u", "water_v", "u10", "v10", "zeta"}
# cmocean's maps for temperature and salinity (forcingkit.render.RAMPS), as the community reads them.
_RAMPS = {
    "temp": "thermal",
    "temperature": "thermal",
    "t2m": "thermal",
    "t": "thermal",
    "salt": "haline",
    "salinity": "haline",
    "h": "cividis",
}


# Store ids and paths ------------------------------------------------------------------------


def resolve_store(dataset_id: str, inventory: Sequence[Any]) -> Path | None:
    """The path of the inventory entry with this id, or None. Only ids the inventory lists
    resolve, so a request cannot name a path of its own."""
    for item in inventory:
        if item.id == dataset_id:
            return Path(item.path)
    return None


# Field description ---------------------------------------------------------------------------


@dataclass
class FieldSpec:
    """How one drawable field is read.

    source: the store variable, or the (u, v) pair for a derived speed. kind: ``"element"``
    (one value per mesh element), ``"node"`` (FVCOM node values, drawn as each element's node
    mean) or ``"cell"`` (one value per grid cell). layered: the field has the store's layers
    (else it is one value per cell at the surface or over the column). timed: it changes by
    record.
    """

    name: str
    source: str | tuple[str, str]
    kind: str
    layered: bool
    timed: bool
    meta: FieldMeta
    ramp: str


def _label_units(name: str, da: xr.DataArray | None) -> tuple[str, str]:
    attrs = da.attrs if da is not None else {}
    label = _LABELS.get(name) or attrs.get("long_name") or name
    units = str(attrs.get("units", ""))
    return str(label), _UNITS.get(units, units)


def _iso(t) -> str:
    return pd.Timestamp(t).strftime("%Y-%m-%dT%H:%M:%SZ")


def _cadence(times: pd.DatetimeIndex) -> str | None:
    if len(times) < 2:
        return None
    step = float(
        np.median(np.diff(times.values).astype("timedelta64[s]").astype(float))
    )
    if step % 3600 == 0:
        return f"every {int(step // 3600)} h"
    if step % 60 == 0:
        return f"every {int(step // 60)} min"
    return f"every {step:g} s"


def _ramp_rgba(meta: FieldMeta, ramp: str) -> bytes:
    """256 RGBA entries, one per byte code: code 0 (no value) transparent, codes 1 to 255 the
    colors of the values they decode to."""
    codes = np.arange(256)
    rgba = np.zeros((256, 4), dtype=np.uint8)
    if ramp in PACKAGED_RAMPS:
        color = lut_color(packaged_ramp_rows(ramp), **PACKAGED_RAMPS[ramp])
        rgba[1:] = color(meta.decode(codes[1:]))
    elif ramp in RAMPS:
        knots = np.array(RAMPS[ramp], dtype=float)
        f = (codes[1:] - 1) / 254.0
        rgba[1:, :3] = np.rint(
            np.stack(
                [np.interp(f, knots[:, 0], knots[:, k]) for k in (1, 2, 3)], axis=1
            )
        ).astype(np.uint8)
        rgba[1:, 3] = 255
    else:
        from matplotlib import colormaps

        rgba[1:] = np.rint(colormaps[ramp]((codes[1:] - 1) / 254.0) * 255).astype(
            np.uint8
        )
    rgba[0] = 0
    return rgba.tobytes()


def _quantile_mesh_scale(
    values: list[np.ndarray], weights: list[np.ndarray], symmetric: bool, log: bool
) -> tuple[float, float]:
    v = np.concatenate(values) if values else np.zeros(0)
    w = np.concatenate(weights) if weights else np.zeros(0)
    ok = np.isfinite(v)
    v, w = v[ok], w[ok]
    if log:
        keep = v > 0
        v, w = v[keep], w[keep]
    if v.size == 0:
        return (0.0, 1.0) if not log else (0.01, 1.0)
    if symmetric:
        hi = weighted_percentile(np.abs(v), w, [99.5])[0] or 1.0
        return -hi, hi
    lo, hi = weighted_percentile(v, w, [1.0, 99.0])
    if hi <= lo:
        hi = lo + (abs(lo) * 0.01 or 1.0)
    return lo, hi


# The view ------------------------------------------------------------------------------------


@dataclass
class StoreView:
    """One store, opened for drawing (see the module docstring). Build with `open_view`."""

    dataset_id: str
    path: Path
    ds: xr.Dataset
    store: str  # "mesh", "grid" or "points"
    times: pd.DatetimeIndex
    snapshot: str | None
    node_lon: np.ndarray = field(default_factory=lambda: np.zeros(0))
    node_lat: np.ndarray = field(default_factory=lambda: np.zeros(0))
    tri: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.int64))
    # Grid: the flat index of each element's cell in the strided (ny, nx) grid.
    elem_cell: np.ndarray | None = None
    grid_shape: tuple[int, int] | None = None
    source_shape: tuple[int, int] | None = None
    stride: int = 1
    ydim: str | None = None
    xdim: str | None = None
    tdim: str | None = None
    vdim: str | None = None
    # Layer i (surface first) is source index layer_src[i] along vdim.
    layer_src: list[int] = field(default_factory=list)
    vertical: str = "surface"
    fields: dict[str, FieldSpec] = field(default_factory=dict)
    sites: list[dict] = field(default_factory=list)
    _base: LayeredMesh | None = None
    medium: str = "ocean"  # or "atmosphere": how layer heights are labeled
    _scales: dict[str, FieldMeta] = field(default_factory=dict)
    _volumes: OrderedDict = field(default_factory=OrderedDict)
    _lock: threading.RLock = field(default_factory=threading.RLock)

    # Sizes and geometry -----------------------------------------------------------------------

    @property
    def n_elem(self) -> int:
        return int(self.tri.shape[0])

    @property
    def n_layer(self) -> int:
        return max(1, len(self.layer_src))

    @property
    def n_records(self) -> int:
        return max(1, len(self.times))

    def base(self) -> LayeredMesh:
        """The cells with no fields: point location and element areas."""
        if self._base is None:
            self._base = LayeredMesh(
                times=pd.DatetimeIndex([pd.Timestamp("2000-01-01")]),
                node_lon=self.node_lon,
                node_lat=self.node_lat,
                tri=self.tri,
                fields={},
            )
        return self._base

    def mesh_bytes(self) -> bytes:
        """Node longitudes and latitudes (Float64, so a page can place cells to the meter) then
        the triangles (Uint32)."""
        return (
            np.asarray(self.node_lon, "<f8").tobytes()
            + np.asarray(self.node_lat, "<f8").tobytes()
            + np.asarray(self.tri, "<u4").tobytes()
        )

    def _record(self, t: int) -> int:
        return int(min(max(t, 0), self.n_records - 1))

    def _layer(self, layer: int) -> int:
        return int(min(max(layer, 0), self.n_layer - 1))

    # Reading ----------------------------------------------------------------------------------

    def _isel(self, da: xr.DataArray, t: int, layer: int | None) -> xr.DataArray:
        sel: dict[str, Any] = {}
        if self.tdim and self.tdim in da.dims:
            sel[self.tdim] = self._record(t)
        if self.vdim and self.vdim in da.dims:
            sel[self.vdim] = self.layer_src[self._layer(layer or 0)]
        da = da.isel(sel) if sel else da
        extra = [
            d for d in da.dims if d not in (self.ydim, self.xdim, "node", "element")
        ]
        if extra:  # a leftover length-one dimension (a GRIB step, a height)
            da = da.isel({d: 0 for d in extra})
        return da

    def _cells(self, da: xr.DataArray) -> np.ndarray:
        """A (ny, nx) grid slice to one value per element."""
        a = np.asarray(da.transpose(self.ydim, self.xdim).values, dtype=np.float32)
        s = self.stride
        a = a[::s, ::s]
        assert self.elem_cell is not None
        return a.ravel()[self.elem_cell]

    def volume(self, name: str, t: int = 0) -> np.ndarray:
        """(n_layer, n_elem) float32, layers surface first ((1, n_elem) for a field without
        layers): the field at one record, read once and kept while it is in use."""
        t = self._record(t) if self.fields[name].timed else 0
        key = (name, t)
        with self._lock:
            hit = self._volumes.get(key)
            if hit is not None:
                self._volumes.move_to_end(key)
                return hit
        spec = self.fields[name]
        if isinstance(spec.source, tuple):
            a = np.hypot(self.volume(spec.source[0], t), self.volume(spec.source[1], t))
            out = a.astype(np.float32)
        else:
            out = self._read_volume(spec, t)
        with self._lock:
            self._volumes[key] = out
            size = sum(v.nbytes for v in self._volumes.values())
            while size > VOLUME_CACHE_BYTES and len(self._volumes) > 1:
                _, old = self._volumes.popitem(last=False)
                size -= old.nbytes
        return out

    def _read_volume(self, spec: FieldSpec, t: int) -> np.ndarray:
        da = self.ds[spec.source]  # type: ignore[index]
        sel: dict[str, Any] = {}
        if self.tdim and self.tdim in da.dims:
            sel[self.tdim] = t
        da = da.isel(sel) if sel else da
        keep = {self.ydim, self.xdim, "node", "element", "river"}
        if spec.layered:
            keep.add(self.vdim)
        extra = [d for d in da.dims if d not in keep]
        if extra:  # a leftover length-one dimension (a GRIB step, a height)
            da = da.isel({d: 0 for d in extra})
        if self.store == "points":
            return np.asarray(da.values, dtype=np.float32).reshape(1, -1)
        if spec.kind == "cell":
            order = ([self.vdim] if spec.layered else []) + [self.ydim, self.xdim]
            a = np.asarray(da.transpose(*order).values, dtype=np.float32)
            if not spec.layered:
                a = a[None]
            a = a[:, :: self.stride, :: self.stride]
            assert self.elem_cell is not None
            a = a.reshape(a.shape[0], -1)[:, self.elem_cell]
        else:
            horiz = "element" if spec.kind == "element" else "node"
            order = ([self.vdim] if spec.layered else []) + [horiz]
            a = np.asarray(da.transpose(*order).values, dtype=np.float32)
            if not spec.layered:
                a = a[None]
            if spec.kind == "node":
                a = a[:, self.tri].mean(axis=2)
        if spec.layered:
            a = a[self.layer_src]
        return np.ascontiguousarray(a, dtype=np.float32)

    def values(self, name: str, t: int = 0, layer: int = 0) -> np.ndarray:
        """(n_elem,) float32: the field at one record and layer, one value per element, NaN
        where the cell holds none. Layer is ignored for a field without layers."""
        vol = self.volume(name, t)
        return vol[self._layer(layer)] if vol.shape[0] > 1 else vol[0]

    def heights(
        self, t: int = 0, layer: int = 0, still: bool = False
    ) -> np.ndarray | None:
        """(n_elem,) height of the layer's center (m, positive up), or None when the store
        holds no heights (sigma layers without the water depth). `still`: with the surface
        elevation taken as zero (for labels)."""
        ds, k = self.ds, self.layer_src[self._layer(layer)] if self.layer_src else 0
        if self.vertical == "surface":
            return np.zeros(self.n_elem, dtype=np.float32)
        if self.store == "mesh":
            sig = np.asarray(ds["siglay"].values[k], dtype=np.float64)
            h = np.asarray(ds["h"].values, dtype=np.float64)
            zeta = (
                np.nan_to_num(
                    np.asarray(self._isel(ds["zeta"], t, None).values, np.float64)
                )
                if "zeta" in ds and not still
                else 0.0
            )
            z = sig * (h + zeta) + zeta
            return z[self.tri].mean(axis=1).astype(np.float32)
        if self.vertical == "z":
            assert self.vdim is not None
            if (
                "z" in ds.data_vars and self.vdim in ds["z"].dims
            ):  # atmosphere level heights
                return self._cells(ds["z"].isel({self.vdim: k}))
            zc = float(ds[self.vdim].values[k])
            positive = str(ds[self.vdim].attrs.get("positive", "")).lower()
            if self.vdim == "depth" and positive != "up" and zc >= 0:
                zc = -zc
            return np.full(self.n_elem, zc, dtype=np.float32)
        if self.vertical == "sigma":
            assert self.vdim is not None
            sig_da = ds["siglay"] if "siglay" in ds.data_vars else ds[self.vdim]
            if sig_da.ndim == 1:
                s = float(sig_da.values[k])
                sig = np.full(self.n_elem, s, dtype=np.float64)
            else:
                sig = self._cells(sig_da.isel({sig_da.dims[0]: k})).astype(np.float64)
            h = self._cells(ds["h"]).astype(np.float64)
            zeta = (
                np.nan_to_num(
                    self._cells(self._isel(ds["zeta"], t, None)).astype(np.float64)
                )
                if "zeta" in ds and not still
                else 0.0
            )
            return (sig * (h + zeta) + zeta).astype(np.float32)
        return None

    def bed(self) -> np.ndarray | None:
        """(n_elem,) height of the bed (or of the ground, for an atmosphere), or None."""
        ds = self.ds
        if self.store == "mesh" and "h" in ds:
            return (
                -np.asarray(ds["h"].values, np.float32)[self.tri].mean(axis=1)
            ).astype(np.float32)
        if self.store == "grid":
            if "zs" in ds.data_vars:
                return self._cells(ds["zs"])
            if "h" in ds.data_vars and ds["h"].dims == (self.ydim, self.xdim):
                return -self._cells(ds["h"])
        return None

    def layer_list(self) -> list[dict]:
        """Each layer, surface first: its source index, a label and its mean height (m)."""
        out = []
        for i, k in enumerate(self.layer_src or [0]):
            entry: dict[str, Any] = {"index": i, "source_index": int(k)}
            z = self.heights(0, i, still=True) if self.store != "points" else None
            if z is not None and z.size and np.isfinite(z).any():
                entry["z_mean"] = round(float(np.nanmean(z)), 2)
            if self.vertical == "surface":
                entry["label"] = "surface"
            elif "z_mean" in entry:
                zm = entry["z_mean"]
                entry["label"] = (
                    f"{zm:.0f} m above sea level"
                    if self.medium == "atmosphere"
                    else f"{max(0.0, -zm):.1f} m deep"
                )
            else:
                s = float(self.ds[self.vdim].values[k]) if self.vdim else 0.0
                entry["label"] = f"sigma {s:.3f}"
            out.append(entry)
        return out

    # Scales -----------------------------------------------------------------------------------

    def scale(self, name: str) -> FieldMeta:
        """The field's fixed color scale over the whole store (sampled records, every layer),
        area weighted, so a cell's color never depends on which record is on screen."""
        with self._lock:
            hit = self._scales.get(name)
            if hit is not None:
                return hit
        spec = self.fields[name]
        meta = spec.meta
        if spec.ramp in PACKAGED_RAMPS:
            rows = packaged_ramp_rows(spec.ramp)
            lo, hi = float(rows[0][0]), float(rows[-1][0])
        else:
            recs = (
                np.unique(
                    np.linspace(0, self.n_records - 1, SCALE_RECORDS)
                    .round()
                    .astype(int)
                )
                if spec.timed
                else np.array([0])
            )
            layers = (
                np.unique(
                    np.linspace(0, self.n_layer - 1, SCALE_LAYERS).round().astype(int)
                )
                if spec.layered
                else np.array([0])
            )
            area = (
                self.base().element_area()
                if self.store != "points"
                else np.ones(len(self.sites))
            )
            vals, wts = [], []
            for t in recs:
                vol = self.volume(name, int(t))
                for k in layers:
                    vals.append(vol[min(int(k), vol.shape[0] - 1)])
                    wts.append(area)
            lo, hi = _quantile_mesh_scale(vals, wts, meta.symmetric, meta.log)
        meta = meta.with_range(lo, hi)
        with self._lock:
            self._scales[name] = meta
        return meta

    def codes(self, name: str, t: int = 0, layer: int = 0) -> bytes:
        """One byte per element on the field's fixed scale (0: no value)."""
        return self.scale(name).encode(self.values(name, t, layer)).tobytes()

    # Manifest ---------------------------------------------------------------------------------

    def manifest(self) -> dict:
        fields = []
        for name, spec in self.fields.items():
            meta = self.scale(name)
            entry = meta.manifest_entry()
            entry.update(
                layered=spec.layered,
                timed=spec.timed,
                symmetric=meta.symmetric,
                ramp=spec.ramp,
                ramp_rgba=base64.b64encode(_ramp_rgba(meta, spec.ramp)).decode(),
                vector=list(spec.source) if isinstance(spec.source, tuple) else None,
            )
            fields.append(entry)
        out: dict[str, Any] = {
            "id": self.dataset_id,
            "store": self.store,
            "times": [_iso(t) for t in self.times],
            "snapshot": self.snapshot,
            "cadence": _cadence(self.times),
            "vertical": self.vertical,
            "layers": self.layer_list() if self.store != "points" else [],
            "fields": fields,
            "vectors": [
                {"u": u, "v": v, "name": n}
                for u, v, n, _ in VECTOR_PAIRS
                if u in self.fields and v in self.fields
            ],
        }
        if self.store == "points":
            out["sites"] = self.sites
            return out
        lon, lat = self.node_lon, self.node_lat
        out.update(
            n_elem=self.n_elem,
            n_node=int(lon.size),
            bbox=[
                float(lon.min()),
                float(lat.min()),
                float(lon.max()),
                float(lat.max()),
            ],
            grid_shape=list(self.grid_shape) if self.grid_shape else None,
            source_shape=list(self.source_shape) if self.source_shape else None,
            stride=self.stride,
            has_bed=self.bed() is not None,
            bed_at_nodes=self.bed_nodes() is not None,
        )
        return out

    # Points, sections and vectors -------------------------------------------------------------

    def point(
        self, lon: float, lat: float, name: str, t: int = 0, layer: int = 0
    ) -> dict:
        """The cell under (lon, lat): the field's value, the layer height and bed, the cell's
        index in the source, and the record time. For a store of sites: the nearest site."""
        if self.store == "points":
            if not self.sites:
                return {"site": None, "time": self._time(t)}
            d = [
                (s["lon"] - lon) ** 2 * math.cos(math.radians(lat)) ** 2
                + (s["lat"] - lat) ** 2
                for s in self.sites
            ]
            i = int(np.argmin(d))
            v = self.values(name, t)[i]
            return {
                "site": i,
                "name": self.sites[i]["name"],
                "value": None if not np.isfinite(v) else float(v),
                "time": self._time(t),
            }
        elem, _ = self.base().locate([lon], [lat])
        e = int(elem[0])
        out: dict[str, Any] = {"element": e, "time": self._time(t)}
        if e < 0:
            return out
        v = self.values(name, t, layer)[e]
        out["value"] = None if not np.isfinite(v) else float(v)
        z = self.heights(t, layer)
        if z is not None and self.fields[name].layered:
            out["z"] = None if not np.isfinite(z[e]) else round(float(z[e]), 2)
        b = self.bed()
        if b is not None:
            out["bed"] = None if not np.isfinite(b[e]) else round(float(b[e]), 2)
        if self.store == "grid" and self.elem_cell is not None and self.source_shape:
            j, i = divmod(int(self.elem_cell[e]), self.grid_shape[1])  # type: ignore[index]
            out["cell"] = [j * self.stride, i * self.stride]
        elif "element_id" in self.ds:
            out["cell"] = int(self.ds["element_id"].values[e])
        else:
            out["cell"] = e
        return out

    def _time(self, t: int) -> str | None:
        if len(self.times):
            return _iso(self.times[self._record(t)])
        return self.snapshot

    def section(
        self, points: Sequence[Sequence[float]], name: str, t: int = 0, n: int = 200
    ) -> dict:
        """The field along a polyline: n samples evenly spaced by distance, each taking its host
        cell's value and heights in every layer (no interpolation)."""
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] != 2:
            raise ValueError("a section needs at least two lon,lat points")
        lon0, lat0 = float(pts[:, 0].mean()), float(pts[:, 1].mean())
        x, y = to_local(pts[:, 0], pts[:, 1], lon0, lat0)
        cum = np.r_[0.0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
        dist = np.linspace(0.0, cum[-1], n)
        lon, lat = from_local(
            np.interp(dist, cum, x), np.interp(dist, cum, y), lon0, lat0
        )
        elem, _ = self.base().locate(lon, lat)
        ok = elem >= 0
        spec = self.fields[name]
        layers = range(self.n_layer) if spec.layered else range(1)
        vals = np.full((len(layers), n), np.nan, dtype=np.float32)
        z = np.full((len(layers), n), np.nan, dtype=np.float32)
        for k in layers:
            vals[k, ok] = self.values(name, t, k)[elem[ok]]
            hk = self.heights(t, k)
            if hk is not None:
                z[k, ok] = hk[elem[ok]]
        bed = np.full(n, np.nan, dtype=np.float32)
        b = self.bed()
        if b is not None:
            bed[ok] = b[elem[ok]]

        def clean(a):
            return [None if not np.isfinite(v) else round(float(v), 4) for v in a]

        return {
            "time": self._time(t),
            "distance": [round(float(d), 1) for d in dist],
            "lon": [round(float(v), 6) for v in lon],
            "lat": [round(float(v), 6) for v in lat],
            "element": elem.tolist(),
            "values": [clean(r) for r in vals],
            "z": (
                [clean(r) for r in z]
                if np.isfinite(z).any() and self.vertical != "surface"
                else None
            ),
            "bed": clean(bed) if np.isfinite(bed).any() else None,
        }

    def bed_nodes(self) -> np.ndarray | None:
        """(n_node,) the bed's height at each mesh node (m, positive up), for a mesh store with
        the water depth at its nodes; None otherwise (grids hold one bed value per cell)."""
        if self.store == "mesh" and "h" in self.ds and self.ds["h"].dims == ("node",):
            return -np.asarray(self.ds["h"].values, np.float32)
        return None

    def tracers(
        self, pair: str, t: int = 0, levels: int = 6, limit: int = 3000
    ) -> dict:
        """Horizontal velocity for tracers drawn inside their own cells: up to `limit` columns
        (the first wet cell in each square of an even grid over the domain, the same columns at
        every level) and up to `levels` layers spread from the surface to the deepest, each
        column's element, layer height and east and north components (null where the cell is
        dry at that layer). Values are the cells' own."""
        match = [p for p in VECTOR_PAIRS if p[2] == pair and p[0] in self.fields]
        if not match:
            raise KeyError(f"no vector pair {pair!r}")
        un, vn = match[0][0], match[0][1]
        layered = self.fields[un].layered
        top_u, top_v = self.values(un, t, 0), self.values(vn, t, 0)
        cand = np.arange(self.n_elem)
        if self.store == "grid":
            cand = cand[::2]  # two triangles per cell
        cand = cand[np.isfinite(top_u[cand]) & np.isfinite(top_v[cand])]
        if cand.size > limit:
            clon = self.node_lon[self.tri[cand]].mean(axis=1)
            clat = self.node_lat[self.tri[cand]].mean(axis=1)
            x, y = to_local(clon, clat, float(clon.mean()), float(clat.mean()))
            # Squares sized from the water's own area (not the bounding box's), so a narrow
            # or ragged domain still gets about `limit` columns.
            area = self.base().element_area()[cand] * (
                2.0 if self.store == "grid" else 1.0
            )
            side = max(math.sqrt(float(area.sum()) / limit), 1.0)
            key = np.floor(x / side).astype(np.int64) * 1_000_003 + np.floor(
                y / side
            ).astype(np.int64)
            _, first = np.unique(key, return_index=True)
            cand = cand[np.sort(first)]
        # Levels per column: up to `levels` layers spread from the surface to the column's own
        # deepest wet layer, so shallow columns get as many tracers as deep ones.
        uvol = self.volume(un, t)[:, cand]
        vvol = self.volume(vn, t)[:, cand]
        n = uvol.shape[0]
        wet = np.isfinite(uvol) & np.isfinite(vvol)
        deepest = np.where(wet.any(axis=0), n - 1 - np.argmax(wet[::-1], axis=0), 0)
        slots = min(levels, n)
        frac = np.linspace(0.0, 1.0, slots) if slots > 1 else np.zeros(1)
        pick = np.rint(frac[:, None] * deepest[None, :]).astype(int)  # (slots, columns)
        cols = np.arange(cand.size)
        zvol = None
        if layered:
            zs = [self.heights(t, k) for k in range(n)]
            if all(z is not None for z in zs):
                zvol = np.stack([z[cand] for z in zs if z is not None])

        def clean(a, digits):
            return [None if not np.isfinite(v) else round(float(v), digits) for v in a]

        # A shallow column has fewer layers than slots: a layer already picked is not repeated.
        repeat = np.zeros_like(pick, dtype=bool)
        repeat[1:] = pick[1:] == pick[:-1]
        out_levels = []
        for s in range(slots):
            k = pick[s]
            u = np.where(repeat[s], np.nan, uvol[k, cols])
            out_levels.append(
                {
                    "layer": k.tolist(),
                    "u": clean(u, 4),
                    "v": clean(vvol[k, cols], 4),
                    "z": clean(zvol[k, cols], 2) if zvol is not None else None,
                }
            )
        return {"time": self._time(t), "element": cand.tolist(), "levels": out_levels}


# Opening a store -----------------------------------------------------------------------------


def _open(path: Path) -> xr.Dataset:
    if path.suffix == ".zarr" or path.is_dir():
        return xr.open_zarr(path, consolidated=None, decode_times=True)
    engine = "cfgrib" if path.suffix in (".grib", ".grib2") else None
    return xr.open_dataset(path, engine=engine)


def _times(ds: xr.Dataset) -> tuple[str | None, pd.DatetimeIndex]:
    for name in _TIME_DIMS:
        if name in ds.dims and name in ds.variables and ds[name].dtype.kind == "M":
            return name, pd.DatetimeIndex(ds[name].values)
    return None, pd.DatetimeIndex([])


def _snapshot(ds: xr.Dataset) -> str | None:
    td = ds.attrs.get("target_date")
    if td:
        try:
            ts = pd.Timestamp(str(td))
            return _iso(ts.tz_convert(None) if ts.tz else ts)
        except ValueError:
            return str(td)
    for name in _TIME_DIMS:
        if name in ds.variables and ds[name].ndim == 0 and ds[name].dtype.kind == "M":
            return _iso(ds[name].values)
    return None


def _wrap(lon: np.ndarray) -> np.ndarray:
    lon = np.asarray(lon, dtype=np.float64)
    return np.where(lon > 180.0, lon - 360.0, lon)


def _horizontal(ds: xr.Dataset):
    """(lon, lat, ydim, xdim) of a grid: 1-D axes or 2-D curvilinear centers, else None."""
    for lo, la in (("lon", "lat"), ("longitude", "latitude"), ("xi", "eta")):
        if (
            lo in ds.variables
            and la in ds.variables
            and ds[lo].dims == (lo,)
            and ds[la].dims == (la,)
        ):
            lon, lat = ds[lo].values, ds[la].values
            if np.nanmax(np.abs(lat)) <= 90 and np.nanmax(np.abs(_wrap(lon))) <= 180:
                return _wrap(lon), np.asarray(lat, np.float64), la, lo
    for lo, la in (("lon_rho", "lat_rho"), ("longitude", "latitude"), ("lon", "lat")):
        if lo in ds.variables and la in ds.variables and ds[lo].ndim == 2:
            ydim, xdim = ds[lo].dims
            return (
                _wrap(ds[lo].values),
                np.asarray(ds[la].values, np.float64),
                ydim,
                xdim,
            )
    return None


def _vertical_dim(ds: xr.Dataset, dims: Sequence[str]) -> str | None:
    for d in dims:
        if d in _VERTICAL_DIMS:
            return d
    return None


def _surface_first(ds: xr.Dataset, vdim: str | None, kind: str) -> list[int]:
    """Source layer indices ordered surface first."""
    if vdim is None:
        return []
    n = int(ds.sizes[vdim])
    if kind == "mesh":  # FVCOM siglay runs surface first
        return list(range(n))
    if vdim in ds.variables and ds[vdim].ndim == 1:
        c = np.asarray(ds[vdim].values, dtype=np.float64)
        positive = str(ds[vdim].attrs.get("positive", "")).lower()
        if vdim == "depth" and positive != "up" and np.nanmin(c) >= 0:
            c = -c  # depths, positive down
        if vdim == "level" and "z" in ds.data_vars:  # atmosphere levels, ground up
            return list(range(n))
        if np.all(np.isfinite(c)):
            return [int(i) for i in np.argsort(-c, kind="stable")]
    return list(range(n))


def _spec(ds, name, kind, layered, timed, label=None, units=None) -> FieldSpec:
    lab, uni = _label_units(name, ds[name] if name in ds.variables else None)
    symmetric = name in _SYMMETRIC
    return FieldSpec(
        name=name,
        source=name,
        kind=kind,
        layered=layered,
        timed=timed,
        meta=FieldMeta(
            name, label or lab, units if units is not None else uni, symmetric=symmetric
        ),
        ramp="RdBu_r" if symmetric else _RAMPS.get(name, "viridis"),
    )


def _add_speeds(view: StoreView) -> None:
    for u, v, name, label in VECTOR_PAIRS:
        if u in view.fields and v in view.fields and name not in view.fields:
            su = view.fields[u]
            water = name == "speed"
            view.fields[name] = FieldSpec(
                name=name,
                source=(u, v),
                kind=su.kind,
                layered=su.layered,
                timed=su.timed,
                meta=FieldMeta(name, label, su.meta.units or "m/s", log=water),
                ramp=WATER_SPEED_RAMP if water else "viridis",
            )


def _order_fields(view: StoreView) -> None:
    """Speeds first, then the rest alphabetically by label."""
    speeds = [n for n in ("speed", "wind_speed") if n in view.fields]
    rest = sorted(
        (n for n in view.fields if n not in speeds),
        key=lambda n: view.fields[n].meta.label,
    )
    view.fields = {n: view.fields[n] for n in speeds + rest}


def _open_mesh(view: StoreView) -> None:
    ds = view.ds
    view.store = "mesh"
    view.node_lon = _wrap(ds["node_lon"].values)
    view.node_lat = np.asarray(ds["node_lat"].values, np.float64)
    view.tri = np.asarray(ds["triangles"].values, dtype=np.int64)
    view.vdim = "siglay" if "siglay" in ds.dims else None
    view.layer_src = _surface_first(ds, view.vdim, "mesh")
    view.vertical = "sigma" if view.vdim and "h" in ds else "surface"
    n_node, n_elem = view.node_lon.size, view.tri.shape[0]
    for name, da in ds.data_vars.items():
        name = str(name)
        if name in _STRUCTURAL or da.dtype.kind not in "fiu":
            continue
        horiz = (
            "element" if "element" in da.dims else "node" if "node" in da.dims else None
        )
        if horiz is None or da.sizes.get(horiz) not in (n_elem, n_node):
            continue
        others = set(da.dims) - {horiz, view.tdim, view.vdim}
        if others:
            continue
        layered = bool(view.vdim and view.vdim in da.dims)
        timed = bool(view.tdim and view.tdim in da.dims)
        view.fields[name] = _spec(ds, name, horiz, layered, timed)
    if "ww" in view.fields:
        view.fields["ww"].meta = replace(
            view.fields["ww"].meta, label="Vertical velocity, up positive"
        )


def _open_grid(view: StoreView, detail: str) -> bool:
    ds = view.ds
    hv = _horizontal(ds)
    if hv is None:
        return False
    lon, lat, ydim, xdim = hv
    ny, nx = int(ds.sizes[ydim]), int(ds.sizes[xdim])
    view.store = "grid"
    view.ydim, view.xdim = ydim, xdim
    view.source_shape = (ny, nx)

    # Fields: every numeric variable on (y, x) with at most time and one vertical dimension.
    candidates = []
    for name, da in ds.data_vars.items():
        name = str(name)
        if name in _STRUCTURAL or da.dtype.kind not in "fiu":
            continue
        if ydim not in da.dims or xdim not in da.dims:
            continue
        rest = [d for d in da.dims if d not in (ydim, xdim)]
        vd = [d for d in rest if d in _VERTICAL_DIMS]
        bad = [d for d in rest if d not in vd and d != view.tdim and da.sizes[d] != 1]
        if bad or len(vd) > 1:
            continue
        candidates.append((name, vd[0] if vd else None))
    vdims = [vd for _, vd in candidates if vd]
    view.vdim = max(set(vdims), key=vdims.count) if vdims else None
    if view.vdim == "level" and "z" in ds.data_vars:
        candidates = [(n, vd) for n, vd in candidates if n not in ("z",)]
    view.layer_src = _surface_first(ds, view.vdim, "grid")
    if view.vdim is None:
        view.vertical = "surface"
    elif view.vdim in ("z", "depth") or (view.vdim == "level" and "z" in ds.data_vars):
        view.vertical = "z"
        if view.vdim == "level":
            view.medium = "atmosphere"
    elif (
        view.vdim in ("s_rho", "siglay", "sigma")
        and "h" in ds.data_vars
        and ds["h"].ndim == 2
    ):
        view.vertical = "sigma"
    else:
        view.vertical = "sigma-index"

    # Cells: water cells only (the store's mask, else where the first field has a value).
    stride = max(1, math.ceil(math.sqrt(ny * nx / MAX_CELLS[detail])))
    view.stride = stride
    lon_s = lon[::stride] if lon.ndim == 1 else lon[::stride, ::stride]
    lat_s = lat[::stride] if lat.ndim == 1 else lat[::stride, ::stride]
    if lon.ndim == 1:
        gy, gx = lat_s.size, lon_s.size
    else:
        gy, gx = lon_s.shape
    view.grid_shape = (gy, gx)
    if gy < 2 or gx < 2:
        return False
    mask = None
    for mname in ("mask", "wet"):
        if mname in ds.data_vars and ds[mname].dims == (ydim, xdim):
            mask = np.asarray(ds[mname].values)[::stride, ::stride] > 0
            break
    if mask is None and candidates:
        first = view._isel(ds[candidates[0][0]], 0, 0)
        a = np.asarray(first.transpose(ydim, xdim).values)[::stride, ::stride]
        mask = np.isfinite(a)
    geo = structured(
        lon_s,
        lat_s,
        mask,
        times=pd.DatetimeIndex([pd.Timestamp("2000-01-01")]),
        fields={},
    )
    view.node_lon, view.node_lat, view.tri = geo.node_lon, geo.node_lat, geo.tri
    view.elem_cell = geo.source_index
    for name, own_vdim in candidates:
        if own_vdim is not None and own_vdim != view.vdim:
            continue  # on a second vertical axis the store's layers do not describe
        timed = bool(view.tdim and view.tdim in ds[name].dims)
        view.fields[name] = _spec(ds, name, "cell", own_vdim is not None, timed)
    return True


def _open_points(view: StoreView) -> bool:
    ds = view.ds
    if not ("mouth_lon" in ds and "mouth_lat" in ds):
        return False
    view.store = "points"
    names = []
    try:
        import json

        names = json.loads(ds.attrs.get("river_names", "[]"))
    except Exception:
        names = []
    lon, lat = _wrap(ds["mouth_lon"].values), ds["mouth_lat"].values
    view.sites = [
        {
            "lon": float(lon[i]),
            "lat": float(lat[i]),
            "name": names[i] if i < len(names) else f"site {i}",
        }
        for i in range(lon.size)
    ]
    for name, da in ds.data_vars.items():
        name = str(name)
        if (
            name in _STRUCTURAL
            or da.dtype.kind not in "fiu"
            or "river" not in da.dims
            or set(da.dims) - {"river", view.tdim}
        ):
            continue
        timed = bool(view.tdim and view.tdim in da.dims)
        view.fields[name] = _spec(ds, name, "site", False, timed)
    return True


def open_view(dataset_id: str, path: Path, detail: str = "2d") -> StoreView:
    """Open a store for drawing. Raises ValueError when the store has nothing to draw."""
    ds = _open(path)
    tdim, times = _times(ds)
    view = StoreView(
        dataset_id=dataset_id,
        path=path,
        ds=ds,
        store="",
        times=times,
        snapshot=None if len(times) else _snapshot(ds),
        tdim=tdim,
    )
    if "triangles" in ds and "node_lon" in ds:
        _open_mesh(view)
    elif not _open_grid(view, detail) and not _open_points(view):
        raise ValueError(
            "this store has no longitude and latitude cells or sites the viewer can draw"
        )
    _add_speeds(view)
    _order_fields(view)
    if not view.fields:
        raise ValueError("this store holds no field the viewer can draw")
    return view


# Cache of open views ------------------------------------------------------------------------

_VIEWS: OrderedDict[tuple, StoreView] = OrderedDict()
_VIEWS_LOCK = threading.Lock()
_VIEWS_MAX = 6


def get_view(dataset_id: str, path: Path, detail: str, key: Any) -> StoreView:
    """An open view, reused while the store is unchanged (`key`, from the inventory)."""
    ck = (dataset_id, detail, key)
    with _VIEWS_LOCK:
        hit = _VIEWS.get(ck)
        if hit is not None:
            _VIEWS.move_to_end(ck)
            return hit
    view = open_view(dataset_id, path, detail)
    with _VIEWS_LOCK:
        # One fixed scale per field whatever the detail, so the map and the 3D view color a
        # value alike: the views of one store share their scale table.
        sibling = next(
            (v for k, v in _VIEWS.items() if k[0] == dataset_id and k[2] == key), None
        )
        if sibling is not None:
            view._scales = sibling._scales
        _VIEWS[ck] = view
        while len(_VIEWS) > _VIEWS_MAX:
            _VIEWS.popitem(last=False)
    return view


def clear_views() -> None:
    with _VIEWS_LOCK:
        _VIEWS.clear()


__all__ = [
    "FieldSpec",
    "StoreView",
    "open_view",
    "get_view",
    "clear_views",
    "resolve_store",
    "nice_ticks",
]
