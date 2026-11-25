"""Doppio (ROMS, Rutgers) full-depth subsets on the model's own grid.

Doppio is a ROMS configuration for the Mid-Atlantic Bight and the Gulf of Maine (about 7 km, 40
terrain-following s-levels) run by the Rutgers Ocean Modeling Group for MARACOOS (Lopez et al.,
2020; Wilkin et al., 2022). Rutgers serves two hourly OPeNDAP aggregations of it:

- the data-assimilative reanalysis DopAnV3R3-ini2007, 2007-01-02 01:00 to 2024-12-31 00:00
  (`REANALYSIS_URL`), and
- the real-time system, 2017-11-02 01:00 to about two days ahead (`REALTIME_URL`, the FMRC "Best"
  time series).

`read_volume` and `volume` return the same fields, names and units as the NECOFS volume
(`forcingkit.fetchers.necofs.volume`, schema volume-v1): `u`, `v`, `ww`, `temp`, `salinity`,
`zeta`, `h`, `siglay` and `siglev`, with the layers surface first and the depth of a layer center
`siglay * (h + zeta) + zeta`. The horizontal layout is the model's curvilinear grid rather than a
mesh: dims (eta_rho, xi_rho) for the index rectangle around the box, with u and v moved from the
C-grid faces to the cell centers and rotated to east and north.
"""

import hashlib
import json
import logging
import os
import shutil
import warnings
from typing import Sequence

import numpy as np
import pandas as pd
import xarray as xr

from forcingkit import settings
from forcingkit.fetchers import necofs

logger = logging.getLogger(__name__)

_TDS = "https://tds.marine.rutgers.edu/thredds/dodsC/roms/doppio"
REANALYSIS_URL = f"{_TDS}/DopAnV3R3-ini2007_da/his"
REALTIME_URL = f"{_TDS}/2017_da/his/History_Best"

# Hourly records each aggregation held when checked on 2026-11-17 (the reanalysis has no gaps).
REANALYSIS_START = pd.Timestamp("2007-01-02 01:00")
REANALYSIS_END = pd.Timestamp("2024-12-31 00:00")
REALTIME_START = pd.Timestamp("2017-11-02 01:00")

# The extent of the rho points, degrees.
DOMAIN_BBOX = [-80.52, 32.24, -59.69, 46.62]

SOURCE = "Doppio ROMS, Rutgers University Ocean Modeling Group for MARACOOS"
REANALYSIS_CITATION = (
    "Wilkin, J. and J. Levin (2022). Outputs from a Regional Ocean Modeling System (ROMS) data "
    "assimilative reanalysis (version DopAnV3R3-ini2007) of ocean circulation in the Mid-Atlantic "
    "Bight and Gulf of Maine for 2007-2021. SEANOE. https://doi.org/10.17882/89673"
)

# Same schema and fields as the NECOFS volume; a change to the layout here bumps it there too.
VOLUME_SCHEMA = "volume-v1"
VOLUME_FIELDS = ("u", "v", "ww", "temp", "salinity", "zeta")

USER_AGENT = "forcingkit (+https://github.com/lhzn-io/forcingkit)"


def get_metadata() -> dict:
    return {
        "id": "doppio",
        "name": "Doppio ROMS (Rutgers, MARACOOS)",
        "resolution_approx_m": 7000.0,
        "type_desc": "Curvilinear Arakawa C-grid, 40 s-levels",
        "domain_bbox": list(DOMAIN_BBOX),
        # Delivered only when a request names it (`donor: "doppio"`); the automatic ranking
        # leaves it out, so no existing request changes donor.
        "auto_rank": False,
    }


def supports_bbox(bbox: Sequence[float]) -> bool:
    min_lon, min_lat, max_lon, max_lat = bbox
    w, s, e, n = DOMAIN_BBOX
    return not (max_lat < s or min_lat > n or max_lon < w or min_lon > e)


def _utc_naive(t) -> pd.Timestamp:
    t = pd.Timestamp(t)
    return t.tz_convert("UTC").tz_localize(None) if t.tzinfo else t


def product_url(start, end) -> str:
    """The aggregation that holds `start` to `end` whole: the reanalysis where it can, else the
    real-time system."""
    t0, t1 = _utc_naive(start), _utc_naive(end)
    if REANALYSIS_START <= t0 and t1 <= REANALYSIS_END:
        return REANALYSIS_URL
    if t0 >= REALTIME_START:
        return REALTIME_URL
    raise ValueError(
        f"Doppio holds {REANALYSIS_START} to {REANALYSIS_END} (reanalysis) and from "
        f"{REALTIME_START} (real time); no one aggregation holds {t0} to {t1}"
    )


def doppio_product(url: str) -> str:
    """Which Doppio product a URL belongs to, for provenance."""
    if url == REANALYSIS_URL:
        return "reanalysis DopAnV3R3-ini2007 (hourly history)"
    if url == REALTIME_URL:
        return "real-time forecast system (FMRC best time series, hourly history)"
    return "Doppio file"


def _open(url: str) -> xr.Dataset:
    """`url` opened lazily, nothing decoded (fill values are masked by hand, so a land column
    reads as NaN): a remote aggregation with xarray's pydap engine and a session that names this
    package, a local file (a test fixture, a downloaded subset) with the default engine."""
    kwargs: dict = {"decode_times": False, "mask_and_scale": False}
    if url.startswith(("http://", "https://")):
        import requests

        session = requests.Session()
        session.headers["User-Agent"] = USER_AGENT
        kwargs.update(engine="pydap", session=session)
        url = url.replace("https://", "dap2://").replace("http://", "dap2://")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return xr.open_dataset(url, **kwargs)


def _times(ds: xr.Dataset) -> tuple[str, pd.DatetimeIndex]:
    """The time dimension of the 3D fields (`ocean_time` in the reanalysis, `time` in the FMRC
    aggregation) and its values as naive UTC; a missing time (NaN) becomes NaT."""
    dim = str(ds["temp"].dims[0])
    var = ds[dim]
    values = np.asarray(var.values, dtype=np.float64)
    units = var.attrs["units"].replace(" UTC", "")
    unit, origin = units.split(" since ")
    step = {"seconds": "s", "hours": "h", "days": "D"}[unit.strip()]
    times = pd.Timestamp(origin.strip()) + pd.to_timedelta(values, unit=step)
    return dim, pd.DatetimeIndex(times).round("s")


def _volume_selection(
    variables: Sequence[str] | None, layers: Sequence[int] | None, n_layers: int
) -> tuple[tuple[str, ...], list[int]]:
    """`variables` checked against VOLUME_FIELDS, in their order there; `layers` as sorted, unique
    surface-first indices (negative ones count from the bottom). As in the NECOFS volume."""
    if variables is None:
        names: tuple[str, ...] = VOLUME_FIELDS
    else:
        unknown = sorted(set(variables) - set(VOLUME_FIELDS))
        if unknown or not variables:
            raise ValueError(
                f"variables must be among {VOLUME_FIELDS}; got {list(variables)}"
            )
        names = tuple(v for v in VOLUME_FIELDS if v in set(variables))
    if layers is None:
        return names, list(range(n_layers))
    ks = [int(k) for k in layers]
    if not ks:
        raise ValueError("layers is empty")
    if any(k >= n_layers or k < -n_layers for k in ks):
        raise ValueError(f"layers must lie in -{n_layers}..{n_layers - 1}; got {ks}")
    return names, sorted({k % n_layers for k in ks})


def _read(da: xr.DataArray, **slabs) -> np.ndarray:
    """One hyperslab of `da` as float32, with the ROMS fill value (1e37) as NaN."""
    a = np.asarray(da.isel(slabs).values, dtype=np.float32)
    fill = da.attrs.get("_FillValue", 1e37)
    a[~np.isfinite(a) | (np.abs(a) >= 0.5 * abs(float(fill)))] = np.nan
    return a


def stretching(
    s: np.ndarray, cs: np.ndarray, h: np.ndarray, hc: float, vtransform: int
) -> np.ndarray:
    """ROMS's terrain-following coordinate as the fraction of the water column NECOFS calls sigma:
    (n_s, ny, nx) with z = sigma * (h + zeta) + zeta. Exact for Vtransform 2, which Doppio uses;
    Vtransform 1 is not of this form."""
    if vtransform != 2:
        raise ValueError(f"Vtransform {vtransform} is not supported (Doppio uses 2)")
    h = np.asarray(h, dtype=np.float64)[None]
    s = np.asarray(s, dtype=np.float64)[:, None, None]
    cs = np.asarray(cs, dtype=np.float64)[:, None, None]
    return (hc * s + h * cs) / (hc + h)


def _faces_to_centers(a: np.ndarray, axis: int, lo: int, n_centers: int) -> np.ndarray:
    """Values on C-grid faces averaged to the cell centers between them, along `axis`.

    `a` holds faces lo, lo + 1, ... (face k lies between centers k and k + 1); center c takes the
    mean of faces c - 1 and c where both exist, and the one face it has at the grid's edge. Land
    faces (NaN) count as zero flow, as ROMS's own u2rho does."""
    a = np.nan_to_num(a, nan=0.0)
    n_faces = a.shape[axis]
    out = []
    for c in range(n_centers):
        idx = [k - lo for k in (c - 1, c) if 0 <= k - lo < n_faces]
        if not idx:
            raise ValueError("a center has no face in the hyperslab")
        out.append(np.take(a, idx, axis=axis).mean(axis=axis))
    return np.stack(out, axis=axis)


VOLUME_VARIABLE_ATTRS = {
    "u": {"units": "m s-1", "long_name": "eastward current at cell layer centers"},
    "v": {"units": "m s-1", "long_name": "northward current at cell layer centers"},
    "ww": {
        "units": "m s-1",
        "long_name": "upward velocity in z at cell layer centers (ROMS w, mean of the "
        "interfaces above and below)",
    },
    "temp": {
        "units": "degree_Celsius",
        "long_name": "potential temperature at cell layer centers",
    },
    "salinity": {"units": "1e-3", "long_name": "salinity at cell layer centers"},
    "zeta": {
        "units": "m",
        "long_name": "free-surface elevation relative to the model mean sea level",
        "positive": "up",
    },
    "h": {
        "units": "m",
        "long_name": "depth below the model mean sea level at cell centers",
        "positive": "down",
    },
    "siglay": {
        "long_name": "fraction of the water column at layer centers, surface first "
        "(ROMS s-coordinate, Vtransform 2)",
        "positive": "up",
    },
    "siglev": {
        "long_name": "fraction of the water column at layer interfaces, surface first",
        "positive": "up",
    },
    "lon": {"units": "degree_east", "long_name": "longitude of cell centers"},
    "lat": {"units": "degree_north", "long_name": "latitude of cell centers"},
    "mask": {"long_name": "cell is water"},
    "angle": {"units": "radian", "long_name": "angle from east to the grid's xi axis"},
    "in_bbox": {"long_name": "cell center lies inside the requested bbox"},
}


def _subset_volume(
    ds: xr.Dataset,
    bbox: Sequence[float],
    wanted: pd.DatetimeIndex,
    url: str,
    variables: Sequence[str] | None = None,
    layers: Sequence[int] | None = None,
) -> xr.Dataset:
    """The full-depth subset of an opened Doppio aggregation (see `read_volume`)."""
    tdim, times = _times(ds)
    lon = np.asarray(ds["lon_rho"].values, dtype=np.float64)
    lat = np.asarray(ds["lat_rho"].values, dtype=np.float64)
    ny_all, nx_all = lon.shape
    min_lon, min_lat, max_lon, max_lat = bbox
    inside = (lon >= min_lon) & (lon <= max_lon) & (lat >= min_lat) & (lat <= max_lat)
    if not inside.any():
        raise ValueError(f"No Doppio cells inside {list(bbox)}")
    jj, ii = np.nonzero(inside)
    j0, j1, i0, i1 = int(jj.min()), int(jj.max()), int(ii.min()), int(ii.max())
    ny, nx = j1 - j0 + 1, i1 - i0 + 1
    rows, cols = slice(j0, j1 + 1), slice(i0, i1 + 1)

    n_s = int(ds.sizes["s_rho"])
    names, ks = _volume_selection(variables, layers, n_s)
    # ROMS counts levels from the bottom; the surface-first layer k is s_rho n_s - 1 - k.
    s_idx = np.array([n_s - 1 - k for k in ks])
    s_lo, s_hi = int(s_idx.min()), int(s_idx.max())
    pick = s_idx - s_lo  # within the read range, in surface-first order

    t_idx = times.get_indexer(wanted)
    keep = t_idx >= 0
    if not keep.any():
        raise RuntimeError(
            f"{url} holds {times.min()} to {times.max()}, none of {wanted[0]} to {wanted[-1]}"
        )
    t_idx, wanted = t_idx[keep], wanted[keep]
    t0, t1 = int(t_idx.min()), int(t_idx.max())
    t_rows = t_idx - t0
    tsl = slice(t0, t1 + 1)

    def static(name):
        return np.asarray(ds[name].values, dtype=np.float64)

    h = static("h")[rows, cols]
    mask = static("mask_rho")[rows, cols] > 0.5
    angle = static("angle")[rows, cols]
    hc = float(ds["hc"].values)
    vtransform = int(ds["Vtransform"].values)
    sig_r = stretching(static("s_rho"), static("Cs_r"), h, hc, vtransform)[::-1]
    sig_w = stretching(static("s_w"), static("Cs_w"), h, hc, vtransform)[::-1]

    def layered(name):
        a = _read(
            ds[name],
            **{tdim: tsl, "s_rho": slice(s_lo, s_hi + 1)},
            eta_rho=rows,
            xi_rho=cols,
        )
        return a[t_rows][:, pick]

    def land(a):
        return np.where(mask, a, np.nan).astype(np.float32)

    data = {}
    if "u" in names or "v" in names:
        # u faces i - 1 .. i1 and v faces j - 1 .. j1 surround the rectangle's centers.
        ua, va = max(i0 - 1, 0), max(j0 - 1, 0)
        ub, vb = min(i1, nx_all - 2), min(j1, ny_all - 2)
        sl = {tdim: tsl, "s_rho": slice(s_lo, s_hi + 1)}
        u_f = _read(ds["u"], **sl, eta_u=rows, xi_u=slice(ua, ub + 1))[t_rows][:, pick]
        v_f = _read(ds["v"], **sl, eta_v=slice(va, vb + 1), xi_v=cols)[t_rows][:, pick]
        u_xi = _faces_to_centers(u_f, -1, ua - i0, nx)
        v_eta = _faces_to_centers(v_f, -2, va - j0, ny)
        cos, sin = np.cos(angle), np.sin(angle)
        east = u_xi * cos - v_eta * sin
        north = u_xi * sin + v_eta * cos
        if "u" in names:
            data["u"] = land(east)
        if "v" in names:
            data["v"] = land(north)
    if "ww" in names:
        w = _read(
            ds["w"],
            **{tdim: tsl, "s_w": slice(s_lo, s_hi + 2)},
            eta_rho=rows,
            xi_rho=cols,
        )[t_rows]
        data["ww"] = land(0.5 * (w[:, :-1] + w[:, 1:])[:, pick])
    if "temp" in names:
        data["temp"] = land(layered("temp"))
    if "salinity" in names:
        data["salinity"] = land(layered("salt"))
    if "zeta" in names:
        z = _read(ds["zeta"], **{tdim: tsl}, eta_rho=rows, xi_rho=cols)[t_rows]
        data["zeta"] = land(z)

    lay, lev, ey, xi = "siglay", "siglev", "eta_rho", "xi_rho"
    fields = {
        name: (
            (("time", ey, xi) if name == "zeta" else ("time", lay, ey, xi)),
            data[name],
        )
        for name in names
    }
    out = xr.Dataset(
        {
            **fields,
            "h": ((ey, xi), h.astype(np.float32)),
            "siglay": ((lay, ey, xi), sig_r[ks].astype(np.float32)),
            "siglev": ((lev, ey, xi), sig_w.astype(np.float32)),
            "lon": ((ey, xi), lon[rows, cols]),
            "lat": ((ey, xi), lat[rows, cols]),
            "mask": ((ey, xi), mask),
            "angle": ((ey, xi), angle),
            "in_bbox": ((ey, xi), inside[rows, cols]),
        },
        coords={
            "time": wanted,
            ey: np.arange(j0, j1 + 1, dtype=np.int32),
            xi: np.arange(i0, i1 + 1, dtype=np.int32),
        },
    )
    # 2D auxiliary coordinates (CF), so viewers place the curvilinear cells correctly.
    out = out.set_coords(["lon", "lat"])
    if layers is not None:
        out = out.assign_coords(layer=(lay, np.asarray(ks, dtype=np.int32)))
        out["layer"].attrs["long_name"] = (
            "zero-based layer index in the full model, surface first"
        )
    out[ey].attrs["long_name"] = "zero-based row index in the full Doppio grid"
    out[xi].attrs["long_name"] = "zero-based column index in the full Doppio grid"
    for name, attrs in VOLUME_VARIABLE_ATTRS.items():
        if name in out:
            out[name].attrs.update(attrs)
    out.attrs.update(
        {
            "type": "Doppio ROMS full-depth subset on the model grid",
            "source": SOURCE,
            "product": doppio_product(url),
            "url": url,
            "requested_bbox": [float(b) for b in bbox],
            "variables": list(names),
            "schema": VOLUME_SCHEMA,
            "vertical_datum": "model mean sea level (the source metadata states no datum)",
            "citation": REANALYSIS_CITATION,
        }
    )
    logger.info(
        f"Doppio volume: {ny} x {nx} cells ({int(inside.sum())} in the box), {len(ks)} of "
        f"{n_s} layers, {', '.join(names)}, {len(wanted)} hours from {url}"
    )
    return out


def read_volume(
    bbox: Sequence[float],
    start_time: str | pd.Timestamp,
    hours: int,
    dap_url: str | None = None,
    variables: Sequence[str] | None = None,
    layers: Sequence[int] | None = None,
) -> xr.Dataset:
    """Every s-level on Doppio's own grid (no horizontal regridding, no vertical conversion) for
    the index rectangle around the cells whose centers lie in `bbox` = [min_lon, min_lat,
    max_lon, max_lat], hourly from `start_time` for `hours` (hours the aggregation does not hold
    are left out).

    Returns dims (time, siglay, siglev, eta_rho, xi_rho), with `eta_rho` and `xi_rho` holding
    the cells' indices in the full grid:

    - `u`, `v` (time, siglay, eta_rho, xi_rho): eastward and northward velocity at the cell
      centers, from the C-grid faces (land faces as zero flow) rotated by the grid `angle`.
    - `ww`: ROMS `w` (upward, in z) averaged from the interfaces to the layer centers.
    - `temp`, `salinity` (time, siglay, eta_rho, xi_rho); `zeta` (time, eta_rho, xi_rho),
      relative to the model mean sea level (no datum in the source metadata); `h` (eta_rho, xi_rho), positive down.
    - `siglay` (siglay, eta_rho, xi_rho), `siglev` (siglev, eta_rho, xi_rho): the fraction of
      the water column, surface first, so the depth of a layer center is
      `siglay * (h + zeta) + zeta`, as in the NECOFS volume.
    - `lon`, `lat`, `mask` (water), `angle`, and `in_bbox` (the center lies in `bbox`; the
      rectangle's corners may not, since the grid is rotated about 37 degrees).

    Land cells are NaN in the time-varying fields. `variables` and `layers` select as in the
    NECOFS volume. Reads `product_url` for the window unless `dap_url` names an aggregation.
    """
    start = _utc_naive(start_time).floor("h")
    wanted = pd.date_range(start, periods=hours, freq="1h")
    url = dap_url or product_url(wanted[0], wanted[-1])
    ds = _open(url)
    return _subset_volume(ds, bbox, wanted, url, variables, layers)


def volume_cache_key(
    bbox: Sequence[float],
    start: pd.Timestamp,
    end: pd.Timestamp,
    url: str,
    variables: Sequence[str] | None = None,
    layers: Sequence[int] | None = None,
) -> str:
    """Cache key of a `volume` request: the schema, the aggregation, the box to 1e-4 degree, the
    window and any selection."""
    fields: dict = {
        "schema": VOLUME_SCHEMA,
        "url": url,
        "bbox": [round(float(b), 4) for b in bbox],
        "start": start.isoformat(),
        "end": end.isoformat(),
    }
    if variables is not None:
        fields["variables"] = sorted(set(variables))
    if layers is not None:
        fields["layers"] = sorted(set(int(k) for k in layers))
    key = json.dumps(fields, sort_keys=True)
    return "doppio_volume_" + hashlib.sha256(key.encode()).hexdigest()[:12]


def volume(
    bbox: Sequence[float],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    cache: bool = True,
    reader=read_volume,
    variables: Sequence[str] | None = None,
    layers: Sequence[int] | None = None,
    dap_url: str | None = None,
) -> xr.Dataset:
    """`read_volume` for every hour from `start` to `end` inclusive (UTC; naive times are UTC).

    Raises rather than returning a short window: every requested hour must be found. With
    `cache`, the result is stored as a zarr store under `cache_dir("doppio", "volume")` and later
    calls with the same box, window, aggregation and selection read it from there. `reader` is
    for tests.
    """
    _volume_selection(variables, None, 1)  # a bad name fails before any read
    t0, t1 = _utc_naive(start).floor("h"), _utc_naive(end)
    if t1 < t0:
        raise ValueError(f"end {t1} precedes start {t0}")
    wanted = pd.date_range(t0, t1, freq="1h")
    url = dap_url or product_url(wanted[0], wanted[-1])

    path = None
    if cache:
        key = volume_cache_key(bbox, t0, t1, url, variables, layers)
        path = settings.cache_dir("doppio", "volume", key + ".zarr")
        if os.path.isdir(path):
            logger.info(f"Doppio volume from cache {path}")
            return xr.open_zarr(path, consolidated=True).load()

    out = reader(
        bbox, wanted[0], len(wanted), dap_url=url, variables=variables, layers=layers
    )
    missing = wanted.difference(pd.DatetimeIndex(out["time"].values))
    if len(missing):
        raise RuntimeError(
            f"Doppio lacks {len(missing)} of {len(wanted)} hours from {t0} to {t1} "
            f"(first missing {missing[0]})"
        )

    if path is not None:
        tmp = path + ".partial"
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        out.to_zarr(tmp, mode="w", consolidated=True, zarr_format=2)
        os.replace(tmp, path)
        logger.info(f"Doppio volume cached at {path}")
    return out


# Parent-ocean delivery (schema z-v3), the NECOFS layout: the same record fields and units, so a
# store from either donor reads the same way.
PARENT_RECORD_DIMS = dict(necofs.PARENT_RECORD_DIMS)
PARENT_UNITS = dict(necofs.PARENT_UNITS)

# w is ROMS's own (upward velocity in z), recorded in the store as ``w_source``.
W_SOURCE = "model"

# The regular grid the parent is interpolated onto, degrees: about 2 km, finer than Doppio's
# 7 km cells so a track box of a few tenths of a degree still spans many points.
PARENT_SPACING_DEG = 0.02

# Hours read from the aggregation per request; each read covers the box for this many hours.
_READ_HOURS = 24

# Beyond the padded box, the native cells read so every target point lies inside the
# triangulation of read cell centers (one Doppio cell is up to about 0.09 degree across).
_READ_MARGIN_DEG = 0.15


def iter_parent(
    start_date: str,
    duration_hours: int,
    bbox: list[float],
    pad_cells: int = 3,
    vertical_spacing_m: float = 2.0,
    reader=None,
):
    """Doppio as a parent ocean, one hourly record at a time, in the NECOFS layout (z-v3).

    Yields `("static", Dataset)` once (lon, lat, z, z_face, h, mask and the store attributes),
    then `("record", time, {u, v, w, temp, salt, zeta})` for each of `duration_hours` hours from
    `start_date`. The fields are read with `read_volume` (u and v already moved to the cell
    centers and rotated to east and north; w is ROMS's w averaged to the layer centers),
    interpolated linearly from the curvilinear cell centers onto a regular grid every
    `PARENT_SPACING_DEG` covering `bbox` plus `pad_cells` grid steps on every side, then from the
    s-levels onto fixed z levels every `vertical_spacing_m` meters, bottom to top, with each
    layer at z = siglay * (h + zeta) + zeta for its column (`necofs.sigma_to_z`). A target point
    in a triangle with a land cell center, outside the read cells, or below the sea floor is NaN.

    The window must lie within one aggregation (`product_url`). Raises rather than returning a
    short record: an unreachable aggregation or a missing hour ends the delivery with an error.
    `reader` (default `read_volume`) is for tests.
    """
    from scipy.spatial import Delaunay

    reader = reader or read_volume
    if not supports_bbox(bbox):
        raise ValueError(f"Bounding box {list(bbox)} outside the Doppio domain.")
    if duration_hours < 1:
        raise ValueError(f"duration_hours must be at least 1, got {duration_hours}")
    min_lon, min_lat, max_lon, max_lat = (float(b) for b in bbox)
    wanted = pd.date_range(
        _utc_naive(start_date).floor("h"), periods=duration_hours, freq="1h"
    )
    url = product_url(wanted[0], wanted[-1])

    d = PARENT_SPACING_DEG
    pad = pad_cells * d
    lon_axis = np.arange(min_lon - pad, max_lon + pad + d / 2, d)
    lat_axis = np.arange(min_lat - pad, max_lat + pad + d / 2, d)
    lon_grid, lat_grid = np.meshgrid(lon_axis, lat_axis)
    target_pts = np.column_stack((lon_grid.ravel(), lat_grid.ravel()))
    shape = lon_grid.shape
    margin = pad + _READ_MARGIN_DEG
    read_box = [min_lon - margin, min_lat - margin, max_lon + margin, max_lat + margin]

    interp = None
    h_grid = siglay_grid = z_centres = None
    for first in range(0, duration_hours, _READ_HOURS):
        hours = wanted[first : first + _READ_HOURS]
        vol = reader(
            read_box,
            hours[0],
            len(hours),
            dap_url=url,
            variables=("u", "v", "ww", "temp", "salinity", "zeta"),
        )
        got = pd.DatetimeIndex(vol["time"].values)
        missing = hours.difference(got)
        if len(missing):
            raise RuntimeError(
                f"{url} lacks {len(missing)} of {len(hours)} hours from {hours[0]} "
                f"(first missing {missing[0]})"
            )
        vol = vol.sel(time=hours)

        if interp is None:
            logger.info(
                "Building Doppio parent interpolation weights and vertical grid..."
            )
            cells = np.column_stack(
                (vol["lon"].values.ravel(), vol["lat"].values.ravel())
            )
            interp = necofs.Barycentric(Delaunay(cells), target_pts, shape)
            water = np.asarray(vol["mask"].values, dtype=bool)
            h_grid = interp(np.where(water, vol["h"].values, np.nan).ravel())
            if not np.any(np.isfinite(h_grid)):
                raise ValueError(f"No Doppio ocean inside {list(bbox)}")
            n_lay = vol.sizes["siglay"]
            siglay_grid = interp(np.asarray(vol["siglay"].values).reshape(n_lay, -1))
            z_centres, z_faces = necofs.z_levels_for_depth(
                float(np.nanmax(h_grid)), vertical_spacing_m
            )
            static = necofs._parent_static(
                lon_axis,
                lat_axis,
                z_centres,
                z_faces,
                h_grid,
                {
                    "type": "Doppio ROMS parent ocean",
                    "doppio_source": SOURCE,
                    "doppio_product": doppio_product(url),
                    "doppio_url": url,
                    "citation": REANALYSIS_CITATION
                    if url == REANALYSIS_URL
                    else SOURCE,
                    "schema": necofs.OBC_SCHEMA,
                    "w_source": W_SOURCE,
                    "pad_cells": pad_cells,
                    "vertical_spacing_m": vertical_spacing_m,
                    "horizontal_spacing_deg": d,
                    "requested_bbox": [float(b) for b in bbox],
                    "vertical_datum": "model mean sea level (the source metadata states "
                    "no datum)",
                },
            )
            static["mask"].attrs["long_name"] = (
                "1 where Doppio has ocean at every surrounding cell center, 0 elsewhere"
            )
            yield ("static", static)
        assert h_grid is not None and siglay_grid is not None and z_centres is not None

        def columns(name, i):
            a = np.asarray(vol[name].values[i], dtype=np.float64)
            return a.reshape(a.shape[0], -1)

        for i, t in enumerate(hours):
            zeta_t = interp(np.asarray(vol["zeta"].values[i], dtype=np.float64).ravel())
            z_bottom = np.where(np.isfinite(zeta_t), -h_grid, np.nan)
            z_layers = siglay_grid * (h_grid + zeta_t)[None] + zeta_t[None]
            fields = {
                "u": interp(columns("u", i)),
                "v": interp(columns("v", i)),
                "w": interp(columns("ww", i)),
                "temp": interp(columns("temp", i)),
                "salt": interp(columns("salinity", i)),
            }
            out = {
                k: necofs.sigma_to_z(v, z_layers, z_bottom, z_centres).astype(
                    np.float32
                )
                for k, v in fields.items()
            }
            out["zeta"] = zeta_t.astype(np.float32)
            yield ("record", t, out)
