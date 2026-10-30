"""Forcing interpolated along a trajectory: a tagged animal, a vessel, a drifter.

The trajectory's bounding box (plus a margin) and time window (whole hours, plus an hour each side)
are fetched once as gridded stores, then each requested variable is interpolated linearly in time,
latitude, longitude and, for 3-D ocean fields, z. The result is a CF discrete sampling geometry of
feature type `trajectory`: one value per point and variable on an `obs` dimension.

Sampling density is the caller's choice. The gridded sources resolve hours and kilometres (HYCOM
is 3-hourly at 0.04 to 0.08 degrees), so a trajectory logged at many hertz loses nothing when
thinned to about one point a minute before it is passed here.

This suits trajectories whose bounding box is one modest store, such as a few hours to a few days
in one region. A track spanning ocean basins needs segmenting, which a later server-side
interpolation endpoint is meant to do.
"""

from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from .schema import wrap_lon

if TYPE_CHECKING:
    from .client import AsyncClient, Client

OBS = "obs"


def _prepare(lon, lat, time, z):
    lon = wrap_lon(np.atleast_1d(lon))
    lat = np.atleast_1d(np.asarray(lat, dtype=np.float64))
    t = pd.to_datetime(np.atleast_1d(np.asarray(time)), utc=True).tz_convert(None)
    z = (
        np.zeros_like(lon)
        if z is None
        else np.atleast_1d(np.asarray(z, dtype=np.float64))
    )
    if z.size == 1:
        z = np.full_like(lon, float(z[0]))
    if not (lon.shape == lat.shape == t.shape == z.shape) or lon.ndim != 1:
        raise ValueError(
            "lon, lat, time and z must be 1-D and of one length, got "
            f"{lon.shape}, {lat.shape}, {t.shape}, {z.shape}"
        )
    if len(lon) == 0:
        raise ValueError("the trajectory has no points")
    if np.any(z > 0):
        raise ValueError("z is metres positive up: points below the surface have z < 0")
    return lon, lat, pd.DatetimeIndex(t), z


def trajectory_window(lon, lat, time, margin_deg: float = 0.1):
    """(bbox, start, hours) of the stores that cover a trajectory."""
    t = pd.DatetimeIndex(time)
    bbox = (
        float(np.min(lon)) - margin_deg,
        float(np.min(lat)) - margin_deg,
        float(np.max(lon)) + margin_deg,
        float(np.max(lat)) + margin_deg,
    )
    start = t.min().floor("h") - pd.Timedelta(hours=1)
    end = t.max().ceil("h") + pd.Timedelta(hours=1)
    return bbox, start, int((end - start) / pd.Timedelta(hours=1))


def _bracket(da: xr.DataArray, dim: str, values: np.ndarray) -> xr.DataArray:
    """`da` cut to the cells around `values` on `dim`, so only what is needed is read."""
    axis = da[dim].values
    if axis.size < 2:
        return da
    if axis[0] > axis[-1]:
        da = da.sortby(dim)
        axis = da[dim].values
    lo = max(int(np.searchsorted(axis, np.min(values), side="right")) - 1, 0)
    hi = min(int(np.searchsorted(axis, np.max(values), side="left")) + 1, axis.size)
    return da.isel({dim: slice(lo, hi)})


def interpolate_points(
    ds: xr.Dataset,
    names: Sequence[str],
    lon: np.ndarray,
    lat: np.ndarray,
    time: pd.DatetimeIndex,
    z: np.ndarray,
) -> Dict[str, xr.DataArray]:
    """Each of `names` in `ds` interpolated (linearly) to the points; NaN outside the grid, on
    land and below the sea floor. z above the shallowest level centre takes the shallowest value,
    the convention the service uses between the surface and its top level."""
    targets = {"lon": lon, "lat": lat, "time": time.values}
    out = {}
    for name in names:
        if name not in ds.data_vars:
            raise KeyError(
                f"{name!r} not in store; it has {sorted(map(str, ds.data_vars))}"
            )
        da = ds[name]
        points = {dim: v for dim, v in targets.items() if dim in da.dims}
        if "z" in da.dims:
            points["z"] = np.minimum(z, float(ds["z"].max()))
        missing = set(da.dims) - set(points)
        if missing:
            raise ValueError(
                f"{name!r} has dimensions {sorted(map(str, missing))} this cannot interpolate over"
            )
        for dim, values in points.items():
            da = _bracket(da, dim, values)
        da = da.load()
        if "time" in points:
            # Interpolate on float seconds: xarray's vectorised interpolation over a datetime axis
            # returns NaN when the axis and the targets carry different datetime units.
            origin = np.datetime64(pd.Timestamp(da["time"].values[0]), "ns")

            def seconds(v: np.ndarray) -> np.ndarray:
                return (v.astype("datetime64[ns]") - origin) / np.timedelta64(1, "s")

            da = da.assign_coords(time=seconds(da["time"].values))
            points["time"] = seconds(points["time"])
        result = da.interp(
            {dim: xr.DataArray(v, dims=OBS) for dim, v in points.items()},
            method="linear",
        )
        result = result.drop_vars(
            [c for c in result.coords if c != OBS], errors="ignore"
        )
        result.attrs = {
            k: v
            for k, v in da.attrs.items()
            if k in ("units", "standard_name", "long_name")
        }
        result.attrs["n_missing"] = int(np.isnan(result.values).sum())
        out[name] = result
    return out


def _provenance(prefix: str, ds: xr.Dataset) -> Dict[str, Any]:
    return {
        f"forcingkit_{prefix}_{key}": ds.attrs.get(f"forcingkit_{key}", "")
        for key in ("store_id", "store_version", "donor", "schema")
    }


def assemble(
    values: Dict[str, xr.DataArray],
    lon: np.ndarray,
    lat: np.ndarray,
    time: pd.DatetimeIndex,
    z: np.ndarray,
    trajectory_id: str,
    attrs: Dict[str, Any],
) -> xr.Dataset:
    """The interpolated values as a CF trajectory (CF 1.11, section 9)."""
    ds = xr.Dataset(
        {name: (OBS, np.asarray(v.values), v.attrs) for name, v in values.items()},
        coords={
            "time": (OBS, time.values, {"standard_name": "time", "axis": "T"}),
            "lat": (
                OBS,
                lat,
                {"standard_name": "latitude", "units": "degrees_north", "axis": "Y"},
            ),
            "lon": (
                OBS,
                lon,
                {"standard_name": "longitude", "units": "degrees_east", "axis": "X"},
            ),
            "z": (
                OBS,
                z,
                {
                    "standard_name": "height",
                    "units": "m",
                    "positive": "up",
                    "axis": "Z",
                },
            ),
        },
    )
    ds["trajectory"] = xr.DataArray(
        trajectory_id,
        attrs={"cf_role": "trajectory_id", "long_name": "trajectory name"},
    )
    ds.attrs.update(
        Conventions="CF-1.11",
        featureType="trajectory",
        source="forcingkit",
        interpolation="linear in time, latitude, longitude and z",
        **attrs,
    )
    return ds


def interpolate_trajectory(
    fk: "Client",
    lon,
    lat,
    time,
    z=None,
    variables: Sequence[str] = ("u", "v"),
    atmosphere: Sequence[str] = (),
    margin_deg: float = 0.1,
    trajectory_id: str = "trajectory",
    **ocean_options: Any,
) -> xr.Dataset:
    """Ocean `variables` (from `fk.ocean`) and `atmosphere` variables (from `fk.atmosphere`, for
    example `u10`, `v10`) interpolated to each point of a trajectory.

    lon, lat: degrees; time: UTC (naive times are taken as UTC); z: metres, positive up, so 25 m
    below the surface is -25 (default: the surface). Returns a CF trajectory Dataset with one
    value per point and variable; each variable's `n_missing` counts points that fell on land,
    below the sea floor or outside the grid, and the global attributes name the stores used.
    """
    lon, lat, t, z = _prepare(lon, lat, time, z)
    bbox, start, hours = trajectory_window(lon, lat, t, margin_deg)
    values: Dict[str, xr.DataArray] = {}
    attrs: Dict[str, Any] = {}
    if variables:
        ocean = fk.ocean(bbox, start, hours, **ocean_options)
        values.update(interpolate_points(ocean, variables, lon, lat, t, z))
        attrs.update(_provenance("ocean", ocean))
    if atmosphere:
        air = fk.atmosphere(bbox, start, hours)
        values.update(interpolate_points(air, atmosphere, lon, lat, t, z))
        attrs.update(_provenance("atmosphere", air))
    return assemble(values, lon, lat, t, z, trajectory_id, attrs)


async def interpolate_trajectory_async(
    fk: "AsyncClient",
    lon,
    lat,
    time,
    z=None,
    variables: Sequence[str] = ("u", "v"),
    atmosphere: Sequence[str] = (),
    margin_deg: float = 0.1,
    trajectory_id: str = "trajectory",
    **ocean_options: Any,
) -> xr.Dataset:
    """`interpolate_trajectory` for an `AsyncClient`."""
    import asyncio

    lon, lat, t, z = _prepare(lon, lat, time, z)
    bbox, start, hours = trajectory_window(lon, lat, t, margin_deg)
    values: Dict[str, xr.DataArray] = {}
    attrs: Dict[str, Any] = {}
    stores: list[tuple[str, Optional[xr.Dataset], Sequence[str]]] = []
    if variables:
        stores.append(
            ("ocean", await fk.ocean(bbox, start, hours, **ocean_options), variables)
        )
    if atmosphere:
        stores.append(
            ("atmosphere", await fk.atmosphere(bbox, start, hours), atmosphere)
        )
    for prefix, ds, names in stores:
        assert ds is not None
        values.update(
            await asyncio.to_thread(interpolate_points, ds, names, lon, lat, t, z)
        )
        attrs.update(_provenance(prefix, ds))
    return assemble(values, lon, lat, t, z, trajectory_id, attrs)
