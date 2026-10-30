"""One layout for every store the client returns.

The service writes two parent-ocean layouts. Schema `z-v3` (streamed donors) is already what the
client returns: `u, v, temp, salt` on `(time, z, lat, lon)` with `z` in metres, positive up, and
longitudes in -180..180. Older one-shot donors write `legacy` stores in the donor's own
conventions: `depth` positive down, `latitude`/`longitude` names, and for HYCOM longitudes in
0..360. `normalise` brings those to the z-v3 conventions and refuses any schema it does not know,
so a layout change fails here with a clear message instead of later on a variable name.
The atmosphere (`hrrr-atm-v1`) and waves (`nwps-waves-v1`) stores are written in that convention
too and pass through unchanged.
"""

import numpy as np
import xarray as xr

KNOWN_SCHEMAS = ("z-v3", "legacy", "hrrr-atm-v1", "nwps-waves-v1")


class SchemaError(ValueError):
    pass


def wrap_lon(lon):
    """Longitudes into -180..180."""
    return (np.asarray(lon, dtype=np.float64) + 180.0) % 360.0 - 180.0


def normalise(ds: xr.Dataset) -> xr.Dataset:
    schema = ds.attrs.get("schema", "legacy")
    if schema not in KNOWN_SCHEMAS:
        raise SchemaError(
            f"store schema {schema!r} is not one this client knows ({', '.join(KNOWN_SCHEMAS)}); "
            "upgrade forcingkit-client"
        )
    if schema != "legacy":
        return ds

    ds = ds.rename(
        {
            old: new
            for old, new in (("latitude", "lat"), ("longitude", "lon"))
            if old in ds.variables
        }
    )
    for name in ("lat", "lon"):
        if name not in ds.coords or ds[name].ndim != 1:
            raise SchemaError(
                f"legacy store has no 1-D {name} axis (a curvilinear donor grid); "
                "this client reads regular longitude-latitude grids only"
            )
    if "depth" in ds.dims:
        ds = ds.assign_coords(depth=-ds["depth"].astype(np.float64)).rename(depth="z")
        ds["z"].attrs.update(units="m", positive="up", axis="Z")
        ds = ds.sortby("z")
    ds = ds.assign_coords(lon=wrap_lon(ds["lon"].values)).sortby("lon")
    return ds
