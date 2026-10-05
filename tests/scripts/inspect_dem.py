import zarr
import numpy as np

ds = zarr.open(
    "/home/lhzn/Projects/lhzn-io/coastal-sim/config/topobathykit/policies/throgs_neck_dem.zarr",
    mode="r",
)
elev = ds["elevation"][:]  # type: ignore
print("Elevation contains NaN:", np.isnan(elev).any())  # type: ignore
print("NaN count:", np.isnan(elev).sum())  # type: ignore
print("Elevation range:", np.nanmin(elev), "to", np.nanmax(elev))  # type: ignore
