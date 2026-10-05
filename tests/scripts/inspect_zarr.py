import os
import xarray as xr

ds = xr.open_zarr(os.path.expanduser("~/.cache/forcingkit/hrrr_2026-03-03_t20z.zarr"))
print("Zarr Output Dimensions:", ds.dims)
print("Bytes:", ds.nbytes)
