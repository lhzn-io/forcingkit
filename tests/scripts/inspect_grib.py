import os
import xarray as xr

ds = xr.open_dataset(
    os.path.expanduser("~/.cache/forcingkit/hrrr_2026-03-03_t20z.grib2"),
    engine="cfgrib",
    backend_kwargs={
        "filter_by_keys": {"typeOfLevel": "surface", "stepType": "instant"}
    },
)
print(ds)
