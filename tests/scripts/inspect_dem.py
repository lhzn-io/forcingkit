"""Report NaN content and range of the `elevation` array in a DEM Zarr store.

Usage: uv run python tests/scripts/inspect_dem.py PATH_TO_DEM.zarr
"""

import sys

import numpy as np
import zarr

if len(sys.argv) != 2:
    sys.exit(__doc__)

ds = zarr.open(sys.argv[1], mode="r")
elev = ds["elevation"][:]  # type: ignore
print("Elevation contains NaN:", np.isnan(elev).any())  # type: ignore
print("NaN count:", np.isnan(elev).sum())  # type: ignore
print("Elevation range:", np.nanmin(elev), "to", np.nanmax(elev))  # type: ignore
