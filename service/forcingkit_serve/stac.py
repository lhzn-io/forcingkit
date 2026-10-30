"""A finished store described as a STAC Item (https://stacspec.org), the shape of a job result.

Each Item carries the store's footprint, time range, schema and donor, and one `zarr` asset whose
href is the store served over HTTP (`/api/v1/stores`). The dict is built by hand so the service
needs no STAC dependency; a catalogue over the cache can later list the same Items.
"""

import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import xarray as xr

STAC_VERSION = "1.1.0"
XARRAY_ASSETS = "https://stac-extensions.github.io/xarray-assets/v1.0.0/schema.json"
ZARR_MEDIA_TYPE = "application/vnd+zarr"


def store_version(zarr_path: str) -> str:
    """Changes whenever the store is rewritten under the same id (after a cache_bust), so a
    client cache keyed by it never serves an older build."""
    meta = os.path.join(zarr_path, ".zmetadata")
    return str(os.stat(meta if os.path.exists(meta) else zarr_path).st_mtime_ns)


def _axis(ds: xr.Dataset, *names: str) -> Optional[np.ndarray]:
    for name in names:
        if name in ds.variables:
            return np.asarray(ds[name].values, dtype=np.float64)
    return None


def _iso(t: Any) -> str:
    return pd.Timestamp(t).strftime("%Y-%m-%dT%H:%M:%SZ")


def store_item(
    zarr_id: str,
    zarr_path: str,
    base_url: str,
    kind: str,
    donor: Optional[str] = None,
) -> Dict[str, Any]:
    """The STAC Item for the finished store at `zarr_path`; `base_url` is the service root."""
    base = base_url.rstrip("/")
    with xr.open_zarr(zarr_path, consolidated=True, zarr_format=2) as ds:
        lon = _axis(ds, "lon", "longitude")
        lat = _axis(ds, "lat", "latitude")
        bbox = None
        if lon is not None and lat is not None:
            lon = (lon + 180.0) % 360.0 - 180.0  # some donors deliver 0..360
            bbox = [
                float(np.nanmin(lon)),
                float(np.nanmin(lat)),
                float(np.nanmax(lon)),
                float(np.nanmax(lat)),
            ]
        times = ds["time"].values if "time" in ds.coords else None
        properties: Dict[str, Any] = {
            "datetime": None,
            "start_datetime": _iso(times[0]) if times is not None else None,
            "end_datetime": _iso(times[-1]) if times is not None else None,
            "forcingkit:kind": kind,
            "forcingkit:schema": ds.attrs.get("schema", "legacy"),
            "forcingkit:donor": donor,
            "forcingkit:store_version": store_version(zarr_path),
        }
    w, s, e, n = bbox if bbox is not None else (None, None, None, None)
    return {
        "type": "Feature",
        "stac_version": STAC_VERSION,
        "stac_extensions": [XARRAY_ASSETS],
        "id": zarr_id,
        "bbox": bbox,
        "geometry": (
            {
                "type": "Polygon",
                "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]],
            }
            if bbox is not None
            else None
        ),
        "properties": properties,
        "links": [
            {
                "rel": "self",
                "href": f"{base}/api/v1/jobs/{zarr_id}",
                "type": "application/json",
            }
        ],
        "assets": {
            "zarr": {
                "href": f"{base}/api/v1/stores/{zarr_id}.zarr",
                "type": ZARR_MEDIA_TYPE,
                "roles": ["data"],
                "xarray:open_kwargs": {"consolidated": True, "zarr_format": 2},
            }
        },
    }
