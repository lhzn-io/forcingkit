"""Write a time series to Zarr one record at a time, and publish it only when it is complete.

Long deliveries (a 168 h parent ocean, an hourly atmosphere) used to be held in memory and written
in one shot, so a failure part way lost everything, and a store left behind by a crashed write was
served as a cache hit by the existence check. Here records are appended to `<path>.partial` with
one time record per chunk (so a reader that needs one hour reads one chunk), the store is checked
for the expected number of contiguous hourly records, marked `complete`, consolidated, and only
then renamed to its final path.
"""

import logging
import os
import shutil
from typing import Any, Optional

import numpy as np
import pandas as pd
import xarray as xr
import zarr

logger = logging.getLogger(__name__)

PARTIAL_SUFFIX = ".partial"


def store_is_complete(path: str, schemas: tuple[str, ...] = ()) -> bool:
    """True when `path` is a finished store a cache may serve.

    A store whose `schema` attribute is one of `schemas` (those written by `StreamingZarrWriter`)
    must also carry `complete = True`; any other store counts as complete if it exists, since it was
    written in one shot after its data was assembled.
    """
    if not os.path.isdir(path):
        return False
    try:
        attrs = dict(zarr.open_group(path, mode="r", zarr_format=2).attrs)
    except Exception:
        return False
    if attrs.get("schema") in schemas:
        return bool(attrs.get("complete", False))
    return True


class StreamingZarrWriter:
    """Append hourly records to a Zarr v2 store and publish it atomically.

    `static` holds the variables without a time dimension (coordinates, masks); `record_dims` maps
    each per-record variable to its dimensions without time. Times are encoded as seconds since
    the first record.
    """

    def __init__(
        self,
        final_path: str,
        static: xr.Dataset,
        record_dims: dict[str, tuple[str, ...]],
        expected_records: int,
        attrs: Optional[dict] = None,
        record_attrs: Optional[dict[str, dict]] = None,
    ):
        self.final_path = final_path
        self.partial_path = final_path + PARTIAL_SUFFIX
        self.static = static
        self.record_dims = record_dims
        self.expected_records = expected_records
        self.attrs = dict(attrs or {})
        self.record_attrs = record_attrs or {}
        self.times: list[pd.Timestamp] = []
        if os.path.exists(self.partial_path):
            shutil.rmtree(self.partial_path)

    def append(self, time: pd.Timestamp, record: dict[str, np.ndarray]) -> None:
        missing = set(self.record_dims) - set(record)
        if missing:
            raise ValueError(f"record at {time} lacks {sorted(missing)}")
        t = pd.Timestamp(time)
        if self.times and t <= self.times[-1]:
            raise ValueError(f"record at {t} is not after {self.times[-1]}")
        data_vars = {
            name: (("time",) + dims, np.asarray(record[name], dtype=np.float32)[None])
            for name, dims in self.record_dims.items()
        }
        ds = xr.Dataset(data_vars, coords={"time": [t.to_datetime64()]})
        for name, a in self.record_attrs.items():
            if name in ds:
                ds[name].attrs.update(a)
        if not self.times:
            ds = xr.merge([ds, self.static])
            ds.attrs.update(self.attrs)
            encoding: dict[str, dict[str, Any]] = {
                name: {"chunks": (1,) + ds[name].shape[1:]} for name in self.record_dims
            }
            encoding["time"] = {
                "units": f"seconds since {t.strftime('%Y-%m-%dT%H:%M:%S')}",
                "dtype": "float64",
            }
            ds.to_zarr(
                self.partial_path,
                mode="w",
                zarr_format=2,
                consolidated=False,
                encoding=encoding,
            )
        else:
            ds.to_zarr(
                self.partial_path,
                append_dim="time",
                zarr_format=2,
                consolidated=False,
            )
        self.times.append(t)

    def close(self) -> str:
        """Verify, mark complete, consolidate and publish. Raises (and removes the partial store)
        if the record count or the hourly spacing is wrong."""
        try:
            if len(self.times) != self.expected_records:
                raise RuntimeError(
                    f"{self.final_path}: {len(self.times)} records written, "
                    f"{self.expected_records} expected"
                )
            gaps = np.diff(np.array([t.value for t in self.times])) / 3.6e12
            if len(gaps) and not np.allclose(gaps, 1.0):
                raise RuntimeError(
                    f"{self.final_path}: records are not hourly and contiguous "
                    f"(spacings {sorted(set(np.round(gaps, 3)))} h)"
                )
            # Appends rewrite the group's attributes from the appended record, so the store's
            # own attributes are set here, once, with the completion mark.
            group = zarr.open_group(self.partial_path, mode="r+", zarr_format=2)
            group.attrs.update(self.attrs)
            group.attrs["complete"] = True
            group.attrs["records"] = len(self.times)
            zarr.consolidate_metadata(self.partial_path, zarr_format=2)
        except Exception:
            self.abort()
            raise
        if os.path.exists(self.final_path):
            shutil.rmtree(self.final_path)
        os.rename(self.partial_path, self.final_path)
        logger.info(f"Published {self.final_path} ({len(self.times)} records)")
        return self.final_path

    def abort(self) -> None:
        if os.path.exists(self.partial_path):
            shutil.rmtree(self.partial_path)
