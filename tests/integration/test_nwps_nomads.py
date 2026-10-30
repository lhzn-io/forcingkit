"""Live NWPS read from NOMADS: a 48 h window for WFO OKX, grid CG1, checked against the GRIB2.

NOMADS keeps about five days, so the window is placed relative to now: it starts at the second
newest cycle that has certainly posted and runs 48 h, which spans that cycle, the next one and the
long leads after it. About 26 MB is read per cycle, and each file is kept in the cache.
"""

import functools
import os
import warnings

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from forcingkit import dispatcher, settings
from forcingkit.fetchers import nwps

pytestmark = pytest.mark.integration

BBOX = [-72.9, 40.95, -72.0, 41.3]  # central Long Island Sound


@functools.cache
def _open(path):
    import cfgrib

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cfgrib.open_datasets(path, backend_kwargs={"indexpath": ""})


def _grib_value(path, short, lead, lat, lon):
    for d in _open(path):
        steps = (np.atleast_1d(d["step"].values) / np.timedelta64(1, "h")).astype(int)
        if lead in steps:
            da = (
                d[short].isel(step=list(steps).index(lead))
                if "step" in d.dims
                else d[short]
            )
            return float(da.sel(latitude=lat, longitude=lon % 360.0, method="nearest"))
    raise AssertionError(f"lead {lead} not in {path}")


def test_48h_window_matches_grib(tmp_path, monkeypatch):
    monkeypatch.delenv("FORCINGKIT_NWPS_ARCHIVE", raising=False)
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    newest = nwps.floor_cycle(now - pd.Timedelta(hours=nwps.POSTING_LAG_HOURS + 6))
    start = newest - pd.Timedelta(hours=nwps.CYCLE_HOURS) + pd.Timedelta(hours=2)
    start_time = start.strftime("%Y-%m-%dT%H:%M:%S")
    latest = nwps.newest_cycle(nwps.record_times(start_time, 48)[-1])
    path = str(tmp_path / "wav_live.zarr")

    dispatcher.dispatch_waves_request(BBOX, start_time, 48, path, latest_cycle=latest)
    ds = xr.open_zarr(path)
    assert ds.attrs["complete"] and ds.sizes["time"] == 51
    assert ds.attrs["files_from_nomads"] >= 2
    assert 0.0 < float(ds["wet"].mean()) < 1.0, "the box holds both sea and land"
    # Cells wet and dry with the water level, so a record's sea is a subset of `wet`.
    finite = np.isfinite(ds["hs"].values)
    assert not (finite & (ds["wet"].values == 0)).any()
    assert finite.any(axis=0).sum() == ds["wet"].values.sum()
    assert ds.attrs["dx"] == pytest.approx(0.0211, abs=1e-3)

    raw = settings.cache_dir("nwps", "okx", "CG1")
    sea = np.argwhere(ds["wet"].values == 1)
    for k, ti in enumerate((0, 20, 50)):
        iy, ix = sea[(k * 37) % len(sea)]
        lat, lon = float(ds["lat"][iy]), float(ds["lon"][ix])
        cycle = pd.Timestamp(ds.attrs["record_cycles"][ti].replace("Z", ":00"))
        lead = int(ds["forecast_hour"].values[ti])
        assert pd.Timestamp(ds["time"].values[ti]) == cycle + pd.Timedelta(hours=lead)
        grib = os.path.join(raw, nwps.file_name("okx", "CG1", cycle))
        for name, short in (("hs", "swh"), ("tp", "perpw"), ("ws", "ws")):
            stored = float(ds[name].values[ti, iy, ix])
            assert stored == pytest.approx(
                _grib_value(grib, short, lead, lat, lon), abs=1e-4
            )
        stored_dir = float(ds["dir"].values[ti, iy, ix])
        source_dir = _grib_value(grib, "dirpw", lead, lat, lon)
        assert stored_dir == pytest.approx((source_dir + 180.0) % 360.0, abs=1e-3)
