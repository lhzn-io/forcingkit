"""The NECOFS whole-mesh cache and the server session, on the synthetic FVCOM-shaped file of test_necofs_volume. No
network: a remote URL only selects the cached path."""

import concurrent.futures
import logging

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from forcingkit.fetchers import necofs
from test_necofs_volume import INNER, fvcom_file

REMOTE = "http://example.invalid/thredds/dodsC/NECOFS_GOM7_2026_04_02.nc"
TIMES = pd.date_range("2026-04-01T01:00", periods=3, freq="1h")


@pytest.fixture(autouse=True)
def cache(monkeypatch, tmp_path):
    monkeypatch.setenv("FORCINGKIT_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(necofs, "_MESHES", {})
    return tmp_path / "necofs" / "mesh"


def subset(ds, url):
    return necofs._subset_volume(ds, TIMES, INNER, TIMES, url)


def test_a_remote_subset_equals_a_local_one_cold_and_from_the_cache(cache):
    ds = fvcom_file(TIMES)
    local = subset(ds, "file://local")
    cold = subset(ds, REMOTE)
    necofs._MESHES.clear()  # a new process: only the file on disk
    warm = subset(ds, REMOTE)
    for out in (cold, warm):
        for name in local.variables:
            np.testing.assert_array_equal(out[name].values, local[name].values)
            assert out[name].dtype == local[name].dtype
    assert len(list(cache.glob("archive-*.npz"))) == 1


def test_the_cache_is_used_instead_of_the_file(cache):
    ds = fvcom_file(TIMES)
    first = subset(ds, REMOTE)
    necofs._MESHES.clear()
    # The same key (nv, lon, lonc and the counts), a different easting.
    moved = ds.assign(x=ds["x"] + 1.0)
    np.testing.assert_array_equal(subset(moved, REMOTE)["x"].values, first["x"].values)


def test_local_files_are_never_cached(cache):
    subset(fvcom_file(TIMES), "file://local")
    assert not cache.exists()


def test_a_renumbered_mesh_misses_the_cache_with_a_warning(cache, caplog):
    ds = fvcom_file(TIMES)
    subset(ds, REMOTE)
    nv = ds["nv"].values.copy()
    nv[:, [0, 1]] = nv[:, [1, 0]]  # two elements swapped: same counts and extents
    renumbered = ds.assign(nv=(ds["nv"].dims, nv))
    with caplog.at_level(logging.WARNING, logger=necofs.logger.name):
        subset(renumbered, REMOTE)
    assert len(list(cache.glob("archive-*.npz"))) == 2
    assert "differs from the cached" in caplog.text


def test_the_forecast_and_the_archive_are_kept_apart(cache):
    ds = fvcom_file(TIMES)
    subset(ds, REMOTE)
    subset(ds, necofs.NECOFS_GOM7_URL)
    assert (
        len(list(cache.glob("archive-*.npz"))) == 1
        and len(list(cache.glob("forecast-*.npz"))) == 1
    )


def test_a_damaged_entry_is_read_again(cache, caplog):
    ds = fvcom_file(TIMES)
    subset(ds, REMOTE)
    (entry,) = cache.glob("archive-*.npz")
    entry.write_bytes(b"not an npz")
    necofs._MESHES.clear()
    with caplog.at_level(logging.WARNING, logger=necofs.logger.name):
        out = subset(ds, REMOTE)
    np.testing.assert_array_equal(
        out["u"].values, subset(ds, "file://local")["u"].values
    )
    assert "not read" in caplog.text
    with np.load(entry) as z:  # rewritten whole
        assert "nbe_all" in z.files


def test_concurrent_builds_leave_one_complete_entry(cache):
    ds = fvcom_file(TIMES)
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        outs = list(pool.map(lambda _: subset(ds, REMOTE), range(8)))
    for out in outs[1:]:
        np.testing.assert_array_equal(
            out["neighbours"].values, outs[0]["neighbours"].values
        )
    assert len(list(cache.glob("archive-*.npz"))) == 1
    assert not list(cache.glob(".*.tmp.npz"))


def test_remote_opens_send_the_user_agent(monkeypatch):
    seen = {}

    def fake_open(url, **kwargs):
        seen[url] = kwargs
        return xr.Dataset()

    monkeypatch.setattr(necofs.xr, "open_dataset", fake_open)
    monkeypatch.setattr(necofs, "_SESSION", None)
    necofs._open_dap(REMOTE)
    necofs._open_dap("/data/local.nc")
    assert seen[REMOTE]["session"].headers["User-Agent"] == necofs.USER_AGENT
    assert "session" not in seen["/data/local.nc"]
