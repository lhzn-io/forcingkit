"""HF radar totals from NDBC: which days are archived, the box, the archive's files and its create-only store, and
reading a window back from the archive and the server. No network: a small stand-in for the server's dataset."""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from forcingkit.fetchers import hfradar

BBOX = [-73.8, 40.85, -71.8, 41.3]


def server(start: str = "2026-09-25T00", hours: int = 48) -> xr.Dataset:
    """The server's layout as the OPeNDAP client returns it with mask_and_scale off: int16 with scale and fill, and
    a signed byte that arrives unsigned."""
    time = pd.date_range(start, periods=hours, freq="h")
    lat = np.round(np.arange(40.80, 41.40, 0.05), 3).astype("float32")
    lon = np.round(np.arange(-74.0, -71.5, 0.1), 3).astype("float32")
    shape = (len(time), len(lat), len(lon))
    u = np.full(shape, -32767, dtype="int16")
    hour_of_record = (
        ((time - pd.Timestamp("2026-09-25")) // pd.Timedelta(hours=1))
        .to_numpy()
        .astype("int16")
    )
    u[:, 3, 5] = (
        hour_of_record  # 0.00, 0.01, ... m/s at one cell, by hours since 25 Sep
    )
    v = np.full(shape, -32767, dtype="int16")
    v[:, 3, 5] = -hour_of_record
    sites = np.full(shape, 129, dtype="uint8")  # -127 as a signed byte
    sites[:, 3, 5] = 2
    vel = {
        "_FillValue": -32767,
        "scale_factor": 0.01,
        "units": "m s-1",
        "coordinates": "depth",
        "_ChunkSizes": [1, 2],
    }
    return xr.Dataset(
        {
            "u": (
                ("time", "lat", "lon"),
                u,
                dict(vel, standard_name="surface_eastward_sea_water_velocity"),
            ),
            "v": (
                ("time", "lat", "lon"),
                v,
                dict(vel, standard_name="surface_northward_sea_water_velocity"),
            ),
            "number_of_sites": (
                ("time", "lat", "lon"),
                sites,
                {"_Unsigned": "false", "_FillValue": -127, "units": "count"},
            ),
            "wgs84": (
                (),
                np.uint8(0),
                {"_Unsigned": "false", "grid_mapping_name": "latitude_longitude"},
            ),
        },
        coords={
            "time": time,
            "lat": ("lat", lat, {"units": "degrees_north"}),
            "lon": ("lon", lon, {"units": "degrees_east"}),
        },
        attrs={
            "title": "Near-Real Time Surface Ocean Velocity",
            "license": "public",
            "id_HFR": "latest hour only",
        },
    )


def test_box_key_round_trips():
    assert hfradar.box_key(BBOX) == "40.850N_41.300N_073.800W_071.800W"
    assert hfradar.parse_box_key(hfradar.box_key(BBOX)) == pytest.approx(tuple(BBOX))
    assert hfradar.parse_box_key("lis") is None
    name = hfradar.day_name("1km", BBOX, pd.Timestamp("2026-09-26"))
    assert (
        name
        == "usegc_1km/40.850N_41.300N_073.800W_071.800W/2026/09/hfradar_usegc_1km_20260926.nc"
    )
    assert hfradar.NAME.fullmatch(name)


def test_ready_days_wait_for_the_server_and_the_lag():
    times = pd.date_range("2026-09-24T21", "2026-09-27T03", freq="h")
    now = datetime(2026, 9, 27, 7, tzinfo=timezone.utc)
    assert hfradar.ready_days(times, now) == [
        pd.Timestamp(d) for d in ("2026-09-24", "2026-09-25", "2026-09-26")
    ]
    assert pd.Timestamp("2026-09-26") not in hfradar.ready_days(
        times, datetime(2026, 9, 27, 5, tzinfo=timezone.utc)
    )
    stalled = pd.date_range(
        "2026-09-24T21", "2026-09-26T22", freq="h"
    )  # not yet past the day's end
    assert pd.Timestamp("2026-09-26") not in hfradar.ready_days(stalled, now)
    one = hfradar.ready_days(
        times, now, start=pd.Timestamp("2026-09-25"), end=pd.Timestamp("2026-09-25")
    )
    assert one == [pd.Timestamp("2026-09-25")]


def test_runs_group_consecutive_days():
    days = [
        pd.Timestamp(d)
        for d in ("2026-09-01", "2026-09-02", "2026-09-03", "2026-09-05")
    ]
    assert hfradar.runs(days, most=2) == [days[:2], days[2:3], days[3:]]


def test_box_slices():
    ds = server()
    ys, xs = hfradar.box_slices(ds["lat"].values, ds["lon"].values, BBOX)
    assert (
        ds["lat"].values[ys].min() >= BBOX[1] and ds["lat"].values[ys].max() <= BBOX[3]
    )
    assert (
        ds["lon"].values[xs].min() >= BBOX[0] and ds["lon"].values[xs].max() <= BBOX[2]
    )
    with pytest.raises(ValueError):
        hfradar.box_slices(
            ds["lat"].values, ds["lon"].values, [-74.0, 10.0, -73.0, 11.0]
        )


def test_day_file_keeps_the_stored_values_and_the_credit(tmp_path):
    ds = hfradar.fetch(BBOX, "2026-09-25", "2026-09-27", src=server())
    path = tmp_path / "day.nc"
    path.write_bytes(hfradar.day_file(ds, pd.Timestamp("2026-09-26")))
    raw = xr.open_dataset(path, mask_and_scale=False)
    assert raw.sizes["time"] == 24 and raw["time"].values[0] == np.datetime64(
        "2026-09-26T00:00"
    )
    j, i = (
        int(np.flatnonzero(raw["lat"].values == np.float32(40.95))[0]),
        int(np.flatnonzero(np.isclose(raw["lon"].values, -73.5))[0]),
    )
    assert raw["u"].dtype == np.int16 and raw["u"].values[0, j, i] == 24
    assert (
        raw["number_of_sites"].dtype == np.int8
        and raw["number_of_sites"].values[0, 0, 0] == -127
    )
    assert "coordinates" not in raw["u"].attrs and "_ChunkSizes" not in raw["u"].attrs
    decoded = xr.open_dataset(path)
    assert decoded["u"].values[23, j, i] == pytest.approx(0.47)
    assert np.isnan(decoded["u"].values[0, 0, 0]) and np.isnan(
        decoded["number_of_sites"].values[0, 0, 0]
    )
    assert (
        "LISICOS" in raw.attrs["source_credit"]
        and "HFRNet" in raw.attrs["source_credit"]
    )
    assert (
        raw.attrs["title"] == "Near-Real Time Surface Ocean Velocity"
        and "id_HFR" not in raw.attrs
    )
    assert raw.attrs["subset_hours"] == 24


def test_local_store_is_create_only(tmp_path):
    store = hfradar.LocalStore(tmp_path)
    name = hfradar.day_name("1km", BBOX, pd.Timestamp("2026-09-26"))
    assert store.put(name, b"first") and store.has(name)
    assert not store.put(name, b"second")
    assert (tmp_path / name).read_bytes() == b"first"
    with pytest.raises(ValueError):
        store.put("../elsewhere.nc", b"x")


def test_local_store_defaults_to_the_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("FORCINGKIT_CACHE_DIR", str(tmp_path))
    assert hfradar.LocalStore().root == tmp_path / "hfradar"


@pytest.fixture
def remote(monkeypatch):
    opened = []

    def fake(grid="1km", session=None):
        opened.append(grid)
        return server(hours=60)  # 25 Sep 00 to 27 Sep 11

    monkeypatch.setattr(hfradar, "open_remote", fake)
    return opened


def test_archive_writes_ready_days_once(tmp_path, remote):
    now = datetime(2026, 9, 27, 12, tzinfo=timezone.utc)
    store = hfradar.LocalStore(tmp_path)
    first = hfradar.archive(BBOX, start="all", grids=("1km",), store=store, now=now)
    assert [n[-11:-3] for n in first["1km"]] == ["20260925", "20260926"]
    assert hfradar.archive(BBOX, start="all", grids=("1km",), store=store, now=now) == {
        "1km": []
    }


def test_load_reads_a_smaller_box_from_the_archive_and_the_rest_from_the_server(
    tmp_path, remote
):
    hfradar.archive(
        BBOX,
        start="2026-09-25",
        end="2026-09-25",
        grids=("1km",),
        store=hfradar.LocalStore(tmp_path),
        now=datetime(2026, 9, 27, 12, tzinfo=timezone.utc),
    )
    inner = [-73.6, 40.9, -73.4, 41.0]
    archived = hfradar.load(
        inner, "2026-09-25T20", "2026-09-26T04", root=tmp_path, remote=False
    )
    assert archived.sizes["time"] == 4  # 20 to 23 on the 25th; the 26th is not archived
    both = hfradar.load(inner, "2026-09-25T20", "2026-09-26T04", root=tmp_path)
    assert both.sizes["time"] == 8
    cell = both["u"].sel(lat=np.float32(40.95), lon=-73.5, method="nearest")
    assert cell.values == pytest.approx(
        [0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27]
    )
    assert np.isnan(both["u"].values).sum() == both["u"].size - 8
    assert "LISICOS" in both.attrs["source_credit"]
    assert (
        hfradar.load(
            [-60.0, 40.0, -59.0, 41.0],
            "2026-09-25",
            "2026-09-26",
            root=tmp_path,
            remote=False,
        ).sizes
        == {}
    )
