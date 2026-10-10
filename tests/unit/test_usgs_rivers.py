"""River discharge from USGS gauges: catalogue, scaling, hourly averaging, gap handling, fill
flags, provisional data and the store round trip. No network: a fake getter serves CSV shaped
like the USGS Water Data API responses."""

import json
from urllib.parse import parse_qs

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from forcingkit import dispatcher
from forcingkit.fetchers import usgs, usgs_rivers

RIVER = {
    "name": "Testriver",
    "mouth": {"lon": -72.5, "lat": 41.2},
    "area_mouth_sqmi": 150.0,
    "gauges": [
        {"usgs_id": "00000001", "name": "Upper", "area_sqmi": 75.0},
        {"usgs_id": "00000002", "name": "Branch", "area_sqmi": 25.0},
    ],
}


class FakeUSGS:
    """Serves 15-minute discharge at a constant value per gauge, with optional holes, daily
    means, temperature and normals."""

    def __init__(
        self,
        cfs,
        holes=(),
        daily=None,
        status="Approved",
        temperature=None,
        median=None,
    ):
        self.cfs = cfs  # usgs_id -> constant ft3/s
        self.holes = holes  # (usgs_id, start, end) with no continuous values
        self.daily = daily or {}  # usgs_id -> {date str: ft3/s}
        self.status = status
        self.temperature = temperature or {}  # usgs_id -> degC
        self.median = median or {}  # usgs_id -> ft3/s
        self.calls = []

    def __call__(self, url, params):
        self.calls.append((url, dict(params)))
        site = params["monitoring_location_id"].removeprefix("USGS-")
        header = "time,value,approval_status"
        if url.endswith("/continuous/items"):
            a, b = (pd.Timestamp(x) for x in params["time"].split("/"))
            if params["parameter_code"] == usgs.TEMPERATURE:
                value = self.temperature.get(site)
            else:
                value = self.cfs.get(site)
            if value is None:
                return header + "\n"
            rows = [header]
            for t in pd.date_range(a, b, freq="15min", inclusive="left"):
                if any(s == site and h0 <= t < h1 for s, h0, h1 in self.holes):
                    continue
                rows.append(
                    f"{t.strftime('%Y-%m-%d %H:%M:%S+00:00')},{value},{self.status}"
                )
            return "\n".join(rows) + "\n"
        if url.endswith("/daily/items"):
            rows = [header] + [
                f"{d},{v},{self.status}" for d, v in self.daily.get(site, {}).items()
            ]
            return "\n".join(rows) + "\n"
        if url == usgs.NORMALS_API:
            m = self.median.get(site)
            days = pd.date_range("2024-01-01", "2024-12-31").strftime("%m-%d")
            return json.dumps(
                {
                    "features": [
                        {
                            "properties": {
                                "data": [
                                    {
                                        "parent_statistic_id": "00003",
                                        "values": [
                                            {
                                                "time_of_year": d,
                                                "percentiles": [25, 50, 75],
                                                "values": [m / 2, m, m * 2],
                                            }
                                            for d in days
                                        ],
                                    }
                                ]
                            }
                        }
                    ]
                }
            )
        raise AssertionError(f"unexpected URL {url} {parse_qs(str(params))}")


UTC = "2026-04-02T00:00:00Z"


def _ts(s):
    return pd.Timestamp(s, tz="UTC")


def test_packaged_mouths_are_listed_with_outlets():
    mouths = usgs_rivers.load_mouths()
    names = [m["name"] for m in mouths]
    assert {"Connecticut", "Housatonic", "Quinnipiac", "Thames"} <= set(names)
    assert all(m.get("comid") for m in mouths)
    assert all("gauges" not in m for m in mouths)


def test_mouths_in_bbox_selects_by_point_and_sorts_west_to_east():
    mouths = usgs_rivers.load_mouths()
    western = usgs_rivers.mouths_in_bbox(
        mouths, [-73.8557, 40.75645, -72.84354, 41.34844]
    )
    assert [m["name"] for m in western] == ["Housatonic", "Quinnipiac"]
    eastern = usgs_rivers.mouths_in_bbox(mouths, [-73.8557, 40.75, -71.9, 41.4])
    assert [m["name"] for m in eastern] == [
        "Housatonic",
        "Quinnipiac",
        "Connecticut",
        "Thames",
    ]


def test_local_mouths_add_to_and_override_the_packaged_ones(tmp_path, monkeypatch):
    (tmp_path / "local.json").write_text(
        json.dumps(
            {
                "mouths": [
                    {
                        "name": "Newriver",
                        "lon": -72.0,
                        "lat": 41.0,
                        "trace_from": "USGS-1",
                    },
                    {
                        "name": "Thames",
                        "lon": -72.08,
                        "lat": 41.32,
                        "comid": 1,
                        "exclude": ["01127500"],
                    },
                ]
            }
        )
    )
    monkeypatch.setenv("FORCINGKIT_RIVER_MOUTHS_DIR", str(tmp_path))
    mouths = {m["name"]: m for m in usgs_rivers.load_mouths()}
    assert "Newriver" in mouths and "Connecticut" in mouths
    assert mouths["Thames"]["exclude"] == ["01127500"]


def test_mouth_entries_are_checked_and_versioned():
    with pytest.raises(ValueError, match="unknown keys"):
        usgs_rivers.validate_mouth({"name": "X", "lon": 0, "lat": 0, "gauge": "1"})
    with pytest.raises(ValueError, match="missing lat"):
        usgs_rivers.validate_mouth({"name": "X", "lon": 0})
    a = [{"name": "X", "lon": 0, "lat": 0, "comid": 1}]
    b = [{"name": "X", "lon": 0, "lat": 0, "comid": 2}]
    noted = [dict(a[0], note="a remark")]
    assert usgs_rivers.mouths_version(a) != usgs_rivers.mouths_version(b)
    assert usgs_rivers.mouths_version(a) == usgs_rivers.mouths_version(noted)


def test_hourly_values_are_centred_on_the_hour():
    t = pd.date_range(_ts("2026-04-02T00:00"), periods=8, freq="15min")
    s = pd.Series([0.0, 0.0, 4.0, 4.0, 8.0, 8.0, 8.0, 8.0], index=t)
    h = usgs_rivers.hourly_centred(s)
    # 01:00 averages 00:30, 00:45, 01:00, 01:15.
    assert h[_ts("2026-04-02T01:00")] == pytest.approx(6.0)


def test_discharge_is_summed_scaled_and_converted():
    get = FakeUSGS({"00000001": 300.0, "00000002": 100.0})
    items = list(
        usgs_rivers.iter_rivers(UTC, 6, [-73, 41, -72, 42], rivers=[RIVER], get=get)
    )
    static, records = items[0][1], items[1:]
    assert len(records) == 7
    q = records[0][2]["discharge"][0]
    assert q == pytest.approx(400.0 * 1.5 * usgs.CFS_TO_M3S)
    assert static.attrs["schema"] == usgs_rivers.RIVER_SCHEMA
    assert static.attrs["provisional"] is False
    assert all(r[2]["fill"][0] == usgs_rivers.FILL_NONE for r in records)
    assert np.isnan(records[0][2]["temperature"][0])


def test_short_gap_is_interpolated_and_flagged():
    holes = [("00000001", _ts("2026-04-02T02:00"), _ts("2026-04-02T04:00"))]
    get = FakeUSGS({"00000001": 300.0, "00000002": 100.0}, holes=holes)
    records = list(
        usgs_rivers.iter_rivers(UTC, 6, [-73, 41, -72, 42], rivers=[RIVER], get=get)
    )[1:]
    flags = [r[2]["fill"][0] for r in records]
    assert usgs_rivers.FILL_INTERPOLATED in flags
    assert all(
        r[2]["discharge"][0] == pytest.approx(400 * 1.5 * usgs.CFS_TO_M3S)
        for r in records
    )


def test_long_gap_takes_the_daily_mean():
    holes = [("00000001", _ts("2026-04-01T12:00"), _ts("2026-04-03T00:00"))]
    get = FakeUSGS(
        {"00000001": 300.0, "00000002": 100.0},
        holes=holes,
        daily={"00000001": {"2026-04-02": 500.0}},
    )
    records = list(
        usgs_rivers.iter_rivers(
            UTC, 6, [-73, 41, -72, 42], rivers=[RIVER], get=get, max_gap_hours=2
        )
    )[1:]
    assert all(r[2]["fill"][0] == usgs_rivers.FILL_DAILY_MEAN for r in records)
    assert records[0][2]["discharge"][0] == pytest.approx(600 * 1.5 * usgs.CFS_TO_M3S)


def test_unfillable_gap_is_an_error_unless_climatology_is_allowed():
    holes = [("00000001", _ts("2026-04-01T00:00"), _ts("2026-04-04T00:00"))]
    get = FakeUSGS(
        {"00000001": 300.0, "00000002": 100.0}, holes=holes, median={"00000001": 200.0}
    )
    with pytest.raises(ValueError, match="no discharge"):
        list(
            usgs_rivers.iter_rivers(UTC, 6, [-73, 41, -72, 42], rivers=[RIVER], get=get)
        )
    items = list(
        usgs_rivers.iter_rivers(
            UTC, 6, [-73, 41, -72, 42], rivers=[RIVER], get=get, allow_climatology=True
        )
    )
    records = items[1:]
    assert all(r[2]["fill"][0] == usgs_rivers.FILL_CLIMATOLOGY for r in records)
    assert records[0][2]["discharge"][0] == pytest.approx(300 * 1.5 * usgs.CFS_TO_M3S)
    prov = json.loads(items[0][1].attrs["provenance"])[0]
    assert prov["hours_filled"]["climatology_median"] == 7


def test_provisional_values_and_temperature_are_carried():
    get = FakeUSGS(
        {"00000001": 300.0, "00000002": 100.0},
        status="Provisional",
        temperature={"00000001": 7.5},
    )
    items = list(
        usgs_rivers.iter_rivers(UTC, 3, [-73, 41, -72, 42], rivers=[RIVER], get=get)
    )
    assert items[0][1].attrs["provisional"] is True
    assert items[1][2]["temperature"][0] == pytest.approx(7.5)


def test_lag_shifts_the_gauge_window():
    river = dict(RIVER, lag_hours=24)
    get = FakeUSGS({"00000001": 300.0, "00000002": 100.0})
    list(usgs_rivers.iter_rivers(UTC, 3, [-73, 41, -72, 42], rivers=[river], get=get))
    first = next(p for u, p in get.calls if u.endswith("/continuous/items"))
    assert first["time"].startswith("2026-03-31T")


def test_no_river_in_bbox_is_an_error():
    with pytest.raises(ValueError, match="no listed river mouth"):
        list(usgs_rivers.iter_rivers(UTC, 3, [0.0, 0.0, 1.0, 1.0], get=FakeUSGS({})))


def test_store_round_trip_and_cache(tmp_path):
    bbox = [-73.0, 41.0, -72.0, 42.0]
    mouths = [{"name": "Testriver", "lon": -72.5, "lat": 41.2, "comid": 1}]
    zid = dispatcher.river_key(bbox, UTC, 6, 6, False, mouths, 0.01)
    assert zid == dispatcher.river_key(bbox, UTC, 6, 6, False, mouths, 0.01)
    assert zid != dispatcher.river_key(bbox, UTC, 7, 6, False, mouths, 0.01)
    assert zid != dispatcher.river_key(bbox, UTC, 6, 6, False, mouths, 0.02)
    moved = [dict(mouths[0], comid=2)]
    assert zid != dispatcher.river_key(bbox, UTC, 6, 6, False, moved, 0.01)
    path = str(tmp_path / f"{zid}.zarr")
    get = FakeUSGS({"00000001": 300.0, "00000002": 100.0})
    out = dispatcher.dispatch_river_request(bbox, UTC, 6, path, get=get, rivers=[RIVER])
    ds = xr.open_zarr(out)
    assert ds.sizes == {"time": 7, "river": 1}
    assert ds.attrs["complete"] is True
    assert ds["discharge"].attrs["units"] == "m3 s-1"
    assert float(ds["discharge"][0, 0]) == pytest.approx(
        400 * 1.5 * usgs.CFS_TO_M3S, rel=1e-6
    )
    assert json.loads(ds.attrs["river_names"]) == ["Testriver"]
    sidecar = json.loads((tmp_path / f"{zid}.json").read_text())
    assert sidecar["rivers"][0]["name"] == "Testriver"
    assert sidecar["schema"] == "river-v1"

    calls = len(get.calls)
    dispatcher.dispatch_river_request(bbox, UTC, 6, path, get=get, rivers=[RIVER])
    assert len(get.calls) == calls  # served from the cache


def test_stale_provisional_store_is_rebuilt(tmp_path):
    bbox = [-73.0, 41.0, -72.0, 42.0]
    path = str(tmp_path / "riv_test.zarr")
    get = FakeUSGS({"00000001": 300.0, "00000002": 100.0}, status="Provisional")
    dispatcher.dispatch_river_request(bbox, UTC, 3, path, get=get, rivers=[RIVER])
    calls = len(get.calls)
    dispatcher.dispatch_river_request(bbox, UTC, 3, path, get=get, rivers=[RIVER])
    assert len(get.calls) == calls  # fresh: served

    import zarr

    zarr.open_group(path, mode="r+", zarr_format=2).attrs["built_utc"] = (
        "2026-01-01T00:00:00Z"
    )
    dispatcher.dispatch_river_request(bbox, UTC, 3, path, get=get, rivers=[RIVER])
    assert len(get.calls) > calls  # stale provisional: rebuilt


# --- deriving a river's gauges ------------------------------------------------------------

NLDI, OGC = usgs.NLDI, usgs.OGC
START = pd.Timestamp("2026-04-02T00:00Z")
END = pd.Timestamp("2026-04-04T00:00Z")
LONG_AGO = "1990-01-01T00:00:00+00:00"
RECENT = "2026-10-01T00:00:00+00:00"

# The outlet basin drains 1000 sq mi (polygon_area_sqmi is patched below).
SITES = {
    # id: (site type, area, record end, sites downstream of it)
    "USGS-A": ("ST", 800.0, RECENT, ["USGS-D"]),  # main stem, lowest tide-free gauge
    "USGS-B": ("ST", 500.0, RECENT, ["USGS-A", "USGS-D"]),  # above A on the main stem
    "USGS-C": ("ST", 100.0, RECENT, ["USGS-D"]),  # tributary joining below A
    "USGS-D": ("ST-TS", 950.0, RECENT, []),  # tidal reach
    "USGS-E": ("ST", 5.0, RECENT, ["USGS-D"]),  # headwater brook, 0.5 percent
    "USGS-F": ("ST", 50.0, "2000-09-30T00:00:00+00:00", ["USGS-D"]),  # discontinued
    "USGS-G": ("LK", 300.0, RECENT, ["USGS-D"]),  # a lake gauge
}


class FakeServices:
    def __init__(self, sites=SITES):
        self.sites = sites
        self.calls = []

    def __call__(self, url, params):
        self.calls.append(url)
        if url == f"{NLDI}/comid/100":
            return _fc(
                [
                    {
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [[-72.1, 41.1], [-72.0, 41.0]],
                        },
                        "properties": {},
                    }
                ]
            )
        if url == f"{NLDI}/comid/100/basin":
            return _fc(
                [
                    {
                        "geometry": {"type": "Polygon", "coordinates": [[]]},
                        "properties": {},
                    }
                ]
            )
        if url == f"{NLDI}/comid/100/navigation/UT/nwissite":
            return _fc([{"properties": {"identifier": s}} for s in self.sites])
        if url == f"{NLDI}/nwissite/USGS-A/navigation/DM/flowlines":
            return _fc(
                [{"properties": {"nhdplus_comid": c}} for c in ("98", "99", "100")]
            )
        if url.startswith(f"{NLDI}/nwissite/") and url.endswith(
            "/navigation/DM/nwissite"
        ):
            site = url.split("/nwissite/")[1].split("/")[0]
            below = self.sites[site][3]
            return _fc([{"properties": {"identifier": s}} for s in [site, *below]])
        if url == f"{OGC}/monitoring-locations/items":
            ids = params["id"].split(",")
            return _fc(
                [
                    {
                        "properties": {
                            "id": s,
                            "site_type_code": self.sites[s][0],
                            "drainage_area": self.sites[s][1],
                            "monitoring_location_name": f"Gauge {s[-1]}",
                        }
                    }
                    for s in ids
                    if s in self.sites
                ]
            )
        if url == f"{OGC}/time-series-metadata/items":
            ids = params["monitoring_location_id"].split(",")
            return _fc(
                [
                    {
                        "properties": {
                            "monitoring_location_id": s,
                            "statistic_id": "00011",
                            "begin_utc": LONG_AGO,
                            "end_utc": self.sites[s][2],
                        }
                    }
                    for s in ids
                ]
            )
        raise AssertionError(f"unexpected {url} {params}")


def _fc(features):
    return json.dumps({"type": "FeatureCollection", "features": features})


@pytest.fixture
def basin_area(monkeypatch):
    monkeypatch.setattr(usgs, "polygon_area_sqmi", lambda geometry: 1000.0)


MOUTH = {"name": "Synthetic", "lon": -72.0, "lat": 41.0, "comid": 100}


def test_keeps_the_lowest_tide_free_gauge_on_each_branch(basin_area):
    r = usgs_rivers.derive_river(MOUTH, START, END, FakeServices())
    assert [g["usgs_id"] for g in r["gauges"]] == ["A", "C"]
    assert r["area_mouth_sqmi"] == 1000.0
    assert r["mouth"] == {"lon": -72.0, "lat": 41.0, "comid": 100}
    d = r["dropped"]
    assert d["USGS-B"] == "upstream of another gauge (USGS-A)"
    assert d["USGS-D"] == "site type ST-TS"
    assert d["USGS-E"].startswith("drains under 1.0%")
    assert d["USGS-F"] == "no discharge record over the window"
    assert d["USGS-G"] == "site type LK"


def test_min_share_and_window_change_the_selection(basin_area):
    # With no area threshold the brook joins; in 1995 the discontinued gauge was running.
    r = usgs_rivers.derive_river(MOUTH, START, END, FakeServices(), min_share=0.0)
    assert [g["usgs_id"] for g in r["gauges"]] == ["A", "C", "E"]
    old = usgs_rivers.derive_river(
        MOUTH,
        pd.Timestamp("1995-04-02T00:00Z"),
        pd.Timestamp("1995-04-04T00:00Z"),
        FakeServices(),
    )
    assert [g["usgs_id"] for g in old["gauges"]] == ["A", "C", "F"]


def test_include_and_exclude_override_the_rules(basin_area):
    r = usgs_rivers.derive_river(dict(MOUTH, exclude=["C"]), START, END, FakeServices())
    assert [g["usgs_id"] for g in r["gauges"]] == ["A"]
    assert r["dropped"]["USGS-C"] == "excluded by the mouth entry"
    # Pinning a gauge upstream of another double counts, which is refused.
    with pytest.raises(ValueError, match="double count"):
        usgs_rivers.derive_river(
            dict(MOUTH, include=["USGS-B"]), START, END, FakeServices()
        )


def test_no_gauge_over_the_window_is_a_distinct_error(basin_area):
    ended = {
        s: (t, a, "2000-01-01T00:00:00+00:00", d) for s, (t, a, _, d) in SITES.items()
    }
    with pytest.raises(usgs_rivers.NoGaugeError):
        usgs_rivers.derive_river(MOUTH, START, END, FakeServices(ended))


def test_outlet_from_a_downstream_trace(basin_area):
    mouth = {"name": "Traced", "lon": -72.0, "lat": 41.0, "trace_from": "USGS-A"}
    assert usgs_rivers.outlet_comid(mouth, FakeServices()) == 100


def test_static_facts_and_downstream_sets_are_cached(basin_area, tmp_path):
    first = FakeServices()
    usgs_rivers.derive_river(MOUTH, START, END, first, cache_dir=str(tmp_path))
    second = FakeServices()
    usgs_rivers.derive_river(MOUTH, START, END, second, cache_dir=str(tmp_path))
    # Only the per-window metadata is fetched again.
    assert not any("/nldi/" in u for u in second.calls)
    assert len(second.calls) < len(first.calls)


def test_record_coverage_allows_a_real_time_lag():
    now = pd.Timestamp.now(tz="UTC")
    lagging = [("00011", pd.Timestamp(LONG_AGO), now - pd.Timedelta(hours=2))]
    assert usgs_rivers.covers(lagging, now - pd.Timedelta(days=2), now)
    assert not usgs_rivers.covers(lagging, pd.Timestamp("1980-01-01T00:00Z"), now)
