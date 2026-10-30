"""NWPS surface waves: run selection, GRIB2 decoding, sources, the store and its cache key.

GRIB2 files are written at test time with ecCodes on a 3 x 2 grid, so no binary fixture is kept.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr

from forcingkit import dispatcher, settings
from forcingkit.fetchers import nwps

BBOX = [-72.6, 40.9, -72.3, 41.2]
# Grid of the synthetic files: 0..360 longitudes as NWPS writes them, ascending latitudes.
LON_360 = np.array([287.4, 287.5, 287.6])
LAT = np.array([41.0, 41.1])

# GRIB2 (discipline, category, number) of each field as NWPS writes it.
GRIB_PARAMS = {
    "swh": (10, 0, 3),
    "perpw": (10, 0, 11),
    "dirpw": (10, 0, 10),
    "shts": (10, 0, 8),
    "ws": (0, 2, 1),
    "wdir": (0, 2, 0),
}


def value(short: str, cycle: pd.Timestamp, lead: int) -> np.ndarray:
    """A field that identifies its cycle and lead, with land (NaN) at the first cell for the
    wave fields."""
    base = {
        "swh": 1.0,
        "perpw": 8.0,
        "dirpw": 10.0,
        "shts": 0.5,
        "ws": 5.0,
        "wdir": 200.0,
    }
    v = np.full((2, 3), base[short] + lead / 100.0 + cycle.hour / 1000.0)
    if short not in ("ws", "wdir"):
        v[0, 0] = np.nan
    return v


def write_grib(path, cycle: pd.Timestamp, steps=nwps.FORECAST_STEPS, analysis=True):
    import eccodes

    sample = eccodes.codes_grib_new_from_samples("regular_ll_sfc_grib2")
    leads = ([0] if analysis else []) + list(range(1, steps + 1))
    with open(path, "wb") as f:
        for lead in leads:
            for short, (disc, cat, num) in GRIB_PARAMS.items():
                g = eccodes.codes_clone(sample)
                for key, v in (
                    ("centre", 7),
                    ("discipline", disc),
                    ("parameterCategory", cat),
                    ("parameterNumber", num),
                    ("typeOfFirstFixedSurface", 1),
                    ("Ni", 3),
                    ("Nj", 2),
                    ("latitudeOfFirstGridPointInDegrees", float(LAT[0])),
                    ("latitudeOfLastGridPointInDegrees", float(LAT[-1])),
                    ("longitudeOfFirstGridPointInDegrees", float(LON_360[0])),
                    ("longitudeOfLastGridPointInDegrees", float(LON_360[-1])),
                    ("iDirectionIncrementInDegrees", 0.1),
                    ("jDirectionIncrementInDegrees", 0.1),
                    ("jScansPositively", 1),
                    ("dataDate", int(cycle.strftime("%Y%m%d"))),
                    ("dataTime", cycle.hour * 100),
                    ("stepUnits", 1),
                    ("forecastTime", lead),
                    ("bitmapPresent", 1),
                    ("missingValue", 9999.0),
                ):
                    eccodes.codes_set(g, key, v)
                vals = value(short, cycle, lead)
                eccodes.codes_set_values(
                    g, np.where(np.isnan(vals), 9999.0, vals).ravel()
                )
                f.write(eccodes.codes_get_message(g))
                eccodes.codes_release(g)
    eccodes.codes_release(sample)
    return path


def fake_cycle(cycle: pd.Timestamp, analysis: bool = True) -> nwps.CycleFields:
    """A decoded cycle without GRIB: every field holds `lead + cycle.hour / 1000`."""
    fields = {}
    for lead in range(0 if analysis else 1, nwps.FORECAST_STEPS + 1):
        v = np.full((2, 3), lead + cycle.hour / 1000.0, dtype=np.float32)
        v[0, 0] = np.nan if lead % 2 else 1.0  # wet in some steps only
        fields[lead] = {name: v.copy() for name in nwps.FIELDS}
    lon = ((LON_360 + 180.0) % 360.0) - 180.0
    return nwps.CycleFields(
        cycle, f"https://example/{cycle:%Y%m%d%H}", LAT, lon, fields
    )


def ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


def test_names_and_urls():
    c = ts("2026-10-30T12:00")
    assert nwps.file_name("okx", "CG1", c) == "okx_nwps_CG1_20261030_1200.grib2"
    assert nwps.nomads_url("okx", "CG1", c) == (
        "https://nomads.ncep.noaa.gov/pub/data/nccf/com/nwps/prod/"
        "er.20261030/okx/12/CG1/okx_nwps_CG1_20261030_1200.grib2"
    )
    assert "/sr.20261030/mfl/" in nwps.nomads_url("mfl", "CG2", c)
    assert nwps.archive_paths("gs://bucket/nwps/", "okx", "CG1", c) == [
        "gs://bucket/nwps/2026/10/okx_nwps_CG1_20261030_1200.grib2",
        "gs://bucket/nwps/20261030/12/okx_nwps_CG1_20261030_1200.grib2",
        "gs://bucket/nwps/okx_nwps_CG1_20261030_1200.grib2",
    ]


def test_office_and_grid_checked():
    assert nwps.check_wfo_grid("OKX", "cg1") == ("okx", "CG1")
    with pytest.raises(ValueError, match="unknown NWPS office"):
        nwps.check_wfo_grid("xyz", "CG1")
    with pytest.raises(ValueError, match="CG1 to CG9"):
        nwps.check_wfo_grid("okx", "CG10")


def test_record_times_bracket_the_window():
    times = nwps.record_times("2026-10-30T06:00:00Z", 4)
    assert len(times) == 7
    assert times[0] == ts("2026-10-30T05:00") and times[-1] == ts("2026-10-30T11:00")
    with pytest.raises(ValueError, match="on the hour"):
        nwps.record_times("2026-10-30T06:30:00", 4)


def test_newest_cycle_respects_window_end_and_posting_lag():
    now = ts("2026-10-30T18:00")
    # A past window: the cycle at or before its end.
    assert nwps.newest_cycle(ts("2026-10-29T05:00"), now) == ts("2026-10-29T00:00")
    # A window into the future: the newest cycle at least POSTING_LAG_HOURS old.
    assert nwps.newest_cycle(ts("2026-11-01T00:00"), now) == ts("2026-10-30T00:00")
    assert nwps.newest_cycle(ts("2026-11-01T00:00"), ts("2026-10-30T19:00")) == ts(
        "2026-10-30T12:00"
    )


def test_plan_chains_cycles_across_midnight():
    calls = []

    def load(c):
        calls.append(c)
        return fake_cycle(c)

    times = pd.date_range("2026-10-29T22:00", "2026-10-30T13:00", freq="1h")
    plan = nwps.plan_records(times, ts("2026-10-30T12:00"), load)
    used = [(t, cf.cycle, lead) for t, cf, lead in plan]
    assert used[0] == (ts("2026-10-29T22:00"), ts("2026-10-29T12:00"), 10)
    assert used[2] == (ts("2026-10-30T00:00"), ts("2026-10-30T00:00"), 0)
    assert used[-1] == (ts("2026-10-30T13:00"), ts("2026-10-30T12:00"), 1)
    assert len(calls) == len(set(calls)) == 3, "each cycle is read once"


def test_plan_without_analysis_uses_previous_cycle():
    def load(c):
        return fake_cycle(c, analysis=c != ts("2026-10-30T00:00"))

    times = pd.date_range("2026-10-29T23:00", "2026-10-30T01:00", freq="1h")
    plan = nwps.plan_records(times, ts("2026-10-30T00:00"), load)
    assert [(cf.cycle.hour, lead) for _, cf, lead in plan] == [
        (12, 11),
        (12, 12),
        (0, 1),
    ]


def test_plan_past_newest_cycle_uses_long_leads_and_skips_missing():
    def load(c):
        return None if c == ts("2026-10-30T12:00") else fake_cycle(c)

    times = pd.date_range("2026-10-30T23:00", "2026-10-31T01:00", freq="1h")
    plan = nwps.plan_records(times, ts("2026-10-30T12:00"), load)
    assert [(cf.cycle, lead) for _, cf, lead in plan] == [
        (ts("2026-10-30T00:00"), 23),
        (ts("2026-10-30T00:00"), 24),
        (ts("2026-10-30T00:00"), 25),
    ]


def test_plan_out_of_reach_is_an_error():
    times = pd.date_range("2026-10-01T00:00", periods=3, freq="1h")
    with pytest.raises(
        LookupError, match="no NWPS cycle holds 2026-10-01T00:00Z.*archive"
    ):
        nwps.plan_records(
            times, ts("2026-10-01T00:00"), lambda c: None, "needs an archive"
        )


def test_decode_cycle(tmp_path):
    cycle = ts("2026-10-30T12:00")
    path = write_grib(tmp_path / "f.grib2", cycle)
    cf = nwps.decode_cycle(str(path), cycle, BBOX)
    assert cf is not None
    np.testing.assert_allclose(cf.lon, [-72.6, -72.5, -72.4])
    np.testing.assert_allclose(cf.lat, LAT)
    assert sorted(cf.fields) == list(range(0, nwps.FORECAST_STEPS + 1))
    f = cf.fields[5]
    assert np.isnan(f["hs"][0, 0]) and np.isnan(f["dir"][0, 0])
    assert f["hs"][1, 2] == pytest.approx(1.0 + 0.05 + 0.012, abs=1e-3)
    assert f["tp"][1, 2] == pytest.approx(8.062, abs=1e-3)
    # NWPS's "from" direction becomes the direction of travel; wind stays "from".
    assert f["dir"][1, 2] == pytest.approx(190.062, abs=1e-3)
    assert f["wdir"][1, 2] == pytest.approx(200.062, abs=1e-3)
    assert (
        not (tmp_path / "f.grib2.idx").exists() and len(list(tmp_path.iterdir())) == 1
    )


def test_decode_crops_to_bbox_plus_one_cell(tmp_path):
    cycle = ts("2026-10-30T12:00")
    path = write_grib(tmp_path / "f.grib2", cycle)
    cf = nwps.decode_cycle(str(path), cycle, [-72.65, 40.9, -72.62, 41.2])
    assert cf is not None
    np.testing.assert_allclose(cf.lon, [-72.6])
    with pytest.raises(ValueError, match="outside the NWPS grid"):
        nwps.decode_cycle(str(path), cycle, [-70.0, 40.9, -69.0, 41.2])


def test_decode_incomplete_file_is_not_a_cycle(tmp_path):
    cycle = ts("2026-10-30T12:00")
    short = write_grib(tmp_path / "short.grib2", cycle, steps=10)
    assert nwps.decode_cycle(str(short), cycle, BBOX) is None
    full = write_grib(tmp_path / "full.grib2", cycle)
    (tmp_path / "cut.grib2").write_bytes(full.read_bytes()[: full.stat().st_size // 2])
    assert nwps.decode_cycle(str(tmp_path / "cut.grib2"), cycle, BBOX) is None
    with pytest.raises(RuntimeError, match="is cycle"):
        nwps.decode_cycle(str(full), ts("2026-10-30T00:00"), BBOX)


def test_decode_without_analysis(tmp_path):
    cycle = ts("2026-10-30T12:00")
    path = write_grib(tmp_path / "f.grib2", cycle, analysis=False)
    cf = nwps.decode_cycle(str(path), cycle, BBOX)
    assert cf is not None and 0 not in cf.fields and 1 in cf.fields


class _Response:
    def __init__(self, status_code, content=b""):
        self.status_code, self.content = status_code, content

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


def test_archive_read_before_nomads(tmp_path, monkeypatch):
    cycle = ts("2026-10-30T12:00")
    archive = tmp_path / "archive"
    (archive / "2026" / "10").mkdir(parents=True)
    write_grib(archive / "2026" / "10" / nwps.file_name("okx", "CG1", cycle), cycle)
    before = sorted(p.name for p in (archive / "2026" / "10").iterdir())
    monkeypatch.setenv("FORCINGKIT_NWPS_ARCHIVE", f"file://{archive}")
    monkeypatch.setattr(
        settings, "cache_dir", lambda *p: str(tmp_path.joinpath("c", *p))
    )
    monkeypatch.setattr(
        nwps.requests,
        "get",
        lambda *a, **k: pytest.fail("NOMADS asked for an archived cycle"),
    )
    cf = nwps.load_cycle("okx", "CG1", cycle, BBOX, str(tmp_path))
    assert cf is not None and cf.origin.startswith("file://")
    assert sorted(p.name for p in (archive / "2026" / "10").iterdir()) == before


def test_nomads_fallback_keeps_only_complete_files(tmp_path, monkeypatch):
    cycle = ts("2026-10-30T12:00")
    full = write_grib(tmp_path / "full.grib2", cycle).read_bytes()
    short = write_grib(tmp_path / "short.grib2", cycle, steps=3).read_bytes()
    monkeypatch.delenv("FORCINGKIT_NWPS_ARCHIVE", raising=False)
    monkeypatch.setattr(
        settings, "cache_dir", lambda *p: str(tmp_path.joinpath("c", *p))
    )
    raw = tmp_path / "c" / "nwps" / "okx" / "CG1"
    responses = [_Response(404), _Response(200, short), _Response(200, full)]
    urls = []

    def get(url, **kwargs):
        urls.append(url)
        assert kwargs["headers"]["User-Agent"].startswith("forcingkit")
        return responses.pop(0)

    monkeypatch.setattr(nwps.requests, "get", get)
    assert nwps.load_cycle("okx", "CG1", cycle, BBOX, str(tmp_path)) is None  # 404
    assert (
        nwps.load_cycle("okx", "CG1", cycle, BBOX, str(tmp_path)) is None
    )  # still posting
    assert list(raw.iterdir()) == []
    cf = nwps.load_cycle("okx", "CG1", cycle, BBOX, str(tmp_path))
    assert cf is not None and cf.origin == nwps.nomads_url("okx", "CG1", cycle)
    assert [p.name for p in raw.iterdir()] == [nwps.file_name("okx", "CG1", cycle)]
    # Kept files are reused without another request.
    assert nwps.load_cycle("okx", "CG1", cycle, BBOX, str(tmp_path)) is not None
    assert len(urls) == 3


def test_iter_waves_static_and_records():
    items = list(
        nwps.iter_waves(
            "2026-10-30T02:00:00",
            3,
            BBOX,
            latest_cycle=ts("2026-10-30T00:00"),
            loader=fake_cycle,
        )
    )
    kind, static = items[0]
    assert kind == "static" and static.attrs["schema"] == nwps.NWPS_SCHEMA
    assert static.attrs["first_cycle"] == "2026-10-30T00Z"
    assert static.attrs["record_cycles"] == ["2026-10-30T00Z"] * 6
    assert (
        static.attrs["dx"] == pytest.approx(0.1) and static.attrs["crs"] == "EPSG:4326"
    )
    # The land cell is wet in even steps, so the union over the cycle marks it wet.
    assert static["wet"].dtype == np.uint8 and static["wet"].values.all()
    records = items[1:]
    assert [r[1] for r in records] == list(
        pd.date_range("2026-10-30T01:00", periods=6, freq="1h")
    )
    assert [float(r[2]["forecast_hour"]) for r in records] == [1, 2, 3, 4, 5, 6]


def test_dispatch_writes_a_complete_store(tmp_path, monkeypatch):
    monkeypatch.setattr(
        nwps, "load_cycle", lambda wfo, grid, c, bbox, workdir: fake_cycle(c)
    )
    path = str(tmp_path / "wav_0123456789ab.zarr")
    out = dispatcher.dispatch_waves_request(
        BBOX, "2026-10-30T02:00:00", 3, path, latest_cycle=ts("2026-10-30T00:00")
    )
    assert out == path
    attrs = dict(zarr.open_group(path, mode="r", zarr_format=2).attrs)
    assert attrs["complete"] is True and attrs["records"] == 6
    assert attrs["schema"] == nwps.NWPS_SCHEMA and attrs["wfo"] == "okx"
    ds = xr.open_zarr(path, decode_times=False)
    assert ds["time"].attrs["units"] == "seconds since 2026-10-30T01:00:00"
    assert ds["dir"].attrs["standard_name"] == "sea_surface_wave_to_direction"
    assert ds["hs"].attrs["units"] == "m" and ds["hs"].dims == ("time", "lat", "lon")
    assert ds["forecast_hour"].dims == ("time",)
    assert ds["wet"].dims == ("lat", "lon")


def test_waves_key():
    args = (BBOX, "2026-10-30T02:00:00", 3, "okx", "CG1", ts("2026-10-30T00:00"))
    key = dispatcher.waves_key(*args)
    assert key.startswith("wav_") and len(key) == 16
    assert dispatcher.waves_key(*args) == key
    assert dispatcher.waves_key(BBOX, *args[1:3], "OKX", "cg1", args[5]) == key
    for i, changed in (
        (1, "2026-10-30T03:00:00"),
        (2, 4),
        (3, "box"),
        (4, "CG2"),
        (5, ts("2026-10-30T12:00")),
    ):
        other = list(args)
        other[i] = changed
        assert dispatcher.waves_key(*other) != key, f"argument {i} is not in the key"
