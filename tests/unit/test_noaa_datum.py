"""CO-OPS water level: NAVD88 where a station has it, MSL where it does not."""

import os

import pytest
from insitukit import coops

from forcingkit.fetchers import noaa

STATION = {"name": "New Haven", "lat": 41.2833, "lng": -72.9083}


def _fetch(tmp_path, monkeypatch, answers):
    """Run the fetcher with insitukit's CO-OPS calls answered by ``answers[datum]``."""
    calls = []

    def fake_get(params):
        calls.append(params["datum"])
        answer = answers[params["datum"]]
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(coops, "_get", fake_get)
    monkeypatch.setattr(coops, "_md", lambda path, params=None: {"stations": [STATION]})
    data = noaa.fetch_noaa_tide_data(
        "8465705",
        "2026-04-02T00:00:00",
        "2026-04-02T04:00:00",
        cache_dir=str(tmp_path),
    )
    return data, calls


ROWS = {
    "data": [
        {"t": "2026-04-02 00:00", "v": "0.1", "s": "0.003", "f": "0,0,0,0", "q": "p"},
        {"t": "2026-04-02 04:00", "v": "", "s": "", "f": "1,1,1,1", "q": "p"},
    ]
}


def test_falls_back_to_msl_when_navd_is_unsupported(tmp_path, monkeypatch):
    unsupported = coops.DatumUnavailable(
        " The supported Datum values are: MHHW, MHW, MTL, MSL"
    )
    data, calls = _fetch(tmp_path, monkeypatch, {"NAVD": unsupported, "MSL": ROWS})
    assert calls == ["NAVD", "MSL"]
    assert data["metadata"]["datum"] == "MSL"


def test_keeps_the_co_ops_shape_with_the_end_included(tmp_path, monkeypatch):
    data, calls = _fetch(tmp_path, monkeypatch, {"NAVD": ROWS})
    assert calls == ["NAVD"]
    assert data["metadata"] == {
        "id": "8465705",
        "name": "New Haven",
        "lat": "41.2833",
        "lon": "-72.9083",
        "datum": "NAVD",
        "quality": "preliminary",
    }
    # The reading at end_time is kept; a missing reading is an empty string, as CO-OPS sends it.
    assert data["data"] == [
        {"t": "2026-04-02 00:00", "v": "0.1", "q": "p"},
        {"t": "2026-04-02 04:00", "v": "", "q": "p"},
    ]


def test_an_empty_window_is_an_error(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="No data"):
        _fetch(tmp_path, monkeypatch, {"NAVD": None})


VERIFIED = {"data": [{**r, "q": "v"} for r in ROWS["data"]]}


def _age(tmp_path, seconds):
    for p in tmp_path.glob("*.json"):
        t = p.stat().st_mtime - seconds
        os.utime(p, (t, t))


def test_a_preliminary_window_expires_and_a_verified_one_is_kept(tmp_path, monkeypatch):
    # Preliminary: cached for a day, then CO-OPS is asked again.
    _, calls = _fetch(tmp_path, monkeypatch, {"NAVD": ROWS})
    assert calls == ["NAVD"]
    _, calls = _fetch(tmp_path, monkeypatch, {"NAVD": ROWS})
    assert calls == []
    _age(tmp_path, noaa.PRELIMINARY_TTL_S + 60)
    # A day on, CO-OPS has verified the window: it replaces the cached copy, kept for good.
    data, calls = _fetch(tmp_path, monkeypatch, {"NAVD": VERIFIED})
    assert calls == ["NAVD"] and data["metadata"]["quality"] == "verified"
    _age(tmp_path, 365 * 86400)
    _, calls = _fetch(tmp_path, monkeypatch, {"NAVD": VERIFIED})
    assert calls == []


def test_a_cache_file_without_a_quality_level_is_fetched_again(tmp_path, monkeypatch):
    # Written by forcingkit before 0.17.0 (raw CO-OPS JSON, or the 0.14 shape): no metadata.quality.
    key = "8465705_2026-04-02T00-00-00_2026-04-02T04-00-00"
    (tmp_path / f"{key}.json").write_text(
        '{"metadata": {"id": "8465705", "datum": "NAVD"}, "data": [{"t": "2026-04-02 00:00", "v": "9.9"}]}'
    )
    data, calls = _fetch(tmp_path, monkeypatch, {"NAVD": VERIFIED})
    assert calls == ["NAVD"] and data["data"][0]["v"] == "0.1"
