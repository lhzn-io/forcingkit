"""CO-OPS water level: NAVD88 where a station has it, MSL where it does not."""

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
    }
    # The reading at end_time is kept; a missing reading is an empty string, as CO-OPS sends it.
    assert data["data"] == [
        {"t": "2026-04-02 00:00", "v": "0.1"},
        {"t": "2026-04-02 04:00", "v": ""},
    ]


def test_an_empty_window_is_an_error(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="No data"):
        _fetch(tmp_path, monkeypatch, {"NAVD": None})
