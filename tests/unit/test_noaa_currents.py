"""CO-OPS harmonic current predictions: parsing, units, caching, and subordinate stations."""

import pytest
from insitukit import coops

from forcingkit.fetchers import noaa

STATION = {"name": "Stratford Shoal, 2 miles south of", "lat": 41.023, "lng": -73.1048}


def answer(monkeypatch, predictions=None, listing=None):
    """Answer insitukit's CO-OPS data and metadata calls; return the list of calls made."""
    calls = []

    def get(params):
        calls.append(("data", params))
        return predictions

    def md(path, params=None):
        calls.append(("md", path, params))
        return listing if path == "stations.json" else {"stations": [STATION]}

    monkeypatch.setattr(coops, "_get", get)
    monkeypatch.setattr(coops, "_md", md)
    return calls


def test_harmonic_series_in_metres_per_second(tmp_path, monkeypatch):
    rows = [
        {
            "meanFloodDir": 272,
            "Bin": "15",
            "meanEbbDir": 79,
            "Time": "2026-04-02 00:00",
            "Depth": "6.7",
            "Velocity_Major": 30.3,
        },
        {
            "meanFloodDir": 272,
            "Bin": "15",
            "meanEbbDir": 79,
            "Time": "2026-04-02 00:30",
            "Depth": "6.7",
            "Velocity_Major": -44.0,
        },
    ]
    calls = answer(
        monkeypatch, {"current_predictions": {"units": "meters, cm/s", "cp": rows}}
    )
    data = noaa.fetch_noaa_current_predictions(
        "LIS1027",
        15,
        "2026-04-02T00:00:00",
        "2026-04-02T00:30:00",
        cache_dir=str(tmp_path),
    )
    assert data["metadata"]["depth_m"] == 6.7
    assert data["metadata"]["flood_dir_deg"] == 272.0
    assert data["metadata"]["lat"] == 41.023
    assert [r["t"] for r in data["data"]] == ["2026-04-02 00:00", "2026-04-02 00:30"]
    assert [r["v"] for r in data["data"]] == pytest.approx([0.303, -0.44])
    request = calls[0][1]
    assert request["bin"] == 15 and request["interval"] == "30"
    assert request["vel_type"] == "default"
    # A second call is served from the cache.
    again = noaa.fetch_noaa_current_predictions(
        "LIS1027",
        15,
        "2026-04-02T00:00:00",
        "2026-04-02T00:30:00",
        cache_dir=str(tmp_path),
    )
    assert again == data
    assert len(calls) == 2


def test_subordinate_station_is_refused(tmp_path, monkeypatch):
    rows = [
        {
            "Type": "flood",
            "meanFloodDir": 267,
            "Bin": "1",
            "meanEbbDir": 80,
            "Time": "2026-04-02 01:32",
            "Depth": "4.6",
            "Velocity_Major": 58.3,
        }
    ]
    answer(monkeypatch, {"current_predictions": {"cp": rows}})
    with pytest.raises(RuntimeError, match="subordinate"):
        noaa.fetch_noaa_current_predictions(
            "ACT2991",
            1,
            "2026-04-02T00:00:00",
            "2026-04-02T06:00:00",
            cache_dir=str(tmp_path),
        )


def test_station_listing_keeps_harmonic_bins_in_the_box(tmp_path, monkeypatch):
    listing = {
        "stations": [
            {
                "id": "LIS1027",
                "type": "H",
                "lat": 41.023,
                "lng": -73.105,
                "currbin": 15,
                "depth": 6.71,
                "name": "Stratford Shoal",
            },
            {
                "id": "LIS1027",
                "type": "H",
                "lat": 41.023,
                "lng": -73.105,
                "currbin": 1,
                "depth": 34.75,
                "name": "Stratford Shoal",
            },
            {
                "id": "ACT2991",
                "type": "S",
                "lat": 41.05,
                "lng": -73.097,
                "currbin": 1,
                "depth": 4.57,
                "name": "subordinate",
            },
            {
                "id": "LIS1028",
                "type": "W",
                "lat": 41.15,
                "lng": -73.18,
                "currbin": 17,
                "depth": 1.22,
                "name": "weak",
            },
            {
                "id": "LIS1001",
                "type": "H",
                "lat": 41.24,
                "lng": -72.06,
                "currbin": 5,
                "depth": 10.0,
                "name": "outside",
            },
        ]
    }
    calls = answer(monkeypatch, listing=listing)
    out = noaa.fetch_noaa_current_stations(
        [-73.86, 40.75, -72.84, 41.35], cache_dir=str(tmp_path)
    )
    assert [(e["id"], e["bin"]) for e in out] == [("LIS1027", 1), ("LIS1027", 15)]
    assert out[1]["depth_m"] == 6.71
    assert calls[0][2] == {"type": "currentpredictions", "units": "metric"}
    # The listing is cached: a second call does not fetch.
    noaa.fetch_noaa_current_stations(
        [-73.86, 40.75, -72.84, 41.35], cache_dir=str(tmp_path)
    )
    assert len(calls) == 1


def test_cached_predictions_expire_after_thirty_days(tmp_path, monkeypatch):
    import os

    rows = [
        {
            "meanFloodDir": 272,
            "Bin": "15",
            "meanEbbDir": 79,
            "Time": "2026-04-02 00:00",
            "Depth": "6.7",
            "Velocity_Major": 30.3,
        }
    ]
    calls = answer(monkeypatch, {"current_predictions": {"cp": rows}})
    args = ("LIS1027", 15, "2026-04-02T00:00:00", "2026-04-02T00:30:00")
    noaa.fetch_noaa_current_predictions(*args, cache_dir=str(tmp_path))
    noaa.fetch_noaa_current_predictions(*args, cache_dir=str(tmp_path))
    assert len(calls) == 2  # one prediction request and one metadata request
    for p in tmp_path.glob("currents_*.json"):
        t = p.stat().st_mtime - noaa.PREDICTIONS_TTL_S - 60
        os.utime(p, (t, t))
    noaa.fetch_noaa_current_predictions(*args, cache_dir=str(tmp_path))
    assert len(calls) == 4
