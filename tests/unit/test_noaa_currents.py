"""CO-OPS harmonic current predictions: parsing, units, caching, and subordinate stations."""

import pytest

from forcingkit.fetchers import noaa


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        pass


STATION = {
    "stations": [
        {"name": "Stratford Shoal, 2 miles south of", "lat": 41.023, "lng": -73.1048}
    ]
}


def fake_get(predictions):
    calls = []

    def get(url, params, timeout):
        calls.append(url)
        return FakeResponse(predictions if "datagetter" in url else STATION)

    return get, calls


def test_harmonic_series_in_metres_per_second(tmp_path):
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
    get, calls = fake_get(
        {"current_predictions": {"units": "meters, cm/s", "cp": rows}}
    )
    data = noaa.fetch_noaa_current_predictions(
        "LIS1027",
        15,
        "2026-04-02T00:00:00",
        "2026-04-02T00:30:00",
        cache_dir=str(tmp_path),
        get=get,
    )
    assert data["metadata"]["depth_m"] == 6.7
    assert data["metadata"]["flood_dir_deg"] == 272.0
    assert data["metadata"]["lat"] == 41.023
    assert [r["v"] for r in data["data"]] == pytest.approx([0.303, -0.44])
    # A second call is served from the cache.
    again = noaa.fetch_noaa_current_predictions(
        "LIS1027",
        15,
        "2026-04-02T00:00:00",
        "2026-04-02T00:30:00",
        cache_dir=str(tmp_path),
        get=get,
    )
    assert again == data
    assert len(calls) == 2


def test_subordinate_station_is_refused(tmp_path):
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
    get, _ = fake_get({"current_predictions": {"cp": rows}})
    with pytest.raises(RuntimeError, match="subordinate"):
        noaa.fetch_noaa_current_predictions(
            "ACT2991",
            1,
            "2026-04-02T00:00:00",
            "2026-04-02T06:00:00",
            cache_dir=str(tmp_path),
            get=get,
        )


def test_station_listing_keeps_harmonic_bins_in_the_box(tmp_path):
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
    calls = []

    def get(url, params, timeout):
        calls.append(url)
        return FakeResponse(listing)

    out = noaa.fetch_noaa_current_stations(
        [-73.86, 40.75, -72.84, 41.35], cache_dir=str(tmp_path), get=get
    )
    assert [(e["id"], e["bin"]) for e in out] == [("LIS1027", 1), ("LIS1027", 15)]
    assert out[1]["depth_m"] == 6.71
    # The listing is cached: a second call does not fetch.
    noaa.fetch_noaa_current_stations(
        [-73.86, 40.75, -72.84, 41.35], cache_dir=str(tmp_path), get=get
    )
    assert len(calls) == 1
