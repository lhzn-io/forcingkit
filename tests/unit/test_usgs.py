"""USGS reads: record parsing, site metadata batching, NLDI navigation and basin areas. No
network: a fake getter serves fixtures shaped like the live responses."""

import json

import pandas as pd
import pytest

from forcingkit.fetchers import usgs


def _fc(features):
    return json.dumps({"type": "FeatureCollection", "features": features})


def test_polygon_area_on_the_sphere():
    box = {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]}
    # A one-degree box at the equator is about 111.2 km on a side.
    assert usgs.polygon_area_sqmi(box) == pytest.approx(111.195**2 / 2.589988, rel=1e-3)
    holed = {
        "type": "MultiPolygon",
        "coordinates": [box["coordinates"]],
    }
    assert usgs.polygon_area_sqmi(holed) == pytest.approx(usgs.polygon_area_sqmi(box))


def test_site_metadata_is_read_in_batches():
    sites = [f"USGS-{n:08d}" for n in range(120)]
    batches = []

    def get(url, params):
        ids = params["id"].split(",")
        batches.append(len(ids))
        return _fc(
            [
                {
                    "properties": {
                        "id": s,
                        "site_type_code": "ST",
                        "drainage_area": 1.0,
                        "monitoring_location_name": s,
                    }
                }
                for s in ids
            ]
        )

    meta = usgs.site_metadata(sites, get)
    assert len(meta) == 120 and batches == [50, 50, 20]
    assert meta["USGS-00000007"]["site_type"] == "ST"


def test_discharge_records_keep_15_minute_and_daily_series_only():
    def get(url, params):
        rows = [
            ("00011", "1990-10-01T05:15:00+00:00", "2026-10-09T05:30:00+00:00"),
            ("00003", "1928-08-01T00:00:00+00:00", "2026-10-07T00:00:00+00:00"),
            (None, "1929-03-25T00:00:00+00:00", "2023-12-19T00:00:00+00:00"),
            ("00011", None, None),
        ]
        return _fc(
            [
                {
                    "properties": {
                        "monitoring_location_id": "USGS-01184000",
                        "statistic_id": st,
                        "begin_utc": b,
                        "end_utc": e,
                    }
                }
                for st, b, e in rows
            ]
        )

    records = usgs.discharge_records(["USGS-01184000"], get)["USGS-01184000"]
    assert [r[0] for r in records] == ["00011", "00003"]
    assert records[0][1] == pd.Timestamp("1990-10-01T05:15:00+00:00")


def test_nldi_navigation():
    def get(url, params):
        if url.endswith("/nwissite/USGS-A/navigation/DM/flowlines"):
            return _fc([{"properties": {"nhdplus_comid": c}} for c in ("7", "8", "9")])
        if url.endswith("/nwissite/USGS-A/navigation/DM/nwissite"):
            return _fc(
                [{"properties": {"identifier": s}} for s in ("USGS-A", "USGS-B")]
            )
        if url.endswith("/comid/9/navigation/UT/nwissite"):
            return _fc(
                [{"properties": {"identifier": s}} for s in ("USGS-B", "USGS-A")]
            )
        if url.endswith("/comid/9"):
            return _fc(
                [
                    {
                        "geometry": {
                            "type": "MultiLineString",
                            "coordinates": [[[0, 0], [1, 1]], [[1, 1], [2, 3]]],
                        }
                    }
                ]
            )
        raise AssertionError(url)

    assert usgs.outlet_from_trace("USGS-A", get) == 9
    assert usgs.sites_downstream("USGS-A", get) == ["USGS-B"]
    assert usgs.sites_upstream(9, get) == ["USGS-A", "USGS-B"]
    assert usgs.flowline_end(9, get) == (2.0, 3.0)
    assert usgs.site_id("01184000") == "USGS-01184000"
    assert usgs.site_id("USGS-01184000") == "USGS-01184000"
