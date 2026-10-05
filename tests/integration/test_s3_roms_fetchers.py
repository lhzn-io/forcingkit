import pytest
import datetime
from forcingkit.fetchers.dbofs import fetch_dbofs_boundary_conditions
from forcingkit.fetchers.nyofs import fetch_nyofs_boundary_conditions


@pytest.mark.integration
def test_dbofs_s3_fetch_small_slice():
    """
    Test that the DBOFS fetcher can successfully connect to AWS S3,
    evaluate the coordinate grid optimally, and slice a tiny bounding box
    for a recent date (1-3 days ago to ensure data availability on S3 or OPeNDAP).
    """
    # 2 days ago to ensure data exists either on OPeNDAP FMRC or AWS S3 FMRC (0 <= age <= 6)
    target_date = (datetime.datetime.utcnow() - datetime.timedelta(days=2)).strftime(
        "%Y-%m-%dT00:00:00Z"
    )

    # A tiny bounding box near the Delaware Bay entrance to ensure we get a few points
    bbox = [-75.1, 38.8, -75.0, 38.9]

    # Request only 3 hours to minimize download/processing time
    duration_hours = 3

    ds = fetch_dbofs_boundary_conditions(target_date, duration_hours, bbox)

    assert ds is not None, (
        "DBOFS fetcher returned None, implying no valid points or a fetch error."
    )
    # The sliced dataset should have the expected variables
    assert "u" in ds.data_vars
    assert "v" in ds.data_vars

    # The time dimension should match duration_hours + 1 (0th hour inclusive) or depend on the interval
    assert len(ds.time) > 0
    # Sanity check dimensions
    assert len(ds.eta) < 50
    assert len(ds.xi) < 50


@pytest.mark.integration
def test_dbofs_s3_fetch_viewer_bbox():
    """
    Test using the exact bounding box from the viewer that failed.
    bbox: [-73.96183, 39.72442, -73.78396, 39.87757]
    """
    target_date = (datetime.datetime.utcnow() - datetime.timedelta(days=2)).strftime(
        "%Y-%m-%dT00:00:00Z"
    )
    bbox = [-73.96183, 39.72442, -73.78396, 39.87757]
    duration_hours = 4

    ds = fetch_dbofs_boundary_conditions(target_date, duration_hours, bbox)

    assert ds is None, (
        "DBOFS fetcher should return None because the offshore NJ bbox is outside the valid dry mask."
    )


@pytest.mark.integration
def test_nyofs_s3_fetch_small_slice():
    """
    Test that the NYOFS fetcher can successfully connect to AWS S3,
    evaluate the coordinate grid optimally, and slice a tiny bounding box.
    """
    # 2 days ago
    target_date = (datetime.datetime.utcnow() - datetime.timedelta(days=2)).strftime(
        "%Y-%m-%dT00:00:00Z"
    )

    # A tiny bounding box near NY Harbor / The Narrows
    bbox = [-74.05, 40.55, -74.0, 40.6]

    # Request only 3 hours
    duration_hours = 3

    ds = fetch_nyofs_boundary_conditions(target_date, duration_hours, bbox)

    assert ds is not None, (
        "NYOFS fetcher returned None, implying no valid points or a fetch error."
    )
    assert "u" in ds.data_vars
    assert "v" in ds.data_vars

    assert len(ds.time) > 0
    assert len(ds.eta) < 50
    assert len(ds.xi) < 50
