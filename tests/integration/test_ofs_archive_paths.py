"""Live reads of the NYOFS and DBOFS nowcast archives, one date per access path.

Each case asks for 3 hours and expects the 4 whole hours back, with water in the box. Dates are
fixed so the path they exercise does not drift as time passes; see docs/source/fetchers.rst
("NOAA OFS archive locations") for the layouts.
"""

import numpy as np
import pandas as pd
import pytest

from forcingkit.fetchers.dbofs import fetch_dbofs_boundary_conditions
from forcingkit.fetchers.nyofs import fetch_nyofs_boundary_conditions

NYOFS_BOX = [-74.10, 40.60, -74.00, 40.70]  # Upper Bay and the Narrows
DBOFS_BOX = [-75.1, 38.8, -75.0, 38.9]  # Delaware Bay entrance

CASES = [
    # AWS S3 per-day layout (from 2024-11-19)
    pytest.param(
        fetch_nyofs_boundary_conditions,
        NYOFS_BOX,
        "2026-09-15T04:00:00Z",
        id="nyofs-aws",
    ),
    pytest.param(
        fetch_dbofs_boundary_conditions,
        DBOFS_BOX,
        "2026-09-15T04:00:00Z",
        id="dbofs-aws",
    ),
    # NCEI, names in use from 2024-09-09
    pytest.param(
        fetch_nyofs_boundary_conditions,
        NYOFS_BOX,
        "2024-10-15T12:00:00Z",
        id="nyofs-ncei-new",
    ),
    pytest.param(
        fetch_dbofs_boundary_conditions,
        DBOFS_BOX,
        "2024-10-15T12:00:00Z",
        id="dbofs-ncei-new",
    ),
    # NCEI, names in use before 2024-09-09
    pytest.param(
        fetch_nyofs_boundary_conditions,
        NYOFS_BOX,
        "2023-12-15T06:00:00Z",
        id="nyofs-ncei-old",
    ),
    pytest.param(
        fetch_dbofs_boundary_conditions,
        DBOFS_BOX,
        "2023-12-15T06:00:00Z",
        id="dbofs-ncei-old",
    ),
]


@pytest.mark.integration
@pytest.mark.parametrize("fetch, bbox, start", CASES)
def test_archive_path_delivers(fetch, bbox, start):
    ds = fetch(start, 3, bbox)

    assert ds is not None
    expected = pd.date_range(pd.Timestamp(start).tz_localize(None), periods=4, freq="h")
    assert list(pd.DatetimeIndex(ds.time.values)) == list(expected)
    assert set(ds.dims) == {"time", "depth", "eta", "xi"}
    u = ds.u.values
    assert np.isfinite(u).any()
    # Real currents, not undecoded fill values.
    assert np.nanmax(np.abs(u)) < 5.0
