import xarray as xr
import pandas as pd
import numpy as np
from unittest.mock import patch

from forcingkit.fetchers.hycom import fetch_hycom_boundary_conditions


def test_hycom_historical_stitch():
    """Test HYCOM stitching logic when crossing experiment boundary."""
    bbox = [-74.0, 40.0, -73.0, 41.0]
    # Crossing 2018-12-04
    start_date = "2018-12-03"
    duration = 48  # 2 days

    with patch("forcingkit.fetchers.hycom._fetch_hycom_data") as mock_fetch:
        # Return dummy datasets
        ds1 = xr.Dataset(
            {"u": (("time", "lat", "lon"), np.ones((24, 5, 5)))},
            coords={"time": pd.date_range(start_date, periods=24, freq="h")},
        )
        ds2 = xr.Dataset(
            {"u": (("time", "lat", "lon"), np.ones((24, 5, 5)))},
            coords={"time": pd.date_range("2018-12-04", periods=24, freq="h")},
        )
        mock_fetch.side_effect = [ds1, ds2]

        ds = fetch_hycom_boundary_conditions(start_date, duration, bbox)

        assert ds is not None
        assert len(ds.time) == 48
        assert mock_fetch.call_count == 2
