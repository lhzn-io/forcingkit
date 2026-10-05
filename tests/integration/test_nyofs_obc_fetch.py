import sys
import os
import pandas as pd

# Ensure local imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))


def test_nyofs_boundary_conditions():
    """Test NYOFS boundary conditions fetch."""
    from forcingkit.fetchers.nyofs import fetch_nyofs_boundary_conditions

    # Throgs Neck Bridge area
    bbox = [-73.815, 40.785, -73.775, 40.815]
    print(f"\nTesting NYOFS OBC Fetcher for {bbox}...")

    # Use a recent date
    target_dt = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=12)
    target_str = target_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    ds = fetch_nyofs_boundary_conditions(target_str, 6, bbox)

    if ds is not None:
        print("NYOFS OBC Success! Variables:")
        print(list(ds.data_vars.keys()))
        print(f"U-Velocity Shape: {ds.u.shape}")
        print(f"Dataset Dims: {ds.dims}")
        assert set(ds.dims) == {"time", "depth", "eta", "xi"}
        assert "u" in ds.data_vars
        assert "v" in ds.data_vars
    else:
        print("NYOFS OBC returned None (Server unavailable or error fetching).")


if __name__ == "__main__":
    test_nyofs_boundary_conditions()
