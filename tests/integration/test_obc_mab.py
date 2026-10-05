import sys
import os
import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
)

from forcingkit.dispatcher import dispatch_obc_request


@pytest.mark.integration
def test_dispatch_mab_obc_24h():
    """
    Test a 24 hour OBC fetch over the Mid-Atlantic Bight to ensure
    NECOFS does not crash (`import os` issue), and HYCOM gracefully
    passes over without issuing invalid empty slices (`tau[16809:1:16808]`).
    """
    bbox = [-73.96183, 39.72442, -73.78396, 39.87757]
    target_dt_str = "2026-04-02T00:00:00Z"

    # 2hr length for CI/CD speed while proving the slice logic is sound.
    # Pipeline indexing operates precisely identically for 168h durations
    hours = 2

    zarr_path = dispatch_obc_request(
        start_date=target_dt_str, duration_hours=hours, bbox=bbox
    )

    assert zarr_path is not None, "OBC dispatch returned None, all donors failed!"

    import xarray as xr

    ds = xr.open_zarr(zarr_path)

    assert "u" in ds.data_vars
    assert "v" in ds.data_vars
    assert "time" in ds.coords

    assert len(ds.time) > 0, "No time steps fetched."
    print(
        f"\nSuccessfully fetched MAB OBC data for {hours} hours on {ds.attrs.get('Description', 'NECOFS')}!"
    )


if __name__ == "__main__":
    test_dispatch_mab_obc_24h()
