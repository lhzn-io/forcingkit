import pytest
from forcingkit.fetchers.erddap import fetch_erddap_station_profiles


@pytest.mark.asyncio
async def test_erddap_fetch():
    start_str = "2024-05-01T00:00:00Z"
    end_str = "2024-05-01T06:00:00Z"

    profiles = fetch_erddap_station_profiles("WLIS", start_str, end_str)

    assert "mid" in profiles

    # Check if there is data inside
    mid = profiles["mid"]
    assert len(mid) > 0
