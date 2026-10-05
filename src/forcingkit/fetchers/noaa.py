import os
import logging
import requests
import json
from datetime import datetime
from forcingkit import settings

logger = logging.getLogger(__name__)


def fetch_noaa_tide_data(
    station_id: str,
    start_time: str,
    end_time: str,
    cache_dir: str = os.path.join(
        settings.cache_dir(),
        "noaa",
    ),
    cache_bust: bool = False,
) -> dict:
    """
    Fetches water level data from NOAA CO-OPS API for a specific station and time window.
    Supports caching to avoid redundant API calls.

    Args:
        station_id: NOAA station ID (e.g., '8516945' for Kings Point)
        start_time: ISO8601 string (YYYY-MM-DDTHH:MM:SSZ)
        end_time: ISO8601 string (YYYY-MM-DDTHH:MM:SSZ)
        cache_dir: Local directory for caching JSON responses
    """
    os.makedirs(cache_dir, exist_ok=True)

    # Generate a cache key based on station and time
    cache_key = f"{station_id}_{start_time}_{end_time}".replace(":", "-")
    cache_path = os.path.join(cache_dir, f"{cache_key}.json")

    if not cache_bust and os.path.exists(cache_path):
        logger.info(f"Cache hit for NOAA station {station_id}: {cache_path}")
        with open(cache_path, "r") as f:
            return json.load(f)

    logger.info(
        f"Fetching NOAA tide data for station {station_id} ({start_time} to {end_time})..."
    )

    # NOAA API expects yyyyMMdd HH:mm
    # Note: We assume the input strings are ISO8601 UTC
    s_dt = datetime.fromisoformat(start_time.replace("Z", ""))
    e_dt = datetime.fromisoformat(end_time.replace("Z", ""))

    begin_str = s_dt.strftime("%Y%m%d %H:%M")
    end_str = e_dt.strftime("%Y%m%d %H:%M")

    url = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
    params = {
        "begin_date": begin_str,
        "end_date": end_str,
        "station": station_id,
        "product": "water_level",
        "datum": "NAVD",
        "units": "metric",
        "time_zone": "gmt",
        "format": "json",
        "application": "lhzn_coastal_sim",
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        if "error" in data:
            logger.warning(
                f"NOAA API returned error: {data['error'].get('message', 'Unknown error')}"
            )
            # Return a dummy fallback or raise? For now, we'll raise to let dispatcher handle it
            raise RuntimeError(f"NOAA API Error: {data['error'].get('message')}")

        # Cache the successful response
        with open(cache_path, "w") as f:
            json.dump(data, f, indent=4)

        return data

    except Exception as e:
        logger.error(f"Failed to fetch NOAA tide data: {e}")
        raise e
