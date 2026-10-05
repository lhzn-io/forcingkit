import os
import logging
import pandas as pd
from forcingkit import settings

logger = logging.getLogger(__name__)

BASE_ERDDAP_URL = "http://merlin.dms.uconn.edu:8080/erddap"


def fetch_erddap_station_profiles(
    station_id: str,
    start_time: str,
    end_time: str,
    cache_dir: str = os.path.join(
        settings.cache_dir(),
        "erddap",
    ),
    cache_bust: bool = False,
) -> dict:
    """
    Fetches 3-depth temperature profiles for a given station.
    Targeting LIS stations via UConn ERDDAP.

    Args:
        station_id: Station ID, e.g., "WLIS", "EXRX"
        start_time: ISO8601 string (e.g., "YYYY-MM-DDTHH:MM:SSZ")
        end_time: ISO8601 string

    Returns:
        dictionary with keys: "surface", "mid", "bottom", each containing a dictionary of time (ISO) -> temperature (float)
    """
    os.makedirs(cache_dir, exist_ok=True)

    station_upper = station_id.upper()
    datasets = {
        "surface": f"{station_upper}_WQ_SFC",
        "mid": f"{station_upper}_WQ_MID",
        "bottom": f"{station_upper}_WQ_BTM",
    }

    # Handle irregular dataset names in LISICOS
    if station_upper == "ARTG":
        datasets["bottom"] = "ARTG_WQ_BTM_1"

    profiles = {}

    for level, dataset_name in datasets.items():
        cache_key = f"{dataset_name}_{start_time}_{end_time}".replace(":", "-").replace(
            "Z", ""
        )
        cache_path = os.path.join(cache_dir, f"{cache_key}.csv")

        if not cache_bust and os.path.exists(cache_path):
            logger.info(f"Cache hit for {dataset_name}: {cache_path}")
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        else:
            query_url = f"{BASE_ERDDAP_URL}/tabledap/{dataset_name}.csvp?time,sea_water_temperature&time%3E={start_time}&time%3C={end_time}"

            logger.info(
                f"Fetching ERDDAP data for {dataset_name} ({start_time} to {end_time})..."
            )
            try:
                df = pd.read_csv(query_url)
                if df is not None and not df.empty:
                    # The headers will be like "time (UTC)" and "sea_water_temperature (celsius)"
                    # We can normalize them
                    df.columns = ["time", "sea_water_temperature"]
                    df = df.set_index("time")
                    df.index = pd.to_datetime(df.index, utc=True)
                    df.to_csv(cache_path)
            except Exception as e:
                logger.error(
                    f"Failed fetching or parsing ERDDAP for {dataset_name}: {e}"
                )
                df = None

        if df is not None and not df.empty:
            df = df.dropna(subset=["sea_water_temperature"])
            # We must convert timestamp back to isoformat strings to match json expectations if returned by API
            time_dict = {
                ts.isoformat(): temp for ts, temp in df["sea_water_temperature"].items()
            }
            profiles[level] = time_dict

    return profiles


KNOWN_STATIONS = {
    "WLIS": {"lat": 40.96, "lon": -73.58},
    "EXRX": {"lat": 40.88, "lon": -73.73},
    "CLIS": {"lat": 41.14, "lon": -72.66},
    "ARTG": {"lat": 41.01, "lon": -73.29},
}


def fetch_erddap_stations_in_bbox(
    bbox: list[float],
    start_time: str,
    end_time: str,
    cache_dir: str = os.path.join(
        settings.cache_dir(),
        "erddap",
    ),
    cache_bust: bool = False,
) -> dict:
    """
    Finds all known LIS stations within the bounding box and fetches their profiles.
    bbox: [max_lat, min_lon, min_lat, max_lon] or [min_lon, min_lat, max_lon, max_lat]
    """
    # Accept standard Julia order (min_lon, min_lat, max_lon, max_lat)
    if len(bbox) == 4:
        min_lon, min_lat, max_lon, max_lat = bbox
    else:
        return {}

    stations_to_fetch = []
    for st_id, coords in KNOWN_STATIONS.items():
        if min_lat <= coords["lat"] <= max_lat and min_lon <= coords["lon"] <= max_lon:
            stations_to_fetch.append(st_id)

    # fallback to EXRX if none in bounding box for testing
    if not stations_to_fetch:
        logger.warning(
            f"No stations found in bbox {bbox}. Falling back to default proxy stations (EXRX, WLIS)"
        )
        stations_to_fetch = ["EXRX", "WLIS"]

    results = {}
    for st_id in stations_to_fetch:
        logger.info(f"Fetching station {st_id} for bounding box request...")
        profile = fetch_erddap_station_profiles(
            st_id, start_time, end_time, cache_dir, cache_bust
        )
        if profile and any(v for v in profile.values()):
            results[st_id] = {
                "lat": KNOWN_STATIONS.get(st_id, {}).get("lat", 0.0),
                "lon": KNOWN_STATIONS.get(st_id, {}).get("lon", 0.0),
                "profiles": profile,
            }

    return results
