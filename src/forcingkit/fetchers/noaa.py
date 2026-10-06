import os
import logging
import requests
import json
from datetime import datetime
from forcingkit import settings

logger = logging.getLogger(__name__)


# Water-level datums to request, in order of preference.
TIDE_DATUMS = ("NAVD", "MSL")


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
        "units": "metric",
        "time_zone": "gmt",
        "format": "json",
        "application": "forcingkit",
    }

    try:
        # NAVD88 where the station has it, so stations share a datum; otherwise MSL (New Haven,
        # 8465705, publishes only tidal datums). The datum used is recorded in the metadata.
        for datum in TIDE_DATUMS:
            response = requests.get(url, params={**params, "datum": datum}, timeout=30)
            data = response.json() if response.content else {}
            message = (
                data.get("error", {}).get("message", "")
                if isinstance(data, dict)
                else ""
            )
            if "datum" in message.lower() and datum != TIDE_DATUMS[-1]:
                logger.info(
                    f"NOAA station {station_id} has no {datum} datum; trying the next"
                )
                continue
            response.raise_for_status()
            break

        if "error" in data:
            logger.warning(
                f"NOAA API returned error: {data['error'].get('message', 'Unknown error')}"
            )
            raise RuntimeError(f"NOAA API Error: {data['error'].get('message')}")
        data.setdefault("metadata", {})["datum"] = datum

        # Cache the successful response
        with open(cache_path, "w") as f:
            json.dump(data, f, indent=4)

        return data

    except Exception as e:
        logger.error(f"Failed to fetch NOAA tide data: {e}")
        raise e


CURRENTS_URL = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
STATION_URL = (
    "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations/{station}.json"
)
STATIONS_URL = "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations.json"


def fetch_noaa_current_stations(
    bbox: list[float],
    cache_dir: str = os.path.join(settings.cache_dir(), "noaa"),
    cache_bust: bool = False,
    get=requests.get,
) -> list[dict]:
    """Every harmonic CO-OPS current-prediction station and depth bin inside ``bbox``
    ([min_lon, min_lat, max_lon, max_lat]).

    Returns one entry per station-bin: ``{"id", "name", "lat", "lon", "bin", "depth_m"}``.
    Harmonic stations (type H) give a predicted series; subordinate (S) and weak or variable (W)
    entries are left out. The full station list is cached for a day.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "currentpredictions_stations.json")
    fresh = os.path.exists(cache_path) and (
        datetime.now().timestamp() - os.path.getmtime(cache_path) < 86400
    )
    if fresh and not cache_bust:
        with open(cache_path) as f:
            stations = json.load(f)
    else:
        response = get(
            STATIONS_URL,
            params={"type": "currentpredictions", "units": "metric"},
            timeout=60,
        )
        response.raise_for_status()
        stations = response.json().get("stations", [])
        with open(cache_path, "w") as f:
            json.dump(stations, f)

    lon0, lat0, lon1, lat1 = bbox
    out = []
    for st in stations:
        if st.get("type") != "H":
            continue
        lat, lon = float(st["lat"]), float(st["lng"])
        if not (lon0 <= lon <= lon1 and lat0 <= lat <= lat1):
            continue
        out.append(
            {
                "id": st["id"],
                "name": st.get("name"),
                "lat": lat,
                "lon": lon,
                "bin": int(st["currbin"]),
                "depth_m": None if st.get("depth") is None else float(st["depth"]),
            }
        )
    return sorted(out, key=lambda e: (e["id"], e["bin"]))


def fetch_noaa_current_predictions(
    station_id: str,
    bin_number: int,
    start_time: str,
    end_time: str,
    interval_minutes: int = 30,
    cache_dir: str = os.path.join(settings.cache_dir(), "noaa"),
    cache_bust: bool = False,
    get=requests.get,
) -> dict:
    """Harmonic current predictions for one CO-OPS station and depth bin.

    Returns ``{"metadata": {...}, "data": [{"t": "YYYY-MM-DD HH:MM", "v": m/s}, ...]}``: the
    predicted velocity along the station's flood-ebb axis, positive on the flood, in m/s, every
    ``interval_minutes``. The metadata carries the station's position, the bin depth (metres
    below the surface), and the mean flood and ebb directions (degrees true, toward which the
    water flows). Only harmonic stations give a series; subordinate stations give slack and
    maximum times only and are refused.
    """
    os.makedirs(cache_dir, exist_ok=True)
    key = f"currents_{station_id}_b{bin_number}_{start_time}_{end_time}_{interval_minutes}"
    cache_path = os.path.join(cache_dir, key.replace(":", "-") + ".json")
    if not cache_bust and os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)

    s_dt = datetime.fromisoformat(start_time.replace("Z", ""))
    e_dt = datetime.fromisoformat(end_time.replace("Z", ""))
    params = {
        "product": "currents_predictions",
        "station": station_id,
        "bin": str(bin_number),
        "begin_date": s_dt.strftime("%Y%m%d %H:%M"),
        "end_date": e_dt.strftime("%Y%m%d %H:%M"),
        "interval": str(interval_minutes),
        "vel_type": "default",
        "units": "metric",
        "time_zone": "gmt",
        "format": "json",
        "application": "lhzn_forcingkit",
    }
    response = get(CURRENTS_URL, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()
    if "error" in payload:
        raise RuntimeError(f"NOAA API Error: {payload['error'].get('message')}")
    rows = payload.get("current_predictions", {}).get("cp", [])
    if not rows:
        raise RuntimeError(f"No current predictions for {station_id} bin {bin_number}")
    if any("Type" in r for r in rows):
        raise RuntimeError(
            f"{station_id} is a subordinate station (slack and maximum times only); "
            "a harmonic station is needed for a time series"
        )

    station = get(STATION_URL.format(station=station_id), params={}, timeout=30)
    station.raise_for_status()
    info = station.json().get("stations", [{}])[0]

    first = rows[0]
    data = {
        "metadata": {
            "id": station_id,
            "name": info.get("name"),
            "lat": float(info["lat"]),
            "lon": float(info["lng"]),
            "bin": int(bin_number),
            "depth_m": float(first["Depth"]),
            "flood_dir_deg": float(first["meanFloodDir"]),
            "ebb_dir_deg": float(first["meanEbbDir"]),
            "units": "m/s along the flood axis, positive on the flood",
        },
        "data": [
            {"t": r["Time"], "v": float(r["Velocity_Major"]) / 100.0} for r in rows
        ],
    }
    with open(cache_path, "w") as f:
        json.dump(data, f)
    return data
