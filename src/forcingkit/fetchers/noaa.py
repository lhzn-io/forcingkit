"""NOAA CO-OPS water level and tidal current predictions, through insitukit.

insitukit's CO-OPS client paces requests (1 s apart, across threads), splits long windows to the API's
range limits, retries server errors and throttling, and names the caller (``insitukit.identify``, set by
the service at start-up). This module keeps forcingkit's call signatures, caching and return shapes.
"""

import json
import logging
import os
from datetime import datetime, timedelta, timezone

import pandas as pd
from insitukit import coops

from forcingkit import settings

logger = logging.getLogger(__name__)


# Water-level datums to request, in order of preference.
TIDE_DATUMS = ("NAVD", "MSL")
# How long a cached water-level window that still holds preliminary readings is served before
# CO-OPS is asked again (a verified window is kept for good).
PRELIMINARY_TTL_S = 24 * 3600


def _utc(stamp: str) -> datetime:
    """An ISO 8601 time, read as UTC (a trailing Z or an offset is accepted)."""
    ts = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    return (
        ts.replace(tzinfo=timezone.utc)
        if ts.tzinfo is None
        else ts.astimezone(timezone.utc)
    )


def _through(stamp: str) -> datetime:
    """The exclusive end for an inclusive ``end_time``: CO-OPS's own end_date includes its minute."""
    return _utc(stamp) + timedelta(minutes=1)


def _t(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y-%m-%d %H:%M")


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
    Fetches water level data from NOAA CO-OPS for a station over [start_time, end_time] (UTC),
    in metres, on NAVD88 where the station has it and MSL otherwise. Supports caching to avoid
    redundant API calls.

    Returns the shape of a CO-OPS response: ``{"metadata": {"id", "name", "lat", "lon", "datum",
    "quality"}, "data": [{"t": "YYYY-MM-DD HH:MM", "v": "0.123", "q": "v"}, ...]}``, with ``lat``,
    ``lon`` and ``v`` as strings (``v`` empty for a missing reading). ``q`` is CO-OPS's quality
    level for the reading, ``p`` (preliminary) or ``v`` (verified); ``metadata.quality`` is
    ``verified`` when every reading is, else ``preliminary``. Raises RuntimeError when the window
    has no data.

    A verified window is cached for good. CO-OPS replaces preliminary readings once it has
    reviewed them, usually within a few months, so a window holding any is served from the cache
    for ``PRELIMINARY_TTL_S`` (a day) and then fetched again, until it is verified. A cached file
    without ``metadata.quality`` (written before 0.17.0) is fetched again.

    Args:
        station_id: NOAA station ID (e.g., '8516945' for Kings Point)
        start_time: ISO8601 string (YYYY-MM-DDTHH:MM:SSZ)
        end_time: ISO8601 string (YYYY-MM-DDTHH:MM:SSZ), inclusive
        cache_dir: Local directory for caching JSON responses
    """
    os.makedirs(cache_dir, exist_ok=True)

    # Generate a cache key based on station and time
    cache_key = f"{station_id}_{start_time}_{end_time}".replace(":", "-")
    cache_path = os.path.join(cache_dir, f"{cache_key}.json")

    if not cache_bust and os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            cached = json.load(f)
        quality = (cached.get("metadata") or {}).get("quality")
        age_s = datetime.now().timestamp() - os.path.getmtime(cache_path)
        if quality == "verified" or (
            quality == "preliminary" and age_s < PRELIMINARY_TTL_S
        ):
            logger.info(f"Cache hit for NOAA station {station_id}: {cache_path}")
            return cached
        logger.info(
            f"Cached NOAA window stale or unverified; fetching again: {cache_path}"
        )

    logger.info(
        f"Fetching NOAA tide data for station {station_id} ({start_time} to {end_time})..."
    )
    start, end = _utc(start_time), _through(end_time)

    try:
        # NAVD88 where the station has it, so stations share a datum; otherwise MSL (New Haven,
        # 8465705, publishes only tidal datums). The datum used is recorded in the metadata.
        for datum in TIDE_DATUMS:
            try:
                df = coops.fetch(
                    "water_level", station_id, start, end, datum=datum, quality=True
                )
                break
            except coops.DatumUnavailable:
                if datum == TIDE_DATUMS[-1]:
                    raise
                logger.info(
                    f"NOAA station {station_id} has no {datum} datum; trying the next"
                )
        if df.empty:
            raise RuntimeError(
                f"NOAA API Error: No data was found for {station_id} "
                f"({start_time} to {end_time})"
            )
        info = coops.metadata(station_id, expand="")
        verified = bool((df["quality"] == "v").all())
        data = {
            "metadata": {
                "id": station_id,
                "name": info.get("name"),
                "lat": str(info["lat"]),
                "lon": str(info["lng"]),
                "datum": datum,
                "quality": "verified" if verified else "preliminary",
            },
            "data": [
                # Written as CO-OPS writes it: 3 decimals at most, no trailing zeros.
                {
                    "t": _t(t),
                    "v": "" if pd.isna(v) else str(round(float(v), 3)),
                    "q": q or "",
                }
                for t, v, q in zip(df["time"], df["water_level"], df["quality"])
            ],
        }

        # Kept for good when verified; a preliminary window expires (see PRELIMINARY_TTL_S).
        with open(cache_path, "w") as f:
            json.dump(data, f, indent=4)

        return data

    except Exception as e:
        logger.error(f"Failed to fetch NOAA tide data: {e}")
        raise e


def fetch_noaa_current_stations(
    bbox: list[float],
    cache_dir: str = os.path.join(settings.cache_dir(), "noaa"),
    cache_bust: bool = False,
    get=None,
) -> list[dict]:
    """Every harmonic CO-OPS current-prediction station and depth bin inside ``bbox``
    ([min_lon, min_lat, max_lon, max_lat]).

    Returns one entry per station-bin: ``{"id", "name", "lat", "lon", "bin", "depth_m"}``.
    Harmonic stations (type H) give a predicted series; subordinate (S) and weak or variable (W)
    entries are left out. The full station list is cached for a day. ``get`` is ignored (kept for
    callers of earlier releases; requests go through insitukit).
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
        stations = coops.stations("currentpredictions", units="metric")
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
    get=None,
) -> dict:
    """Harmonic current predictions for one CO-OPS station and depth bin over
    [start_time, end_time] (UTC).

    Returns ``{"metadata": {...}, "data": [{"t": "YYYY-MM-DD HH:MM", "v": m/s}, ...]}``: the
    predicted velocity along the station's flood-ebb axis, positive on the flood, in m/s, every
    ``interval_minutes``. The metadata carries the station's position, the bin depth (metres
    below the surface), and the mean flood and ebb directions (degrees true, toward which the
    water flows). Only harmonic stations give a series; subordinate stations give slack and
    maximum times only and are refused. ``get`` is ignored (kept for callers of earlier
    releases; requests go through insitukit).
    """
    os.makedirs(cache_dir, exist_ok=True)
    key = f"currents_{station_id}_b{bin_number}_{start_time}_{end_time}_{interval_minutes}"
    cache_path = os.path.join(cache_dir, key.replace(":", "-") + ".json")
    if not cache_bust and os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)

    try:
        df = coops.current_predictions(
            station_id,
            _utc(start_time),
            _through(end_time),
            bin=int(bin_number),
            interval=int(interval_minutes),
            vel_type="default",
        )
    except coops.CoopsError as exc:
        if "subordinate" in str(exc):
            raise RuntimeError(
                f"{station_id} is a subordinate station (slack and maximum times only); "
                "a harmonic station is needed for a time series"
            ) from exc
        raise RuntimeError(f"NOAA API Error: {exc}") from exc
    if df.empty:
        raise RuntimeError(f"No current predictions for {station_id} bin {bin_number}")

    info = coops.metadata(station_id, expand="")
    data = {
        "metadata": {
            "id": station_id,
            "name": info.get("name"),
            "lat": float(info["lat"]),
            "lon": float(info["lng"]),
            "bin": int(bin_number),
            "depth_m": df.attrs["depth_m"],
            "flood_dir_deg": df.attrs["flood_dir_deg"],
            "ebb_dir_deg": df.attrs["ebb_dir_deg"],
            "units": "m/s along the flood axis, positive on the flood",
        },
        "data": [
            {"t": _t(t), "v": float(v)}
            for t, v in zip(df["time"], df["velocity_major"])
        ],
    }
    with open(cache_path, "w") as f:
        json.dump(data, f)
    return data
