"""USGS water data: stream gauge records, site metadata, and the Network-Linked Data Index.

Plain reads, with no forcing logic (river discharge for a model is `usgs_rivers`):

- the USGS Water Data APIs (OGC API - Features, `api.waterdata.usgs.gov`): 15-minute
  (`continuous`) and daily mean (`daily`) values, day-of-year statistics, monitoring-location
  metadata (site type, drainage area) and time-series metadata (the period each record covers);
- the Network-Linked Data Index (NLDI, `api.water.usgs.gov/nldi`) over NHDPlus: an outlet
  flowline from a point or a downstream trace, the flowline's basin, and the USGS sites upstream
  or downstream of a flowline or site.

Every read takes a `get(url, params) -> text` so tests can serve fixtures; `session_get` is the
real one. An API key (`USGS_API_KEY`, free from api.waterdata.usgs.gov) raises the rate limit of
the Water Data APIs; without one the requests still work. The legacy `waterservices.usgs.gov`,
which USGS plans to retire in 2027, is not used.
"""

import io
import json
import logging
import math
import time as _time
from typing import Callable

import pandas as pd
import requests

from forcingkit import settings

logger = logging.getLogger(__name__)

API = "https://api.waterdata.usgs.gov/ogcapi/v1/collections"
OGC = "https://api.waterdata.usgs.gov/ogcapi/v0/collections"
NORMALS_API = "https://api.waterdata.usgs.gov/statistics/v0/observationNormals"
NLDI = "https://api.water.usgs.gov/nldi/linked-data"
USER_AGENT = "forcingkit (+https://github.com/lhzn-io/forcingkit)"
DISCHARGE = "00060"  # ft3 s-1
TEMPERATURE = "00010"  # degC
CFS_TO_M3S = 0.028316846592
STREAM_SITE_TYPE = "ST"
CHUNK_DAYS = 30
PAGE_LIMIT = 10000
BATCH = 50
SQKM_PER_SQMI = 2.589988110336
EARTH_RADIUS_KM = 6371.0088

Getter = Callable[[str, dict], str]


def session_get(url: str, params: dict, retries: int = 4) -> str:
    """GET returning the body as text, with backoff on rate limiting and server errors."""
    headers = {"User-Agent": USER_AGENT}
    key = (settings.env("USGS_API_KEY") or "").strip()
    if key:
        headers["X-Api-Key"] = key
    delay = 5.0
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=120)
        except requests.RequestException as exc:
            logger.warning(
                f"USGS request failed ({exc.__class__.__name__}), attempt {attempt}"
            )
        else:
            if resp.status_code == 200:
                return resp.text
            logger.warning(
                f"USGS HTTP {resp.status_code}, attempt {attempt}: {resp.url}"
            )
            if resp.status_code not in (429, 500, 502, 503, 504):
                resp.raise_for_status()
        if attempt < retries:
            _time.sleep(delay)
            delay *= 2
    raise RuntimeError(f"USGS: giving up after {retries} attempts: {url}")


def _json(get: Getter, url: str, params: dict) -> dict:
    return json.loads(get(url, params))


def site_id(s: str) -> str:
    """A USGS site number as an NLDI / Water Data id (`USGS-<number>`)."""
    return s if s.upper().startswith("USGS-") else f"USGS-{s}"


# --- gauge records -----------------------------------------------------------------------------


def _iso(t: pd.Timestamp) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def _frame(text: str) -> pd.DataFrame:
    text = text.strip()
    if not text or "\n" not in text:
        return pd.DataFrame(columns=["time", "value", "approval_status"])
    return pd.read_csv(io.StringIO(text))


def _clean(parts: list[pd.DataFrame], parameter: str) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame(
            {"value": [], "approval_status": []}, index=pd.DatetimeIndex([], tz="UTC")
        )
    df = pd.concat(parts)
    out = pd.DataFrame(
        {
            "value": pd.to_numeric(df["value"], errors="coerce").values,
            "approval_status": df.get(
                "approval_status", pd.Series([""] * len(df))
            ).values,
        },
        index=pd.to_datetime(df["time"], utc=True),
    ).sort_index()
    out = out[~out.index.duplicated(keep="last")].dropna(subset=["value"])
    if parameter == DISCHARGE:
        out = out[out["value"] >= 0]
    return out


def continuous(
    usgs_id: str, parameter: str, start: pd.Timestamp, end: pd.Timestamp, get: Getter
) -> pd.DataFrame:
    """Instantaneous values for start <= time < end: a frame of `value` and `approval_status`
    indexed by UTC time. Negative discharge is dropped."""
    parts = []
    a = start
    while a < end:
        b = min(a + pd.Timedelta(days=CHUNK_DAYS), end)
        df = _frame(
            get(
                f"{API}/continuous/items",
                {
                    "f": "csv",
                    "monitoring_location_id": f"USGS-{usgs_id}",
                    "parameter_code": parameter,
                    "time": f"{_iso(a)}/{_iso(b)}",
                    "properties": "time,value,approval_status",
                    "limit": PAGE_LIMIT,
                },
            )
        )
        if len(df) >= PAGE_LIMIT:
            raise RuntimeError(
                f"USGS {usgs_id} {parameter}: page limit reached for {a}/{b}"
            )
        if not df.empty:
            parts.append(df)
        a = b
    return _clean(parts, parameter)


def daily(
    usgs_id: str, start: pd.Timestamp, end: pd.Timestamp, get: Getter
) -> pd.DataFrame:
    """Daily mean discharge for the days from start to end, indexed by UTC midnight."""
    df = _frame(
        get(
            f"{API}/daily/items",
            {
                "f": "csv",
                "monitoring_location_id": f"USGS-{usgs_id}",
                "parameter_code": DISCHARGE,
                "statistic_id": "00003",
                "time": f"{start:%Y-%m-%d}/{end:%Y-%m-%d}",
                "properties": "time,value,approval_status",
                "limit": PAGE_LIMIT,
            },
        )
    )
    return _clean([df] if not df.empty else [], DISCHARGE)


def daily_median(usgs_id: str, get: Getter) -> dict[str, float]:
    """The USGS day-of-year median of daily mean discharge, keyed by 'MM-DD'."""
    body = _json(
        get,
        NORMALS_API,
        {
            "monitoring_location_id": f"USGS-{usgs_id}",
            "parameter_code": DISCHARGE,
            "start_date": "01-01",
            "end_date": "12-31",
            "computation_type": "percentile",
        },
    )
    out: dict[str, float] = {}
    for feature in body.get("features") or []:
        for series in feature["properties"].get("data", []):
            if series.get("parent_statistic_id") != "00003":
                continue
            for v in series.get("values", []):
                for p, x in zip(v.get("percentiles", []), v.get("values", [])):
                    if str(p) == "50" and x is not None:
                        out[v["time_of_year"]] = float(x)
    return out


# --- site metadata -----------------------------------------------------------------------------


def site_metadata(sites: list[str], get: Getter) -> dict[str, dict]:
    """Site type, drainage area (sq mi) and name of each USGS site (`USGS-<number>` ids)."""
    out: dict[str, dict] = {}
    for i in range(0, len(sites), BATCH):
        chunk = sites[i : i + BATCH]
        body = _json(
            get,
            f"{OGC}/monitoring-locations/items",
            {
                "f": "json",
                "id": ",".join(chunk),
                "properties": "id,site_type_code,drainage_area,monitoring_location_name",
                "limit": len(chunk),
            },
        )
        for f in body.get("features") or []:
            p = f["properties"]
            out[p["id"]] = {
                "site_type": p.get("site_type_code"),
                "area_sqmi": p.get("drainage_area"),
                "name": p.get("monitoring_location_name"),
            }
    return out


def discharge_records(sites: list[str], get: Getter) -> dict[str, list[tuple]]:
    """Each site's discharge records as (statistic, begin, end), for 15-minute (00011) and
    daily mean (00003) values."""
    out: dict[str, list[tuple]] = {}
    for i in range(0, len(sites), BATCH):
        chunk = sites[i : i + BATCH]
        body = _json(
            get,
            f"{OGC}/time-series-metadata/items",
            {
                "f": "json",
                "monitoring_location_id": ",".join(chunk),
                "parameter_code": DISCHARGE,
                "properties": "monitoring_location_id,statistic_id,begin_utc,end_utc",
                "limit": 20 * len(chunk),
            },
        )
        for f in body.get("features") or []:
            p = f["properties"]
            if p.get("statistic_id") not in ("00011", "00003"):
                continue
            if not p.get("begin_utc") or not p.get("end_utc"):
                continue
            out.setdefault(p["monitoring_location_id"], []).append(
                (
                    p["statistic_id"],
                    pd.Timestamp(p["begin_utc"]),
                    pd.Timestamp(p["end_utc"]),
                )
            )
    return out


# --- NLDI --------------------------------------------------------------------------------------


def _ring_area_km2(ring: list) -> float:
    total = 0.0
    for (lon1, lat1), (lon2, lat2) in zip(ring, ring[1:]):
        total += math.radians(lon2 - lon1) * (
            2 + math.sin(math.radians(lat1)) + math.sin(math.radians(lat2))
        )
    return abs(total) * EARTH_RADIUS_KM**2 / 2


def polygon_area_sqmi(geometry: dict) -> float:
    """Area of a GeoJSON Polygon or MultiPolygon in lon/lat, on a sphere, in square miles."""
    polys = (
        [geometry["coordinates"]]
        if geometry["type"] == "Polygon"
        else geometry["coordinates"]
    )
    km2 = sum(
        _ring_area_km2(p[0]) - sum(_ring_area_km2(h) for h in p[1:]) for p in polys
    )
    return km2 / SQKM_PER_SQMI


def _identifiers(body: dict) -> set[str]:
    return {f["properties"]["identifier"] for f in body.get("features") or []}


def outlet_from_trace(feature: str, get: Getter) -> int:
    """The last NHDPlus flowline downstream of an NLDI feature: a USGS site (`USGS-<number>`)
    or a flowline (its comid)."""
    source = "nwissite" if feature.upper().startswith("USGS-") else "comid"
    body = _json(
        get,
        f"{NLDI}/{source}/{feature}/navigation/DM/flowlines",
        {"distance": 9999, "f": "json"},
    )
    features = body.get("features") or []
    if not features:
        raise ValueError(f"NLDI: no flowlines downstream of {feature}")
    return int(features[-1]["properties"]["nhdplus_comid"])


def flowline_at(lon: float, lat: float, get: Getter) -> int:
    """The NHDPlus flowline NLDI snaps a point to."""
    body = _json(
        get, f"{NLDI}/comid/position", {"coords": f"POINT({lon} {lat})", "f": "json"}
    )
    return int(body["features"][0]["properties"]["comid"])


def flowline_end(comid: int, get: Getter) -> tuple[float, float]:
    """The downstream end (lon, lat) of a flowline."""
    body = _json(get, f"{NLDI}/comid/{comid}", {"f": "json"})
    geometry = body["features"][0]["geometry"]
    coords = geometry["coordinates"]
    end = coords[-1] if geometry["type"] == "LineString" else coords[-1][-1]
    return float(end[0]), float(end[1])


def basin_area_sqmi(comid: int, get: Getter) -> float:
    """Area of the basin draining to a flowline, in square miles."""
    body = _json(
        get, f"{NLDI}/comid/{comid}/basin", {"simplified": "false", "f": "json"}
    )
    return polygon_area_sqmi(body["features"][0]["geometry"])


def sites_upstream(comid: int, get: Getter) -> list[str]:
    """Every USGS site on the network upstream of a flowline (main stem and tributaries)."""
    body = _json(
        get,
        f"{NLDI}/comid/{comid}/navigation/UT/nwissite",
        {"distance": 9999, "f": "json"},
    )
    return sorted(_identifiers(body))


def sites_downstream(site: str, get: Getter) -> list[str]:
    """Every USGS site on the main stem downstream of a site, excluding the site itself."""
    body = _json(
        get,
        f"{NLDI}/nwissite/{site}/navigation/DM/nwissite",
        {"distance": 9999, "f": "json"},
    )
    return sorted(_identifiers(body) - {site})
