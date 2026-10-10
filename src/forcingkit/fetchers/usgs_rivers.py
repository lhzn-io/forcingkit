"""River discharge at the mouth, for point sources in an ocean model, from USGS stream gauges
(schema `river-v1`).

The method follows common practice for gauge-driven river forcing (the NOAA Operational Forecast
Systems' river control files; drainage-area scaling as in regional estuary models): the discharge
at a river's mouth is the sum of the lowest tide-free gauge on each branch, scaled by the ratio of
the drainage area at the mouth to the gauged area. Gauges below the head of tide are never used for
discharge, since their flow reverses with the tide.

Only the mouths are listed (`forcingkit/data/rivers/*.json`, plus any `*.json` in the directory
named by `FORCINGKIT_RIVER_MOUTHS_DIR`, or mouths passed with a request). `derive_river` chooses
each river's gauges and scale factor for the request's window, so a new river needs a mouth entry
and no code, and a gauge that opens or closes is picked up or dropped by the next request whose
window it affects. From the mouth's outlet flowline (NHDPlus):

1. the drainage area at the mouth is the area of the outlet's NLDI basin;
2. the candidate gauges are every USGS site NLDI finds upstream of the outlet;
3. a candidate is kept if USGS classes it as a stream site (site type ``ST``; tidal streams
   ``ST-TS`` and estuaries ``ES`` are left out), it has a drainage area, and its discharge record
   (15-minute or daily) covers the window;
4. candidates draining less than ``min_share`` of the mouth's area are dropped;
5. of those, a gauge is kept only if no other kept gauge lies downstream of it, which leaves the
   lowest gauge on each branch, so the kept gauges measure disjoint parts of the basin;
6. the scale factor is the mouth's area over the summed area of the kept gauges.

A mouth may pin gauges (``include``) or rule them out (``exclude``). Discharge is the 15-minute
record averaged to hourly values centred on the hour. Short gaps (up to `max_gap_hours`) are
interpolated; longer ones take that day's mean; days with neither take the USGS day-of-year median
only when the caller allows it, and are otherwise an error. Every fill is flagged per river and
hour. Salinity of river water is zero; water temperature is carried where a gauge on the river
measures it (parameter 00010), else NaN. The reads themselves are in `usgs`.
"""

import hashlib
import json
import logging
import os
from typing import Iterator, Optional

import numpy as np
import pandas as pd
import xarray as xr

from forcingkit import settings
from forcingkit.fetchers import usgs
from forcingkit.fetchers.usgs import CFS_TO_M3S, DISCHARGE, TEMPERATURE, Getter

logger = logging.getLogger(__name__)

RIVER_SCHEMA = "river-v1"
MOUTHS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "rivers")
DEFAULT_MIN_SHARE = 0.01
# Facts about a mouth (outlet, basin area, upstream sites, which sites lie downstream of which)
# change only when NHDPlus or the gauge network does; they are cached this long.
STATIC_MAX_AGE_DAYS = 30
# A gauge's record "covers" a window ending within this long of now if it runs to within this of
# the window's end: real-time records lag by an hour or so, and a day's slack absorbs it.
RECENT_SLACK = pd.Timedelta(hours=24)

# Fill flags, per river and hour: the largest flag among the river's gauges.
FILL_NONE, FILL_INTERPOLATED, FILL_DAILY_MEAN, FILL_CLIMATOLOGY = 0, 1, 2, 3
FILL_MEANINGS = "measured interpolated daily_mean climatology_median"

RECORD_DIMS: dict[str, tuple[str, ...]] = {
    "discharge": ("river",),
    "temperature": ("river",),
    "fill": ("river",),
}
UNITS = {
    "discharge": ("m3 s-1", "water_volume_transport_in_river_channel"),
    "temperature": ("degC", "water_temperature"),
    "fill": ("1", "status_flag"),
}


class NoGaugeError(ValueError):
    """No USGS stream gauge upstream of a mouth has a discharge record over the window."""


# --- mouths ------------------------------------------------------------------------------------

MOUTH_KEYS = {
    "name",
    "lon",
    "lat",
    "comid",
    "trace_from",
    "include",
    "exclude",
    "lag_hours",
    "note",
}


def mouth_paths() -> list[str]:
    """The packaged mouth files, then those in FORCINGKIT_RIVER_MOUTHS_DIR."""
    dirs = [MOUTHS_DIR]
    extra = settings.env("FORCINGKIT_RIVER_MOUTHS_DIR")
    if extra:
        dirs.append(os.path.expanduser(extra))
    return [
        os.path.join(d, f)
        for d in dirs
        if os.path.isdir(d)
        for f in sorted(os.listdir(d))
        if f.endswith(".json")
    ]


def validate_mouth(mouth: dict) -> dict:
    """Check a mouth entry: a name, a point, and only known keys."""
    unknown = set(mouth) - MOUTH_KEYS
    if unknown:
        raise ValueError(f"mouth {mouth.get('name')!r}: unknown keys {sorted(unknown)}")
    for key in ("name", "lon", "lat"):
        if key not in mouth:
            raise ValueError(f"mouth {mouth.get('name')!r}: missing {key}")
    return mouth


def load_mouths(paths: Optional[list[str]] = None) -> list[dict]:
    """Every listed mouth. A later file's entry replaces an earlier one of the same name, so a
    local file can override a packaged mouth."""
    by_name: dict[str, dict] = {}
    for path in mouth_paths() if paths is None else paths:
        with open(path) as f:
            for m in json.load(f)["mouths"]:
                by_name[m["name"]] = validate_mouth(m)
    return list(by_name.values())


def mouths_in_bbox(mouths: list[dict], bbox: list[float]) -> list[dict]:
    """Mouths whose point lies in [min_lon, min_lat, max_lon, max_lat], west to east."""
    inside = [
        m
        for m in mouths
        if bbox[0] <= m["lon"] <= bbox[2] and bbox[1] <= m["lat"] <= bbox[3]
    ]
    return sorted(inside, key=lambda m: m["lon"])


def mouths_version(mouths: list[dict]) -> str:
    """A short hash of the mouth entries, for store ids: editing a mouth invalidates its stores."""
    canonical = json.dumps(
        [{k: v for k, v in m.items() if k != "note"} for m in mouths], sort_keys=True
    )
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


# --- deriving a river's gauges -----------------------------------------------------------------


class _Cache:
    """JSON files under a directory, each with the time it was written."""

    def __init__(self, directory: Optional[str]):
        self.directory = directory
        if directory:
            os.makedirs(directory, exist_ok=True)

    def _path(self, key: str) -> Optional[str]:
        if not self.directory:
            return None
        digest = hashlib.sha256(key.encode()).hexdigest()[:16]
        return os.path.join(self.directory, f"{digest}.json")

    def get(self, key: str, max_age: Optional[pd.Timedelta]):
        path = self._path(key)
        if not path or not os.path.exists(path):
            return None
        with open(path) as f:
            entry = json.load(f)
        if entry.get("key") != key:
            return None
        written = pd.Timestamp(entry["written_utc"])
        if max_age is not None and pd.Timestamp.now(tz="UTC") - written > max_age:
            return None
        return entry["value"]

    def put(self, key: str, value) -> None:
        path = self._path(key)
        if not path:
            return
        entry = {
            "key": key,
            "written_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "value": value,
        }
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(entry, f)
        os.replace(tmp, path)


def outlet_comid(mouth: dict, get: Getter) -> int:
    """The NHDPlus flowline at the mouth.

    In order of preference: the mouth's ``comid``; the end of the downstream trace from a feature
    on the river (``trace_from``, such as ``USGS-01184000``); the flowline NLDI snaps the mouth's
    point to, which at an estuary can be a small coastal flowline rather than the river, so it is
    logged as a warning.
    """
    if mouth.get("comid"):
        return int(mouth["comid"])
    if mouth.get("trace_from"):
        return usgs.outlet_from_trace(mouth["trace_from"], get)
    comid = usgs.flowline_at(mouth["lon"], mouth["lat"], get)
    logger.warning(
        f"River {mouth['name']}: outlet snapped from its point to flowline {comid}; set the "
        f"mouth's comid or trace_from so it cannot land on a coastal flowline"
    )
    return comid


def mouth_facts(mouth: dict, get: Getter, cache: _Cache) -> dict:
    """Outlet flowline, mouth position, drainage area at the mouth, and the USGS sites upstream."""
    key = f"mouth:{json.dumps(mouth, sort_keys=True)}"
    cached = cache.get(key, pd.Timedelta(days=STATIC_MAX_AGE_DAYS))
    if cached is not None:
        return cached
    comid = outlet_comid(mouth, get)
    lon, lat = usgs.flowline_end(comid, get)
    facts = {
        "comid": comid,
        "lon": lon,
        "lat": lat,
        "area_mouth_sqmi": usgs.basin_area_sqmi(comid, get),
        "upstream_sites": usgs.sites_upstream(comid, get),
    }
    cache.put(key, facts)
    return facts


def _downstream(site: str, get: Getter, cache: _Cache) -> set[str]:
    key = f"downstream:{site}"
    cached = cache.get(key, pd.Timedelta(days=STATIC_MAX_AGE_DAYS))
    if cached is None:
        cached = usgs.sites_downstream(site, get)
        cache.put(key, cached)
    return set(cached)


def covers(records: list[tuple], start: pd.Timestamp, end: pd.Timestamp) -> bool:
    """True if any discharge record runs from before start to (nearly) end."""
    target = min(end, pd.Timestamp.now(tz="UTC")) - RECENT_SLACK
    return any(b <= start and e >= target for _, b, e in records)


def derive_river(
    mouth: dict,
    start: pd.Timestamp,
    end: pd.Timestamp,
    get: Getter,
    *,
    min_share: float = DEFAULT_MIN_SHARE,
    cache_dir: Optional[str] = None,
) -> dict:
    """The gauges and scale factor for `mouth` over [start, end]: a dict holding the river's
    name, outlet, mouth position and area, the kept gauges, and every dropped candidate with the
    reason."""
    cache = _Cache(cache_dir)
    facts = mouth_facts(mouth, get, cache)
    a_mouth = facts["area_mouth_sqmi"]
    include = {usgs.site_id(s) for s in mouth.get("include", [])}
    exclude = {usgs.site_id(s) for s in mouth.get("exclude", [])}
    sites = sorted(set(facts["upstream_sites"]) | include)

    meta = usgs.site_metadata(sites, get)
    dropped: dict[str, str] = {}
    candidates = []
    for s in sites:
        m = meta.get(s)
        if s in exclude:
            dropped[s] = "excluded by the mouth entry"
        elif m is None:
            dropped[s] = "no USGS monitoring-location record"
        elif m["site_type"] != usgs.STREAM_SITE_TYPE and s not in include:
            dropped[s] = f"site type {m['site_type']}"
        elif not m["area_sqmi"]:
            dropped[s] = "no drainage area"
        elif m["area_sqmi"] < min_share * a_mouth and s not in include:
            dropped[s] = f"drains under {min_share:.1%} of the mouth's area"
        else:
            candidates.append(s)

    records = usgs.discharge_records(candidates, get)
    active = []
    for s in candidates:
        if covers(records.get(s, []), start, end):
            active.append(s)
        else:
            dropped[s] = "no discharge record over the window"
    unknown_includes = include - set(active)
    if unknown_includes:
        raise ValueError(
            f"river {mouth['name']}: included gauge(s) {sorted(unknown_includes)} have no "
            f"discharge record over the window"
        )

    # Lowest gauge on each branch: a gauge downstream has the larger area, so only gauges with
    # a larger area need checking as possible downstream neighbours.
    by_area = sorted(active, key=lambda s: meta[s]["area_sqmi"])
    kept = []
    for n, s in enumerate(by_area):
        larger = set(by_area[n + 1 :])
        below = _downstream(s, get, cache) & larger if larger else set()
        if below and s not in include:
            dropped[s] = f"upstream of another gauge ({sorted(below)[0]})"
        else:
            kept.append(s)
    if not kept:
        raise NoGaugeError(
            f"river {mouth['name']}: no USGS stream gauge upstream of the mouth has a "
            f"discharge record over the window"
        )

    gauged = sum(meta[s]["area_sqmi"] for s in kept)
    if gauged > a_mouth * 1.02:
        raise ValueError(
            f"river {mouth['name']}: the kept gauges drain {gauged:.0f} sq mi, more than the "
            f"{a_mouth:.0f} sq mi at the mouth; a pinned gauge may double count"
        )
    gauges = [
        {
            "usgs_id": s.removeprefix("USGS-"),
            "name": meta[s]["name"],
            "area_sqmi": meta[s]["area_sqmi"],
        }
        for s in sorted(kept, key=lambda s: -meta[s]["area_sqmi"])
    ]
    logger.info(
        f"River {mouth['name']}: {len(sites)} upstream sites, {len(kept)} gauges kept "
        f"({', '.join(g['usgs_id'] for g in gauges)}), {gauged:.0f} of {a_mouth:.0f} sq mi "
        f"gauged, scale {a_mouth / gauged:.4f}"
    )
    return {
        "name": mouth["name"],
        "mouth": {"lon": facts["lon"], "lat": facts["lat"], "comid": facts["comid"]},
        "area_mouth_sqmi": round(a_mouth, 1),
        "gauges": gauges,
        "lag_hours": mouth.get("lag_hours", 0),
        "min_share": min_share,
        "upstream_sites": len(sites),
        "dropped": dict(sorted(dropped.items())),
    }


def scale_factor(river: dict) -> float:
    """Drainage area at the mouth over the summed area of the river's gauges."""
    return river["area_mouth_sqmi"] / sum(g["area_sqmi"] for g in river["gauges"])


# --- hourly series -----------------------------------------------------------------------------


def hourly_centred(values: pd.Series) -> pd.Series:
    """Mean over [t - 30 min, t + 30 min) for each hour t."""
    if values.empty:
        return values
    shifted = values.copy()
    shifted.index = shifted.index + pd.Timedelta(minutes=30)
    return shifted.resample("h").mean().dropna()


def _runs(missing: np.ndarray) -> list[tuple[int, int]]:
    """(start, stop) index pairs of each run of True."""
    runs, i = [], 0
    while i < len(missing):
        if missing[i]:
            j = i
            while j < len(missing) and missing[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def gauge_discharge(
    usgs_id: str,
    hours_index: pd.DatetimeIndex,
    get: Getter,
    max_gap_hours: int,
    allow_climatology: bool,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Hourly discharge (ft3 s-1) at a gauge over `hours_index`, its fill flags, and whether any
    value used is provisional."""
    pad = pd.Timedelta(hours=max_gap_hours + 1)
    raw = usgs.continuous(
        usgs_id, DISCHARGE, hours_index[0] - pad, hours_index[-1] + pad, get
    )
    provisional = (
        bool((raw["approval_status"] != "Approved").any()) if len(raw) else False
    )
    hourly = hourly_centred(raw["value"]) if len(raw) else pd.Series(dtype=float)
    full = hourly.reindex(hours_index.union(hourly.index))
    flags = pd.Series(FILL_NONE, index=full.index, dtype=np.int8)

    # Interpolate gaps of up to max_gap_hours that have measured values on both sides.
    missing = full.isna().to_numpy()
    for a, b in _runs(missing):
        if a > 0 and b < len(full) and (b - a) <= max_gap_hours:
            flags.iloc[a:b] = FILL_INTERPOLATED
    interp = full.interpolate(method="time", limit_area="inside")
    full = full.where(flags != FILL_INTERPOLATED, interp)

    full = full.reindex(hours_index)
    flags = flags.reindex(hours_index).fillna(FILL_NONE).astype(np.int8)
    if full.isna().any():
        days = usgs.daily(
            usgs_id, hours_index[0].floor("D"), hours_index[-1].floor("D"), get
        )
        if len(days):
            provisional |= bool((days["approval_status"] != "Approved").any())
        by_day = (
            days["value"].groupby(days.index.floor("D")).first() if len(days) else None
        )
        for t in full.index[full.isna()]:
            day = t.floor("D")
            if by_day is not None and day in by_day.index:
                full[t] = by_day[day]
                flags[t] = FILL_DAILY_MEAN
    if full.isna().any():
        gap = full.index[full.isna()]
        if not allow_climatology:
            raise ValueError(
                f"USGS {usgs_id}: no discharge for {len(gap)} h from {gap[0]} to {gap[-1]} "
                f"(no continuous or daily values); pass allow_climatology to fill with the "
                f"day-of-year median"
            )
        medians = usgs.daily_median(usgs_id, get)
        for t in gap:
            m = medians.get(t.strftime("%m-%d"))
            if m is None:
                raise ValueError(f"USGS {usgs_id}: no day-of-year median for {t:%m-%d}")
            full[t] = m
            flags[t] = FILL_CLIMATOLOGY
        logger.warning(
            f"USGS {usgs_id}: {len(gap)} h filled with the day-of-year median"
        )
    return full.to_numpy(dtype=float), flags.to_numpy(dtype=np.int8), provisional


def gauge_temperature(
    usgs_id: str, hours_index: pd.DatetimeIndex, get: Getter, max_gap_hours: int
) -> Optional[np.ndarray]:
    """Hourly water temperature at a gauge, or None if it has no record in the window. Gaps
    longer than max_gap_hours stay NaN."""
    pad = pd.Timedelta(hours=max_gap_hours + 1)
    raw = usgs.continuous(
        usgs_id, TEMPERATURE, hours_index[0] - pad, hours_index[-1] + pad, get
    )
    if raw.empty:
        return None
    hourly = hourly_centred(raw["value"])
    full = hourly.reindex(hours_index.union(hourly.index))
    full = full.interpolate(method="time", limit=max_gap_hours, limit_area="inside")
    return full.reindex(hours_index).to_numpy(dtype=float)


# --- delivery ----------------------------------------------------------------------------------


def _count_reasons(dropped: dict[str, str]) -> dict[str, int]:
    """Dropped candidates counted by reason, with the site in a reason left out."""
    counts: dict[str, int] = {}
    for reason in dropped.values():
        r = reason.split(" (")[0]
        counts[r] = counts.get(r, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def iter_rivers(
    start_time: str,
    hours: int,
    bbox: list[float],
    *,
    max_gap_hours: int = 6,
    allow_climatology: bool = False,
    mouths: Optional[list[dict]] = None,
    min_share: float = DEFAULT_MIN_SHARE,
    skip_ungauged: bool = False,
    rivers: Optional[list[dict]] = None,
    get: Optional[Getter] = None,
    cache_dir: Optional[str] = None,
) -> Iterator[tuple]:
    """Yield ("static", Dataset) and then ("record", time, {name: array over river}) for each
    hour from start_time to start_time + hours inclusive (hours + 1 records).

    `mouths` defaults to the listed mouths in `bbox`. Each mouth's gauges and scale factor are
    derived for the window (`derive_river`, with `min_share`). A mouth with no gauge over the
    window is an error, or, with `skip_ungauged`, left out and listed in the store's
    `skipped_rivers`. `rivers` passes already-derived rivers instead (tests use it to stay
    offline)."""
    get = get or usgs.session_get
    start = pd.Timestamp(start_time)
    start = (
        start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    )
    index = pd.date_range(start, periods=hours + 1, freq="h")
    skipped: list[str] = []
    if rivers is None:
        if mouths is None:
            mouths = mouths_in_bbox(load_mouths(), bbox)
        if not mouths:
            raise ValueError(f"no listed river mouth lies in bbox {bbox}")
        derivation_cache = os.path.join(
            cache_dir or settings.cache_dir("usgs"), "derivations"
        )
        rivers = []
        for m in mouths:
            lag = pd.Timedelta(hours=m.get("lag_hours", 0))
            try:
                rivers.append(
                    derive_river(
                        validate_mouth(m),
                        index[0] - lag,
                        index[-1] - lag,
                        get,
                        min_share=min_share,
                        cache_dir=derivation_cache,
                    )
                )
            except NoGaugeError as exc:
                if not skip_ungauged:
                    raise
                logger.warning(f"{exc}; left out (skip_ungauged)")
                skipped.append(m["name"])
    if not rivers:
        raise ValueError(f"no river mouth lies in bbox {bbox}")

    discharge = np.zeros((len(index), len(rivers)))
    temperature = np.full((len(index), len(rivers)), np.nan)
    fill = np.zeros((len(index), len(rivers)), dtype=np.int8)
    provenance = []
    provisional_any = False
    for n, river in enumerate(rivers):
        lag = pd.Timedelta(hours=river.get("lag_hours", 0))
        gauge_index = index - lag
        total = np.zeros(len(index))
        for g in river["gauges"]:
            q, flags, provisional = gauge_discharge(
                g["usgs_id"], gauge_index, get, max_gap_hours, allow_climatology
            )
            total += q
            fill[:, n] = np.maximum(fill[:, n], flags)
            provisional_any |= provisional
            if np.isnan(temperature[:, n]).all():
                t = gauge_temperature(g["usgs_id"], gauge_index, get, max_gap_hours)
                if t is not None:
                    temperature[:, n] = t
        k = scale_factor(river)
        discharge[:, n] = total * k * CFS_TO_M3S
        provenance.append(
            {
                "name": river["name"],
                "gauges": [g["usgs_id"] for g in river["gauges"]],
                "gauged_area_sqmi": sum(g["area_sqmi"] for g in river["gauges"]),
                "area_mouth_sqmi": river["area_mouth_sqmi"],
                "scale_factor": round(k, 6),
                "lag_hours": river.get("lag_hours", 0),
                "outlet_comid": river["mouth"].get("comid"),
                "min_share": river.get("min_share"),
                "upstream_sites": river.get("upstream_sites"),
                "dropped_by_reason": _count_reasons(river.get("dropped", {})),
                "hours_filled": {
                    "interpolated": int((fill[:, n] == FILL_INTERPOLATED).sum()),
                    "daily_mean": int((fill[:, n] == FILL_DAILY_MEAN).sum()),
                    "climatology_median": int((fill[:, n] == FILL_CLIMATOLOGY).sum()),
                },
            }
        )
        logger.info(
            f"River {river['name']}: {len(river['gauges'])} gauges, scale {k:.4f}, "
            f"mean {discharge[:, n].mean():.1f} m3/s"
        )

    static = xr.Dataset(
        {
            "mouth_lon": (
                "river",
                np.array([r["mouth"]["lon"] for r in rivers], dtype=np.float64),
            ),
            "mouth_lat": (
                "river",
                np.array([r["mouth"]["lat"] for r in rivers], dtype=np.float64),
            ),
            "scale_factor": ("river", np.array([scale_factor(r) for r in rivers])),
            "salinity": ("river", np.zeros(len(rivers), dtype=np.float32)),
        },
        coords={"river": np.arange(len(rivers))},
        attrs={
            "schema": RIVER_SCHEMA,
            "source": __name__,
            "river_names": json.dumps([r["name"] for r in rivers]),
            "provenance": json.dumps(provenance),
            "provisional": bool(provisional_any),
            "skipped_rivers": json.dumps(skipped),
            # Every candidate gauge each river's derivation dropped, and why.
            "derivation_dropped": json.dumps(
                {r["name"]: r.get("dropped", {}) for r in rivers}
            ),
            "max_gap_hours": max_gap_hours,
            "allow_climatology": bool(allow_climatology),
            "fill_flag_values": "0 1 2 3",
            "fill_flag_meanings": FILL_MEANINGS,
            "method": (
                "Per river, the lowest tide-free USGS stream gauge on each branch with a record "
                "over the window, derived from USGS NLDI and site metadata; their sum scaled by "
                "drainage area at the mouth over gauged area; 15-minute values averaged to "
                "hourly values centred on the hour."
            ),
        },
    )
    static["mouth_lon"].attrs = {"units": "degrees_east", "standard_name": "longitude"}
    static["mouth_lat"].attrs = {"units": "degrees_north", "standard_name": "latitude"}
    static["salinity"].attrs = {"units": "1e-3", "standard_name": "sea_water_salinity"}
    yield ("static", static)
    for i, t in enumerate(index):
        yield (
            "record",
            t.tz_convert(None),
            {"discharge": discharge[i], "temperature": temperature[i], "fill": fill[i]},
        )
