"""NDBC buoy observations for model validation: standard meteorology (sea and air temperature,
pressure, wind, waves) and near-surface ADCP currents.

Files come from the NDBC data server in the order they are published: the per-month file for
months of the current year (`data/<product>/<Mon>/<id><m><yyyy>.txt.gz`), then the yearly
historical file (`data/historical/<product>/<id>h<yyyy>.txt.gz`), then the rolling 45-day
real-time file (`data/realtime2/<id>.txt` or `.adcp`). Missing-value sentinels (99, 999, 9999,
MM) become NaN. These observations validate a model; they never force it.
"""

import gzip
import json
import logging
import math
import os
import re
from typing import Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

NDBC = "https://www.ndbc.noaa.gov"
PRODUCTS = ("stdmet", "adcp")
REALTIME_SUFFIX = {"stdmet": "txt", "adcp": "adcp"}
MONTH_ABBR = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]
# Column sentinels NDBC uses for "not measured".
SENTINELS = {99.0, 999.0, 9999.0, 99.00}
# stdmet columns carried through, with units.
STDMET_COLUMNS = {
    "WTMP": "degC",  # sea surface temperature
    "ATMP": "degC",  # air temperature
    "PRES": "hPa",  # sea level pressure
    "WSPD": "m s-1",  # wind speed
    "WDIR": "degree from true north (direction the wind comes from)",
    "GST": "m s-1",
    "WVHT": "m",
    "DPD": "s",
}


def month_code(month: int) -> str:
    """NDBC's month code in per-month file names: 1 to 9, then a, b, c for October to December."""
    return str(month) if month < 10 else "abc"[month - 10]


def candidate_urls(station: str, product: str, year: int, month: int) -> list[str]:
    """URLs that may hold `product` for `station` in the given month, in the order to try."""
    station = station.lower()
    return [
        f"{NDBC}/data/{product}/{MONTH_ABBR[month - 1]}/{station}{month_code(month)}{year}.txt.gz",
        f"{NDBC}/data/historical/{product}/{station}h{year}.txt.gz",
        f"{NDBC}/data/realtime2/{station.upper()}.{REALTIME_SUFFIX[product]}",
    ]


def parse_ndbc_text(text: str) -> pd.DataFrame:
    """Parse an NDBC whitespace-delimited file (two `#` header rows) into a frame indexed by UTC
    time, sentinels and `MM` as NaN. Handles both four- and two-digit years, with or without a
    minute column."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines or not lines[0].startswith("#"):
        raise ValueError("not an NDBC data file: no header row")
    names = lines[0].lstrip("#").split()
    rows = [ln.split() for ln in lines if not ln.startswith("#")]
    df = pd.DataFrame(rows, columns=names[: len(rows[0])] if rows else names)
    year_col = names[0]
    years = df[year_col].astype(int)
    years = years.where(years >= 100, years + 1900)
    minutes = df["mm"].astype(int) if "mm" in df else 0
    df.index = pd.to_datetime(
        dict(
            year=years,
            month=df["MM"].astype(int),
            day=df["DD"].astype(int),
            hour=df["hh"].astype(int),
            minute=minutes,
        )
    )
    df = df.drop(columns=[c for c in (year_col, "MM", "DD", "hh", "mm") if c in df])
    df = df.replace("MM", np.nan).apply(pd.to_numeric, errors="coerce")
    return df.mask(df.isin(SENTINELS))


def _get(url: str, timeout: float = 60.0) -> Optional[str]:
    r = requests.get(url, timeout=timeout)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    if url.endswith(".gz"):
        return gzip.decompress(r.content).decode("utf-8", errors="replace")
    return r.text


def _utc_naive(t) -> pd.Timestamp:
    t = pd.Timestamp(t)
    return t if t.tzinfo is None else t.tz_convert("UTC").tz_localize(None)


def station_position(station: str, get=_get) -> tuple[float, float]:
    """(lat, lon) of an NDBC station from its station page."""
    html = get(f"{NDBC}/station_page.php?station={station.lower()}") or ""
    m = re.search(r"(\d+\.\d+)\s*([NS])\s+(\d+\.\d+)\s*([EW])", html)
    if not m:
        raise ValueError(f"No position found for NDBC station {station}")
    lat = float(m.group(1)) * (1 if m.group(2) == "N" else -1)
    lon = float(m.group(3)) * (1 if m.group(4) == "E" else -1)
    return lat, lon


def fetch_product(
    station: str, product: str, start: pd.Timestamp, end: pd.Timestamp, get=_get
):
    """`product` for `station` over [start, end]: (frame, source URLs). Each month is taken from
    the first source that has it."""
    frames, sources = [], []
    for month_start in pd.date_range(start.normalize().replace(day=1), end, freq="MS"):
        for url in candidate_urls(
            station, product, month_start.year, month_start.month
        ):
            text = get(url)
            if text is None:
                continue
            df = parse_ndbc_text(text)
            df = df[(df.index >= start) & (df.index <= end)]
            if len(df):
                frames.append(df)
                sources.append(url)
                break
    if not frames:
        return pd.DataFrame(), sources
    df = pd.concat(frames)
    return df[~df.index.duplicated(keep="first")].sort_index(), sources


def _list(values) -> list:
    return [
        None if (v is None or (isinstance(v, float) and math.isnan(v))) else float(v)
        for v in values
    ]


def fetch_ndbc(
    station: str,
    start_time: str,
    end_time: str,
    products: tuple[str, ...] = PRODUCTS,
    cache_dir: Optional[str] = None,
    get=_get,
    position: Optional[tuple[float, float]] = None,
) -> dict:
    """Observations at an NDBC station over [start_time, end_time] (UTC) as a JSON-ready dict.

    stdmet: times plus the columns of `STDMET_COLUMNS`. adcp: times, the bin depth and the
    current as eastward and northward components (m/s); NDBC reports speed in cm/s and the
    direction the current flows toward, in degrees true. Missing values are null. A window that
    ended more than two days ago is cached as JSON under `cache_dir`. `get` (a URL to text or
    None) and `position` exist so tests can run without the network.
    """
    start, end = _utc_naive(start_time), _utc_naive(end_time)
    unknown = set(products) - set(PRODUCTS)
    if unknown:
        raise ValueError(f"unknown NDBC products {sorted(unknown)}")

    cache_path = None
    if cache_dir:
        os.makedirs(os.path.join(cache_dir, "ndbc"), exist_ok=True)
        key = f"{station.lower()}_{start:%Y%m%dT%H%M}_{end:%Y%m%dT%H%M}_{'-'.join(sorted(products))}"
        cache_path = os.path.join(cache_dir, "ndbc", f"{key}.json")
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                return json.load(f)

    lat, lon = position if position is not None else station_position(station, get)
    out: dict = {
        "station_id": station.upper(),
        "lat": lat,
        "lon": lon,
        "start_time": f"{start:%Y-%m-%dT%H:%M:%SZ}",
        "end_time": f"{end:%Y-%m-%dT%H:%M:%SZ}",
        "products": {},
        "sources": [],
    }
    for product in products:
        df, sources = fetch_product(station, product, start, end, get=get)
        out["sources"] += sources
        times = [f"{t:%Y-%m-%dT%H:%M:%SZ}" for t in df.index]
        if product == "stdmet":
            out["products"]["stdmet"] = {
                "times": times,
                "units": {c: u for c, u in STDMET_COLUMNS.items() if c in df},
                **{c: _list(df[c]) for c in STDMET_COLUMNS if c in df},
            }
        else:
            speed = df["SPD01"] / 100.0 if "SPD01" in df else pd.Series(dtype=float)
            toward = (
                np.deg2rad(df["DIR01"]) if "DIR01" in df else pd.Series(dtype=float)
            )
            out["products"]["adcp"] = {
                "times": times,
                "depth_m": _list(df["DEP01"]) if "DEP01" in df else [],
                "u": _list(speed * np.sin(toward)),
                "v": _list(speed * np.cos(toward)),
                "units": {
                    "u": "m s-1 eastward",
                    "v": "m s-1 northward",
                    "depth_m": "m",
                },
            }

    if cache_path and end < pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(
        days=2
    ):
        with open(cache_path, "w") as f:
            json.dump(out, f)
    return out
