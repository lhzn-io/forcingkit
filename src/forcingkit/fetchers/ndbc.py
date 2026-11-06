"""NDBC buoy observations for model validation: standard meteorology (sea and air temperature,
pressure, wind, waves) and near-surface ADCP currents, through insitukit.

insitukit's NDBC client reads each month from the first file NDBC publishes it in: the per-month
file for months of the current year, then the yearly historical file, then the rolling 45-day
real-time file. It downloads each file once per call, paces and identifies its requests, and turns
NDBC's missing-value sentinels (99, 999, 9999, MM) into NaN. This module keeps forcingkit's
function names and the JSON shape of ``fetch_ndbc``. These observations validate a model; they
never force it.
"""

import json
import math
import os
from typing import Optional

import numpy as np
import pandas as pd
from insitukit import ndbc as _ndbc

PRODUCTS = _ndbc.PRODUCTS
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

month_code = _ndbc.month_code


def candidate_urls(station: str, product: str, year: int, month: int) -> list[str]:
    """URLs that may hold `product` for `station` in the given month, in the order to try."""
    return _ndbc.urls(station, product, year, month)


def parse_ndbc_text(text: str) -> pd.DataFrame:
    """Parse an NDBC whitespace-delimited file into a frame indexed by UTC time (naive),
    sentinels and `MM` as NaN."""
    df = _ndbc.parse(text)
    return df.set_index(pd.DatetimeIndex(df.pop("time")).tz_localize(None))


def _utc_naive(t) -> pd.Timestamp:
    t = pd.Timestamp(t)
    return t if t.tzinfo is None else t.tz_convert("UTC").tz_localize(None)


def station_position(station: str, get=None) -> tuple[float, float]:
    """(lat, lon) of an NDBC station from its station page."""
    return _ndbc.position(station, get=get)


def fetch_product(
    station: str, product: str, start: pd.Timestamp, end: pd.Timestamp, get=None
):
    """`product` for `station` over [start, end] (naive UTC, both included): (frame indexed by
    naive UTC time, source URLs). Each month is taken from the first source that has it."""
    t0 = _utc_naive(start).tz_localize("UTC")
    t1 = _utc_naive(end).tz_localize("UTC") + pd.Timedelta(minutes=1)
    df = _ndbc.fetch(station, t0, t1, product, get=get)
    sources = list(df.attrs.get("sources", []))
    if df.empty:
        return pd.DataFrame(), sources
    df = df.set_index(pd.DatetimeIndex(df.pop("time")).tz_localize(None))
    return df[df.index <= _utc_naive(end)], sources


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
    get=None,
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
