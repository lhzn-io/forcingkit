"""NDBC buoy observations: file naming, parsing, sentinels, current components and source
fallback. No network: files are served from fixtures shaped like the April 2026 44091 files."""

import numpy as np
import pandas as pd
import pytest

from ecodata_cache.fetchers import ndbc

STDMET = """#YY  MM DD hh mm WDIR WSPD GST  WVHT   DPD   APD MWD   PRES  ATMP  WTMP  DEWP  VIS  TIDE
#yr  mo dy hr mn degT m/s  m/s     m   sec   sec deg    hPa  degC  degC  degC  nmi    ft
2026 04 01 23 30 999 99.0 99.0  1.94  7.14  5.13 176 9999.0   8.5   5.9 999.0 99.0 99.00
2026 04 02 00 00 200  6.0  8.0  1.82  7.41  5.14 170 1020.4   9.2   6.0 999.0 99.0 99.00
2026 04 02 00 30 210  7.0  9.0    MM  7.41  5.14 170 1020.6   9.0   6.1 999.0 99.0 99.00
2026 04 03 00 00 220  8.0 10.0  1.50  7.00  5.00 170 1021.0   8.0   6.2 999.0 99.0 99.00
"""

ADCP = """#YY  MM DD hh mm DEP01 DIR01 SPD01
#yr  mo dy hr mn     m   deg  cm/s
2026 04 02 00 00     1.0    90  20.0
2026 04 02 00 30     1.0   180  10.0
"""


def test_month_codes_run_one_to_nine_then_letters():
    assert [ndbc.month_code(m) for m in (1, 4, 9, 10, 11, 12)] == [
        "1",
        "4",
        "9",
        "a",
        "b",
        "c",
    ]
    urls = ndbc.candidate_urls("44091", "stdmet", 2026, 4)
    assert urls[0].endswith("/data/stdmet/Apr/4409142026.txt.gz")
    assert urls[1].endswith("/data/historical/stdmet/44091h2026.txt.gz")
    assert urls[2].endswith("/data/realtime2/44091.txt")
    assert ndbc.candidate_urls("44091", "adcp", 2025, 11)[0].endswith(
        "/data/adcp/Nov/44091b2025.txt.gz"
    )


def test_parse_maps_sentinels_and_mm_to_nan():
    df = ndbc.parse_ndbc_text(STDMET)
    assert df.index[1] == pd.Timestamp("2026-04-02T00:00")
    assert np.isnan(df["WSPD"].iloc[0]) and np.isnan(df["PRES"].iloc[0])
    assert np.isnan(df["WVHT"].iloc[2])  # MM
    assert df["WTMP"].iloc[1] == 6.0


def test_two_digit_years_without_minutes():
    old = "#YY MM DD hh WTMP\n#yr mo dy hr degC\n98 04 02 00 5.5\n"
    df = ndbc.parse_ndbc_text(old)
    assert df.index[0] == pd.Timestamp("1998-04-02T00:00")


def _fixture_get(files):
    def get(url):
        for key, text in files.items():
            if key in url:
                return text
        return None

    return get


def test_window_currents_and_fallback():
    # Only the historical files exist here, so the monthly URL (tried first) returns None.
    get = _fixture_get({"historical/stdmet": STDMET, "historical/adcp": ADCP})
    out = ndbc.fetch_ndbc(
        "44091",
        "2026-04-02T00:00:00Z",
        "2026-04-02T23:59:00Z",
        get=get,
        position=(39.772, -73.769),
    )
    st = out["products"]["stdmet"]
    assert st["times"] == ["2026-04-02T00:00:00Z", "2026-04-02T00:30:00Z"]
    assert st["WTMP"] == [6.0, 6.1]
    assert st["WVHT"][1] is None
    # 20 cm/s toward 90 deg is 0.2 m/s east; 10 cm/s toward 180 deg is 0.1 m/s south.
    ad = out["products"]["adcp"]
    assert ad["u"][0] == pytest.approx(0.2) and ad["v"][0] == pytest.approx(
        0.0, abs=1e-12
    )
    assert ad["u"][1] == pytest.approx(0.0, abs=1e-12) and ad["v"][1] == pytest.approx(
        -0.1
    )
    assert all("historical" in s for s in out["sources"])
    assert out["lat"] == 39.772


def test_unknown_product_is_refused():
    with pytest.raises(ValueError):
        ndbc.fetch_ndbc(
            "44091", "2026-04-02", "2026-04-03", ("waves",), position=(0, 0)
        )
