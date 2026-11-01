"""NECOFS file selection for past dates: GOM7 daily history, then the SMAST monthly hindcast, then
the rolling forecast; and a parent delivery read from a monthly file. No network."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from forcingkit.fetchers import necofs

BASE = necofs.NECOFS_HINDCAST_BASE
BBOX = [-69.9, 40.1, -69.1, 40.9]


@pytest.fixture(autouse=True)
def _fresh_hindcast_cache(monkeypatch):
    monkeypatch.setattr(necofs, "_HINDCAST_PRESENT", set())


def _probe(present: set[str], calls: list[str]):
    def exists(url, timeout):
        calls.append(url)
        return url in present

    return exists


@pytest.mark.parametrize(
    "t, expected",
    [
        ("2022-08-13 05:00", f"{BASE}/necofs_2022/NECOFS_202208.nc"),
        ("2017-01-01", f"{BASE}/necofs_2017/NECOFS_201701.nc"),
        ("2016-12-31 23:00", f"{BASE}/gom3_201612.nc"),
        ("1978-01-01", f"{BASE}/gom3_197801.nc"),
    ],
)
def test_hindcast_url_for_each_era(t, expected):
    assert necofs.necofs_hindcast_url(pd.Timestamp(t)) == expected


def test_product_labels():
    assert necofs.necofs_product(necofs.NECOFS_GOM7_URL) == "GOM7 rolling forecast"
    assert necofs.necofs_product(f"{BASE}/gom3_201608.nc").startswith("GOM3 monthly")
    assert necofs.necofs_product(f"{BASE}/necofs_2022/NECOFS_202208.nc").startswith(
        "GOM7-mesh monthly"
    )
    daily = "http://x/NECOFS/Archive/necofs_history/NECOFS_GOM7_2026_03_03.nc"
    assert necofs.necofs_product(daily) == "GOM7 daily history"
    assert necofs.necofs_product("/data/local.nc") == "NECOFS file"


def test_daily_history_wins_when_present(monkeypatch):
    calls: list[str] = []
    daily = "NECOFS_GOM7_2026_03_03.nc"
    monkeypatch.setattr(
        necofs,
        "_dap_exists",
        lambda url, timeout: calls.append(url) or url.endswith(daily),
    )
    url = necofs.get_necofs_url(pd.Timestamp("2026-03-02"))
    assert url.endswith(daily)
    assert len(calls) == 1  # the hindcast is not probed


def test_past_date_uses_the_monthly_hindcast(monkeypatch):
    calls: list[str] = []
    month = f"{BASE}/necofs_2022/NECOFS_202208.nc"
    monkeypatch.setattr(necofs, "_dap_exists", _probe({month}, calls))
    assert necofs.get_necofs_url(pd.Timestamp("2022-08-12")) == month
    assert len(calls) == 2  # daily history, then the hindcast
    # A later day of the same month needs no second hindcast probe.
    assert necofs.get_necofs_url(pd.Timestamp("2022-08-13")) == month
    assert calls[2:] == [calls[0].replace("2022_08_13", "2022_08_14")]


def test_last_day_of_a_month_reads_that_month(monkeypatch):
    # Day D's hours run to the next midnight, the last record of D's monthly file.
    july = f"{BASE}/necofs_2022/NECOFS_202207.nc"
    monkeypatch.setattr(necofs, "_dap_exists", _probe({july}, []))
    assert necofs.get_necofs_url(pd.Timestamp("2022-07-31")) == july


def test_forecast_when_neither_archive_has_the_day(monkeypatch):
    monkeypatch.setattr(necofs, "_dap_exists", _probe(set(), []))
    assert necofs.get_necofs_url(pd.Timestamp("2025-06-01")) == necofs.NECOFS_GOM7_URL


def test_missing_hindcast_is_probed_again(monkeypatch):
    # Only files seen to exist are remembered; a month published later is found.
    calls: list[str] = []
    monkeypatch.setattr(necofs, "_dap_exists", _probe(set(), calls))
    necofs.get_necofs_url(pd.Timestamp("2025-06-01"))
    necofs.get_necofs_url(pd.Timestamp("2025-06-02"))
    assert sum("Seaplan" in c for c in calls) == 2


# --- parent delivery from a monthly file -------------------------------------------------------


def _fvcom(times: pd.DatetimeIndex, n: int = 5, nsig: int = 3) -> xr.Dataset:
    """A FVCOM-shaped file on an n x n node grid over BBOX (plus a margin), uniform u = 0.3 and
    v = -0.1 m/s, 50 m deep, with CF times in seconds since 1858-11-17 as SMAST writes them."""
    lon1 = np.linspace(BBOX[0] - 0.2, BBOX[2] + 0.2, n)
    lat1 = np.linspace(BBOX[1] - 0.2, BBOX[3] + 0.2, n)
    lon, lat = (a.ravel() for a in np.meshgrid(lon1, lat1))
    corners: list[list[int]] = []
    for j in range(n - 1):
        for i in range(n - 1):
            sw, se = j * n + i, j * n + i + 1
            corners += [[sw, se, se + n], [sw, se + n, sw + n]]
    tri = np.array(corners)
    n_node, n_ele, nt = lon.size, len(tri), len(times)
    siglay = np.repeat(-(np.arange(nsig) + 0.5)[:, None] / nsig, n_node, axis=1)
    seconds = (times - pd.Timestamp("1858-11-17")) / pd.Timedelta(seconds=1)
    return xr.Dataset(
        {
            "nv": (("three", "nele"), (tri.T + 1).astype(np.int32)),
            "lon": (("node",), lon.astype(np.float32)),
            "lat": (("node",), lat.astype(np.float32)),
            "lonc": (("nele",), lon[tri].mean(axis=1).astype(np.float32)),
            "latc": (("nele",), lat[tri].mean(axis=1).astype(np.float32)),
            "h": (("node",), np.full(n_node, 50.0, np.float32)),
            "siglay": (("siglay", "node"), siglay.astype(np.float32)),
            "u": (
                ("time", "siglay", "nele"),
                np.full((nt, nsig, n_ele), 0.3, np.float32),
            ),
            "v": (
                ("time", "siglay", "nele"),
                np.full((nt, nsig, n_ele), -0.1, np.float32),
            ),
            "temp": (
                ("time", "siglay", "node"),
                np.full((nt, nsig, n_node), 20.0, np.float32),
            ),
            "salinity": (
                ("time", "siglay", "node"),
                np.full((nt, nsig, n_node), 35.0, np.float32),
            ),
            "zeta": (("time", "node"), np.zeros((nt, n_node), np.float32)),
            "time": (
                ("time",),
                seconds.values,
                {"units": "seconds since 1858-11-17 00:00:00"},
            ),
        }
    )


def _month_times(month: str) -> pd.DatetimeIndex:
    start = pd.Timestamp(month)
    return pd.date_range(start, start + pd.offsets.MonthBegin(1), freq="1h")


def _serve(monkeypatch, files: dict[str, xr.Dataset]):
    monkeypatch.setattr(necofs, "_dap_exists", lambda url, timeout: url in files)
    monkeypatch.setattr(necofs, "_open_dap", lambda url: files[url])


def test_parent_from_a_monthly_hindcast(monkeypatch):
    month = f"{BASE}/necofs_2022/NECOFS_202208.nc"
    _serve(monkeypatch, {month: _fvcom(_month_times("2022-08-01"))})
    items = list(
        necofs.iter_parent(
            "2022-08-13T01:00", 12, BBOX, pad_cells=1, vertical_spacing_m=10.0
        )
    )
    static = items[0][1]
    assert static.attrs["necofs_url"] == month
    assert static.attrs["necofs_product"].startswith("GOM7-mesh monthly hindcast")
    records = items[1:]
    assert [r[1] for r in records] == list(
        pd.date_range("2022-08-13T01:00", periods=12, freq="1h")
    )
    u = records[0][2]["u"]
    assert np.nanmean(u) == pytest.approx(0.3, abs=1e-6)
    assert np.nanmean(records[-1][2]["v"]) == pytest.approx(-0.1, abs=1e-6)


def test_parent_window_across_months(monkeypatch):
    july = f"{BASE}/necofs_2022/NECOFS_202207.nc"
    august = f"{BASE}/necofs_2022/NECOFS_202208.nc"
    _serve(
        monkeypatch,
        {
            july: _fvcom(_month_times("2022-07-01")),
            august: _fvcom(_month_times("2022-08-01")),
        },
    )
    items = list(necofs.iter_parent("2022-07-31T22:00", 5, BBOX, pad_cells=1))
    times = [r[1] for r in items[1:]]
    assert times == list(pd.date_range("2022-07-31T22:00", periods=5, freq="1h"))


def test_parent_refuses_a_window_across_the_mesh_change(monkeypatch):
    dec = f"{BASE}/gom3_201612.nc"
    jan = f"{BASE}/necofs_2017/NECOFS_201701.nc"
    _serve(
        monkeypatch,
        {
            dec: _fvcom(_month_times("2016-12-01"), n=4),
            jan: _fvcom(_month_times("2017-01-01")),
        },
    )
    with pytest.raises(RuntimeError, match="different NECOFS mesh"):
        list(necofs.iter_parent("2016-12-31T22:00", 6, BBOX, pad_cells=1))
