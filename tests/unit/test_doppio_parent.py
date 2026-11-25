"""Doppio as a parent ocean (`doppio.iter_parent`, schema z-v3) on a synthetic ROMS history file,
and naming a donor in `dispatcher.dispatch_obc_request`. No network.

The file is a small unrotated grid (angle 0) with linear s-levels (Cs = s, so the fraction of the
water column at a level is s itself), a flat 100 m bed, a land corner, uniform currents and a
temperature linear in depth, so every interpolated value is known."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from forcingkit import dispatcher
from forcingkit.fetchers import doppio, necofs

NY, NX, N = 8, 10, 4
LON0, LAT0, STEP = -69.9, 39.9, 0.05
DEPTH = 100.0
U, V, W = 0.3, -0.1, 1e-4
START = pd.Timestamp("2022-08-13 02:00")
BOX = [-69.65, 40.1, -69.5, 40.2]  # inside the water cells, clear of the land corner
FILL = 1e37
READ_VOLUME = doppio.read_volume  # the real reader, before any test patches the module


def _temp(z):
    return 20.0 + 0.1 * z  # warmer at the surface, z negative below it


def _roms(times: pd.DatetimeIndex) -> xr.Dataset:
    """A ROMS history file in the reanalysis's names and layout."""
    nt = len(times)
    lon = LON0 + STEP * np.arange(NX)
    lat = LAT0 + STEP * np.arange(NY)
    lon_rho, lat_rho = np.meshgrid(lon, lat)
    mask = np.ones((NY, NX))
    mask[:3, :3] = 0.0  # a land corner, south-west
    s_w = np.linspace(-1.0, 0.0, N + 1)
    s_rho = 0.5 * (s_w[:-1] + s_w[1:])
    z_rho = s_rho[:, None, None] * DEPTH * np.ones((NY, NX))  # zeta = 0
    temp = np.broadcast_to(_temp(z_rho), (nt, N, NY, NX)).copy()
    temp[:, :, mask == 0] = FILL
    land_u = (mask[:, :-1] == 0) | (mask[:, 1:] == 0)
    land_v = (mask[:-1, :] == 0) | (mask[1:, :] == 0)
    u = np.full((nt, N, NY, NX - 1), U)
    u[:, :, land_u] = FILL
    v = np.full((nt, N, NY - 1, NX), V)
    v[:, :, land_v] = FILL
    w = np.full((nt, N + 1, NY, NX), W)
    w[:, :, mask == 0] = FILL
    zeta = np.zeros((nt, NY, NX))
    zeta[:, mask == 0] = FILL
    seconds = (times - pd.Timestamp("2011-01-01")).total_seconds().values
    rho = ("eta_rho", "xi_rho")
    four = ("ocean_time", "s_rho") + rho
    fill = {"_FillValue": FILL}
    return xr.Dataset(
        {
            "lon_rho": (rho, lon_rho),
            "lat_rho": (rho, lat_rho),
            "h": (rho, np.full((NY, NX), DEPTH)),
            "mask_rho": (rho, mask),
            "angle": (rho, np.zeros((NY, NX))),
            "hc": ((), 10.0),
            "Vtransform": ((), 2),
            "s_rho": (("s_rho",), s_rho),
            "Cs_r": (("s_rho",), s_rho),
            "s_w": (("s_w",), s_w),
            "Cs_w": (("s_w",), s_w),
            "temp": (four, temp, fill),
            "salt": (four, np.where(temp == FILL, FILL, 32.0), fill),
            "u": (("ocean_time", "s_rho", "eta_u", "xi_u"), u, fill),
            "v": (("ocean_time", "s_rho", "eta_v", "xi_v"), v, fill),
            "w": (("ocean_time", "s_w") + rho, w, fill),
            "zeta": (("ocean_time",) + rho, zeta, fill),
        },
        coords={
            "ocean_time": (
                ("ocean_time",),
                seconds,
                {"units": "seconds since 2011-01-01 00:00:00"},
            )
        },
    )


@pytest.fixture
def roms_file(tmp_path):
    """Path of a synthetic file holding 6 hours from START, and a reader for it."""
    path = str(tmp_path / "doppio_synthetic.nc")
    _roms(pd.date_range(START, periods=6, freq="1h")).to_netcdf(path)
    calls = []

    def reader(box, start, hours, dap_url=None, variables=None):
        calls.append((list(box), pd.Timestamp(start), hours, dap_url))
        return READ_VOLUME(box, start, hours, dap_url=path, variables=variables)

    return path, reader, calls


def _collect(reader, hours=3, bbox=BOX, **kwargs):
    items = list(
        doppio.iter_parent(
            str(START),
            hours,
            bbox,
            pad_cells=1,
            vertical_spacing_m=10.0,
            reader=reader,
            **kwargs,
        )
    )
    assert items[0][0] == "static"
    return items[0][1], items[1:]


def test_parent_static_matches_the_necofs_layout(roms_file):
    _, reader, calls = roms_file
    static, records = _collect(reader)
    assert static.attrs["schema"] == necofs.OBC_SCHEMA == "z-v3"
    assert static.attrs["w_source"] == "model" == doppio.W_SOURCE
    assert static.attrs["doppio_url"] == doppio.REANALYSIS_URL
    assert "reanalysis" in static.attrs["doppio_product"]
    assert static.attrs["horizontal_spacing_deg"] == doppio.PARENT_SPACING_DEG
    assert doppio.PARENT_RECORD_DIMS == necofs.PARENT_RECORD_DIMS
    assert doppio.PARENT_UNITS == necofs.PARENT_UNITS
    for name in ("h", "mask", "z_face", "z", "lat", "lon"):
        assert name in static.variables
    # The regular grid covers the box plus one 0.02 degree step on every side.
    assert static["lon"].values[0] == pytest.approx(BOX[0] - 0.02)
    assert static["lat"].values[-1] == pytest.approx(BOX[3] + 0.02)
    # z levels every 10 m from the deepest bed up, bottom to top.
    assert static["z_face"].values[0] <= -DEPTH and static["z_face"].values[-1] == 0.0
    np.testing.assert_allclose(static["h"].values, DEPTH)
    # One read for the whole window, from the reanalysis.
    assert len(calls) == 1 and calls[0][2] == 3 and calls[0][3] == doppio.REANALYSIS_URL


def test_parent_records_carry_the_model_fields(roms_file):
    _, reader, _ = roms_file
    static, records = _collect(reader)
    assert [r[1] for r in records] == list(pd.date_range(START, periods=3, freq="1h"))
    z = static["z"].values
    for _, _, rec in records:
        assert set(rec) == set(doppio.PARENT_RECORD_DIMS)
        assert rec["u"].shape == (z.size, static.sizes["lat"], static.sizes["lon"])
        assert rec["zeta"].shape == (static.sizes["lat"], static.sizes["lon"])
        assert all(rec[k].dtype == np.float32 for k in rec)
        np.testing.assert_allclose(rec["u"], U, rtol=1e-6)
        np.testing.assert_allclose(rec["v"], V, rtol=1e-6)
        np.testing.assert_allclose(rec["w"], W, rtol=1e-5)
        np.testing.assert_allclose(rec["salt"], 32.0, rtol=1e-6)
        np.testing.assert_allclose(rec["zeta"], 0.0, atol=1e-7)
        # Linear in depth between the shallowest and deepest layer centers (-12.5 to -87.5 m).
        mid = (z > -87.5) & (z < -12.5)
        expected = _temp(z[mid])[:, None, None]
        np.testing.assert_allclose(
            rec["temp"][mid],
            np.broadcast_to(expected, rec["temp"][mid].shape),
            rtol=1e-5,
        )


def test_land_and_cells_beside_it_are_nan(roms_file):
    _, reader, _ = roms_file
    # A box over the land corner (-69.9..-69.8, 39.9..40.0).
    static, records = _collect(reader, bbox=[-69.88, 39.92, -69.7, 40.05])
    h = static["h"].values
    lon, lat = np.meshgrid(static["lon"].values, static["lat"].values)
    on_land = (lon < -69.8) & (lat < 40.0)
    assert np.isnan(h[on_land]).all() and (static["mask"].values[on_land] == 0).all()
    assert np.isfinite(h).any()
    u = records[0][2]["u"]
    assert np.isnan(u[:, on_land]).all()
    assert np.isfinite(u[:, ~np.isnan(h)]).any()


def test_a_missing_hour_ends_the_delivery(roms_file):
    _, reader, _ = roms_file
    with pytest.raises(RuntimeError, match="lacks 2 of 8 hours"):
        _collect(reader, hours=8)


def test_a_box_outside_the_domain_is_refused(roms_file):
    _, reader, _ = roms_file
    with pytest.raises(ValueError, match="outside the Doppio domain"):
        _collect(reader, bbox=[-50.0, 40.0, -49.0, 41.0])


def test_long_windows_are_read_a_day_at_a_time(roms_file, monkeypatch):
    _, reader, calls = roms_file
    monkeypatch.setattr(doppio, "_READ_HOURS", 2)
    _, records = _collect(reader, hours=5)
    assert len(records) == 5
    assert [c[2] for c in calls] == [2, 2, 1]
    assert [c[1] for c in calls] == list(pd.date_range(START, periods=3, freq="2h"))


def test_parent_from_the_recorded_reanalysis_corner():
    """The recorded 6 x 6 cell corner of the reanalysis (rotated grid, 90 to 1950 m deep, near
    RED001's track): every target inside the corner is ocean, and the values are the model's."""
    from pathlib import Path

    fixture = str(
        Path(__file__).parent / "data" / "doppio_reanalysis_20220811_corner.nc"
    )
    t0 = pd.Timestamp("2022-08-11 00:00")

    def reader(box, start, hours, dap_url=None, variables=None):
        return READ_VOLUME(box, start, hours, dap_url=fixture, variables=variables)

    items = list(
        doppio.iter_parent(
            str(t0),
            2,
            [-69.65, 39.9, -69.55, 39.97],
            pad_cells=0,
            vertical_spacing_m=25.0,
            reader=reader,
        )
    )
    static, records = items[0][1], items[1:]
    assert len(records) == 2
    h = static["h"].values
    assert np.isfinite(h).all() and 89.0 < h.min() and h.max() < 1954.0
    native = READ_VOLUME(
        [-69.9, 39.7, -69.3, 40.2], t0, 1, dap_url=fixture, variables=("u", "v", "ww")
    )
    for _, _, rec in records:
        wet = np.isfinite(rec["u"])
        assert wet[-1].all()  # the top level is ocean everywhere
        assert np.isfinite(rec["w"][wet]).all() and np.isfinite(rec["temp"][wet]).all()
        # Interpolation stays within the native range of each field.
        for k, name in (("u", "u"), ("v", "v"), ("w", "ww")):
            lo, hi = np.nanmin(native[name].values), np.nanmax(native[name].values)
            assert lo - 1e-6 <= np.nanmin(rec[k]) and np.nanmax(rec[k]) <= hi + 1e-6, k


# Naming a donor in the dispatcher.


def test_doppio_is_a_donor_but_never_ranked():
    assert "doppio" in dispatcher.obc_donor_ids()
    assert doppio.get_metadata()["auto_rank"] is False
    for bbox in (BOX, [-75.5, 38.5, -74.5, 39.5], [-70.9, 41.3, -70.6, 41.6]):
        ids = [meta["id"] for _, _, meta in dispatcher._rank_obc_candidates(bbox)]
        assert "doppio" not in ids
    assert dispatcher.predict_obc_donor(BOX)["id"] == "necofs"


def test_a_named_donor_is_checked_before_anything_is_fetched():
    with pytest.raises(ValueError, match="Unknown parent-ocean donor 'roms'"):
        dispatcher.resolve_obc_donor("roms", BOX)
    with pytest.raises(ValueError, match="outside the Doppio"):
        dispatcher.resolve_obc_donor("doppio", [-50.0, 40.0, -49.0, 41.0])
    module, func, meta = dispatcher.resolve_obc_donor("doppio", BOX)
    assert module is doppio and func is None and meta["id"] == "doppio"


def test_dispatch_delivers_the_named_donor(roms_file, monkeypatch, tmp_path):
    _, reader, _ = roms_file
    monkeypatch.setattr(doppio, "read_volume", reader)
    path = dispatcher.dispatch_obc_request(
        str(START),
        3,
        BOX,
        cache_dir=str(tmp_path),
        pad_cells=1,
        vertical_spacing_m=10.0,
        donor="doppio",
    )
    assert dispatcher.delivered_obc_donor(path) == "doppio"
    assert dispatcher.parent_store_is_current(path, necofs.OBC_SCHEMA)
    ds = xr.open_zarr(path, consolidated=True, zarr_format=2)
    assert ds.attrs["w_source"] == "model" and ds.attrs["schema"] == "z-v3"
    assert ds["w"].attrs["standard_name"] == "upward_sea_water_velocity"
    assert ds.sizes["time"] == 3
    np.testing.assert_allclose(ds["u"].values, U, rtol=1e-6)


def test_a_named_donor_has_its_own_store_and_never_falls_back(monkeypatch, tmp_path):
    from forcingkit.fetchers import hycom

    seen = []

    def fake_stream(module, zarr_path, *args, **kwargs):
        seen.append((module.__name__, zarr_path))
        raise RuntimeError("Doppio unreachable")

    def no_hycom(*args, **kwargs):
        raise AssertionError("a named donor fell back to HYCOM")

    monkeypatch.setattr(dispatcher, "_stream_parent", fake_stream)
    monkeypatch.setattr(hycom, "fetch_hycom_boundary_conditions", no_hycom)
    args = (str(START), 3, BOX)
    with pytest.raises(RuntimeError, match="Doppio ROMS .* parent delivery failed"):
        dispatcher.dispatch_obc_request(*args, cache_dir=str(tmp_path), donor="doppio")
    with pytest.raises(RuntimeError):
        dispatcher.dispatch_obc_request(*args, cache_dir=str(tmp_path), donor="necofs")
    with pytest.raises(RuntimeError):
        # The ranked request streams NECOFS first, then falls back (here to the patched HYCOM).
        dispatcher.dispatch_obc_request(*args, cache_dir=str(tmp_path))
    paths = [p for _, p in seen]
    assert [m for m, _ in seen][:2] == [
        "forcingkit.fetchers.doppio",
        "forcingkit.fetchers.necofs",
    ]
    # Doppio, named NECOFS and ranked NECOFS each have a store of their own.
    assert len({paths[0], paths[1], paths[2]}) == 3
