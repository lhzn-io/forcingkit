"""The cache viewer's store views: cells, layers, heights, fixed scales, points and sections."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

sys.path.insert(0, "service")
from forcingkit_serve import view_store  # noqa: E402
from forcingkit_serve.view_store import open_view, resolve_store  # noqa: E402

TIMES = pd.date_range("2026-04-02", periods=4, freq="h")


def _write(ds: xr.Dataset, tmp_path: Path, name: str) -> Path:
    path = tmp_path / f"{name}.zarr"
    ds.to_zarr(path, consolidated=False)
    return path


def fvcom_store(tmp_path: Path) -> Path:
    """Two triangles sharing an edge, three sigma layers surface first, element u and v, node
    temperature that falls with depth."""
    node_lon = np.array([-71.0, -70.9, -71.0, -70.9])
    node_lat = np.array([41.0, 41.0, 41.1, 41.1])
    tri = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    siglay = np.repeat(np.array([[-1 / 6], [-0.5], [-5 / 6]]), 4, axis=1)
    siglev = np.repeat(np.array([[0.0], [-1 / 3], [-2 / 3], [-1.0]]), 4, axis=1)
    h = np.array([10.0, 20.0, 10.0, 20.0])
    u = np.zeros((4, 3, 2), dtype=np.float32)
    u[:, :, 1] = 0.5  # element 1 flows east at 0.5 m/s in every layer
    v = np.zeros_like(u)
    temp = np.broadcast_to(np.array([12.0, 10.0, 8.0])[None, :, None], (4, 3, 4))
    ds = xr.Dataset(
        {
            "node_lon": ("node", node_lon),
            "node_lat": ("node", node_lat),
            "triangles": (("element", "vertex"), tri),
            "element_id": ("element", np.array([100, 101])),
            "h": ("node", h.astype(np.float32), {"units": "m"}),
            "siglay": (("siglay", "node"), siglay.astype(np.float32)),
            "siglev": (("siglev", "node"), siglev.astype(np.float32)),
            "zeta": (("time", "node"), np.zeros((4, 4), np.float32), {"units": "m"}),
            "u": (("time", "siglay", "element"), u, {"units": "m s-1"}),
            "v": (("time", "siglay", "element"), v, {"units": "m s-1"}),
            "temp": (
                ("time", "siglay", "node"),
                temp.astype(np.float32),
                {"units": "degree_Celsius"},
            ),
        },
        coords={"time": TIMES},
    )
    return _write(ds, tmp_path, "necofs_volume_000000000001")


def zgrid_store(tmp_path: Path) -> Path:
    """A 3 x 4 regular grid on true z levels (positive up, listed deepest first), a land cell,
    u growing with record index."""
    lon = np.array([-71.0, -70.99, -70.98, -70.97])
    lat = np.array([41.0, 41.01, 41.02])
    z = np.array([-15.0, -5.0])
    mask = np.ones((3, 4), dtype=np.int8)
    mask[0, 0] = 0
    u = np.zeros((4, 2, 3, 4), dtype=np.float32)
    for t in range(4):
        u[t] = 0.1 * (t + 1)
    u[:, 0] = 0.05  # the deep level is slow
    u[:, :, 0, 0] = np.nan
    temp = np.zeros_like(u)
    temp[:, 0] = 6.0
    temp[:, 1] = 9.0
    ds = xr.Dataset(
        {
            "u": (("time", "z", "lat", "lon"), u, {"units": "m s-1"}),
            "v": (("time", "z", "lat", "lon"), np.zeros_like(u), {"units": "m s-1"}),
            "temp": (("time", "z", "lat", "lon"), temp, {"units": "degree_Celsius"}),
            "mask": (("lat", "lon"), mask),
            "h": (("lat", "lon"), np.full((3, 4), 20.0, np.float32)),
        },
        coords={
            "time": TIMES,
            "z": ("z", z, {"positive": "up"}),
            "lat": lat,
            "lon": lon,
        },
    )
    return _write(ds, tmp_path, "obc_000000000002")


def sigma_snapshot_store(tmp_path: Path) -> Path:
    """An initial condition on s_rho (bottom first) with 2-D centers and no water depth."""
    ny, nx = 3, 3
    lon2, lat2 = np.meshgrid(np.linspace(-71, -70.98, nx), np.linspace(41, 41.02, ny))
    s = np.array([-0.75, -0.25], dtype=np.float32)
    temp = np.stack([np.full((ny, nx), 5.0), np.full((ny, nx), 9.0)]).astype(np.float32)
    ds = xr.Dataset(
        {
            "temp": (("s_rho", "eta_rho", "xi_rho"), temp),
            "lon_rho": (("eta_rho", "xi_rho"), lon2),
            "lat_rho": (("eta_rho", "xi_rho"), lat2),
        },
        coords={"s_rho": s},
        attrs={"target_date": "2026-03-02T15:00:00Z"},
    )
    return _write(ds, tmp_path, "ic_000000000003")


def test_resolve_store_only_lists_inventory_ids(tmp_path):
    items = [
        SimpleNamespace(
            id="necofs_volume_x", path=str(tmp_path / "necofs/volume/x.zarr")
        )
    ]
    assert resolve_store("necofs_volume_x", items) == tmp_path / "necofs/volume/x.zarr"
    assert resolve_store("../etc/passwd", items) is None


def test_fvcom_cells_layers_and_heights(tmp_path):
    view = open_view("necofs_volume_000000000001", fvcom_store(tmp_path))
    m = view.manifest()
    assert m["store"] == "mesh" and m["n_elem"] == 2 and m["vertical"] == "sigma"
    assert [f["name"] for f in m["fields"]][0] == "speed"
    # Surface first: layer 0 is the warmest and the shallowest.
    assert view.values("temp", 0, 0)[0] == pytest.approx(12.0)
    assert view.values("temp", 0, 2)[0] == pytest.approx(8.0)
    z0 = view.heights(0, 0)
    assert z0[0] == pytest.approx(-1 / 6 * np.mean([10.0, 20.0, 10.0]), rel=1e-5)
    assert m["layers"][0]["label"].endswith("m deep")
    assert view.bed()[1] == pytest.approx(-np.mean([20.0, 20.0, 10.0]))
    pt = view.point(-70.91, 41.09, "u", 0, 0)
    assert (
        pt["element"] == 1 and pt["value"] == pytest.approx(0.5) and pt["cell"] == 101
    )


def test_water_speed_uses_the_packaged_ramp_at_its_own_speeds(tmp_path):
    view = open_view("necofs_volume_000000000001", fvcom_store(tmp_path))
    meta = view.scale("speed")
    assert meta.log and meta.lo == pytest.approx(0.02) and meta.hi == pytest.approx(3.1)
    codes = np.frombuffer(view.codes("speed", 0, 0), np.uint8)
    assert codes[0] == 1  # still water: the floor color, not "no value"
    assert codes[1] > 1


def test_z_levels_are_ordered_surface_first_with_true_heights(tmp_path):
    view = open_view("obc_000000000002", zgrid_store(tmp_path))
    m = view.manifest()
    assert m["vertical"] == "z" and m["store"] == "grid"
    assert [ly["z_mean"] for ly in m["layers"]] == [-5.0, -15.0]
    assert m["layers"][0]["label"] == "5.0 m deep"
    # The land cell is not drawn: 11 water cells, two triangles each.
    assert m["n_elem"] == 22
    assert np.all(view.values("temp", 0, 0) == 9.0)
    assert np.all(view.values("temp", 0, 1) == 6.0)
    assert view.bed() is not None and np.all(view.bed() == -20.0)


def test_scale_is_fixed_across_records(tmp_path):
    view = open_view("obc_000000000002", zgrid_store(tmp_path))
    first = view.scale("u")
    view.values("u", 3, 0)
    assert view.scale("u") is first
    assert first.symmetric and first.lo == -first.hi
    # The same value encodes to the same byte whichever record it is in.
    a = np.frombuffer(view.codes("u", 0, 1), np.uint8)
    b = np.frombuffer(view.codes("u", 3, 1), np.uint8)
    assert np.array_equal(a, b)


def test_sigma_without_depth_says_so_and_has_no_heights(tmp_path):
    view = open_view("ic_000000000003", sigma_snapshot_store(tmp_path))
    m = view.manifest()
    assert m["vertical"] == "sigma-index"
    assert m["snapshot"] == "2026-03-02T15:00:00Z" and m["times"] == []
    assert [ly["label"] for ly in m["layers"]] == ["sigma -0.250", "sigma -0.750"]
    assert view.heights(0, 0) is None
    assert np.all(view.values("temp", 0, 0) == 9.0)


def test_section_takes_host_cell_values_in_every_layer(tmp_path):
    view = open_view("obc_000000000002", zgrid_store(tmp_path))
    s = view.section([[-70.995, 41.01], [-70.965, 41.01]], "temp", 0, n=10)
    assert len(s["values"]) == 2
    assert all(v == 9.0 for v in s["values"][0] if v is not None)
    assert all(v == -5.0 for v in s["z"][0] if v is not None)
    assert all(b == -20.0 for b in s["bed"] if b is not None)


def test_large_grids_are_strided_and_say_so(tmp_path, monkeypatch):
    monkeypatch.setitem(view_store.MAX_CELLS, "2d", 3)
    view = open_view("obc_000000000002", zgrid_store(tmp_path), "2d")
    m = view.manifest()
    assert (
        m["stride"] == 2 and m["source_shape"] == [3, 4] and m["grid_shape"] == [2, 2]
    )


def test_ramp_has_transparent_no_value_and_one_entry_per_code(tmp_path):
    import base64

    view = open_view("obc_000000000002", zgrid_store(tmp_path))
    temp = next(f for f in view.manifest()["fields"] if f["name"] == "temp")
    rgba = np.frombuffer(base64.b64decode(temp["ramp_rgba"]), np.uint8).reshape(256, 4)
    assert rgba[0, 3] == 0 and np.all(rgba[1:, 3] == 255)
    # Temperature on cmocean thermal: its first and last knots at the ends of the codes.
    assert tuple(rgba[1, :3]) == (4, 35, 51) and tuple(rgba[255, :3]) == (232, 250, 91)
    assert temp["ramp"] == "thermal"


@pytest.mark.asyncio
async def test_api_serves_a_store_in_a_subfolder(tmp_path, monkeypatch):
    from httpx import ASGITransport, AsyncClient

    from forcingkit_serve.main import app

    sub = tmp_path / "necofs" / "volume"
    sub.mkdir(parents=True)
    fvcom_store(sub)
    monkeypatch.setenv("FORCINGKIT_CACHE_DIR", str(tmp_path))
    view_store.clear_views()
    sid = "necofs_volume_000000000001"
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        m = (await ac.get(f"/api/v1/cache/view/{sid}/manifest")).json()
        assert m["n_elem"] == 2
        mesh = (await ac.get(f"/api/v1/cache/view/{sid}/mesh")).content
        assert len(mesh) == 4 * 8 * 2 + 2 * 3 * 4
        frame = await ac.get(f"/api/v1/cache/view/{sid}/frame?field=temp&layer=2&t=1")
        assert frame.status_code == 200 and len(frame.content) == 2
        pt = (
            await ac.get(f"/api/v1/cache/view/{sid}/point?lon=-70.91&lat=41.09&field=u")
        ).json()
        assert pt["value"] == pytest.approx(0.5)
        missing = await ac.get(f"/api/v1/cache/view/{sid}/frame?field=nope")
        assert missing.status_code == 404
        unknown = await ac.get("/api/v1/cache/view/not_a_store/manifest")
        assert unknown.status_code == 404


@pytest.mark.asyncio
async def test_api_reuses_the_store_list_between_requests(tmp_path, monkeypatch):
    """Drawing a record must not crawl the whole cache each time; a removed store is looked up
    again and then reported missing."""
    import shutil

    from httpx import ASGITransport, AsyncClient

    from forcingkit_serve.main import app
    from forcingkit_serve.routers import view as view_api
    from forcingkit_serve.routers import viewer

    path = fvcom_store(tmp_path)
    monkeypatch.setenv("FORCINGKIT_CACHE_DIR", str(tmp_path))
    view_store.clear_views()
    view_api.forget_stores()
    crawls = []
    real = viewer._crawl_inventory

    def counting():
        crawls.append(1)
        return real()

    monkeypatch.setattr(view_api, "_crawl_inventory", counting)
    sid = path.stem
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        for t in range(4):
            r = await ac.get(f"/api/v1/cache/view/{sid}/frame?field=temp&layer=0&t={t}")
            assert r.status_code == 200
        assert len(crawls) == 1
        shutil.rmtree(path)
        gone = await ac.get(f"/api/v1/cache/view/{sid}/manifest")
        assert gone.status_code == 404
        assert len(crawls) == 2
    view_api.forget_stores()


def test_curvilinear_sigma_store_with_depths_gets_true_heights(tmp_path):
    """A Doppio-shaped volume: 2-D centers as coordinates, sigma per cell surface first, the water
    depth and a land mask."""
    ny, nx = 3, 4
    lon2, lat2 = np.meshgrid(np.linspace(-70.0, -69.9, nx), np.linspace(40.0, 40.1, ny))
    sig = np.stack([np.full((ny, nx), -0.25), np.full((ny, nx), -0.75)]).astype(
        np.float32
    )
    h = np.full((ny, nx), 100.0, np.float32)
    mask = np.ones((ny, nx), bool)
    mask[2, 3] = False
    temp = np.zeros((4, 2, ny, nx), np.float32)
    temp[:, 0], temp[:, 1] = 15.0, 8.0
    ds = xr.Dataset(
        {
            "temp": (("time", "siglay", "eta_rho", "xi_rho"), temp),
            "siglay": (("siglay", "eta_rho", "xi_rho"), sig),
            "h": (("eta_rho", "xi_rho"), h),
            "mask": (("eta_rho", "xi_rho"), mask),
            "lon": (("eta_rho", "xi_rho"), lon2),
            "lat": (("eta_rho", "xi_rho"), lat2),
        },
        coords={"time": TIMES},
    ).set_coords(["lon", "lat"])
    view = open_view("doppio_volume_000000000004", _write(ds, tmp_path, "doppio"))
    m = view.manifest()
    assert m["vertical"] == "sigma" and m["n_elem"] == 2 * 11
    assert [ly["z_mean"] for ly in m["layers"]] == [-25.0, -75.0]
    assert np.all(view.values("temp", 2, 1) == 8.0)
    assert np.all(view.bed() == -100.0)


def test_tracers_spread_levels_down_each_column_without_repeats(tmp_path):
    view = open_view("obc_000000000002", zgrid_store(tmp_path))
    d = view.tracers("speed", t=1, levels=6)
    assert len(d["element"]) == 11  # one per water cell
    # Two layers in this store: two slots kept per column, the rest left empty.
    filled = [sum(u is not None for u in lv["u"]) for lv in d["levels"]]
    assert filled[0] == 11 and sum(filled) == 22
    assert d["levels"][0]["layer"][0] == 0 and d["levels"][-1]["layer"][0] == 1
    assert d["levels"][-1]["z"][0] == -15.0


def test_mesh_bed_at_nodes(tmp_path):
    view = open_view("necofs_volume_000000000001", fvcom_store(tmp_path))
    assert list(view.bed_nodes()) == [-10.0, -20.0, -10.0, -20.0]
    assert view.manifest()["bed_at_nodes"] is True
