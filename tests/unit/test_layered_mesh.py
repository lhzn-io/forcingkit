"""Layered-mesh contract, sections, shared scales, tracks and adapters, on synthetic meshes (no
network)."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from forcingkit import layered_mesh as lm
from forcingkit.mesh_frames import MeshFrames

TIMES = pd.date_range("2026-04-02", periods=2, freq="1h")


def square(n_layer: int = 3) -> lm.LayeredMesh:
    """The square [0, 0.1] x [0, 0.1] degrees split along its SW-NE diagonal: element 0 south-east
    of it, element 1 north-west. Sigma layers over h = 10 m (west) to 20 m (east), zeta 0.5 m at
    the second time; element field E = element index + 10 * layer, node field N = node longitude
    in degrees (linear, so interpolation reproduces it exactly)."""
    lon = np.array([0.0, 0.1, 0.1, 0.0])
    lat = np.array([0.0, 0.0, 0.1, 0.1])
    tri = np.array([[0, 1, 2], [0, 2, 3]])
    k = np.arange(n_layer)
    E = np.broadcast_to(
        np.arange(2)[None, None, :] + 10.0 * k[None, :, None], (2, n_layer, 2)
    )
    N = np.broadcast_to(lon[None, None, :], (2, n_layer, 4))
    siglev = -np.linspace(0.0, 1.0, n_layer + 1)
    siglay = 0.5 * (siglev[1:] + siglev[:-1])
    zeta = np.array([[0.0] * 4, [0.5] * 4])
    return lm.LayeredMesh(
        TIMES,
        lon,
        lat,
        tri,
        {"E": E, "N": N},
        h=10.0 + 100.0 * lon,
        siglay=siglay,
        siglev=siglev,
        zeta=zeta,
    )


# Contract --------------------------------------------------------------------------------------


def test_contract_infers_kinds_and_checks_shapes():
    m = square()
    assert str(m.times.tz) == "UTC" and len(m) == 2
    assert (m.n_node, m.n_elem, m.n_layer, m.vertical) == (4, 2, 3, "sigma")
    assert m.meta["E"].kind == lm.ELEMENT and m.meta["N"].kind == lm.NODE
    assert m.siglay.shape == (3, 4) and m.siglev.shape == (4, 4)
    with pytest.raises(ValueError, match="layers"):
        lm.LayeredMesh(
            TIMES,
            m.node_lon,
            m.node_lat,
            m.tri,
            {"E": m.fields["E"], "F": m.fields["E"][:, :2]},
        )
    with pytest.raises(ValueError, match="needs sigma"):
        lm.LayeredMesh(TIMES, m.node_lon, m.node_lat, m.tri, {"E": m.fields["E"]})
    with pytest.raises(ValueError, match="needs h"):
        lm.LayeredMesh(
            TIMES, m.node_lon, m.node_lat, m.tri, {"E": m.fields["E"]}, siglay=m.siglay
        )
    with pytest.raises(ValueError, match="columns"):
        lm.LayeredMesh(
            TIMES,
            m.node_lon,
            m.node_lat,
            m.tri,
            {"E": m.fields["N"]},
            meta={"E": lm.FieldMeta("E", kind="element")},
            h=m.h,
            siglay=m.siglay,
        )
    with pytest.raises(ValueError, match="kind"):
        lm.FieldMeta("x", kind="cell")


def test_single_layer_needs_no_vertical_and_takes_2d_fields():
    m = square()
    s = lm.LayeredMesh(TIMES, m.node_lon, m.node_lat, m.tri, {"E": m.fields["E"][:, 0]})
    assert (s.n_layer, s.vertical) == (1, "surface")
    assert s.fields["E"].shape == (2, 1, 2)
    assert np.all(s.layer_heights() == 0.0)


def test_element_values_average_node_fields_and_derive_speed():
    m = square()
    np.testing.assert_allclose(
        m.element_values("N")[0, 0], [0.2 / 3, 0.1 / 3], rtol=1e-6
    )
    u = np.full((2, 3, 2), 3.0)
    v = np.full((2, 3, 2), 4.0)
    m2 = lm.LayeredMesh(
        TIMES, m.node_lon, m.node_lat, m.tri, {"u": u, "v": v}, h=m.h, siglay=m.siglay
    )
    assert np.all(m2.element_values("speed") == 5.0)
    assert m2.field_meta("speed").log
    with pytest.raises(KeyError):
        m.element_values("speed")


def test_layer_heights_follow_sigma_and_zeta():
    m = square()
    z = m.layer_heights()
    h = m.h[m.tri].mean(axis=1)
    np.testing.assert_allclose(z[0, 0], m.siglay[0, 0] * h)
    np.testing.assert_allclose(z[1, -1], m.siglay[-1, 0] * (h + 0.5) + 0.5)


def test_element_area_in_square_metres():
    m = square()
    side = lm.EARTH_RADIUS * np.deg2rad(0.1)
    area = m.element_area()
    # Each half of a 0.1 degree square near the equator (cos(lat) within 2e-6 of 1).
    np.testing.assert_allclose(area, 0.5 * side**2, rtol=1e-5)


def test_surface_is_mesh_frames():
    m = square()
    u = np.random.default_rng(0).normal(size=(2, 3, 2))
    m2 = lm.LayeredMesh(
        TIMES, m.node_lon, m.node_lat, m.tri, {"u": u, "v": -u}, h=m.h, siglay=m.siglay
    )
    f = m2.surface()
    assert isinstance(f, MeshFrames)
    np.testing.assert_allclose(f.u, u[:, 0], rtol=1e-6)


# Sections --------------------------------------------------------------------------------------


def test_section_samples_evenly_and_takes_host_values():
    m = square()
    # West to east along lat 0.03: element 1 (north-west of the diagonal) until lon 0.03,
    # element 0 after.
    s = m.section([(0.005, 0.03), (0.095, 0.03)], n=10)
    assert s.distance[0] == 0.0
    np.testing.assert_allclose(np.diff(s.distance), s.distance[-1] / 9)
    np.testing.assert_allclose(s.lon, np.linspace(0.005, 0.095, 10), atol=1e-12)
    np.testing.assert_array_equal(s.element, np.where(s.lon < 0.03, 1, 0))
    np.testing.assert_array_equal(
        s.values["E"][0, 2], np.where(s.lon < 0.03, 21.0, 20.0)
    )
    # Node field: flat per element by default, linear on request.
    np.testing.assert_allclose(
        s.values["N"][0, 0], m.element_values("N")[0, 0][s.element], rtol=1e-6
    )
    lin = m.section([(0.005, 0.03), (0.095, 0.03)], n=10, node_values="linear")
    np.testing.assert_allclose(lin.values["N"][0, 1], s.lon, atol=1e-6)


def test_section_heights_and_bed_interpolate_in_the_element():
    m = square()
    s = m.section([(0.02, 0.05), (0.08, 0.05)], n=4)
    h = 10.0 + 100.0 * s.lon
    np.testing.assert_allclose(s.bed_z, -h)
    np.testing.assert_allclose(s.z[0], m.siglay[:, :1] * h[None])
    np.testing.assert_allclose(s.z[1], m.siglay[:, :1] * (h[None] + 0.5) + 0.5)
    np.testing.assert_allclose(s.z_interfaces[1, 0], 0.5)
    np.testing.assert_allclose(s.z_interfaces[1, -1], -h)


def test_section_follows_a_polyline_and_marks_points_outside():
    m = square()
    s = m.section([(-0.05, 0.05), (0.05, 0.05), (0.05, 0.15)], n=21, fields=["E"])
    assert set(s.values) == {"E"}
    side = lm.EARTH_RADIUS * np.deg2rad(0.1)
    np.testing.assert_allclose(s.distance[-1], 2 * side, rtol=1e-4)
    out = s.element < 0
    assert out[0] and out[-1] and not out[10]
    assert np.isnan(s.values["E"][:, :, out]).all() and np.isnan(s.bed_z[out]).all()
    assert np.isfinite(s.values["E"][:, :, ~out]).all()


def test_brute_force_location_agrees_with_trapezoid_map():
    m = square()
    rng = np.random.default_rng(1)
    lon, lat = rng.uniform(-0.02, 0.12, 200), rng.uniform(-0.02, 0.12, 200)
    e1, w1 = m.locate(lon, lat)
    m._finder = False
    e2, w2 = m.locate(lon, lat)
    inside = (lon > 0) & (lon < 0.1) & (lat > 0) & (lat < 0.1)
    np.testing.assert_array_equal(e1 >= 0, inside)
    np.testing.assert_array_equal(e1, e2)
    np.testing.assert_allclose(w1[inside].sum(axis=1), 1.0)


# Structured grids ------------------------------------------------------------------------------


def test_structured_regular_grid_carries_one_value_per_cell():
    lon, lat = np.array([0.0, 1.0, 2.0]), np.array([10.0, 11.0])
    mask = np.array([[True, True, False], [True, True, True]])
    val = np.arange(6.0).reshape(1, 2, 3)  # (t, ny, nx): cell (j, i) = 3j + i
    m = lm.structured(
        lon,
        lat,
        mask,
        times=TIMES[:1],
        fields={"T": val},
        meta={"T": lm.FieldMeta("T", "Temperature", "°C", lm.NODE)},
    )
    assert (m.n_elem, m.n_layer, m.vertical) == (10, 1, "surface")
    assert m.meta["T"].kind == lm.ELEMENT and m.meta["T"].label == "Temperature"
    np.testing.assert_array_equal(m.source_index, np.repeat([0, 1, 3, 4, 5], 2))
    np.testing.assert_array_equal(m.fields["T"][0, 0], np.repeat([0, 1, 3, 4, 5], 2))
    # Corners midway between centres, extended half a cell.
    assert m.node_lon.min() == -0.5 and m.node_lon.max() == 2.5
    assert m.node_lat.min() == 9.5 and m.node_lat.max() == 11.5
    np.testing.assert_allclose(
        m.element_area().sum(), 5 * m.element_area()[:2].sum(), rtol=0.02
    )
    s = m.section([(-0.4, 10.0), (2.4, 10.0)], n=30)  # no sample on a cell edge
    # Row 0: cells 0, 1, then the masked cell (no value).
    expect = np.where(s.lon < 0.5, 0.0, np.where(s.lon < 1.5, 1.0, np.nan))
    np.testing.assert_array_equal(s.values["T"][0, 0], expect)


def test_structured_descending_axis_and_curvilinear_corners():
    lon = np.array([0.0, 1.0, 2.0])
    lat = np.array([11.0, 10.0])  # north first
    val = np.arange(6.0).reshape(1, 2, 3)
    m = lm.structured(lon, lat, times=TIMES[:1], fields={"F": val})
    e, _ = m.locate([0.0, 2.0], [11.0, 10.0])
    np.testing.assert_array_equal(m.fields["F"][0, 0][e], [0.0, 5.0])
    # A sheared grid given as 2-D centres keeps corners midway between them.
    lon2, lat2 = np.meshgrid(lon, [10.0, 11.0, 12.0])
    lon2 = lon2 + 0.2 * (lat2 - 10.0)
    c = lm._corners(lon2)
    np.testing.assert_allclose(c[1, 1], 0.5 + 0.2 * 0.5)
    m2 = lm.structured(lon2, lat2, times=TIMES[:1], fields={"F": np.zeros((1, 3, 3))})
    assert m2.n_elem == 18


def test_structured_fixed_levels_and_bed():
    lon, lat = np.array([0.0, 1.0]), np.array([0.0, 1.0])
    z = np.array([-1.0, -5.0, -20.0])
    f = np.arange(2 * 3 * 2 * 2, dtype=float).reshape(2, 3, 2, 2)
    bed = np.array([[-10.0, -30.0], [-30.0, -30.0]])
    f[:, 2, 0, 0] = np.nan  # below the bed of cell (0, 0)
    m = lm.structured(lon, lat, None, z, times=TIMES, fields={"F": f}, bed_z=bed)
    assert (m.vertical, m.n_layer) == ("z", 3)
    s = m.section([(0.0, 0.0), (1.0, 0.0)], n=2)
    np.testing.assert_array_equal(s.z[1, :, 0], z)
    np.testing.assert_array_equal(s.bed_z, [-10.0, -30.0])
    assert np.isnan(s.values["F"][0, 2, 0]) and s.values["F"][1, 1, 1] == f[1, 1, 0, 1]
    with pytest.raises(ValueError, match="z must be"):
        lm.structured(lon, lat, None, np.zeros((3, 4)), times=TIMES, fields={"F": f})


# Field metadata and shared scales --------------------------------------------------------------


def test_field_meta_codes_and_manifest_entry():
    t = lm.FieldMeta("T", "Temperature", "°C", lm.NODE).with_range(1.7406, 7.9932)
    codes = t.encode([0.0, 1.7406, 7.9932, 100.0, np.nan])
    np.testing.assert_array_equal(codes, [1, 1, 255, 255, 0])
    v = np.linspace(2.0, 7.9, 9)
    np.testing.assert_allclose(
        t.decode(t.encode(v)), v, atol=0.5 * (t.hi - t.lo) / 254 + 1e-9
    )
    assert np.isnan(t.decode(0))
    entry = t.manifest_entry()
    assert list(entry) == ["name", "label", "units", "lo", "hi", "log", "ticks"]
    assert entry["ticks"] == [2.0, 4.0, 6.0]
    sp = lm.SPEED_META.with_range(0.02, 3.1)
    np.testing.assert_allclose(sp.decode(sp.encode([0.1, 1.0])), [0.1, 1.0], rtol=0.011)
    assert sp.tick_values() == [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
    with pytest.raises(ValueError, match="no colour scale"):
        lm.FieldMeta("x").encode([1.0])


def test_nice_ticks_round_legend_values():
    assert lm.nice_ticks(28.4723, 34.2236) == [30.0, 32.0, 34.0]
    assert lm.nice_ticks(-3.7407, 3.7407) == [-2.0, 0.0, 2.0]


def test_weighted_percentile():
    x = np.array([5.0, 1.0, 3.0])
    assert lm.weighted_percentile(x, np.ones(3), [0, 50, 100]) == [1.0, 3.0, 5.0]
    assert lm.weighted_percentile(x, np.array([1.0, 1.0, 98.0]), [2, 50]) == [3.0, 3.0]


def test_shared_scale_weights_by_area_across_meshes():
    # Four 1 degree cells at 10 degC; 100 cells of 0.001 degree at 30 degC, which outnumber
    # them but hold under 1% of the area.
    big = lm.structured(
        np.array([0.0, 1.0]),
        np.array([0.0, 1.0]),
        times=TIMES,
        fields={"T": np.full((2, 2, 2), 10.0)},
    )
    small = lm.structured(
        np.linspace(5.0, 5.009, 10),
        np.linspace(5.0, 5.009, 10),
        times=TIMES,
        fields={"T": np.full((2, 10, 10), 30.0)},
    )
    t = lm.shared_scale([big, small], "T")
    assert (t.lo, t.hi) == (10.0, 10.0)
    by_count = lm.weighted_percentile(
        np.r_[big.fields["T"].ravel(), small.fields["T"].ravel()],
        np.ones(big.fields["T"].size + small.fields["T"].size),
        [99],
    )
    assert by_count == [30.0]
    with pytest.raises(KeyError):
        lm.shared_scale([big], "S")


def test_shared_scales_symmetric_fields_and_window():
    m = square()
    w = np.zeros((2, 3, 2))
    w[0] = -1.0
    w[1] = 4.0
    mesh = lm.LayeredMesh(
        TIMES,
        m.node_lon,
        m.node_lat,
        m.tri,
        {"w": w, "N": m.fields["N"]},
        meta={"w": lm.FieldMeta("w", "Vertical velocity", "m/s", symmetric=True)},
        h=m.h,
        siglay=m.siglay,
    )
    s = lm.shared_scales([mesh])
    assert (s["w"].lo, s["w"].hi) == (-4.0, 4.0)
    assert s["N"].lo >= 0.0 and s["N"].hi <= 0.1
    first = lm.shared_scales([mesh], ["w"], window=(TIMES[0], TIMES[0]))
    assert (first["w"].lo, first["w"].hi) == (-1.0, 1.0)
    assert lm.shared_scale([mesh], "w", symmetric=False).lo == -1.0


# Tracks ----------------------------------------------------------------------------------------


def test_tracks_quantize_within_half_a_step_and_zero_the_dead():
    rng = np.random.default_rng(2)
    xyz = np.stack(
        [
            rng.uniform(-72.2, -72.0, (5, 7)),
            rng.uniform(41.1, 41.3, (5, 7)),
            rng.uniform(-40, 0, (5, 7)),
        ],
        axis=-1,
    )
    alive = np.ones((5, 7), dtype=bool)
    alive[3, 2] = False
    xyz[4, 6] = np.nan
    tr = lm.Tracks(pd.date_range("2026-04-02", periods=5, freq="2min"), xyz, alive)
    assert tr.n_particle == 7 and not tr.alive[4, 6] and not tr.alive[3, 2]
    codes, q = tr.quantize(-72.1, 41.2)
    assert codes.dtype == np.int16 and codes.shape == (5, 7, 3)
    assert np.all(codes[3, 2] == 0) and np.all(codes[4, 6] == 0)
    local = tr.local_xyz(-72.1, 41.2)
    back = np.asarray(q["offset"]) + np.asarray(q["scale"]) * codes
    err = np.abs(back - local)[tr.alive]
    assert np.all(err <= 0.5 * np.asarray(q["scale"]) + 1e-9)
    assert np.abs(codes).max() == 32767
    with pytest.raises(ValueError):
        lm.Tracks(tr.times, xyz[:, :, :2])


# NECOFS adapter --------------------------------------------------------------------------------


def necofs_like() -> xr.Dataset:
    """A dataset shaped like `necofs.volume`: two elements, four nodes, two layers."""
    m = square(2)
    nt, nl = 2, 2
    rng = np.random.default_rng(3)
    lay, lev = ("time", "siglay", "element"), ("time", "siglay", "node")
    return xr.Dataset(
        {
            "u": (lay, rng.normal(size=(nt, nl, 2))),
            "v": (lay, rng.normal(size=(nt, nl, 2))),
            "ww": (lay, rng.normal(size=(nt, nl, 2)) * 1e-3),
            "temp": (lev, rng.uniform(2, 8, (nt, nl, 4))),
            "salinity": (lev, rng.uniform(28, 34, (nt, nl, 4))),
            "zeta": (("time", "node"), rng.normal(size=(nt, 4)) * 0.3),
            "h": (("node",), m.h),
            "siglay": (("siglay", "node"), m.siglay),
            "siglev": (("siglev", "node"), m.siglev),
            "node_lon": (("node",), m.node_lon),
            "node_lat": (("node",), m.node_lat),
            "triangles": (("element", "vertex"), m.tri.astype(np.int32)),
            "element_id": (("element",), np.array([1001, 1002])),
        },
        coords={"time": TIMES.tz_localize(None)},
        attrs={"requested_bbox": [-1.0, -1.0, 1.0, 1.0]},
    )


def test_necofs_adapter_maps_the_volume():
    ds = necofs_like()
    m = lm.from_necofs_volume(ds)
    assert (m.vertical, m.n_layer, m.n_elem, m.n_node) == ("sigma", 2, 2, 4)
    assert {k: v.kind for k, v in m.meta.items()} == {
        "u": "element",
        "v": "element",
        "w": "element",
        "T": "node",
        "S": "node",
    }
    np.testing.assert_allclose(m.fields["w"], ds["ww"].values, rtol=1e-6)
    np.testing.assert_array_equal(m.source_index, [1001, 1002])
    assert m.attrs["model"] == "necofs" and m.attrs["bbox"] == [-1.0, -1.0, 1.0, 1.0]
    assert m.meta["w"].symmetric and m.meta["T"].units == "°C"
    assert m.meta is not lm.NECOFS_META and m.meta["T"] is not lm.NECOFS_META["T"]
    s = m.section([(0.02, 0.05), (0.08, 0.05)], n=3, fields=["speed", "T"])
    assert s.values["speed"].shape == (2, 2, 3)


def test_necofs_reads_through_volume(monkeypatch):
    from forcingkit.fetchers import necofs as fetch

    calls = []

    def fake_volume(bbox, start, end, cache=True):
        calls.append((bbox, start, end, cache))
        return necofs_like()

    monkeypatch.setattr(fetch, "volume", fake_volume)
    m = lm.necofs(
        (-73.4, 40.9, -72.8, 41.2), "2026-04-02T02:00", "2026-04-02T21:00", cache=False
    )
    assert calls == [
        ([-73.4, 40.9, -72.8, 41.2], "2026-04-02T02:00", "2026-04-02T21:00", False)
    ]
    assert m.n_layer == 2
