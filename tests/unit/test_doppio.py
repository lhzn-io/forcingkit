"""Doppio full-depth subset (`read_volume`, `volume`) on a recorded corner of the reanalysis, and
the C-grid and s-coordinate helpers on synthetic arrays. No network.

The fixture `data/doppio_reanalysis_20220811_corner.nc` holds a 6 x 6 cell corner of the
DopAnV3R3-ini2007 history aggregation (eta_rho 49..54, xi_rho 129..134, 2022-08-11 00:00 and
01:00), in the aggregation's own names and layout, with the u and v the fetcher returned from the
whole grid at the inner 4 x 4 cells as `expected_u`, `expected_v`."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from forcingkit.fetchers import doppio, necofs

FIXTURE = str(Path(__file__).parent / "data" / "doppio_reanalysis_20220811_corner.nc")
START = pd.Timestamp("2022-08-11 00:00")


def fixture_bbox(margin: float = 1e-6) -> list[float]:
    """A box around every cell center of the fixture."""
    ds = xr.open_dataset(FIXTURE, decode_times=False)
    lon, lat = ds["lon_rho"].values, ds["lat_rho"].values
    return [
        float(lon.min()) - margin,
        float(lat.min()) - margin,
        float(lon.max()) + margin,
        float(lat.max()) + margin,
    ]


@pytest.fixture(scope="module")
def vol() -> xr.Dataset:
    return doppio.read_volume(fixture_bbox(), START, 2, dap_url=FIXTURE)


def test_volume_matches_the_necofs_schema(vol):
    assert vol.attrs["schema"] == necofs.VOLUME_SCHEMA == doppio.VOLUME_SCHEMA
    assert doppio.VOLUME_FIELDS == necofs.VOLUME_FIELDS
    for name in doppio.VOLUME_FIELDS + ("h", "siglay", "siglev"):
        assert name in vol.variables
    for name in doppio.VOLUME_FIELDS + ("h",):
        assert (
            vol[name].attrs["units"] == necofs.VOLUME_VARIABLE_ATTRS[name]["units"]
        ), name
    assert vol["u"].dims == ("time", "siglay", "eta_rho", "xi_rho")
    assert vol["zeta"].dims == ("time", "eta_rho", "xi_rho")
    assert vol.sizes == {
        "time": 2,
        "siglay": 40,
        "siglev": 41,
        "eta_rho": 6,
        "xi_rho": 6,
    }
    assert list(pd.DatetimeIndex(vol["time"].values)) == [
        START,
        START + pd.Timedelta(hours=1),
    ]
    for name in doppio.VOLUME_FIELDS:
        assert vol[name].dtype == np.float32


def test_interior_velocity_matches_the_whole_grid(vol):
    """The inner cells have both faces in the corner, so their east and north velocity must equal
    what the fetcher returned from the whole grid (where it agreed with Rutgers's own u_eastward,
    v_northward to 6e-8 m/s)."""
    raw = xr.open_dataset(FIXTURE, decode_times=False)
    inner = dict(eta_rho=slice(1, 5), xi_rho=slice(1, 5))
    np.testing.assert_allclose(
        vol["u"].isel(inner).values, raw["expected_u"].values, atol=1e-6
    )
    np.testing.assert_allclose(
        vol["v"].isel(inner).values, raw["expected_v"].values, atol=1e-6
    )


def test_layers_are_surface_first_and_depths_follow_the_necofs_formula(vol):
    raw = xr.open_dataset(FIXTURE, decode_times=False, mask_and_scale=False)
    # Surface first: layer 0 is the top s_rho level of the file.
    np.testing.assert_array_equal(
        vol["temp"].values[:, 0], raw["temp"].values[:, -1].astype(np.float32)
    )
    np.testing.assert_array_equal(
        vol["salinity"].values[:, -1], raw["salt"].values[:, 0].astype(np.float32)
    )
    sig = vol["siglay"].values
    assert np.all(np.diff(sig, axis=0) < 0) and np.all((sig < 0) & (sig > -1))
    np.testing.assert_allclose(vol["siglev"].values[0], 0.0, atol=1e-7)
    np.testing.assert_allclose(vol["siglev"].values[-1], -1.0, atol=1e-7)
    # ROMS Vtransform 2 depth, computed independently.
    hc = float(raw["hc"])
    h = raw["h"].values
    zeta = raw["zeta"].values[0]
    s, cs = raw["s_rho"].values[-1], raw["Cs_r"].values[-1]
    z_roms = zeta + (zeta + h) * (hc * s + h * cs) / (hc + h)
    z = (
        vol["siglay"].values[0] * (vol["h"].values + vol["zeta"].values[0])
        + vol["zeta"].values[0]
    )
    np.testing.assert_allclose(z, z_roms, rtol=1e-5, atol=1e-4)


def test_ww_is_the_mean_of_the_interfaces(vol):
    raw = xr.open_dataset(FIXTURE, decode_times=False, mask_and_scale=False)
    w = raw["w"].values[:, ::-1]  # surface first
    np.testing.assert_allclose(
        vol["ww"].values, 0.5 * (w[:, :-1] + w[:, 1:]), rtol=1e-6, atol=1e-9
    )


def test_selection_reads_the_named_layers_and_fields():
    sel = doppio.read_volume(
        fixture_bbox(),
        START,
        1,
        dap_url=FIXTURE,
        variables=["temp", "u"],
        layers=[0, -1],
    )
    assert set(sel.data_vars) >= {"u", "temp"}
    assert "v" not in sel and "salinity" not in sel
    assert sel["layer"].values.tolist() == [0, 39]
    full = doppio.read_volume(fixture_bbox(), START, 1, dap_url=FIXTURE)
    np.testing.assert_array_equal(
        sel["temp"].values, full["temp"].isel(siglay=[0, 39]).values
    )
    np.testing.assert_allclose(
        sel["u"].values, full["u"].isel(siglay=[0, 39]).values, atol=1e-7
    )
    with pytest.raises(ValueError, match="variables must be among"):
        doppio.read_volume(fixture_bbox(), START, 1, dap_url=FIXTURE, variables=["T"])
    with pytest.raises(ValueError, match="layers must lie"):
        doppio.read_volume(fixture_bbox(), START, 1, dap_url=FIXTURE, layers=[40])


def test_box_selects_an_index_rectangle():
    ds = xr.open_dataset(FIXTURE, decode_times=False)
    lon, lat = ds["lon_rho"].values, ds["lat_rho"].values
    # A box around cell (2, 3) alone.
    bbox = [lon[2, 3] - 1e-4, lat[2, 3] - 1e-4, lon[2, 3] + 1e-4, lat[2, 3] + 1e-4]
    one = doppio.read_volume(bbox, START, 1, dap_url=FIXTURE)
    assert one["eta_rho"].values.tolist() == [2]
    assert one["xi_rho"].values.tolist() == [3]
    assert bool(one["in_bbox"].values.all())
    with pytest.raises(ValueError, match="No Doppio cells"):
        doppio.read_volume([0, 0, 1, 1], START, 1, dap_url=FIXTURE)


def test_missing_hours_are_left_out_by_read_and_refused_by_volume(
    tmp_path, monkeypatch
):
    part = doppio.read_volume(fixture_bbox(), START, 5, dap_url=FIXTURE)
    assert part.sizes["time"] == 2
    monkeypatch.setenv("FORCINGKIT_CACHE_DIR", str(tmp_path))
    with pytest.raises(RuntimeError, match="lacks 3 of 5 hours"):
        doppio.volume(
            fixture_bbox(), START, START + pd.Timedelta(hours=4), dap_url=FIXTURE
        )


def test_volume_caches(tmp_path, monkeypatch):
    monkeypatch.setenv("FORCINGKIT_CACHE_DIR", str(tmp_path))
    calls = []

    def reader(*args, **kwargs):
        calls.append(kwargs)
        return doppio.read_volume(*args, **kwargs)

    end = START + pd.Timedelta(hours=1)
    first = doppio.volume(fixture_bbox(), START, end, reader=reader, dap_url=FIXTURE)
    again = doppio.volume(fixture_bbox(), START, end, reader=reader, dap_url=FIXTURE)
    assert len(calls) == 1
    xr.testing.assert_allclose(first.reset_coords(), again.reset_coords())
    assert list((tmp_path / "doppio" / "volume").iterdir())


def test_product_url_prefers_the_reanalysis():
    assert doppio.product_url("2022-08-11", "2022-08-14") == doppio.REANALYSIS_URL
    assert doppio.product_url("2025-06-01", "2025-06-02") == doppio.REALTIME_URL
    assert doppio.product_url("2024-12-30", "2025-01-02") == doppio.REALTIME_URL
    with pytest.raises(ValueError):
        doppio.product_url("2006-06-01", "2006-06-02")
    assert doppio.supports_bbox([-69.89, 39.66, -69.26, 40.14])
    assert not doppio.supports_bbox([-125.0, 35.0, -120.0, 40.0])


def test_layered_mesh_adapter(vol):
    from forcingkit.layered_mesh import from_doppio_volume

    mesh = from_doppio_volume(vol)
    assert mesh.n_elem == 2 * int(vol["mask"].values.sum())
    assert mesh.n_layer == 40
    assert mesh.attrs["model"] == "doppio"
    assert set(mesh.fields) == {"u", "v", "w", "T", "S"}


def test_faces_to_centers():
    # Faces 0..2 between four centers on a full grid row: edges take their one face.
    faces = np.array([[1.0, 3.0, np.nan]])
    out = doppio._faces_to_centers(faces, -1, 0, 4)
    np.testing.assert_allclose(out, [[1.0, 2.0, 1.5, 0.0]])
    # A hyperslab starting one face before the first center.
    out = doppio._faces_to_centers(np.array([[2.0, 4.0, 6.0]]), -1, -1, 2)
    np.testing.assert_allclose(out, [[3.0, 5.0]])


def test_stretching_vtransform_2():
    s = np.array([-0.75, -0.25])
    cs = np.array([-0.5, -0.1])
    h = np.array([[50.0, 1000.0]])
    sig = doppio.stretching(s, cs, h, 250.0, 2)
    expected = (250.0 * s[:, None, None] + h[None] * cs[:, None, None]) / (250.0 + h)
    np.testing.assert_allclose(sig, expected)
    with pytest.raises(ValueError):
        doppio.stretching(s, cs, h, 250.0, 1)
