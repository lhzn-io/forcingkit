import numpy as np
import pandas as pd
import pytest
from conftest import T0, hycom_legacy_store, linear_field, z_v3_store

from forcingkit_client import interpolate_points, normalise, trajectory_window
from forcingkit_client.trajectory import _prepare, assemble

# Interior points, off every grid node, at sub-hourly times and between z levels.
LON = np.array([-70.93, -70.81, -70.66, -70.57])
LAT = np.array([41.27, 41.39, 41.52, 41.63])
TIME = T0 + pd.to_timedelta([0.25, 1.6, 3.1, 5.9], unit="h")
Z = np.array([-8.2, -6.5, -4.4, -1.7])
HOURS = (TIME - T0) / pd.Timedelta(hours=1)


@pytest.mark.parametrize("store", [z_v3_store, lambda: normalise(hycom_legacy_store())])
def test_linear_field_is_reproduced(store):
    lon, lat, t, z = _prepare(LON, LAT, TIME, Z)
    out = interpolate_points(store(), ["u", "zeta"], lon, lat, t, z)
    np.testing.assert_allclose(
        out["u"].values, linear_field(LON, LAT, Z, HOURS), rtol=0, atol=1e-4
    )
    # zeta is the top level of u: 2-D fields ignore z.
    np.testing.assert_allclose(
        out["zeta"].values, linear_field(LON, LAT, -1.0, HOURS), atol=1e-4
    )
    assert out["u"].attrs["units"] == "m s-1" and out["u"].attrs["n_missing"] == 0


def test_points_given_in_0_360_are_wrapped():
    lon, lat, t, z = _prepare(LON % 360, LAT, TIME, Z)
    out = interpolate_points(z_v3_store(), ["u"], lon, lat, t, z)
    np.testing.assert_allclose(
        out["u"].values, linear_field(LON, LAT, Z, HOURS), atol=1e-4
    )


def test_above_top_level_takes_the_shallowest_value():
    lon, lat, t, z = _prepare(LON, LAT, TIME, np.zeros(4))
    out = interpolate_points(z_v3_store(), ["u"], lon, lat, t, z)
    np.testing.assert_allclose(
        out["u"].values, linear_field(LON, LAT, -1.0, HOURS), atol=1e-4
    )


def test_land_seabed_and_outside_are_missing():
    ds = z_v3_store()
    ds["u"][:, :, :, 0] = np.nan  # a land column along the western edge
    lon = np.array([-70.99, -70.8, -70.8, -69.0])
    z = np.array(
        [-2.0, -9.5, -2.0, -2.0]
    )  # below the deepest level; inside; outside the grid
    p = _prepare(lon, LAT, TIME, z)
    out = interpolate_points(ds, ["u"], *p)
    assert np.isnan(out["u"].values[[0, 1, 3]]).all() and np.isfinite(
        out["u"].values[2]
    )
    assert out["u"].attrs["n_missing"] == 3


def test_cf_trajectory_layout():
    lon, lat, t, z = _prepare(
        LON, LAT, TIME.tz_localize("UTC").tz_convert("US/Eastern"), Z
    )
    np.testing.assert_array_equal(t.values, TIME.values)  # aware times arrive in UTC
    values = interpolate_points(z_v3_store(), ["u", "v"], lon, lat, t, z)
    ds = assemble(
        values, lon, lat, t, z, "tag-01", {"forcingkit_ocean_store_id": "obc_x"}
    )
    assert (
        ds.attrs["featureType"] == "trajectory" and ds.attrs["Conventions"] == "CF-1.11"
    )
    assert (
        ds["trajectory"].attrs["cf_role"] == "trajectory_id"
        and ds["trajectory"].item() == "tag-01"
    )
    assert ds["u"].dims == ("obs",) and set(ds.coords) == {"time", "lat", "lon", "z"}
    assert ds["z"].attrs["positive"] == "up"
    assert ds.attrs["forcingkit_ocean_store_id"] == "obc_x"


def test_window_covers_trajectory():
    bbox, start, hours = trajectory_window(LON, LAT, TIME, 0.1)
    assert bbox == pytest.approx((-71.03, 41.17, -70.47, 41.73))
    assert start == T0 - pd.Timedelta(hours=1) and hours == 8


@pytest.mark.parametrize(
    "z, message",
    [(np.array([1.0, 0, 0, 0]), "positive up"), (np.zeros(3), "one length")],
)
def test_bad_input(z, message):
    with pytest.raises(ValueError, match=message):
        _prepare(LON, LAT, TIME, z)
