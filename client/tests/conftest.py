import functools
import http.server
import threading

import numpy as np
import pandas as pd
import pytest
import xarray as xr

T0 = pd.Timestamp("2026-06-01T00:00:00")


def linear_field(lon, lat, z, hours):
    """Linear in every axis, so linear interpolation must reproduce it exactly."""
    return 2.0 * lon + 3.0 * lat + 0.5 * z + 0.1 * hours


def z_v3_store(nt: int = 7) -> xr.Dataset:
    """A small parent store in the z-v3 layout."""
    lon = np.linspace(-71.0, -70.5, 6)
    lat = np.linspace(41.2, 41.7, 6)
    z = np.array([-9.0, -7.0, -5.0, -3.0, -1.0])
    time = pd.date_range(T0, periods=nt, freq="1h")
    hours = np.arange(nt, dtype=np.float64)
    H, Z, Y, X = np.meshgrid(hours, z, lat, lon, indexing="ij")
    u = linear_field(X, Y, Z, H).astype(np.float32)
    ds = xr.Dataset(
        {
            "u": (
                ("time", "z", "lat", "lon"),
                u,
                {"units": "m s-1", "standard_name": "eastward_sea_water_velocity"},
            ),
            "v": (("time", "z", "lat", "lon"), -u, {"units": "m s-1"}),
            "zeta": (("time", "lat", "lon"), u[:, -1], {"units": "m"}),
        },
        coords={"time": time, "z": z, "lat": lat, "lon": lon},
        attrs={"schema": "z-v3", "complete": True},
    )
    return ds


def hycom_legacy_store() -> xr.Dataset:
    """The same field in HYCOM's legacy layout: depth positive down, longitudes 0..360."""
    ds = z_v3_store()
    ds = ds.assign_coords(lon=ds["lon"] % 360, z=-ds["z"]).rename(z="depth")
    ds = ds.sortby("depth")
    ds.attrs = {"schema": "legacy", "source": "forcingkit.fetchers.hycom"}
    return ds


class _Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        self.server.requests.append(self.path)  # type: ignore[attr-defined]
        super().do_GET()


@pytest.fixture
def store_server(tmp_path):
    """A static HTTP server over `tmp_path/srv`, recording every request path."""
    root = tmp_path / "srv"
    root.mkdir()
    server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(_Handler, directory=str(root))
    )
    server.requests = []  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield root, f"http://127.0.0.1:{server.server_address[1]}", server.requests  # type: ignore[attr-defined]
    server.shutdown()
