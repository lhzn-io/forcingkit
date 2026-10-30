# forcingkit-client

A lightweight Python client for a [forcingkit](https://github.com/lhzn-io/forcingkit) service. It
returns ocean, atmosphere and station forcing as xarray objects, and interpolates forcing along a
trajectory (a tagged animal, a vessel, a drifter). It depends on `httpx`, `numpy`, `pandas`,
`scipy`, `xarray` and `zarr` only, none of the service's data-access stack, so it installs into a
simulator or an application's embedded Python (3.11 or newer).

```python
from forcingkit_client import Client

fk = Client("http://localhost:9598")
box = (-70.9, 41.3, -70.6, 41.6)  # west, south, east, north
ocean = fk.ocean(box, "2025-07-01", 24)  # u, v, temp, salt, zeta
air = fk.atmosphere(box, "2025-07-01", 24)  # HRRR u10, v10, t2m, ...

# lon, lat, time, z: one value per point; z in metres, positive up
track = fk.interpolate_trajectory(
    lon, lat, time, z, variables=("u", "v"), atmosphere=("u10", "v10")
)
```

Store builds run as jobs on the service; the client polls them and opens the finished store over
HTTP through a local cache (`~/.cache/forcingkit-client`), so only the chunks read are fetched,
once. `AsyncClient` has the same methods for callers that must not block an event loop.

Units are SI; z is metres, positive up. Documentation: <https://forcingkit.docs.lhzn.io/client.html>.
