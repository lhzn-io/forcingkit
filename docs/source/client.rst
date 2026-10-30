Python client and trajectories
==============================

``forcingkit-client`` is a separate, lightweight distribution published from this repository
(``client/``). It reaches a running forcingkit service over HTTP and returns xarray objects. It
depends on ``httpx``, ``numpy``, ``pandas``, ``scipy``, ``xarray`` and ``zarr`` only, none of the
service's data-access stack (GRIB, OPeNDAP, S3), so it installs into a simulator or an
application's embedded Python (3.11 or newer).

.. code-block:: python

   from forcingkit_client import Client

   fk = Client("http://localhost:9598")
   box = (-70.9, 41.3, -70.6, 41.6)  # west, south, east, north

   ocean = fk.ocean(box, "2025-07-01", 24)        # u, v, temp, salt, zeta on (time, z, lat, lon)
   air = fk.atmosphere(box, "2025-07-01", 24)     # HRRR u10, v10, t2m, q2m, sp, dswrf, dlwrf, prate
   waves = fk.waves(box, "2025-07-01", 24, wfo="box", grid="CG1")  # NWPS hs, tp, dir, shts, ws, wdir
   buoy = fk.ndbc("44013", "2025-07-01", "2025-07-02", products=("stdmet", "adcp"))
   elevation = fk.bathymetry(box, resolution_m=30)

``AsyncClient`` has the same methods as coroutines, for callers that must not block an event
loop. Units are SI throughout; z is metres, positive up.

How a request is served
-----------------------

Building a store can take minutes on a cold cache, so the client does not hold a request open.
It submits a job and polls it:

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Route
     - Purpose
   * - ``POST /api/v1/jobs/obc``
     - Queue the build that ``POST /api/v1/obc`` runs (same request body). Returns ``202`` with
       ``job_id``, ``status`` and ``status_url``.
   * - ``POST /api/v1/jobs/atmosphere``
     - The same for ``POST /api/v1/atmosphere``.
   * - ``POST /api/v1/jobs/waves``
     - The same for ``POST /api/v1/waves``.
   * - ``GET /api/v1/jobs/{job_id}``
     - ``status`` is ``queued``, ``running``, ``succeeded`` or ``failed`` (with ``error``). Once
       succeeded, ``result`` is a STAC Item describing the store.
   * - ``GET /api/v1/stores/{zarr_id}.zarr/{key}``
     - The finished store's files, read-only, for reading the store remotely.

The job id is the store id, which is a hash of the request. Identical submissions therefore share
one build, a request for a store that already exists succeeds at once, and a job whose store
finished before a service restart is still reported as succeeded. The service runs
``FORCINGKIT_JOB_WORKERS`` builds at a time (default 2). The synchronous routes are unchanged.

The client opens the store named in the Item over HTTP through a local cache
(``~/.cache/forcingkit-client``), keyed by store id and store version. Only the chunks actually
read are transferred, each once; a store the service rebuilds under the same id (after
``cache_bust``) has a new version and is fetched afresh.

The result as a STAC Item
-------------------------

Each succeeded job's ``result`` is a `STAC <https://stacspec.org>`__ Item, so a store describes
itself in the vocabulary that catalogue tools (``pystac``, ``pystac-client``, ``odc-stac``) read:

.. code-block:: json

   {
     "type": "Feature", "stac_version": "1.1.0", "id": "obc_1f3a9c0e5b72",
     "bbox": [-71.0, 41.2, -70.5, 41.7],
     "properties": {
       "datetime": null,
       "start_datetime": "2025-07-01T00:00:00Z", "end_datetime": "2025-07-02T00:00:00Z",
       "forcingkit:kind": "ocean", "forcingkit:schema": "z-v3",
       "forcingkit:donor": "necofs", "forcingkit:store_version": "1761830400000000000"
     },
     "assets": {
       "zarr": {
         "href": "http://localhost:9598/api/v1/stores/obc_1f3a9c0e5b72.zarr",
         "type": "application/vnd+zarr", "roles": ["data"],
         "xarray:open_kwargs": {"consolidated": true, "zarr_format": 2}
       }
     }
   }

Any Zarr reader can open the asset directly, for example
``xr.open_zarr(href, consolidated=True, zarr_format=2)``.

Store layouts
-------------

The client returns one layout whatever the donor. ``z-v3`` stores already have it. Stores from
donors not yet converted to ``z-v3`` (schema ``legacy``: HYCOM, NYOFS, DBOFS) are normalised on
opening: ``depth`` (positive down) becomes ``z`` (positive up), ``latitude`` and ``longitude``
become ``lat`` and ``lon``, and HYCOM's 0 to 360 longitudes are wrapped to -180 to 180. A legacy
store on a curvilinear grid, and any schema the client does not know, is refused with a message
naming the schema.

Forcing along a trajectory
--------------------------

``interpolate_trajectory`` returns forcing at each point of a moving platform's path, such as a
tagged animal, a vessel or a drifter:

.. code-block:: python

   track = fk.interpolate_trajectory(
       lon, lat, time, z,                  # degrees; UTC; metres, positive up (-25 is 25 m down)
       variables=("u", "v", "temp"),       # from the parent ocean
       atmosphere=("u10", "v10"),          # from HRRR, optional
       trajectory_id="tag-01",
   )

It fetches the trajectory's bounding box (plus ``margin_deg``, default 0.1 degree) and time window
(whole hours, plus one each side) once, then interpolates each variable linearly in time,
latitude, longitude and, for 3-D fields, z. Points above the shallowest level take its value,
matching the service's own surface convention. Points on land, below the sea floor or outside the
grid are NaN, and each variable's ``n_missing`` attribute counts them.

The result is a CF discrete sampling geometry (CF 1.11, ``featureType = "trajectory"``): one value
per point and variable on an ``obs`` dimension, with ``time``, ``lat``, ``lon`` and ``z``
coordinates, a ``trajectory`` variable carrying ``cf_role = "trajectory_id"``, and global
attributes naming each store used (id, version, donor, schema). ``to_netcdf`` writes a file that
CF-aware tools read as a trajectory.

**Sampling density is the caller's choice.** The gridded sources resolve hours and kilometres
(HYCOM is 3-hourly at 0.04 to 0.08 degrees), so a trajectory logged at many hertz loses nothing
when thinned to about one point a minute first: about 550 points for a 9 hour deployment.

**Extent.** One bounding box suits trajectories of hours to days in one region. A multi-week track
spanning hundreds of kilometres would pull a store far larger than its samples; segmenting such
tracks server-side is on the :doc:`roadmap`.
