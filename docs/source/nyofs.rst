NYOFS - NOAA New York/New Jersey Operational Forecast System
==============================================================

System Overview
---------------

NYOFS is the NOAA-operated successor to the Stevens NYHOPS system, providing high-resolution 3D hydrodynamic forecasts and hindcasts for the New York/New Jersey Harbor region and surrounding coastal areas.

**Advantages over previous systems:**

- **Resolution**: 70–150m in inner nest (vs. legacy NYHOPS at ~500m)
- **Coverage**: Dedicated operational system for NY Harbor, Kill van Kull, Jamaica Bay, Newark Bay, Arthur Kill, Raritan Bay, and NY Bight approaches
- **Data Access**: Modern NOAA THREDDS infrastructure with FMRC aggregation for recent data and NCEI archive for historical
- **Operational Status**: Actively maintained by NOAA CO-OPS

Underlying Model
~~~~~~~~~~~~~~~~

- **Hydrodynamic Model**: Princeton Ocean Model (POM), 3D primitive equation free-surface model
- **Grid Type**: Structured curvilinear orthogonal (Arakawa C-grid)
- **Vertical Coordinate**: Terrain-following sigma coordinates (20 sigma layers)
- **Time Steps**: 6-hourly model cycles (runs at 00Z, 06Z, 12Z, 18Z UTC)
- **Output Frequency**: Hourly nowcast/forecast files per cycle

Spatial Domain & Resolution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

  Domain Bounding Box: [-74.3°, 40.2°, -73.3°, 41.1°] (lon/lat)

  Outer (coarse) grid: ~150–1000m resolution, broader NY Bight
  Inner (fine) grid:   ~70–150m resolution, harbor channels & bays ← *Primary for coastal-sim*

Data Access Strategy
--------------------

ecodata-cache uses a **tiered URL resolution** approach:

1. **Recent Data (< 31 days)**: CO-OPS THREDDS FMRC Aggregation
   - Single virtual OPeNDAP dataset consolidating rolling 7-day window
   - URL: ``https://opendap.co-ops.nos.noaa.gov/thredds/dodsC/NYOFS/fmrc/Aggregated_7_day_NYOFS_Fields_Forecast_best.ncd``
   - Single time slice → no concatenation required

2. **Historical Data (> 31 days)**: NCEI THREDDS File-Per-Hour
   - Individual hourly files organized by run cycle (00Z, 06Z, 12Z, 18Z)
   - URL pattern (post–Sept 9, 2024): ``https://www.ncei.noaa.gov/thredds/dodsC/model-nyofs-files/{yyyy}/{mm}/{dd}/nyofs.t{cc}z.{yyyymmdd}.fields.n{hhh:03d}.nc``
   - URL pattern (pre–Sept 9, 2024):  ``https://www.ncei.noaa.gov/thredds/dodsC/model-nyofs-files/{yyyy}/{mm}/nos.nyofs.fields.n{hhh:03d}.{yyyymmdd}.t{cc}z.nc``
   - Multi-file concatenation required for durations > 1 hour

Grid & Variable Reference
--------------------------

Coordinates
~~~~~~~~~~~

::

  lon(eta, xi)        2D array, range [-74.3, -73.3]
  lat(eta, xi)        2D array, range [40.2, 41.1]
  sigma(k)            1D array, k ∈ [0, -1] (surface to seabed)
  time                 ISO 8601 UTC seconds since model run start
  mask(eta, xi)        1 = ocean, 0 = land

Variables
~~~~~~~~~

======================== ============== ==================================
Field                    NYOFS Name     Notes
======================== ============== ==================================
Eastward Velocity        ``u``          At xi-edges (C-grid stagger)
Northward Velocity       ``v``          At eta-edges (C-grid stagger)
Temperature              ``temp``       Units: °C
Salinity                 ``salt``       Units: PSU
Surface Elevation        ``zeta``       Units: m
Bathymetric Depth        ``depth``      At rho-points (cell centers)
======================== ============== ==================================

**Important**: u and v are on **staggered C-grid** edges and must be interpolated to rho-points (cell centers) before use in coastal-sim. ecodata-cache handles this automatically.

Processing Pipeline
-------------------

Open Boundary Conditions (OBC)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Load & Concatenate** NYOFS hourly files spanning [start_date, start_date + duration_hours]
2. **Spatial Subset** via 2D curvilinear mask (bbox + land mask)
3. **C-Grid → Rho Interpolation**
4. **Sigma → Pseudo-Depth**
5. **Output** xarray Dataset with dims ``(time, depth, eta, xi)``

Dispatcher Integration
----------------------

NYOFS is **automatically ranked first** for any OBC request with a bounding box inside the NYOFS domain. The dispatcher's fallback chain is:

::

  OBC Fallback Chain:
    1. NYOFS       (100m, ~1°² domain)   ← Primary for NY Harbor
    2. NECOFS      (200m, 132°² domain)  ← Regional fallback
    3. HYCOM       (9km, global)         ← Global fallback

Ranking is determined by:
- **Resolution**: Closer to 0m resolution wins
- **Domain Size**: Smaller domain wins (more specialized)

For Throgs Neck Bridge (bbox: [-73.815, 40.785, -73.775, 40.815]):
- NYOFS (~100m in 1°² domain) **beats** NECOFS (200m in 132°² domain)
- NYOFS **beats** HYCOM (9km in 64800°² domain)

API Usage
---------

Open Boundary Conditions
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

  from ecodata_cache.fetchers.nyofs import fetch_nyofs_boundary_conditions

  bbox = [-73.815, 40.785, -73.775, 40.815]
  start = pd.Timestamp("2026-03-30T12:00:00Z")
  duration_hours = 24

  ds = fetch_nyofs_boundary_conditions(start.isoformat(), duration_hours, bbox)
  # ds.u, ds.v available with dims (time, depth, eta, xi)

Dispatcher Integration
~~~~~~~~~~~~~~~~~~~~~~

The dispatcher automatically selects NYOFS when appropriate:

.. code-block:: python

  from ecodata_cache.dispatcher import dispatch_obc_request

  obc_zarr = dispatch_obc_request("2026-03-30T12:00:00Z", 24,
                                   [-73.815, 40.785, -73.775, 40.815])
  # Automatically uses NYOFS, returns the Zarr cache path

Testing & Verification
----------------------

Unit Tests
~~~~~~~~~~

Location: ``tests/unit/test_nyofs.py``

- Metadata validation (domain bbox, resolution)
- URL resolution logic (FMRC vs NCEI, naming conventions)
- NCEI file enumeration across cycles
- C-grid interpolation with synthetic data
- OBC fetching (mocked OPeNDAP)
- Error handling (pydap failures → graceful fallback to None)
- Dispatcher integration and ranking

Run with:

.. code-block:: bash

  uv run pytest tests/unit/test_nyofs.py -v

Integration Tests
~~~~~~~~~~~~~~~~~

Location: ``tests/integration/test_ic_fetch.py::test_nyofs`` and ``test_nyofs_boundary_conditions``

- Live CO-OPS THREDDS connection
- Actual data download and processing
- Verifies output schema and data types

Run with:

.. code-block:: bash

  uv run pytest tests/integration/test_ic_fetch.py::test_nyofs -v

References
----------

- **NOAA NYOFS Page**: https://tidesandcurrents.noaa.gov/ofs/nyofs/nyofs.html
- **Technical Report**: `NOAA Technical Report NOS CO-OPS 37 (POM Model Description) <https://tidesandcurrents.noaa.gov/publications/techrpt37.pdf>`_
- **OFS FAQ**: https://tidesandcurrents.noaa.gov/ofs/ofs_faq.html
- **CO-OPS THREDDS Catalog**: https://opendap.co-ops.nos.noaa.gov/thredds/catalog.html
- **NCEI NYOFS Archive**: https://www.ncei.noaa.gov/thredds/catalog/model/nyofs_catalog.html
- **NOAA NODD OFS Reference**: https://github.com/NOAA-Big-Data-Program/nodd-data-docs/blob/main/OFS/README.md
