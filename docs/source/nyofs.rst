NYOFS: NOAA New York/New Jersey Operational Forecast System
===========================================================

As of 2026-10-06. Grid and file facts on this page were read from the NYOFS output itself
(CO-OPS FMRC, NCEI and AWS copies of the 2026-09-30 and 2026-10-01 cycles); the status of each
access path was tested live against ``forcingkit.fetchers.nyofs`` on the same day.

.. warning::
   **NYOFS delivers, but not as a z-v3 parent store.** Every access path (CO-OPS FMRC,
   AWS S3 and NCEI) returns data for 2014 to the present. The output is the legacy
   ``(time, depth, eta, xi)`` layout rather than the ``z-v3`` parent store. See `Known issues`_
   below; use :doc:`necofs` for the parent ocean in this region until NYOFS writes ``z-v3``.

System overview
---------------

`NYOFS <https://tidesandcurrents.noaa.gov/ofs/nyofs/nyofs.html>`__ is the operational
nowcast/forecast system of the NOAA `Center for Operational Oceanographic Products and Services
<https://tidesandcurrents.noaa.gov/>`__ (CO-OPS) for the Port of New York and New Jersey
(implementation plan: `NOAA Technical Report NOS CO-OPS 37
<https://tidesandcurrents.noaa.gov/publications/techrpt37.pdf>`__, 2002). Its grid covers the
Upper and Lower Bays, the lower Hudson, the East River to the western end of Long Island Sound,
Newark Bay, the Kills and Raritan Bay. It forecasts water level and currents; its output carries
no temperature or salinity.

Underlying model
~~~~~~~~~~~~~~~~

- **Hydrodynamic model**: the Princeton Ocean Model (POM), 3D, free surface, as described by
  `Blumberg and Mellor (1987) <https://doi.org/10.1029/CO004p0001>`__.
- **Grids**: two curvilinear grids, a coarse grid for the harbour and its approaches and a fine
  grid over the Kill van Kull and its approaches.
- **Vertical coordinate**: sigma, 7 uniformly spaced levels from 0 to 1 (positive down: 0 at the
  surface, 1 at the bed).
- **Cycles**: four a day, at 05, 11, 17 and 23 UTC.
- **Output**: per cycle, one *nowcast* file holding the 6 hours up to the cycle time (the 23 UTC
  nowcast holds 18:00 to 23:00, hourly) and one *forecast* file running forward from it. Files
  are netCDF3 (classic format), COARDS conventions, time in days since 2008-01-01.

Grids
~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 18 32 32

   * - Grid
     - Size (ny x nx)
     - Extent of wet cells (lon/lat)
     - Cell size over water
   * - Coarse (``nyofs``)
     - 134 x 73; 2840 wet
     - [-74.343, 40.417, -73.776, 40.907]
     - 140 to 1040 m, median 340 m
   * - Fine (``nyofs_fg``)
     - 40 x 126; 2443 wet
     - [-74.198, 40.631, -74.024, 40.682]
     - 70 to 151 m, median 117 m

forcingkit reads the coarse grid only. The deepest wet cell is 24.8 m on the coarse grid and
16.9 m on the fine grid.

Variables
~~~~~~~~~

Both grids use the same fields-file layout:

.. list-table::
   :header-rows: 1
   :widths: 16 24 60

   * - Name
     - Dimensions
     - Meaning
   * - ``lon``, ``lat``
     - (ny, nx)
     - Cell positions, degrees.
   * - ``mask``
     - (ny, nx)
     - 1 for water, 0 for land.
   * - ``depth``
     - (ny, nx)
     - Bathymetry, metres, positive down.
   * - ``sigma``
     - (sigma)
     - Level positions, 0 to 1, positive down.
   * - ``zeta``
     - (time, ny, nx)
     - Water surface elevation, metres.
   * - ``u``, ``v``
     - (time, sigma, ny, nx)
     - Eastward and northward water velocity, m/s.
   * - ``w``
     - (time, sigma, ny, nx)
     - Upward water velocity, m/s.
   * - ``air_u``, ``air_v``
     - (time, ny, nx)
     - Eastward and northward wind forcing, m/s.

``u`` and ``v`` are earth-referenced (``eastward_sea_water_velocity`` and
``northward_sea_water_velocity``) and share the ``(ny, nx)`` points of ``lon`` and ``lat``: they
are not on staggered C-grid faces, so no face-to-centre interpolation is needed. Missing values
are ``-99999.0``.

Related systems
~~~~~~~~~~~~~~~

The `New York Harbor Observing and Prediction System
<https://hudson.dl.stevens-tech.edu/maritimeforecast/>`__ (NYHOPS), run by the `Davidson
Laboratory <https://www.stevens.edu/davidson-laboratory>`__ at `Stevens Institute of Technology
<https://www.stevens.edu/>`__, is an independent system for the same waters: the Hudson and East
Rivers, the NY/NJ estuary, Raritan Bay, Long Island Sound and the New Jersey coast, with
forecasts to 72 hours. Unlike NYOFS it includes temperature, salinity and waves. NYHOPS is not
integrated into forcingkit. The Davidson Laboratory also runs the `Stevens Flood Advisory System
<https://hudson.dl.stevens-tech.edu/sfas/>`__.

Where the data lives
--------------------

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Location
     - Span
     - Files
   * - CO-OPS FMRC aggregation
     - Rolling 7 days (428 hourly steps on 2026-10-06)
     - ``https://opendap.co-ops.nos.noaa.gov/thredds/dodsC/NYOFS/fmrc/Aggregated_7_day_NYOFS_Fields_Forecast_best.ncd``:
       the coarse grid as one OPeNDAP dataset, with ``time``, ``time_run`` and ``time_offset``.
   * - NCEI THREDDS
     - 2014 to the present month
     - ``https://www.ncei.noaa.gov/thredds/dodsC/model-nyofs-files/{yyyy}/{mm}/<file>``, all files
       of a month in one directory. Coarse-grid field nowcasts only (plus station files), four a
       day. Up to the 2024-09-08 cycles the name is
       ``nos.nyofs.fields.nowcast.{yyyymmdd}.t{cc}z.nc``; from 2024-09-09 it is
       ``nyofs.t{cc}z.{yyyymmdd}.fields.nowcast.nc``. A few days are missing (2024-06 has 25
       of 30).
   * - AWS S3 ``noaa-nos-ofs-pds``
     - Complete from 2024-11-19; parts of 2024 before that
     - From 2024-11-19 (that day holds only the 23 UTC cycle),
       ``nyofs/netcdf/{yyyy}/{mm}/{dd}/nyofs.t{cc}z.{yyyymmdd}.fields.{nowcast|forecast}.nc``,
       ``nyofs_fg...`` for the fine grid, and station files. Before that,
       ``nyofs/netcdf/{yyyymm}/`` with the older names, for 2024-04 and 2024-07 to 2024-12 only.
       A coarse nowcast file is about 5.8 MB, a coarse forecast file about 51 MB.

A continuous hindcast is the chain of nowcast files: cycle ``t{cc}z`` supplies hours ``cc - 5``
to ``cc``, so the four files of a day cover it without overlap.

How forcingkit reads it
-----------------------

``forcingkit.fetchers.nyofs.fetch_nyofs_boundary_conditions(start_date, duration_hours, bbox)``:

1. **Domain check.** The box must lie inside the declared domain
   [-74.475, 40.389, -73.743, 40.940], the extent of the coarse grid.
2. **Source.** Start times up to 6 days old use the FMRC aggregation. Older windows are read as
   the chain of nowcast files, one per cycle (``_nowcast_cycles``). For each cycle the fetcher
   tries AWS S3 first when the date is 2024-11-19 or later, then NCEI (``_nowcast_file_urls``).
   AWS files are netCDF3 and are read with the ``scipy`` engine; NCEI is read over OPeNDAP.
   If any cycle in the window is missing from both, the fetcher returns nothing, so the
   dispatcher falls back rather than writing a shortened store.
3. **Times.** Record times are rounded to the hour (NYOFS stores them up to 15 s off). Where
   the FMRC series repeats an hour, the first non-missing value is kept.
4. **Subset.** The bounding rectangle of the wet cells inside the box is read; cells outside
   the box or on land are NaN. ``u`` and ``v`` are passed through as stored, since they are
   already earth-referenced at the cell positions.
5. **Output.** A Dataset with ``u`` and ``v`` only, dimensions ``(time, depth, eta, xi)``.
   ``eta`` and ``xi`` are integer indices. ``depth`` is ``linspace(-50, 0, 7)``, a placeholder
   for the 7 sigma levels that does not follow the local water depth.

The dispatcher then resamples to hourly, casts to Float32, and writes the store in one piece with
``schema = "legacy"``.

Known issues
~~~~~~~~~~~~

Tested live on 2026-10-06 with the box [-74.10, 40.60, -74.00, 40.70] (Upper Bay and the
Narrows), 3 hours; each returned 4 hourly records x 7 levels x 62 x 29:

.. list-table::
   :header-rows: 1
   :widths: 26 30 44

   * - Start
     - Path
     - Files read
   * - 2026-10-04 (2 days old)
     - FMRC
     - The aggregation.
   * - 2026-09-15 04 UTC
     - AWS S3, per-day layout
     - ``nyofs.t05z`` and ``nyofs.t11z.20260915.fields.nowcast.nc``.
   * - 2024-11-19 20 UTC
     - AWS S3, first per-day date
     - ``nyofs.t23z.20241119.fields.nowcast.nc``.
   * - 2024-10-15 12 UTC
     - NCEI, current names
     - ``nyofs.t17z.20241015.fields.nowcast.nc``.
   * - 2023-12-15 06 UTC
     - NCEI, older names
     - ``nos.nyofs.fields.nowcast.20231215.t11z.nc``.

Fixed on 2026-10-06: the file names and cycle times (NYOFS publishes one nowcast file per cycle,
on 05/11/17/23 UTC, not one file per hour on 00/06/12/18 UTC), the AWS cutover (per-day layout
only from 2024-11-19; earlier dates go to NCEI), the netCDF3 reader for AWS files, the velocity
averaging (removed: ``u`` and ``v`` were being smoothed by one cell), the declared domain (it
reached 40.2 N and 73.3 W, past the grid, so boxes in that margin ranked NYOFS first and fell
back), and duplicate FMRC hours (one copy of a repeated hour can be all fill, which left
all-NaN records in the output).

Still open:

- **Output layout.** The legacy ``(time, depth, eta, xi)`` store has no geographic coordinates
  and no true depths, and lacks ``zeta``, ``temp``, ``salt``, ``h`` and ``mask``, all of which a
  ``z-v3`` parent store carries. NYOFS has no ``temp`` or ``salt`` to give, so a ``z-v3`` NYOFS
  store would need them from another donor. A model that reads ``z-v3`` parents cannot use a
  NYOFS store until this is done; see `Converting to z-v3`_.
- **Placeholder depths.** ``depth`` is ``linspace(-50, 0, 7)`` in sigma-index order, but NYOFS
  sigma index 0 is the surface, so the first level is labelled -50 m.
- **Fine grid.** The fine grid over the Kill van Kull (about 117 m) is not read.
- **Archive gaps.** A window that includes a cycle missing from both AWS and NCEI (2024-06 lacks
  five days) is not served by NYOFS; the dispatcher falls back to NECOFS.

The ``/api/v1/obc`` response's ``donor`` field names the donor that delivered (read from the
store's ``source`` attribute), and ``predicted_donor`` names the one ranked first.

Converting to z-v3
~~~~~~~~~~~~~~~~~~

Scope of the follow-up that would make NYOFS a ``z-v3`` parent, modelled on
``forcingkit.fetchers.necofs.iter_parent``:

1. **Streaming interface.** Add ``iter_parent(start_date, duration_hours, bbox, pad_cells,
   vertical_spacing_m)`` yielding the static fields once and then one record per hour, with
   ``PARENT_RECORD_DIMS`` and ``PARENT_UNITS``; the dispatcher already streams any donor that
   has it. The nowcast chain fits this well: one 6 MB file gives six records.
2. **Static fields.** ``lon``, ``lat`` (full precision), ``h`` from ``depth``, ``mask``, and
   ``z``/``z_face`` on the requested vertical spacing, over the box padded by ``pad_cells``.
   The coarse grid is curvilinear, so the regridding onto the parent's regular lon/lat grid
   that NECOFS does for its unstructured mesh is needed here too (bilinear on the curvilinear
   grid, or the same scattered-point interpolator).
3. **Vertical.** Convert sigma to depth per cell and hour, ``z = zeta - sigma * (h + zeta)``
   with sigma positive down, then interpolate ``u`` and ``v`` onto the fixed ``z`` levels.
   This also removes the placeholder depths.
4. **Tracers.** NYOFS has no ``temp`` or ``salt``. Options, to be decided: take them from
   NECOFS over the same box and times (a second donor inside one store, recorded in the
   attributes), or write a constant-salinity, constant-temperature parent and flag it. The first
   is more faithful; the second is simpler and adequate for barotropic boundary forcing.
5. **Tests.** Unit tests for the sigma-to-z conversion and the regridding on a synthetic
   curvilinear grid, and a live integration test that streams a few hours and checks the store
   against the ``z-v3`` contract (``store_is_complete``, attributes, units).

Rough size: the NECOFS parent path (``z_levels_for_depth``, ``sigma_to_z``, the scattered-point
interpolators, ``_parent_static`` and ``iter_parent``) is about 300 lines, with about 180 lines
of tests. NYOFS can reuse ``z_levels_for_depth``, ``sigma_to_z`` (which, like NYOFS, puts level 0 at
the surface and takes the layer depths from step 3), the interpolator and ``_parent_static``,
and the dispatcher's streaming writer needs no change, so the new code is mainly the reader loop
and the tracer source.

Python usage
------------

.. code-block:: python

   from forcingkit.fetchers.nyofs import fetch_nyofs_boundary_conditions

   # A box in the Upper Bay; any start time from 2014 on.
   ds = fetch_nyofs_boundary_conditions(
       "2023-12-15T06:00:00Z", 6, [-74.10, 40.60, -74.00, 40.70]
   )
   # ds.u, ds.v: (time, depth, eta, xi), or None if nothing could be read

Through the dispatcher, which ranks NYOFS first inside its domain and falls back on failure:

.. code-block:: python

   from forcingkit.dispatcher import dispatch_obc_request

   path = dispatch_obc_request(
       "2026-10-04T00:00:00Z", 6, [-74.10, 40.60, -74.00, 40.70]
   )

Tests
-----

- ``tests/unit/test_nyofs.py``: metadata and domain checks, access-mode resolution, the
  hour-to-cycle mapping, file names for each archive era, reader choice and fallback between
  AWS and NCEI, time rounding and FMRC duplicate hours, velocity pass-through, and dispatcher
  registration. No network.
- ``tests/integration/test_ofs_archive_paths.py``: live reads of one fixed date per archive
  path (AWS per-day, NCEI current names, NCEI older names) for NYOFS and DBOFS.
- ``tests/integration/test_nyofs_obc_fetch.py::test_nyofs_boundary_conditions``: a live fetch
  12 hours back, which exercises the FMRC path.

.. code-block:: bash

   uv run pytest tests/unit/test_nyofs.py -v
   uv run pytest tests/integration/test_ofs_archive_paths.py tests/integration/test_nyofs_obc_fetch.py -v

References
----------

Citations
~~~~~~~~~

- Blumberg, A. F. and G. L. Mellor (1987). A description of a three-dimensional coastal ocean
  circulation model. In N. S. Heaps (ed.), *Three-Dimensional Coastal Ocean Models*, Coastal and
  Estuarine Sciences 4, American Geophysical Union, 1-16.
  `doi:10.1029/CO004p0001 <https://doi.org/10.1029/CO004p0001>`__
- NOAA (2002). *Implementation Plan, Port of New York and New Jersey Operational Forecast System
  (NYOFS)*. NOAA Technical Report NOS CO-OPS 37, Silver Spring, MD.
  `PDF <https://tidesandcurrents.noaa.gov/publications/techrpt37.pdf>`__

Institutions and data services
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- `NOAA CO-OPS <https://tidesandcurrents.noaa.gov/>`__: `NYOFS
  <https://tidesandcurrents.noaa.gov/ofs/nyofs/nyofs.html>`__, `OFS FAQ
  <https://tidesandcurrents.noaa.gov/ofs/ofs_faq.html>`__, `THREDDS
  <https://opendap.co-ops.nos.noaa.gov/thredds/catalog/catalog.html>`__
- `NOAA NCEI <https://www.ncei.noaa.gov/>`__: `NYOFS files
  <https://www.ncei.noaa.gov/thredds/catalog/model-nyofs-files/catalog.html>`__
- `NOAA Open Data Dissemination <https://www.noaa.gov/information-technology/open-data-dissemination>`__:
  `OFS on AWS <https://github.com/NOAA-Big-Data-Program/nodd-data-docs/blob/main/OFS/README.md>`__
- `Stevens Institute of Technology, Davidson Laboratory <https://www.stevens.edu/davidson-laboratory>`__:
  `NYHOPS <https://hudson.dl.stevens-tech.edu/maritimeforecast/>`__
