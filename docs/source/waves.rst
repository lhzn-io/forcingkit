Surface Waves (NWPS)
====================

As of 2026-10-30. ``POST /api/v1/waves`` serves surface waves from NOAA's `Nearshore Wave
Prediction System <https://polar.ncep.noaa.gov/nwps/>`__ (NWPS) as a Zarr store (schema
``nwps-waves-v1``), for a viewer that shows surface conditions over a coastal window or for a
downstream ocean model that needs wave height and direction.

Provenance
----------

NWPS runs the SWAN spectral wave model for each coastal NWS Weather Forecast Office (WFO), on
that office's computational grids, at 00 and 12 UTC. Each grid's output is published on `NOMADS
<https://nomads.ncep.noaa.gov/pub/data/nccf/com/nwps/prod/>`__ as one GRIB2 file per cycle,
holding the analysis and 144 hourly forecast steps on a regular longitude-latitude grid, with land
as missing values::

   <region>.YYYYMMDD/<wfo>/HH/<GRID>/<wfo>_nwps_<GRID>_YYYYMMDD_HH00.grib2

``<region>`` is the NWS region (``er`` for Eastern, ``sr``, ``wr``, ``ar``, ``pr``). CG1 is the
office's whole domain; for OKX (New York) it covers 39.75 to 41.40 N and 74.45 to 71.05 W at
0.0211 by 0.0163 degrees (about 1.8 km), 162 by 102 cells, and a file is about 26 MB. NOAA open
data: free to use, with attribution.

How a store is built
--------------------

A request names one office and one grid (``wfo``, default ``okx``; ``grid``, default ``CG1``),
a box, a start on the hour and a number of hours. Records run hourly from one hour before the
start to one hour after the end (``hours + 3`` records), as for the atmosphere.

*Cycle per hour.* Each hour is taken from the newest cycle at or before it whose file holds it:
leads 0 to 11 of successive cycles for a window in the past, and the newest cycle's longer leads
(up to 144 h) for hours beyond it. A cycle without an analysis (it is sometimes missing) gives its
first hour to the previous cycle's lead 12. Each record carries its lead as ``forecast_hour``, and
the store's ``record_cycles`` attribute lists the cycle of every record.

*Complete files only.* NOMADS answers a request for a file that is still being written, and a
truncated download decodes too, both with fewer steps. A file counts only when it decodes with
every field at all 144 forecast steps; otherwise the cycle is treated as not yet posted. A cycle
younger than 7 hours is not asked for. Each file is fetched once, whole (there is no ``.idx``),
and kept under ``<cache>/nwps/<wfo>/<GRID>/``.

*Sources.* An archive named by ``FORCINGKIT_NWPS_ARCHIVE`` is read first, then NOMADS. The
archive is a directory, a ``file://`` prefix or a ``gs://`` prefix (install ``forcingkit[gcs]``);
it is only ever read. A cycle's file is looked for at ``YYYY/MM/<name>``, then
``YYYYMMDD/HH/<name>``, then ``<name>`` directly under the prefix.

*Reach.* NOMADS keeps about five days. An hour that neither the archive nor NOMADS holds fails
the request with ``404`` and a message naming the hour, rather than a shorter store.

*Grid.* The store stays on the office's grid, cropped to the box plus one cell on each side;
nothing is regridded. Longitudes are converted from 0..360 to -180..180.

*Cache key.* The store id hashes the source, schema, office, grid, box, start, hours and the
newest cycle the window may use. A window in the past therefore always maps to the same store,
and a forecast window asked for again after a newer cycle has posted becomes a new store instead
of a stale one.

Store layout
------------

.. list-table::
   :header-rows: 1
   :widths: 16 16 18 50

   * - Variable
     - Dimensions
     - Units
     - Meaning (GRIB2 field)
   * - ``hs``
     - time, lat, lon
     - m
     - Significant height of combined wind waves and swell (``SWH``);
       ``sea_surface_wave_significant_height``
   * - ``tp``
     - time, lat, lon
     - s
     - Peak period (``PERPW``, which NWPS's GRIB2 labels "primary wave mean period");
       ``sea_surface_wave_period_at_variance_spectral_density_maximum``
   * - ``dir``
     - time, lat, lon
     - degree
     - Primary wave direction (``DIRPW``), converted from the direction waves come from to the
       direction they travel to, clockwise from true north; ``sea_surface_wave_to_direction``
   * - ``shts``
     - time, lat, lon
     - m
     - Significant height of total swell (``SHTS``); ``sea_surface_swell_wave_significant_height``
   * - ``ws``, ``wdir``
     - time, lat, lon
     - m/s, degree
     - The office's forcing wind (``WS``, ``WDIR``), which explains the waves; ``wdir`` keeps the
       meteorological "from" convention (``wind_from_direction``). Defined over land too.
   * - ``forecast_hour``
     - time
     - hours
     - Lead of the record from its cycle (``forecast_period``); cycle = time - forecast_hour
   * - ``wet``
     - lat, lon
     - 1
     - 1 where any step of any cycle used has a wave height. Cells near the shore wet and dry
       with the water level, so one record's sea can be smaller.

Wave fields are float32, NaN on land. ``time`` is CF ``seconds since`` the first record (UTC).
Store attributes: ``schema``, ``wfo``, ``grid``, ``crs`` (``EPSG:4326``), ``dx`` and ``dy``
(degrees), ``requested_bbox``, ``start_time``, ``first_cycle``, ``last_cycle``,
``record_cycles``, ``source_urls``, ``files_from_archive``, ``files_from_nomads``,
``direction_convention``, ``complete`` and ``records``.

Not served: the surface current and water level in the same files (OKX's current is zero; NWPS is
not coupled to an ocean model there), spectral output, and any grid the request does not name.

Example
-------

.. code-block:: bash

   curl -s -X POST localhost:9598/api/v1/jobs/waves -H 'Content-Type: application/json' -d '{
     "bbox": {"min_lon": -72.9, "min_lat": 40.95, "max_lon": -72.0, "max_lat": 41.3},
     "start_time": "2026-10-29T00:00:00", "hours": 48, "wfo": "okx", "grid": "CG1"}'

From Python, ``forcingkit_client.Client.waves(box, start, hours, wfo="okx", grid="CG1")``
(:doc:`client`).
