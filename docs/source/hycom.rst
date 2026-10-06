HYCOM: global ocean analyses and reanalysis
===========================================

As of 2026-10-06. Dataset paths, time ranges, grids and variable layouts on this page were read
from the HYCOM consortium THREDDS server (``tds.hycom.org``) on that date; the fetcher was tested
live against it on the same day.

HYCOM is forcingkit's global parent ocean and the dispatcher's last-resort donor: it accepts any
box and is tried after NYOFS, DBOFS and NECOFS (:doc:`fetchers`). It covers 1994 to the present
by switching between several experiments, and delivers the legacy output layout rather than the
``z-v3`` parent store.

System overview
---------------

The `HYCOM consortium <https://www.hycom.org/>`__ (US Navy, NOAA and academic partners)
publishes global runs of the HYbrid Coordinate Ocean Model (`Bleck, 2002
<https://doi.org/10.1016/S1463-5003(01)00012-9>`__) with data assimilation (`Chassignet et al.,
2007 <https://doi.org/10.1016/j.jmarsys.2005.09.016>`__), interpolated from the model's hybrid
layers onto 40 fixed depths from 0 to 5000 m. Each run is published as a numbered *experiment*
on one of several grids. No single experiment spans the whole record: the 1994 to 2015
reanalysis, a sequence of analysis experiments from 2014 to 2024, and, since August 2024, the
Navy ESPC-D-V02 analysis each cover part of it. forcingkit chains them by date.

Experiments forcingkit reads
----------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 26 26 26

   * - Dataset (path under ``thredds/dodsC/``)
     - On the server (first to last step)
     - forcingkit uses it from
     - Longitudes; time steps
   * - ``GLBv0.08/expt_53.X``
     - 1994-01-01 (catalogue) to 2015-12-31 09:00
     - 1994-01-01
     - -180..180; 63,341
   * - ``GLBv0.08/expt_56.3``
     - 2014-07-01 12:00 to 2016-09-30 09:00
     - 2015-12-31
     - -180..180; 6,451
   * - ``GLBv0.08/expt_57.2``
     - 2016-05-01 12:00 to 2017-02-01 09:00
     - 2016-05-01 12:00
     - -180..180; 2,208
   * - ``GLBv0.08/expt_92.8``
     - 2017-02-01 12:00 to 2017-06-01 09:00
     - 2017-02-01 12:00
     - 0..360; 952
   * - ``GLBv0.08/expt_57.7``
     - 2017-06-01 12:00 to 2017-10-01 09:00
     - 2017-06-01 12:00
     - -180..180; 976
   * - ``GLBv0.08/expt_92.9``
     - 2017-10-01 12:00 to 2018-03-20 09:00
     - 2017-10-01 12:00
     - 0..360; 1,270
   * - ``GLBv0.08/expt_93.0``
     - 2018-01-01 12:00 to 2020-02-19 09:00
     - 2018-01-01 12:00
     - 0..360; 6,128
   * - ``GLBy0.08/expt_93.0``
     - 2018-12-04 12:00 to 2024-09-05 09:00
     - 2018-12-04 12:00
     - 0..360; 16,809
   * - ``ESPC-D-V02/{u3z,v3z,t3z,s3z,ssh}``
     - 2024-08-10 12:00 to the present (2026-10-06 09:00 when read)
     - 2024-09-05
     - 0..360; 6,280 (3-hourly), ssh hourly

Each experiment serves from the time in the third column until the next one starts. Where two
overlap, the switch is at the newer one's first time step, with one exception: GLBy0.08
``expt_93.0`` is kept to its last step on 2024-09-05 although ESPC-D-V02 begins on 2024-08-10.
Times before 1994-01-01 are not served.

Grids
~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Grid
     - Latitude
     - Longitude and depth
   * - GLBv0.08
     - 3251 rows, 80S to 90N: 0.08 degree between 40S and 40N, 0.04 degree poleward
     - 4500 columns at 0.08 degree; 40 depths, 0 to 5000 m
   * - GLBy0.08, ESPC-D-V02
     - 4251 rows, 80S to 90N at 0.04 degree
     - 4500 columns at 0.08 degree; 40 depths, 0 to 5000 m

The two grids coincide poleward of 40 degrees, so for the US Northeast a GLBv0.08 to GLBy0.08
switch changes nothing spatially.

Variables
~~~~~~~~~

forcingkit reads five fields and renames them: ``water_u`` to ``u``, ``water_v`` to ``v``,
``water_temp`` to ``temp``, ``salinity`` to ``salt`` and ``surf_el`` to ``zeta``. The GLB
experiments carry all five in one dataset, 3-hourly, plus ``tau`` (analysis time) and four
``*_bottom`` fields, which forcingkit does not download. ESPC-D-V02 publishes one dataset per
field: ``u3z``, ``v3z``, ``t3z`` and ``s3z`` are 3-hourly; ``ssh`` is hourly. All use the time
unit "hours since 2000-01-01 00:00:00".

Where the data lives
--------------------

All datasets are read over OPeNDAP from ``https://tds.hycom.org/thredds/dodsC/`` (`catalogue
<https://tds.hycom.org/thredds/catalog.html>`__). ESPC-D-V02 is also published as one dataset
per field and year (``ESPC-D-V02/u3z/2025`` and so on) and as a forecast collection; forcingkit
uses the multi-year aggregations, which need no change at the turn of a year. The catalogue also
lists combined ``uv3z`` and ``ts3z`` collections and an experimental all-variables collection;
on 2026-10-06 the combined ones contained no datasets.

How forcingkit reads it
-----------------------

``forcingkit.fetchers.hycom.fetch_hycom_boundary_conditions(start_date, duration_hours, bbox)``:

1. **Experiment by date.** The window is split at experiment switches (``_split_window``) and
   each piece is read from its own experiment.
2. **Subset.** Times are selected by position, not label, so a time axis that steps backwards
   still works. Latitude is sliced to the box plus 0.1 degree. Longitude is sliced in the
   dataset's own convention (-180..180 or 0..360, detected per dataset); a box that crosses the
   dataset's seam is read in two parts. Only the five fields above are downloaded.
3. **ESPC-D-V02 merge.** The five per-field datasets are read separately and merged on the
   times they share, so hourly ``zeta`` is reduced to the 3-hourly steps.
4. **Clean-up.** Longitudes are returned on 0..360 for every experiment. Times are sorted,
   duplicates dropped, and decoded to datetimes.
5. **Stitch.** Pieces from different experiments are concatenated in time. A piece on a
   different grid is interpolated (bilinear) onto the grid of the first piece. The step at a
   switch is read from both experiments; the newer one's copy is kept.
6. **All or nothing.** If any piece or field fails, the fetch returns nothing, so the
   dispatcher reports the failure rather than writing a shortened store.

Output is a Dataset with ``u``, ``v``, ``temp``, ``salt`` (``time, depth, lat, lon``) and
``zeta`` (``time, lat, lon``). Land and cells below the sea floor are NaN. The dispatcher then
resamples to hourly by linear interpolation, casts to Float32, and writes the store with
``schema = "legacy"``.

Known issues
~~~~~~~~~~~~

- **Slow, easily saturated server.** On 2026-10-06 a small read from ``tds.hycom.org`` usually
  took 0.2 to 2 s, but some requests stalled past the 120 s read timeout, for metadata and data
  alike, and the same request would succeed minutes later. Stalls were far more frequent while
  other requests from the same machine were running in parallel; fetches run one at a time
  succeeded. One ESPC-D-V02 window needs at least five data requests (one per field), so it is
  the most exposed. A failed fetch can usually be retried; avoid running several HYCOM fetches
  at once.
- **Gaps inside an experiment.** Most steps are 3 hours apart, but every experiment has a few
  longer gaps, up to 51 hours (for example in GLBv0.08 ``expt_93.0``, ``expt_92.9`` and
  ESPC-D-V02). They are passed through unfilled, and the dispatcher's hourly interpolation then
  bridges them linearly.
- **GLBv0.08 ``expt_93.0`` steps back.** Its time axis goes from 2018-06-21 09:00 to 06:00,
  repeating 06:00. The fetcher selects by position and drops the duplicate.
- **Dead path for ``expt_53.X``.** ``GLBy0.08/expt_53.X`` answers HTTP 200 with an empty
  dataset description; the reanalysis lives under ``GLBv0.08``. forcingkit read the GLBy0.08
  path before 2026-10-06, so HYCOM served nothing before 2018-12-04 until then.
- **GLBy0.08 ``expt_93.0`` starts at 12:00.** Its first step is 2018-12-04 12:00, not the
  midnight that the catalogue date suggests; the switch from GLBv0.08 ``expt_93.0`` is at
  12:00.

Tested live
~~~~~~~~~~~

On 2026-10-06, with the box [-72.6, 40.6, -72.4, 40.8] (south shore of Long Island), each of
these returned all five fields on 10 x 5 cells and 40 depths:

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Window
     - Experiment
     - Result
   * - 2010-06-01 00:00, 6 h
     - GLBv0.08 ``expt_53.X``
     - 3 steps; -180..180 longitudes returned on 0..360.
   * - 2018-12-03 18:00, 12 h
     - GLBv0.08 ``expt_93.0``
     - 5 steps, in 12 s.
   * - 2026-10-01 00:00, 6 h
     - ESPC-D-V02 (five datasets)
     - 3 steps, in 216 s; ``zeta`` reduced from hourly to 3-hourly.

A window across the 2024-09-05 switch was tried four times and one across 2018-12-04 12:00
once. Every piece (GLBv0.08 and GLBy0.08 ``expt_93.0``, and each ESPC-D-V02 field) was read
successfully in at least one attempt, but no attempt completed all of its pieces: each stopped
on one read timeout (see `Known issues`_). The stitching itself is covered by the unit tests
below.

Tests
-----

- ``tests/unit/test_hycom.py``: experiment resolution and window splitting at every switch,
  the URLs for GLBv0.08 and the ESPC-D-V02 fields, stitching across 2018-12-04 (regridding and
  the duplicated switch step) and across 2024-09-05 (merged ESPC-D-V02 fields), the all or
  nothing rule, both longitude conventions and the seam, and the non-monotonic time axis. No
  network.

.. code-block:: bash

   uv run pytest tests/unit/test_hycom.py -v

References
----------

Citations
~~~~~~~~~

- Bleck, R. (2002). An oceanic general circulation model framed in hybrid isopycnic-Cartesian
  coordinates. *Ocean Modelling*, 4(1), 55-88.
  `doi:10.1016/S1463-5003(01)00012-9 <https://doi.org/10.1016/S1463-5003(01)00012-9>`__
- Chassignet, E. P., H. E. Hurlburt, O. M. Smedstad, et al. (2007). The HYCOM (HYbrid
  Coordinate Ocean Model) data assimilative system. *Journal of Marine Systems*, 65(1-4),
  60-83. `doi:10.1016/j.jmarsys.2005.09.016 <https://doi.org/10.1016/j.jmarsys.2005.09.016>`__

Institutions and data services
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- `HYCOM consortium <https://www.hycom.org/>`__: `THREDDS catalogue
  <https://tds.hycom.org/thredds/catalog.html>`__
