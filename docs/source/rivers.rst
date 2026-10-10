River Discharge
===============

``POST /api/v1/rivers`` delivers hourly freshwater discharge at the mouth of every listed river
in a box (or at mouths passed with the request), for point or surface sources in an ocean model. A nested regional model otherwise
receives river water only through its open boundaries and initial state, so a river that enters
inside the domain is missing as a source and its freshwater decays over the run.

Method
------

The method is common practice for gauge-driven river forcing. The NOAA Operational Forecast
Systems map US Geological Survey real-time gauges onto model cells with a per-river scale factor
and fall back to climatology. Regional estuary models scale gauged flow to ungauged area by the
ratio of drainage areas.

1. **Gauges, derived for every request.** Only a river's mouth is listed. From its NHDPlus
   outlet flowline, forcingkit asks the USGS `NLDI <https://waterdata.usgs.gov/blog/nldi-intro/>`__
   for every USGS site upstream, then keeps a site if:

   - USGS classes it as a stream site (``ST``). Tidal streams (``ST-TS``) and estuaries (``ES``)
     are left out, because their flow reverses with the tide; on the Connecticut River that rules
     out Hartford and Middle Haddam;
   - it has a drainage area, and a discharge record (15-minute or daily) covering the window;
   - it drains at least ``min_share`` of the mouth's area (default 0.01), so headwater brooks
     drop out;
   - no other kept site lies downstream of it, which leaves the lowest gauge on each branch.

   The kept gauges then measure disjoint parts of the basin. The selection follows the window: a
   gauge that opened in 2010 is used from 2010 on, and one that closed is dropped from then on.
2. **Discharge at the mouth.** The sum of the river's gauges, scaled by the drainage area at the
   mouth over the gauged area:

   .. math::

      Q_\text{mouth} = \frac{A_\text{mouth}}{\sum_k A_k} \sum_k Q_k

   Gauge areas come from the USGS monitoring-location records; the mouth's area is that of the
   outlet's NLDI basin.
3. **Time base.** The 15-minute record from the USGS Water Data APIs (``continuous``), averaged to
   hourly values centred on the hour. The store holds ``hours + 1`` records, from ``start_time``
   to ``start_time + hours``.
4. **Gaps.** A gap of up to ``max_gap_hours`` (default 6) with values on both sides is
   interpolated. A longer gap takes that day's mean from the ``daily`` record. A day with neither
   is an error, unless ``allow_climatology`` is set; then it takes the USGS day-of-year median of
   daily flow. Every filled hour is flagged in ``fill`` (0 measured, 1 interpolated, 2 daily
   mean, 3 climatology median), and the store's ``provenance`` attribute counts them per river.
5. **Water properties.** Salinity is zero. Water temperature (``temperature``) comes from the first
   gauge on the river that measures it (USGS parameter 00010), and is NaN where none does.
6. **Provisional data.** USGS values are provisional until approved, typically months later, and
   provisional values are revised. A store containing any provisional value is marked
   ``provisional = True`` and is served from the cache for one day only; after that a request
   rebuilds it.
7. **Travel time.** None by default. A mouth entry may set ``lag_hours``; gauge values are then
   taken that many hours earlier.

Mouths
------

Mouths are listed in ``forcingkit/data/rivers/*.json`` and in any ``*.json`` file in the directory
named by ``FORCINGKIT_RIVER_MOUTHS_DIR``; a later entry with the same name replaces an earlier one.
A request may instead pass its own ``mouths``. Adding a river needs a mouth entry and no code or
release:

.. code-block:: json

   {"mouths": [
     {"name": "Connecticut", "lon": -72.3305, "lat": 41.2726, "comid": 7703046},
     {"name": "Saugatuck", "lon": -73.36, "lat": 41.10, "trace_from": "USGS-01208990"}
   ]}

- ``name``, ``lon``, ``lat`` (required): the point places the mouth for the bbox test.
- ``comid``: the NHDPlus outlet flowline. Otherwise ``trace_from`` names any NLDI feature on the
  river (a USGS site such as ``USGS-01184000``) and the outlet is the end of its downstream trace.
  Failing both, the point is snapped to the nearest flowline, with a warning: at an estuary the
  nearest flowline can be a short coastal one rather than the river.
- ``include`` and ``exclude``: USGS site numbers to pin or rule out where judgement beats the rule
  (for example a gauge below a dam whose releases should not drive the forcing). A pinned set that
  drains more than the mouth's area is refused as double counting.
- ``lag_hours``: travel time from the gauges to the mouth, applied to the gauge window.

The id of a store includes the mouth entries and ``min_share``, but not the derived gauges, which
are recorded in the store instead. NLDI facts about a mouth (its outlet, basin area, upstream sites
and which sites lie downstream of which) are cached for 30 days under ``usgs/derivations`` in the
cache directory, so only the first request for a mouth takes long (about a minute for the
Connecticut, whose basin holds 1,375 USGS sites).

The packaged mouths are the rivers into Long Island Sound. For a request over the Sound in April
2026 the derivation chose:

.. list-table::
   :header-rows: 1
   :widths: 16 46 13 13 12

   * - River
     - USGS gauges kept
     - Gauged (mi²)
     - Mouth (mi²)
     - Scale
   * - Housatonic
     - 01205500 at Stevenson; 01208500 Naugatuck at Beacon Falls
     - 1,804
     - 1,944
     - 1.08
   * - Quinnipiac
     - 01196500 at Wallingford; 01196620 Mill River near Hamden; 01196561 Muddy River near East
       Wallingford
     - 148
     - 204
     - 1.37
   * - Connecticut
     - 01184000 at Thompsonville; 01189995 Farmington at Tariffville
     - 10,237
     - 11,242
     - 1.10
   * - Thames
     - 01127000 Quinebaug at Jewett City; 011230695 Shetucket at Taftville; 01127500 Yantic at
       Yantic
     - 1,314
     - 1,470
     - 1.12

The Quinnipiac's outlet basin is New Haven Harbor's, so its scaled flow also stands for the
harbour's other streams. The Housatonic at Stevenson is below a hydroelectric dam, so its hourly
flow follows the dam's releases. The Bronx River gauge (01302020) has no record before the 2000s,
so a request for an earlier window fails unless ``skip_ungauged`` is set, which leaves such a
river out and lists it in the store's ``skipped_rivers``.

Request and store
-----------------

.. code-block:: bash

   curl -s -X POST localhost:9598/api/v1/rivers -H 'Content-Type: application/json' -d '{
     "bbox": {"min_lon": -73.855, "min_lat": 40.75, "max_lon": -71.9, "max_lat": 41.45},
     "start_time": "2026-04-02T00:00:00Z", "hours": 48}'

Optional fields: ``mouths``, ``min_share`` (default 0.01), ``max_gap_hours`` (default 6),
``allow_climatology`` and ``skip_ungauged`` (both default false), and ``cache_bust``.

For that request (four rivers, 49 records), the mean discharges over the 48 hours were 1,553 m³/s
for the Connecticut, 147 for the Housatonic, 148 for the Thames and 13 for the Quinnipiac.

The store (schema ``river-v1``) has a ``river`` dimension:

- ``discharge`` (time, river): m³ s⁻¹, ``water_volume_transport_in_river_channel``.
- ``temperature`` (time, river): °C.
- ``fill`` (time, river): the flags above.
- ``mouth_lon``, ``mouth_lat``, ``scale_factor``, ``salinity`` (river).
- Attributes: ``river_names``; ``provenance`` (JSON: per river, the gauges kept, the areas, the
  scale factor, the outlet, and the dropped candidates counted by reason); ``derivation_dropped``
  (JSON: every dropped candidate with its reason); ``skipped_rivers``; ``provisional``; and
  ``built_utc``.

A JSON sidecar next to the store repeats the request, the mouths and the derivation. ``GET /api/v1/rivers/download/{zarr_id}``
returns the store as a zip.

Credentials
-----------

None are required. A free USGS Water Data API key (``USGS_API_KEY``, from
`api.waterdata.usgs.gov/signup <https://api.waterdata.usgs.gov/signup/>`__) raises the hourly rate
limit; it is sent in the ``X-Api-Key`` header. forcingkit uses the modernized APIs, not the
legacy ``waterservices.usgs.gov``, which USGS plans to retire in 2027.

Limits
------

- Mouths are not found automatically within a box: each river needs a mouth entry. Finding
  coastal outlets from NHDPlus would remove that step.
- The selection rule cannot tell a regulated gauge from a natural one; use ``exclude`` where dam
  operations should not reach the forcing.
- Gauges measure surface runoff. Groundwater discharge (most of Long Island's freshwater input) is
  not represented.
- A model that applies discharge as a surface freshwater flux on fixed levels dilutes salinity
  but adds no volume. Whether the volume is added depends on the model's vertical coordinate, not
  on this store.
