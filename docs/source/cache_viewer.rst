Cache viewer
============

The service serves a page at ``/ui/`` that lists every store in ``FORCINGKIT_CACHE_DIR``: the
Zarr stores it built and the raw GRIB and NetCDF files it fetched. The page polls
``/api/v1/cache/inventory`` every two seconds, so new stores appear while a request is still
running.

Coverage
--------

The **Coverage** tab answers which stores cover a place and a time:

- **Map.** Each store's footprint is drawn as an outline, colored by its source family (NECOFS,
  Doppio, HRRR, HYCOM, HF radar and so on). Hovering an outline names the store, its kind and its
  time span; clicking it selects the store and opens its preview.
- **Coverage strip.** One row per source family and one column per UTC day. The shade is the
  number of hours of that day covered by the stores in the current map view, so panning or
  zooming the map changes the strip. Clicking a cell filters the list and the map to that day.

The filters above the list apply to both: the family chips (click to hide or show a family,
double-click to show only it), the kind, a date range, and **In map view**, which lists only
the stores that overlap the map.

Inventory fields
----------------

Besides the display strings, each inventory entry carries structured fields for other tools:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Field
     - Meaning
   * - ``bbox``
     - ``[west, south, east, north]`` in degrees (longitudes in -180 to 180), from the store's
       longitude and latitude coordinates or variables (mesh nodes and river mouths included),
       else from its ``bbox`` or ``requested_bbox`` attribute. ``null`` when neither exists.
   * - ``time_start``, ``time_end``
     - ISO 8601 UTC times of the first and last records, or the target date of a snapshot.
   * - ``family``
     - The source model or provider, read from the store's path and its ``type``, ``source``
       and donor attributes.
   * - ``kind``
     - ``ocean boundary``, ``ocean volume``, ``initial condition``, ``atmosphere``,
       ``atmosphere levels``, ``waves``, ``rivers``, ``surface currents (radar)`` or
       ``raw donor file``.

Entries are cached per store and rebuilt only when the store's metadata files change.

Basemap
-------

By default the map uses the `OpenFreeMap <https://openfreemap.org/>`__ dark vector style
(OpenStreetMap data), which needs no key. To draw your own relief instead, put PMTiles archives of raster tiles (WebP or PNG, 256 px) in a folder and point the
service at it:

.. code-block:: bash

   export FORCINGKIT_BASEMAP_DIR=/path/to/basemaps
   export FORCINGKIT_BASEMAP_ATTRIBUTION="Relief: NOAA, USGS"

Every ``<name>.pmtiles`` in the folder is drawn, coarsest first (by deepest zoom), dimmed for the
dark page; a ``<name>.coast.geojson`` beside it is drawn as a shoreline. The page reads the
archives in byte ranges from ``/api/v1/cache/basemap/files/<file>``, and
``/api/v1/cache/basemap`` lists them.
