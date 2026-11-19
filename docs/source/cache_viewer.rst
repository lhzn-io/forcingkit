Cache viewer
============

The service serves a page at ``/ui/`` that lists every store in ``FORCINGKIT_CACHE_DIR``: the
Zarr stores it built and the raw GRIB and NetCDF files it fetched. The page polls
``/api/v1/cache/inventory`` every five seconds while it is visible, so new stores appear while a
request is still running. The response carries an ETag, and a poll that sends it back in
``If-None-Match`` gets 304 with no body when nothing changed.

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

Preview
-------

The **Preview** tab draws the selected store on its own cells: an unstructured mesh (NECOFS
FVCOM) by its triangles, a regular or curvilinear grid (HRRR, NWPS, Doppio, the parent-ocean
stores) by its cells, and a store of sites (river mouths) as one circle per site. It reads every
store the inventory lists, including those in subfolders.

- **Map.** One color per cell over the basemap, one record at a time. Choose the field (a speed
  is derived from each east and north pair), the layer and the record; **Play** steps through the
  records. The readout under the pointer gives the cell's value, the layer's depth, the bed and
  the cell's index in the source.
- **3D.** The surface cells, see-through and with faint cell edges, over a lit bed tinted from
  shallow to deep: a mesh's own faceted bed where the store holds the depth at its nodes, else
  each cell flat at its own bed with walls between neighbors. Tracers show the current or wind
  below: in a thinned set of columns and up to six levels down each, one segment per cell moves
  along that cell's velocity and wraps inside the cell, colored by depth in the column. Up to
  four level maps draw whole layers at their heights. Navigation is
  `vantagekit <https://github.com/lhzn-io/vantagekit>`__: drag to orbit, keys while the view has
  focus, double-click to turn about a point.
- **Section.** Click points on the map and double-click (or press Enter) to finish: the panel
  shows the field along the line in every layer, with the bed, and the 3D view hangs it as a
  curtain.

The link in the address bar (``#<store>&f=<field>&k=<layer>&t=<record>&v=3d``) reopens the same
view.

The preview draws what the store holds and nothing more:

- **No interpolation.** A cell is drawn flat in its own color; a section sample takes its host
  cell's value. Records are held, not blended, during playback.
- **Fixed scales.** A field's scale is set once per store from the 1st to the 99th percentile of
  its values weighted by cell area (symmetric about zero, at the 99.5th percentile of the
  magnitude, for signed fields), so a color means the same value in every record and layer.
  Water speed uses the packaged ``currents_tempo_dark`` ramp at its own speeds (0.02 to 3.1 m/s on
  a log axis), so different stores compare directly. Temperature is drawn on cmocean "thermal" and
  salinity on "haline", the maps oceanographers know for them.
- **True heights.** Layers run from the surface down and are labeled with their mean depth:
  fixed levels from the store's vertical coordinate, sigma layers from ``siglay`` and the water
  depth ``h``, atmosphere levels from their heights. Sigma layers in a store without ``h`` are
  labeled by sigma and stacked evenly in 3D, and the legend says so.
- **Large grids.** A grid with more than 400,000 cells (120,000 in 3D) is shown every *n*-th
  cell in each direction; each drawn cell carries one source cell's value, never an average, and
  the legend says when.

Drawing API
-----------

The page draws through ``/api/v1/cache/view/{id}/...``; other tools can use it too. ``detail``
(``2d`` or ``3d``) picks the cell budget, and element numbering is fixed for a store and detail.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Route
     - Returns
   * - ``manifest``
     - JSON: fields (label, units, scale ``lo`` and ``hi``, ``log``, ticks, and a 256-entry RGBA
       ramp indexed by byte code), layers surface first with mean heights, record times and
       cadence, cell counts, stride and bounding box.
   * - ``mesh``
     - Binary: node longitudes and latitudes (Float64) then triangles (Uint32, three per element).
   * - ``frame?field=&layer=&t=``
     - Binary: one byte per element on the field's scale (0: no value; 1 to 255 from ``lo`` to
       ``hi``), as ``forcingkit.layered_mesh.FieldMeta.encode`` writes it.
   * - ``heights?layer=&t=``, ``bed``
     - Binary Float32 per element: the layer center's height and the bed's (m, positive up).
       ``bed?at=node`` gives the bed per mesh node where the store holds the depth at its nodes.
   * - ``point?lon=&lat=&field=&layer=&t=``
     - JSON: the cell under the point, its value, height, bed and source index.
   * - ``section?pts=lon,lat;...&field=&t=&n=``
     - JSON: values and heights in every layer at ``n`` points along the polyline.
   * - ``tracers?pair=&t=&levels=&limit=``
     - JSON: up to ``limit`` columns thinned evenly (their elements, the same at every level) and
       up to ``levels`` layers from the surface down, each with east and north components and
       heights per column.

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

By default the maps use `OpenFreeMap <https://openfreemap.org/>`__ vector tiles
(OpenStreetMap data), which need no key, cut down to land, water, the shoreline and faint
rivers: no roads, buildings, place names or boundaries, so the data drawn over it carries the
map. The credit sits behind the (i) button in the map's corner. To draw your own relief instead, put PMTiles archives of raster tiles (WebP or PNG, 256 px) in a folder and point the
service at it:

.. code-block:: bash

   export FORCINGKIT_BASEMAP_DIR=/path/to/basemaps
   export FORCINGKIT_BASEMAP_ATTRIBUTION="Relief: NOAA, USGS"

Every ``<name>.pmtiles`` in the folder is drawn, coarsest first (by deepest zoom), dimmed for the
dark page; a ``<name>.coast.geojson`` beside it is drawn as a shoreline. The page reads the
archives in byte ranges from ``/api/v1/cache/basemap/files/<file>``, and
``/api/v1/cache/basemap`` lists them.
