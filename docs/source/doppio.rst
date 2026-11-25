Doppio (ROMS, Mid-Atlantic Bight and Gulf of Maine)
===================================================

Doppio is a `ROMS <https://www.myroms.org/>`__ configuration for the Mid-Atlantic Bight and the
Gulf of Maine, run by the `Rutgers University Ocean Modeling Group
<https://rucool.marine.rutgers.edu/>`__ for `MARACOOS <https://maracoos.org/>`__ (López et al.,
2020). Rutgers runs it in two forms, both with data assimilation and both served hourly from the
`Rutgers THREDDS server <https://tds.marine.rutgers.edu/thredds/catalog/roms/doppio/catalog.html>`__:

- the **reanalysis** DopAnV3R3-ini2007 (4D-Var; Wilkin et al., 2022), 2007-01-02 01:00 to
  2024-12-31 00:00 UTC, every hour with no gaps; and
- the **real-time system** (2017_da), 2017-11-02 01:00 UTC to about two days ahead, as the "Best"
  time series of a forecast model run collection.

``forcingkit.fetchers.doppio`` reads either as a full-depth subset in the same layout as the
NECOFS volume (:doc:`necofs`, schema ``volume-v1``), so a comparison between the two models, or
between either and a downstream ocean model, can treat them alike. It is also a parent-ocean
donor in the NECOFS ``z-v3`` layout, delivered only when a request names it (see `Parent
ocean`_).

Underlying model
----------------

Checked against the aggregations' OPeNDAP metadata on 2026-11-17.

- **Grid**: curvilinear Arakawa C-grid, 106 x 242 rho points (``eta_rho``, ``xi_rho``), about
  7 km, from 80.5 W to 59.7 W and 32.2 N to 46.6 N. The grid's xi axis is turned about 37 degrees
  from east (``angle``), so ``u`` and ``v`` in the files are along the grid, not east and north.
- **Vertical coordinate**: 40 terrain-following s-levels (``s_rho``, bottom first), Vtransform 2,
  Vstretching 4, theta_s 7, theta_b 2, hc 250 m.
- **Fields**: ``u`` (on u faces), ``v`` (on v faces), ``w`` (on the 41 ``s_w`` interfaces),
  ``temp`` (potential temperature, Celsius), ``salt``, ``zeta``, plus ``ubar``, ``vbar``, mixing
  coefficients and surface fluxes. The real-time aggregation also carries ``u_eastward`` and
  ``v_northward`` at rho points, but they are empty (fill values) for 2022 and populated for
  recent dates.
- **Datum**: the free surface is relative to the model mean sea level; the Rutgers metadata
  states no geodetic datum, so do not compare it with NAVD88 water levels without an offset.
- **Time**: ``ocean_time`` in seconds since 2006-01-01 (reanalysis); ``time`` in hours since
  2017-11-01 (real time).

The full-depth subset
---------------------

``volume(bbox, start, end)`` returns every s-level for the index rectangle around the cells
whose centers lie in the box, hourly from ``start`` to ``end`` inclusive, with no horizontal
regridding and no vertical conversion. ``read_volume(bbox, start_time, hours, dap_url=None)``
reads one aggregation.

.. code-block:: python

   from forcingkit.fetchers.doppio import volume

   bbox = [-69.89, 39.66, -69.26, 40.14]  # Veatch Canyon mouth and the shelf break above it
   vol = volume(bbox, "2022-08-11T00Z", "2022-08-14T00Z")  # 73 hourly records

.. list-table::
   :header-rows: 1
   :widths: 22 30 48

   * - Name
     - Dimensions
     - Meaning
   * - ``u``, ``v``
     - (time, siglay, eta_rho, xi_rho)
     - Eastward and northward velocity at the cell centers, m s-1: the two faces either side of
       each center averaged (a land face counts as zero flow, as in ROMS's own ``u2rho``), then
       rotated by ``angle``. On 2026-11-10 this matched the real-time aggregation's own
       ``u_eastward`` and ``v_northward`` to 6e-8 m s-1.
   * - ``ww``
     - (time, siglay, eta_rho, xi_rho)
     - ROMS ``w`` (upward velocity in z), averaged from the interfaces above and below each
       layer center; the name follows the NECOFS volume.
   * - ``temp``, ``salinity``
     - (time, siglay, eta_rho, xi_rho)
     - Potential temperature (degree_Celsius) and salinity (1e-3).
   * - ``zeta``; ``h``
     - (time, eta_rho, xi_rho); (eta_rho, xi_rho)
     - Surface elevation relative to the model mean sea level (m, positive up); depth (m, positive down).
   * - ``siglay``, ``siglev``
     - (siglay, eta_rho, xi_rho), (siglev, eta_rho, xi_rho)
     - The fraction of the water column at layer centers and interfaces, surface first:
       ``(hc * s + h * Cs) / (hc + h)``, which for Vtransform 2 makes the depth of a layer center
       :math:`\sigma (h + \zeta) + \zeta`, the NECOFS formula.
   * - ``lon``, ``lat``, ``mask``, ``angle``
     - (eta_rho, xi_rho)
     - Cell centers in degrees (``lon`` and ``lat`` are 2D auxiliary coordinates, so CF tools
       and the cache viewer place the cells), water mask, and the grid angle in radians.
   * - ``in_bbox``
     - (eta_rho, xi_rho)
     - The cell center lies in the box. The rectangle's corners may not, since the grid is
       rotated: for the box above, 59 of the 100 cells.
   * - ``eta_rho``, ``xi_rho`` (coordinates)
     - (eta_rho), (xi_rho)
     - Each row's and column's index in the full grid.

Field names and units match the NECOFS volume (``u``, ``v``, ``ww``, ``temp``, ``salinity``,
``zeta``, ``h``, ``siglay``, ``siglev``). The horizontal layout differs: NECOFS returns elements
and nodes of an unstructured mesh, Doppio rows and columns of a curvilinear grid, with every field
at the cell centers. Land cells are NaN in the time-varying fields.

``variables`` and ``layers`` select as for NECOFS: ``variables`` among ``u``, ``v``, ``ww``,
``temp``, ``salinity``, ``zeta``; ``layers`` as zero-based indices, surface first, negative ones
from the bottom, returned with a ``layer`` coordinate.

``volume`` uses the reanalysis when it holds the whole window, else the real-time system, else
raises; ``dap_url`` names an aggregation explicitly. It raises if any requested hour is missing.
Results are cached as Zarr under ``$FORCINGKIT_CACHE_DIR/doppio/volume/``, keyed by the schema,
the aggregation, the box, the window and the selection. The 73 hours above (10 x 10 cells, 40
levels, 6 MB) took 45 s from Rutgers on 2026-11-17, and 0.1 s from the cache.

``forcingkit.layered_mesh.doppio(bbox, start, end)`` returns the same subset as a
``LayeredMesh`` (see :doc:`layered_mesh`) for rendering and sections next to NECOFS.

Parent ocean
------------

``doppio.iter_parent(start_date, duration_hours, bbox, pad_cells, vertical_spacing_m)`` streams
Doppio in the parent-ocean layout NECOFS writes (schema ``z-v3``, :doc:`necofs`): ``u``, ``v``,
``w``, ``temp``, ``salt`` on ``(time, z, lat, lon)``, ``zeta`` on ``(time, lat, lon)``, and
``h``, ``mask``, ``z_face``, with the same units and standard names, so a store from either
model reads the same way. The service delivers it only for a request that names it:

.. code-block:: json

   {"bbox": {"min_lon": -69.72, "min_lat": 39.80, "max_lon": -69.30, "max_lat": 40.05},
    "start_date": "2022-08-13T02:00:00", "duration_hours": 12, "donor": "doppio"}

to ``POST /api/v1/jobs/obc`` (or ``fk.ocean(box, start, hours, donor="doppio")`` from the
client). The automatic ranking never picks Doppio, so requests that name no donor keep their
donor and their store ids. Steps:

1. **Read.** ``read_volume`` over the box plus the padding and 0.15 degree more, so every
   target lies among read cell centers, 24 hours per request, from the reanalysis if it holds
   the whole window, else the real-time system (a window across neither raises). ``u`` and
   ``v`` arrive at the cell centers, rotated to east and north by the grid ``angle``; ``w`` is
   ROMS's ``w`` averaged from the interfaces to the layer centers. A missing hour is an error.
2. **Horizontal.** Linear interpolation from the curvilinear cell centers (a Delaunay
   triangulation) onto a regular grid every 0.02 degree (about 2 km, finer than the 7 km
   cells) covering the box plus ``pad_cells`` grid steps on every side. A target in a triangle
   with a land cell center is NaN, so the coast is drawn up to one cell conservatively.
3. **Vertical.** Each s-level sits at ``siglay * (h + zeta) + zeta`` for its column and is
   interpolated onto fixed levels every ``vertical_spacing_m`` (``necofs.sigma_to_z``), bottom to
   top; levels below the sea floor are NaN.

The store records ``w_source = "model"``, ``doppio_product``, ``doppio_url``, ``citation``
and ``horizontal_spacing_deg``; ``forcingkit:donor`` in its STAC Item is ``doppio``.

RED001's box ([-69.72, 39.80, -69.30, 40.05], the slope south of Georges Bank) for 12 h from
2022-08-13 02:00, with 20 m levels and one cell of padding, read 2026-11-24, next to the
NECOFS store for the same request:

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Over the box
     - Doppio
     - NECOFS
   * - Build time
     - 10 s (reanalysis)
     - 132 s (GOM7 monthly hindcast)
   * - Grid points, z levels
     - 13 x 22, 95
     - 125 x 211, 91
   * - Bed depth (min, median, max)
     - 109, 514, 1784 m
     - 100, 417, 1804 m
   * - Ocean at the surface
     - 100 %
     - 100 %
   * - Surface speed (median, 90th percentile)
     - 0.32, 0.55 m/s
     - 0.21, 0.39 m/s
   * - Speed at 510 m (median)
     - 0.28 m/s
     - 0.11 m/s
   * - ``|w|`` over wet cells (median, 99th percentile)
     - 1.1e-3, 7.6e-3 m/s
     - 1.0e-3, 1.0e-2 m/s

The two models agree on the bathymetry and on the size of ``w`` (both about 1e-3 m/s below
500 m, where the slope's internal tide moves water up and down), but not on the currents. The
vector difference between them at 10 m has a median of 0.25 m/s against a NECOFS median speed
of 0.21 m/s, and Doppio's box-mean surface current turns through about three quarters of a
circle in 9 hours (a semidiurnal tidal ellipse) where NECOFS's barely turns. A track corrected
with one or the other will differ by more than either model's own uncertainty suggests.

Limits
------

- **Resolution.** Cells are about 7 km across (6 to 7 km over the box above), so features under
  about 15 to 20 km, such as the flow inside a single shelf-break canyon, are not resolved. Use
  Doppio for the slope current, stratification and tides at shelf scale, and NECOFS or a finer
  model inside canyons.
- **Sea level datum.** ``zeta`` is relative to the model's own mean sea level; the Rutgers
  metadata names no geodetic datum.
- **Cache viewer.** The cache viewer (:doc:`cache_viewer`) lists the stores under
  ``doppio/volume/``, draws their footprints on its coverage map and previews them on their
  own curvilinear cells, every layer at its height from ``siglay`` and ``h``.

Credit
------

The Doppio catalog describes the data as "Freely available". Cite the model (López et al.,
2020), the reanalysis (Wilkin et al., 2022) and, for the reanalysis output, the dataset as its
catalog asks: Wilkin, J. and J. Levin (2022), SEANOE,
`doi:10.17882/89673 <https://doi.org/10.17882/89673>`__. The observations it assimilated are on
`Rutgers ERDDAP
<https://tds.marine.rutgers.edu/erddap/tabledap/DOPPIO_REANALYSIS_Ver3_OBS.graph>`__. See
:doc:`shelf_models` for the other models of the region.
