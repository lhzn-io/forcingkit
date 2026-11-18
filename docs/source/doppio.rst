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
between either and a downstream ocean model, can treat them alike. It is a library function: no
service route or parent-ocean (``z-v3``) delivery uses it yet.

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
