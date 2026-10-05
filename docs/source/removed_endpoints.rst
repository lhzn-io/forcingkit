Removed Endpoints
=================

The routes below were removed on 2026-10-05. The model core (coastal-sim) and its viewer hub had
moved to the parent-ocean and atmosphere stores, and no caller remained in the lhzn-io
repositories.

Until the next release, each removed route still answers, with **410 Gone** and a JSON body
naming its replacement. The response carries ``Deprecation`` (RFC 9745) and ``Sunset``
(RFC 8594) headers, both at the removal date, and a ``Link`` to this page. OpenAPI lists the
routes as deprecated. The next release drops the stubs (``service/forcingkit_serve/routers/removed.py``),
and the routes then answer 404.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Removed
     - Use instead
   * - ``POST /api/v1/ic/generate``, ``POST /api/v1/ic/regrid``
     - ``POST /api/v1/obc``: the parent ocean on true z (schema ``z-v3``). Its first record is
       the initial condition, already on regular lon/lat and fixed z levels, so no separate
       regrid step remains.
   * - ``POST /api/v1/ic/cache``, ``POST /api/v1/ic/predict-donor``,
       ``GET /api/v1/ic/download/{zarr_id}``
     - ``POST /api/v1/obc/cache``, ``POST /api/v1/obc/predict-donor``,
       ``GET /api/v1/obc/download/{zarr_id}``
   * - ``POST /api/v1/bc/generate``, ``POST /api/v1/bc/cache``,
       ``GET /api/v1/bc/download/{zarr_id}``
     - ``POST /api/v1/atmosphere`` and ``GET /api/v1/atmosphere/download/{zarr_id}``: HRRR
       hourly surface fields (10 m wind, 2 m temperature and humidity, pressure, radiation,
       precipitation) on a regular 0.03 degree grid. See :doc:`atmospheric_forcing`.
   * - ``POST /api/v1/bc/predict-donor``
     - None. The atmosphere is HRRR from 2014-07-30; earlier runs use ERA5 through
       NumericalEarth in the model, which needs no fetch from this service.
   * - ``POST /api/v1/harmonics``
     - ``POST /api/v1/tide`` for observed water level at a CO-OPS station, or the ``zeta``
       variable of the ``/api/v1/obc`` parent store for modelled sea surface height.
   * - ``GET /api/v2/coupled-boundaries/{region}``
     - ``POST /api/v1/obc``

The code behind them (the initial-condition fetchers, the 10 m wind fetcher, the ERA5 and
regridder modules, and the pyTMD harmonics) was removed with them, together with the ``cdsapi``
and ``pytmd`` dependencies.
