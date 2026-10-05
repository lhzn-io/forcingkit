forcingkit Documentation
==============================

``forcingkit`` is a Python-based microservice designed to fetch, harmonize,
and serve multi-domain Earth-system observations, atmospheric telemetry, and parent-ocean boundary conditions for high-fidelity coupled physical simulations.

It acts as a caching proxy designed to alleviate the complexities of remote netCDF, OPeNDAP, and HRRR queries by producing robust local Zarr caches that can be seamlessly consumed by external computational systems.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   fetchers
   atmospheric_forcing
   nyofs
   removed_endpoints
