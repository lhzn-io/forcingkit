Data Fetchers
=============

The Data Fetchers module contains implementations for communicating with external data providers. We use tiered fallback logic and cache everything locally to minimize bandwidth usage and guarantee reproducible simulations.

Atmospheric Forcing Tiers
--------------------------

.. note::
   This section describes the legacy wind forcing of ``/api/v1/bc/generate``. Ocean models take
   the full HRRR surface atmosphere from ``/api/v1/atmosphere``; see :doc:`atmospheric_forcing`
   for that path and a comparison of the available datasets.

Wind forcing is dispatched by how far the target date is from the present:

.. list-table::
   :header-rows: 1
   :widths: 10 20 15 55

   * - Tier
     - Source
     - Resolution
     - When used
   * - IV
     - HRRR (NOAA AWS S3)
     - ~3 km
     - Any date from **2014-07-30 to present**. The ``noaa-hrrr-bdp-pds`` S3 bucket
       serves both the rolling operational window and the full historical archive with
       identical path structure. Near-real-time requests walk back up to 12 init cycles
       to find the latest published run; archive requests target the 0z cycle directly.
   * - I
     - ERA5 Final (ECMWF CDS)
     - ~31 km
     - Dates **before 2014-07-30** (pre-HRRR archive). Requires ``CDSAPI_KEY``.
   * - II
     - ERA5T Preliminary (ECMWF CDS)
     - ~31 km
     - Preserved as a fallback path; effectively unreachable for dates within the
       HRRR archive. Requires ``CDSAPI_KEY``.
   * - III
     - ERA5T / HRES fallback
     - ~31 km
     - ECMWF HRES not yet implemented. Falls back to ERA5T. Requires ``CDSAPI_KEY``.

.. note::
   ERA5T and ERA5 Final are both ~31 km - roughly 10× coarser than HRRR.
   For small offshore domains (< 20 km), ERA5 forcing is nearly spatially uniform.
   CDS credentials (``CDSAPI_URL`` / ``CDSAPI_KEY``) are required for all ERA5 tiers;
   see ``.env.template`` for setup.

Available Sources
-----------------

Atmospheric Forcing:
- **HRRR**: High-Resolution Rapid Refresh (AWS S3, ``noaa-hrrr-bdp-pds``) - 3 km surface forcing. Archive spans 2014-07-30 to present. No authentication required.
- **ERA5T / ERA5**: ECMWF Copernicus Reanalysis - ~31 km surface and **3D volumetric pressure levels** (u, v, w, T, Z, q). Requires CDS credentials.

Ocean Boundary & Initial Conditions:
- **NYOFS** (Primary for NY/NJ Harbor): NOAA New York/New Jersey Operational Forecast System (70–150m resolution, Princeton Ocean Model/POM). Historical data >= 2024 sourced from ``noaa-nos-ofs-pds`` AWS S3 bucket.
- **DBOFS** (Primary for Mid-Atlantic Bight): NOAA Delaware Bay Operational Forecast System (ROMS, ~100m resolution, domain [-76.5, 37.5, -73.0, 40.0]). Historical data >= 2024 sourced from ``noaa-nos-ofs-pds`` AWS S3 bucket.
- **NECOFS**: New England Coastal and Ocean Forecasting System (FVCOM, 200m).
- **TPXO10**: Global Tidal Harmonics (OTIS format) via ``pyTMD`` v3. Reserved as a future roadmap item given usage rights; not yet integrated.

.. note::
   NOAA Big Data Program (NODD) migrated the historic OFS archives from NCEI THREDDS to AWS S3 starting in 2024. The fetchers for NYOFS and DBOFS automatically retrieve pre-2024 data via NCEI OPeNDAP and 2024+ data via AWS S3 utilizing `h5netcdf`.

   **Known Data Gap:** There is a known missing data gap for December 2023 (12/2023). During the infrastructure migration, NCEI stopped archiving at the end of November 2023, while the new AWS S3 pipeline did not begin archiving until January 2024. Hindcasts attempting to span this window through NYOFS or DBOFS may fail.

   For details on the AWS S3 structure, see the `NOAA NODD OFS documentation <https://github.com/NOAA-Big-Data-Program/nodd-data-docs/blob/main/OFS/README.md>`_.
- **NERACOOS** / **MARACOOS**: Regional IOOS endpoints.
- **HYCOM**: Global 9km ocean state. v2.0 adds support for continuous **60-day historical hindcasts** by stitching experiment series (e.g., expt_53.X and expt_93.0).
- **GOT4.10c**: Global tidal harmonic extraction from NASA/Goddard via ``pyTMD``. Constituent set: M2, S2, N2, K2, K1, O1, P1, Q1. Used as default ``model_name`` in ``/api/v1/harmonics``. Requires model files in ``~/.pytmd/`` (or ``PYTMD_DATA_DIR``). Generates raw amplitudes and phases. Resolution ~0.5°.
- **EOT20**: DTU/DGFI global empirical ocean tide model. Same constituent set as GOT4.10c. Alternative to GOT4.10c. Specify ``"model_name": "EOT20"`` in harmonics request. Resolution ~0.125°.

V2.0 Processing Features
------------------------

- **Temporal Alignment**: Linear interpolation of disparate sources (where applicable) to a strictly hourly shared temporal index, simplifying timestep integration for the end client.
