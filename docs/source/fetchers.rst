Data Fetchers
=============

The Data Fetchers module contains implementations for communicating with external data providers. We use tiered fallback logic and cache everything locally to minimize bandwidth usage and guarantee reproducible simulations.

Available Sources
-----------------

Atmospheric Forcing:
- **HRRR**: High-Resolution Rapid Refresh (AWS S3, ``noaa-hrrr-bdp-pds``) - 3 km surface fields, served by ``/api/v1/atmosphere`` on a regular 0.03 degree grid. Archive spans 2014-07-30 to present. No authentication required. See :doc:`atmospheric_forcing`; earlier dates use ERA5 through NumericalEarth in the model.

Parent Ocean (``/api/v1/obc``):
- **NYOFS** (Primary for NY/NJ Harbor): NOAA New York/New Jersey Operational Forecast System (70–150m resolution, Princeton Ocean Model/POM). Historical data >= 2024 sourced from ``noaa-nos-ofs-pds`` AWS S3 bucket.
- **DBOFS** (Primary for Mid-Atlantic Bight): NOAA Delaware Bay Operational Forecast System (ROMS, ~100m resolution, domain [-76.5, 37.5, -73.0, 40.0]). Historical data >= 2024 sourced from ``noaa-nos-ofs-pds`` AWS S3 bucket.
- **NECOFS**: New England Coastal and Ocean Forecasting System (FVCOM, 200m).
- **HYCOM**: Global 9km ocean state. v2.0 adds support for continuous **60-day historical hindcasts** by stitching experiment series (e.g., expt_53.X and expt_93.0).
- **TPXO10**: Global Tidal Harmonics (OTIS format) via ``pyTMD`` v3. Reserved as a future roadmap item given usage rights; not yet integrated.

.. note::
   NOAA Big Data Program (NODD) migrated the historic OFS archives from NCEI THREDDS to AWS S3 starting in 2024. The fetchers for NYOFS and DBOFS automatically retrieve pre-2024 data via NCEI OPeNDAP and 2024+ data via AWS S3 utilizing `h5netcdf`.

   **Known Data Gap:** There is a known missing data gap for December 2023 (12/2023). During the infrastructure migration, NCEI stopped archiving at the end of November 2023, while the new AWS S3 pipeline did not begin archiving until January 2024. Hindcasts attempting to span this window through NYOFS or DBOFS may fail.

   For details on the AWS S3 structure, see the `NOAA NODD OFS documentation <https://github.com/NOAA-Big-Data-Program/nodd-data-docs/blob/main/OFS/README.md>`_.

V2.0 Processing Features
------------------------

- **Temporal Alignment**: Linear interpolation of disparate sources (where applicable) to a strictly hourly shared temporal index, simplifying timestep integration for the end client.
