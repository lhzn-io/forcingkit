Northeast Shelf Models and Reanalyses
=====================================

Ocean models and reanalyses that cover the US Northeast continental shelf (the Gulf of Maine to
the Mid-Atlantic Bight), as candidates for a second opinion beside NECOFS (:doc:`necofs`) or for
years NECOFS does not cover. A downstream ocean model that disagrees with one parent can be set
against several, built on different grids and vertical coordinates: a feature that survives
FVCOM's sigma layers, ROMS's s-levels and a z-level model is less likely to be a numerical
artifact of any one of them.

Each entry was checked against the provider's live servers (THREDDS catalogs, OPeNDAP metadata,
S3 listings, public STAC metadata) on 2026-11-17. "Not verified" marks what could not be.

.. list-table:: At a glance
   :header-rows: 1
   :widths: 16 22 20 22 20

   * - Source
     - Model and grid
     - Vertical
     - Coverage and interval
     - In forcingkit
   * - :doc:`Doppio reanalysis <doppio>`
     - ROMS 4D-Var, curvilinear, about 7 km
     - 40 s-levels
     - 2007-01-02 to 2024-12-31, hourly
     - ``fetchers.doppio.volume``
   * - :doc:`Doppio real time <doppio>`
     - ROMS with assimilation, same grid
     - 40 s-levels
     - 2017-11-02 to two days ahead, hourly
     - ``fetchers.doppio.volume``
   * - MOM6-NWA12
     - MOM6, curvilinear C-grid, 1/12 degree
     - 52 z levels (remapped)
     - 1993 to 2023, monthly (3D)
     - Not yet
   * - GLORYS12
     - NEMO with assimilation, regular 1/12 degree
     - 50 z levels
     - 1993 to about two months ago, daily and monthly
     - Not yet (needs an account)
   * - HYCOM GOFS 3.1 reanalysis
     - HYCOM, regular 0.08 degree
     - 40 z levels
     - 1994 to 2015, 3-hourly
     - :doc:`hycom` (parent ocean)
   * - ESPreSSO
     - ROMS with assimilation, curvilinear
     - 36 s-levels
     - 2009 to 2018 (discontinued)
     - Not yet
   * - NOS OFS (GoMOFS, DBOFS, CBOFS, NYOFS)
     - ROMS (POM for NYOFS)
     - 30, 10, 20 s-levels; 7 sigma
     - 2015 or 2018 onward, hourly or 3-hourly
     - :doc:`nyofs`, DBOFS (parent ocean)
   * - ECCOFS
     - ROMS 4D-Var, 3 km
     - 50 s-levels
     - Developmental, from 2026
     - Not yet

Doppio
------

See :doc:`doppio` for the reader.

- **Access**: Rutgers THREDDS, OPeNDAP. Reanalysis history (hourly):
  ``https://tds.marine.rutgers.edu/thredds/dodsC/roms/doppio/DopAnV3R3-ini2007_da/his``; daily
  averages at ``.../DopAnV3R3-ini2007_da/avg``, with monthly and yearly averages and atmospheric
  fluxes beside them. Real time:
  ``https://tds.marine.rutgers.edu/thredds/dodsC/roms/doppio/2017_da/his/History_Best`` (and
  ``.../2017_da/avg``). The reanalysis is also listed on the `AWS Open Data registry
  <https://registry.opendata.aws/noaa-ioos-roms-doppio/>`__ (not checked here).
- **Grid**: 106 x 242 rho points, about 7 km, C-grid, rotated about 37 degrees from east;
  40 s-levels (Vtransform 2, Vstretching 4, theta_s 7, theta_b 2, hc 250 m).
- **Variables**: ``u``, ``v`` (grid-relative, on faces), ``w``, ``temp``, ``salt``, ``zeta``,
  ``ubar``, ``vbar``, mixing coefficients, surface fluxes; the real-time aggregation adds
  ``u_eastward``, ``v_northward`` at rho points (empty for 2022).
- **Time**: reanalysis 2007-01-02 01:00 to 2024-12-31 00:00, 157,752 hourly records without a gap;
  real time 2017-11-02 01:00 onward, hourly. Surface elevation relative to NAVD88.
- **License and credit**: "Freely available" in the catalog. Cite López et al. (2020), Wilkin et
  al. (2022), and for the reanalysis output Wilkin and Levin (2022),
  `doi:10.17882/89673 <https://doi.org/10.17882/89673>`__.

MOM6-NWA12 (NOAA CEFI)
----------------------

The Northwest Atlantic 1/12 degree configuration of MOM6 with COBALT biogeochemistry, from NOAA
GFDL for NOAA's Changing Ecosystems and Fisheries Initiative (Ross et al., 2023).

- **Access**: NOAA PSL THREDDS,
  ``https://psl.noaa.gov/thredds/catalog/Projects/CEFI/regional_mom6/cefi_portal/northwest_atlantic/full_domain/catalog.xml``,
  with products ``hindcast``, ``seasonal_forecast``, ``seasonal_reforecast``,
  ``decadal_forecast``, ``multi_decadal_outlook`` and ``long_term_projection``, each split into
  daily, monthly and yearly, ``raw`` (model grid) and ``regrid`` (regular grid), and by release
  (``r20230520``, ``r20250715``; ``latest`` is ``r20250715``). One file per variable, for example
  ``hindcast/monthly/raw/r20250715/uo.nwa.full.hcast.monthly.raw.r20250715.199301-202312.nc``.
  The same tree is in the anonymous S3 bucket ``noaa-oar-cefi-regional-mom6-pds``, with
  kerchunk sidecars and Zarr stores under ``arraylake/``. PSL returned HTTP 429 after a few
  quick requests, so a reader should pace itself or use the bucket.
- **Grid**: native C-grid, 845 x 775 tracer points (``yh``, ``xh``), 98 W to 36 W and 5 N to
  52 N; curvilinear (2D ``geolon``, ``geolat`` in ``ocean_static.nc``). The regridded files are
  on a regular 844 x 774 grid.
- **Vertical**: remapped to 52 z levels (``z_l``, 2.5 m to 6250 m; 5 m spacing to 50 m), with
  ``z_i`` interfaces. The native coordinate is not published.
- **Variables**: ``uo``, ``vo`` (grid-relative, on faces), ``uo_rotate``, ``vo_rotate`` (east and
  north at tracer points), ``thetao``, ``so``, ``zos``, ``ssh``, surface and bottom fields, mixed
  layer depth.
- **Time**: 3D physics only as **monthly means**, 1993-01 to 2023-12 (``r20250715``). The daily
  output holds surface and bottom fields: ``r20230520`` (1993 to 2019) has daily SSH and surface
  currents; ``r20250715`` daily has biogeochemistry only.
- **License and credit**: no license attribute in the files; NOAA's open data terms for its
  buckets presumably apply (not verified). Cite Ross et al. (2023) and the data DOI
  `10.5281/zenodo.7893386 <https://doi.org/10.5281/zenodo.7893386>`__, both named in the files.
- **Use here**: monthly means cannot resolve tides or an eddy that lives a few days, so
  NWA12 is a climatological check, not an hourly parent.

GLORYS12 (Copernicus Marine)
----------------------------

Mercator Ocean's global reanalysis for the `Copernicus Marine Service
<https://marine.copernicus.eu/>`__, product GLOBAL_MULTIYEAR_PHY_001_030 (Lellouche et al., 2021).

- **Access**: Copernicus Marine (the ``copernicusmarine`` toolbox, ARCO Zarr), which **needs a
  free account**. forcingkit does not create one; a user who wants GLORYS registers and supplies
  their own credentials. Public metadata (no login):
  ``https://stac.marine.copernicus.eu/metadata/GLOBAL_MULTIYEAR_PHY_001_030/product.stac.json``.
  Dataset IDs ``cmems_mod_glo_phy_my_0.083deg_P1D-m_202311`` (daily) and
  ``cmems_mod_glo_phy_my_0.083deg_P1M-m_202311`` (monthly), with static coordinate, bathymetry
  and mean dynamic topography sets.
- **Grid**: regular 1/12 degree, 80 S to 90 N; 50 z levels, 0.494 m to 5727.9 m.
- **Variables**: ``uo``, ``vo`` (east, north), ``thetao`` (potential temperature), ``so``,
  ``zos``, ``bottomT``, ``mlotst``, sea ice.
- **Time**: daily means 1993-01-01 to 2026-10-06 and monthly means to 2026-09 in the STAC
  metadata; the recent months are the interim extension, folded into the same dataset IDs (the
  date where the reanalysis becomes the interim run is not verified). Daily means, so tides come
  from elsewhere.
- **License and credit**: the Copernicus Marine Service licence (STAC ``"license":
  "proprietary"``, linking to the service commitments and licence page). Products must be
  credited as "Generated using E.U. Copernicus Marine Service Information", with the product DOI
  `10.48670/moi-00021 <https://doi.org/10.48670/moi-00021>`__; publications state "This study
  has been conducted using E.U. Copernicus Marine Service Information". Cite Lellouche et al.
  (2021). Whether the ARCO stores can be read anonymously was not tested.

HYCOM GOFS 3.1 reanalysis
-------------------------

The HYCOM consortium's global reanalysis, GLBv0.08 ``expt_53.X``, which :doc:`hycom` already uses
as the first segment of its parent-ocean record.

- **Access**: ``https://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_53.X/data/<year>`` per year
  (1994 to 2015) and the aggregation ``GLBv0.08/expt_53.X``; the raw 3-hourly files are also in
  the S3 bucket ``hycom-gofs-3pt1-reanalysis`` (us-west-2, about 4.8 GB each).
- **Grid**: regular, 4500 x 3251, 0.08 degree in longitude (0.04 degree in latitude poleward of
  40 degrees); 40 standard z levels, 0 m to 5000 m (the model's 41 hybrid layers interpolated).
- **Variables**: ``water_u``, ``water_v`` (east, north), ``water_temp``, ``salinity``,
  ``surf_el``, and ``*_bottom`` variants, stored as scaled Int16. HYCOM states that the temperature
  is in situ, not potential.
- **Time**: 1994-01-01 12:00 to 2015-12-31 09:00, 3-hourly, with gaps (2010 holds 2888 of 2920
  steps, 2015 holds 2861). The analysis experiments that follow (``expt_56.3`` to ``expt_93.0``,
  then ESPC-D-V02 from 2024-08-10) are listed in :doc:`hycom`.
- **License and credit**: "Approved for public release. Distribution unlimited." (file
  metadata); the AWS registry entry states no restrictions on use. No acknowledgment sentence was
  found; cite Bleck (2002) and Chassignet et al. (2007), listed in :doc:`sources`.
- **Use here**: 9 km, so it resolves the shelf break and Gulf Stream rings but not canyon-scale
  flow.

ESPreSSO (Rutgers, discontinued)
--------------------------------

Doppio's predecessor, also ROMS with data assimilation for MARACOOS.

- **Access**: ``https://tds.marine.rutgers.edu/thredds/catalog/roms/espresso/catalog.xml``,
  marked discontinued in favor of Doppio. Version 2 (2013 to 2018):
  ``.../dodsC/roms/espresso/2013_da/avg/ESPRESSO_Real-Time_v2_Averages_Best`` and
  ``.../2013_da/his/ESPRESSO_Real-Time_v2_History_Best``; version 1 (2009 to 2013) under
  ``roms/espresso/2009_da``; hindcast experiments under ``roms/esp2006``.
- **Grid**: 82 x 130 rho points, C-grid, 36 s-levels; the catalog says about 6 km (not verified
  against the grid metrics).
- **Variables**: ``zeta``, ``u``, ``v``, ``temp``, ``salt``, ``h``.
- **Time**: version 2 daily averages 2013-05-19 to 2018-09-19 (the last days may be forecast).
- **License and credit**: no license attribute found. The paper to cite was not verified.

NOAA NOS Operational Forecast Systems
-------------------------------------

NOAA CO-OPS's regional forecast systems. Besides NECOFS (served by SMAST; see :doc:`necofs`), four
cover parts of the region.

- **Access**: the anonymous S3 bucket ``noaa-nos-ofs-pds`` (per model, per day, for example
  ``gomofs/netcdf/2026/11/15/gomofs.t00z.20261115.fields.f003.nc``; older months in
  ``YYYYMM/`` folders, from 2022-01 for CBOFS and 2024 for the others); the 7-day CO-OPS THREDDS
  aggregations
  ``https://opendap.co-ops.nos.noaa.gov/thredds/dodsC/<MODEL>/fmrc/Aggregated_7_day_<MODEL>_Fields_Forecast_best.ncd``;
  and NCEI before that, ``https://www.ncei.noaa.gov/thredds/catalog/model-<model>-files/``.
- **Systems**:

  .. list-table::
     :header-rows: 1
     :widths: 14 26 14 46

     * - System
       - Model and grid
       - Levels
       - Output and archive
     * - GoMOFS
       - ROMS, 777 x 1173, Gulf of Maine (about 700 m, not verified)
       - 30 s
       - 3-hourly fields (hourly 2D surface files); NCEI from 2018-01-10
     * - DBOFS
       - ROMS, 732 x 119, Delaware Bay
       - 10 s
       - Hourly fields; NCEI from 2015-08-12
     * - CBOFS
       - ROMS, 291 x 332, Chesapeake Bay
       - 20 s
       - Hourly fields; NCEI from 2015-08-12
     * - NYOFS
       - POM, 134 x 73, NY/NJ Harbor
       - 7 sigma
       - Nowcast and forecast files four times a day; no temperature or salinity; NCEI from
         2015-08-19

- **Variables**: ``zeta``, ``u``, ``v``, ``temp``, ``salt`` (ROMS, grid-relative ``u``, ``v``
  on faces, so the same destaggering and rotation as Doppio); NYOFS ``zeta``, ``u``, ``v``, ``w``.
  The ROMS systems' Vtransform and Vstretching were not checked.
- **License and credit**: NOAA data on NODD "are open to the public and can be used as desired";
  NOAA requests attribution (AWS registry). Cite the model papers in :doc:`sources`.

ECCOFS (developmental)
----------------------

The East Coast Coastal Ocean Forecast System, a ROMS 4D-Var system by Rutgers, UC Santa Cruz and
NOS at 3 km with 50 s-levels, from the Grand Banks to Venezuela, planned for operations at NCEP
in 2028. Output is in the S3 bucket ``noaa-nos-eccofs-pds`` (``avg/``, ``his/``, and others; for
example ``avg/eccofs_avg_20260608_0001.nc``, 7.6 GB a day) and on Rutgers THREDDS under
``roms/eccofs``. Too young for a reanalysis window, but a future successor to Doppio.

Next steps
----------

- MOM6-NWA12 monthly 3D fields through ``uo_rotate``, ``vo_rotate``, ``thetao``, ``so`` from the
  S3 mirror, as a climatological comparison.
- ESPreSSO and the NOS ROMS systems through the Doppio reader, once their vertical transform
  parameters are checked (the reader supports Vtransform 2 only).
- GLORYS12, for a user with a Copernicus Marine account.

References
----------

- Lellouche, J.-M., E. Greiner, R. Bourdallé-Badie, et al. (2021). The Copernicus global 1/12°
  oceanic and sea ice GLORYS12 reanalysis. *Frontiers in Earth Science*, 9, 698876.
  `doi:10.3389/feart.2021.698876 <https://doi.org/10.3389/feart.2021.698876>`__
- López, A. G., J. L. Wilkin and J. C. Levin (2020). Doppio: a ROMS (v3.6)-based circulation
  model for the Mid-Atlantic Bight and Gulf of Maine: configuration and comparison to integrated
  coastal observing network observations. *Geoscientific Model Development*, 13(8), 3709-3729.
  `doi:10.5194/gmd-13-3709-2020 <https://doi.org/10.5194/gmd-13-3709-2020>`__
- Ross, A. C., C. A. Stock, A. Adcroft, et al. (2023). A high-resolution ocean model for the
  Northwest Atlantic: model description and evaluation (MOM6-COBALT-NWA12 v1.0). *Geoscientific
  Model Development*, 16, 6943-6985.
  `doi:10.5194/gmd-16-6943-2023 <https://doi.org/10.5194/gmd-16-6943-2023>`__
- Wilkin, J. and J. Levin (2022). Outputs from a Regional Ocean Modeling System (ROMS) data
  assimilative reanalysis (version DopAnV3R3-ini2007) of ocean circulation in the Mid-Atlantic
  Bight and Gulf of Maine for 2007-2021. SEANOE. `doi:10.17882/89673 <https://doi.org/10.17882/89673>`__
- Wilkin, J., J. Levin, A. Moore, H. Arango, A. López and E. Hunter (2022). A data-assimilative
  model reanalysis of the U.S. Mid Atlantic Bight and Gulf of Maine: Configuration and
  comparison to observations and global ocean models. *Progress in Oceanography*, 209, 102919.
  `doi:10.1016/j.pocean.2022.102919 <https://doi.org/10.1016/j.pocean.2022.102919>`__
