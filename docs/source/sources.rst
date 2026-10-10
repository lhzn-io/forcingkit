Sources and Citations
=====================

forcingkit produces no data of its own: every value in a store comes from a model or network run
by the institutions below. If you publish work that uses a forcingkit store, cite the source
models and acknowledge the providers named in the store's ``source`` attribute, as you would if
you had fetched the data yourself. Terms of use are the providers'; check their pages for the
current wording.

Providers
---------

.. list-table::
   :header-rows: 1
   :widths: 16 34 50

   * - Source
     - Institutions
     - Cite or acknowledge
   * - HRRR
     - `NOAA Global Systems Laboratory <https://gsl.noaa.gov/>`__ (development) and
       `NOAA NCEP <https://www.nco.ncep.noaa.gov/>`__ (operations); distributed by `NOAA Open Data
       Dissemination <https://www.noaa.gov/information-technology/open-data-dissemination>`__
     - Dowell et al. (2022); James et al. (2022). NOAA asks for attribution and that modified
       products not be presented as NOAA's.
   * - NECOFS
     - `UMass Dartmouth SMAST <https://www.umassd.edu/smast/>`__ and `Woods Hole Oceanographic
       Institution <https://www.whoi.edu/>`__; operation supported by `NERACOOS
       <https://neracoos.org/>`__
     - Chen, Liu and Beardsley (2003) for FVCOM; acknowledge `NECOFS
       <https://fvcom.smast.umassd.edu/?p=20>`__ and SMAST.
   * - NYOFS, DBOFS
     - `NOAA CO-OPS <https://tidesandcurrents.noaa.gov/>`__; archived by `NOAA NCEI
       <https://www.ncei.noaa.gov/>`__ and on AWS through NOAA Open Data Dissemination
     - Blumberg and Mellor (1987) for POM (NYOFS); Shchepetkin and McWilliams (2005) for ROMS
       (DBOFS); NOAA attribution as for HRRR.
   * - HYCOM
     - The `HYCOM consortium <https://www.hycom.org/>`__ (US Navy, NOAA and academic partners);
       from 2024-09-05 the US Navy's ESPC-D-V02 analysis, served by the consortium
     - Bleck (2002) for the model; Chassignet et al. (2007) for the data-assimilative system.
   * - ERA5 (used through NumericalEarth, not served here)
     - `ECMWF <https://www.ecmwf.int/>`__ for the `Copernicus Climate Change Service
       <https://climate.copernicus.eu/>`__
     - Hersbach et al. (2020). The Copernicus licence requires the statement "Contains modified
       Copernicus Climate Change Service information [year]".
   * - CO-OPS water level
     - `NOAA CO-OPS <https://tidesandcurrents.noaa.gov/>`__
     - NOAA attribution.
   * - NDBC buoys
     - `NOAA National Data Buoy Center <https://www.ndbc.noaa.gov/>`__
     - NOAA attribution.
   * - River discharge
     - `US Geological Survey <https://www.usgs.gov/>`__ stream gauges, through the `USGS Water
       Data APIs <https://api.waterdata.usgs.gov/>`__; drainage areas from the USGS `NLDI
       <https://waterdata.usgs.gov/blog/nldi-intro/>`__ and NHDPlus
     - Cite the gauges as "U.S. Geological Survey, USGS Water Data for the Nation", with the site
       numbers in the store's ``provenance``. USGS data are in the public domain; provisional
       values are subject to revision.
   * - Satellite surface fields
     - `NOAA NESDIS CoastWatch <https://coastwatch.noaa.gov/>`__ ERDDAP servers: VIIRS
       chlorophyll-a and Kd490 (S-NPP, NOAA-20, NOAA-21), Sentinel-3 OLCI chlorophyll-a, and the
       gap-filled DINEOF products; MUR sea surface temperature from `NASA JPL
       <https://podaac.jpl.nasa.gov/>`__ (PO.DAAC), served through CoastWatch
     - Acknowledge NOAA NESDIS CoastWatch and the datasets named in the store's ``products``
       attribute. MUR: JPL MUR MEaSUREs Project (2015) and Chin et al. (2017). OLCI and DINEOF
       products require "Contains modified Copernicus Sentinel data [year]".
   * - Water-column profiles
     - `University of Connecticut Department of Marine Sciences
       <https://marinesciences.uconn.edu/>`__ (`ERDDAP
       <http://merlin.dms.uconn.edu:8080/erddap/index.html>`__)
     - Acknowledge the station operator named in each dataset's ERDDAP metadata.

Related systems not served by forcingkit: NYHOPS from the `Davidson Laboratory
<https://www.stevens.edu/davidson-laboratory>`__ at `Stevens Institute of Technology
<https://www.stevens.edu/>`__ (see :doc:`nyofs`), and the atmospheric datasets compared in
:doc:`atmospheric_forcing`.

Software
--------

- `Oceananigans.jl <https://github.com/CliMA/Oceananigans.jl>`__, from the `Climate Modeling
  Alliance <https://clima.caltech.edu/>`__: Ramadhan et al. (2020).
- `NumericalEarth <https://github.com/NumericalEarth/NumericalEarth.jl>`__: Earth-system
  components for Oceananigans, including the ERA5 and JRA55-do atmospheres.
- `FVCOM <https://fvcom.smast.umassd.edu/>`__ and `ROMS <https://www.myroms.org/>`__: the source
  models of NECOFS and DBOFS.

References
----------

- Beardsley, R. C., C. Chen and Q. Xu (2013). Coastal flooding in Scituate (MA): A FVCOM study
  of the 27 December 2010 nor'easter. *Journal of Geophysical Research: Oceans*, 118(11),
  6030-6045. `doi:10.1002/2013JC008862 <https://doi.org/10.1002/2013JC008862>`__
- Bleck, R. (2002). An oceanic general circulation model framed in hybrid isopycnic-Cartesian
  coordinates. *Ocean Modelling*, 4(1), 55-88.
  `doi:10.1016/S1463-5003(01)00012-9 <https://doi.org/10.1016/S1463-5003(01)00012-9>`__
- Blumberg, A. F. and G. L. Mellor (1987). A description of a three-dimensional coastal ocean
  circulation model. In N. S. Heaps (ed.), *Three-Dimensional Coastal Ocean Models*, Coastal and
  Estuarine Sciences 4, American Geophysical Union, 1-16.
  `doi:10.1029/CO004p0001 <https://doi.org/10.1029/CO004p0001>`__
- Chassignet, E. P., H. E. Hurlburt, O. M. Smedstad, et al. (2007). The HYCOM (HYbrid Coordinate
  Ocean Model) data assimilative system. *Journal of Marine Systems*, 65(1-4), 60-83.
  `doi:10.1016/j.jmarsys.2005.09.016 <https://doi.org/10.1016/j.jmarsys.2005.09.016>`__
- Chen, C., H. Liu and R. C. Beardsley (2003). An unstructured grid, finite-volume,
  three-dimensional, primitive equations ocean model: application to coastal ocean and
  estuaries. *Journal of Atmospheric and Oceanic Technology*, 20(1), 159-186.
  `doi:10.1175/1520-0426(2003)020<0159:AUGFVT>2.0.CO;2
  <https://doi.org/10.1175/1520-0426(2003)020%3C0159:AUGFVT%3E2.0.CO;2>`__
- Chin, T. M., J. Vazquez-Cuervo and E. M. Armstrong (2017). A multi-scale high-resolution
  analysis of global sea surface temperature. *Remote Sensing of Environment*, 200, 154-169.
  `doi:10.1016/j.rse.2017.07.029 <https://doi.org/10.1016/j.rse.2017.07.029>`__
- Dowell, D. C., C. R. Alexander, E. P. James, et al. (2022). The High-Resolution Rapid Refresh
  (HRRR): An hourly updating convection-allowing forecast model. Part I: Motivation and system
  description. *Weather and Forecasting*, 37(8), 1371-1395.
  `doi:10.1175/WAF-D-21-0151.1 <https://doi.org/10.1175/WAF-D-21-0151.1>`__
- Hersbach, H., B. Bell, P. Berrisford, et al. (2020). The ERA5 global reanalysis. *Quarterly
  Journal of the Royal Meteorological Society*, 146(730), 1999-2049.
  `doi:10.1002/qj.3803 <https://doi.org/10.1002/qj.3803>`__
- James, E. P., C. R. Alexander, D. C. Dowell, et al. (2022). The High-Resolution Rapid Refresh
  (HRRR): An hourly updating convection-allowing forecast model. Part II: Forecast performance.
  *Weather and Forecasting*, 37(8), 1397-1417.
  `doi:10.1175/WAF-D-21-0130.1 <https://doi.org/10.1175/WAF-D-21-0130.1>`__
- JPL MUR MEaSUREs Project (2015). GHRSST Level 4 MUR Global Foundation Sea Surface
  Temperature Analysis (v4.1). PO.DAAC, CA, USA.
  `doi:10.5067/GHGMR-4FJ04 <https://doi.org/10.5067/GHGMR-4FJ04>`__
- NOAA (2002). *Implementation Plan, Port of New York and New Jersey Operational Forecast System
  (NYOFS)*. NOAA Technical Report NOS CO-OPS 37, Silver Spring, MD.
  `PDF <https://tidesandcurrents.noaa.gov/publications/techrpt37.pdf>`__
- Ramadhan, A., G. L. Wagner, C. Hill, et al. (2020). Oceananigans.jl: Fast and friendly
  geophysical fluid dynamics on GPUs. *Journal of Open Source Software*, 5(53), 2018.
  `doi:10.21105/joss.02018 <https://doi.org/10.21105/joss.02018>`__
- Shchepetkin, A. F. and J. C. McWilliams (2005). The regional oceanic modeling system (ROMS): a
  split-explicit, free-surface, topography-following-coordinate oceanic model. *Ocean
  Modelling*, 9(4), 347-404.
  `doi:10.1016/j.ocemod.2004.08.002 <https://doi.org/10.1016/j.ocemod.2004.08.002>`__
- Tsujino, H., S. Urakawa, H. Nakano, et al. (2018). JRA-55 based surface dataset for driving
  ocean-sea-ice models (JRA55-do). *Ocean Modelling*, 130, 79-139.
  `doi:10.1016/j.ocemod.2018.07.002 <https://doi.org/10.1016/j.ocemod.2018.07.002>`__
