Roadmap and Contributing
========================

As of 2026-10-09. This page covers the kinds of forcing forcingkit delivers: the **parent
ocean** for a model's open boundaries (``/api/v1/obc``), the **surface atmosphere**
(``/api/v1/atmosphere``) and **river discharge** (``/api/v1/rivers``). New sources are ranked by
two questions:

1. **Coverage first.** Does it serve a region, or a span of years, that no current source does?
2. **Resolution second.** Where something already serves, does it resolve the coast better?

Archive depth counts as much as grid spacing: most requests are hindcasts, so a source with only
its latest runs online can serve forecasts but not the past. Every span below was read from the
provider's archive or catalogue on the date above. Contributions are welcome at any scale; see
`Contributing`_.

Current coverage
----------------

.. list-table::
   :header-rows: 1
   :widths: 16 34 26 24

   * - Source
     - Region
     - Span
     - Grid
   * - :doc:`NECOFS <necofs>`
     - Gulf of Maine to the Mid-Atlantic Bight
     - 2025-01-01 on (66 days missing)
     - FVCOM, about 130 m to 2.8 km
   * - :doc:`NYOFS <nyofs>`
     - NY/NJ Harbor
     - 2014 on
     - POM, about 140 to 1000 m
   * - DBOFS
     - Delaware Bay and offshore New Jersey
     - 2014 on
     - ROMS, about 100 m
   * - :doc:`HYCOM <hycom>`
     - Global
     - 1994 on
     - 1/12 degree
   * - :doc:`HRRR <atmospheric_forcing>`
     - Contiguous US (atmosphere)
     - 2014-07-30 on
     - 3 km
   * - :doc:`USGS gauges <rivers>`
     - Rivers into Long Island Sound (river discharge)
     - 15-minute values online from about 2007; daily means fill in further back
     - Five river mouths

So today the only parent ocean outside the US Northeast is HYCOM, at 1/12 degree and in the
legacy output layout, and there is no atmosphere outside the contiguous US: there, and before
2014-07-30, the atmosphere comes from ERA5 through NumericalEarth in the model.

Planned work
------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Item
     - Detail
   * - HYCOM as a ``z-v3`` parent
     - The cheapest conversion, and the one that gives a ``z-v3`` parent everywhere: HYCOM is
       already on a regular longitude/latitude grid at fixed depths in metres, with temperature
       and salinity, so it needs regridding onto the ``z-v3`` grid and an hour-by-hour
       ``iter_parent``, but no sigma-to-depth step. Tidal content changes at 2024-09-05
       (:doc:`hycom`), so the store should record it.
   * - NYOFS and DBOFS as ``z-v3`` parents
     - Deliver both on true depths with geographic axes, like :doc:`necofs`. NYOFS carries no
       temperature or salinity, so those would come from another donor. The scope is in
       :doc:`nyofs`.
   * - HRRR forecast mode
     - One cycle's f01 to f48 for forecasts, alongside the chained one-hour forecasts used for
       hindcasts (:doc:`atmospheric_forcing`).
   * - Cache policy
     - Invalidate stores built from forecasts when the upstream system publishes a newer cycle.
   * - A second global parent
     - HYCOM is the only parent ocean for current dates outside the US Northeast, and its server
       (``tds.hycom.org``) often stalls past the read timeout (:doc:`hycom`). Copernicus GLO12
       (see `Global parent ocean`_) would give an independent fallback for 2020-11-01 on.
   * - Warn on gaps in a donor's record
     - Every HYCOM experiment has a few gaps of up to 51 hours in its 3-hourly record, and the
       hourly resampling bridges them by linear interpolation without notice. Log a warning,
       and record the gap in the store's attributes, when a window contains a step longer than
       about 6 hours.
   * - HYCOM fetcher clean-up
     - Remove the unused ``_normalize_lons``. Select the initial-condition time by position, as
       the boundary path does: the nearest-label lookup fails on the non-monotonic time axis of
       GLBv0.08 ``expt_93.0``.
   * - River mouths for more regions, and found automatically
     - :doc:`rivers` derives each river's gauges per request, so a new river needs only a mouth
       entry. Mouths for the Hudson, Raritan and Delaware would cover the NY Bight and Delaware Bay
       parents already served; finding coastal outlets in a box from NHDPlus would remove the
       entries altogether.
   * - Ungauged rivers
     - Where no tide-free gauge exists, the `National Water Model
       <https://water.noaa.gov/about/nwm>`__ retrospective and operational streamflow (on AWS)
       could supply discharge at the outlet reach.
   * - mypy hook without an active environment
     - The ``mypy`` pre-commit hook runs the ``mypy`` on ``PATH``, so a commit made without the
       project environment activated fails with "Executable ``mypy`` not found". Run it through
       ``uv run`` so it works from any shell.
   * - Fetcher module names
     - Modules in ``forcingkit.fetchers`` are named for their source (``necofs``, ``hycom``,
       ``ndbc``, ``coastwatch``, ``usgs``), and a model-ready delivery for the source and what it
       delivers (``hrrr_atmosphere``, ``usgs_rivers``). Three predate the convention:
       ``noaa.py`` serves CO-OPS water levels and current predictions (``coops.py``),
       ``erddap.py`` serves the UConn profile stations (named for the protocol, not the source),
       and ``hydrography.py`` finds heads of tide from OpenStreetMap (``osm.py`` or similar).
       Renaming them breaks imports from earlier releases, so it belongs in a release that says
       so, with the old names kept as deprecated aliases for one release.

Candidate sources: coverage
---------------------------

Global parent ocean
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 48 30

   * - Source
     - What it offers
     - Adds
   * - **Copernicus GLO12** (`Mercator Ocean International <https://www.mercator-ocean.eu/>`__
       for the `Copernicus Marine Service <https://marine.copernicus.eu/>`__,
       `GLOBAL_ANALYSISFORECAST_PHY_001_024
       <https://data.marine.copernicus.eu/product/GLOBAL_ANALYSISFORECAST_PHY_001_024/description>`__)
     - NEMO, 1/12 degree, 50 levels, hourly and daily fields, from 2020-11-01 to 10 days ahead,
       global. Free registration; cloud-native (ARCO Zarr) access.
     - Hourly fields and a second, independent model to set against HYCOM; the backbone for
       any region outside the US.
   * - **GLORYS12 reanalysis** (Mercator Ocean, `GLOBAL_MULTIYEAR_PHY_001_030
       <https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030/description>`__)
     - NEMO with data assimilation, 1/12 degree, 50 levels, daily means from 1993-01-01 to
       2026-08 (Lellouche et al., 2021).
     - One continuous reanalysis from 1993, where HYCOM chains several experiments from 1994;
       daily, so tides must come from elsewhere.
   * - **NOAA Global RTOFS** (`on AWS <https://registry.opendata.aws/noaa-rtofs/>`__)
     - HYCOM-based, 1/12 degree. Besides the global output, 6-hourly netCDF already on z levels
       for three US subdomains (``US_east``, ``US_west``, ``alaska``), from 2024-01-27.
     - US subsets already on z levels, so they need no vertical regridding.

US coasts without a parent
~~~~~~~~~~~~~~~~~~~~~~~~~~

The rest of the NOAA Operational Forecast Systems. Each is on AWS (``noaa-nos-ofs-pds``) per day
from late 2024 (2024-10-01 for CBOFS, 2024-11-19 for most) and at NCEI before that, in the
layouts forcingkit's NYOFS and DBOFS readers already handle. The ROMS systems can share the DBOFS
reader and the FVCOM systems the NECOFS mesh interpolation, so each is mostly configuration and
a domain.

.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - System
     - Model
     - Region
   * - `WCOFS <https://tidesandcurrents.noaa.gov/ofs/wcofs/wcofs.html>`__
     - ROMS, 4D-Var
     - US West Coast, 18 N to 56 N, about 4 km.
   * - `NGOFS2 <https://tidesandcurrents.noaa.gov/ofs/ngofs2/ngofs2.html>`__
     - FVCOM
     - Northern Gulf coast.
   * - `CBOFS <https://tidesandcurrents.noaa.gov/ofs/cbofs/cbofs.html>`__
     - ROMS
     - Chesapeake Bay.
   * - `SSCOFS <https://tidesandcurrents.noaa.gov/ofs/sscofs/sscofs.html>`__
     - FVCOM
     - Salish Sea and the Columbia River. Its predecessor for the river, CREOFS, is on AWS
       only to 2024-10.
   * - `SFBOFS <https://tidesandcurrents.noaa.gov/ofs/sfbofs/sfbofs.html>`__
     - FVCOM
     - San Francisco Bay and approaches.
   * - `CIOFS <https://tidesandcurrents.noaa.gov/ofs/ciofs/ciofs.html>`__
     - ROMS
     - Cook Inlet, Alaska.
   * - `TBOFS <https://tidesandcurrents.noaa.gov/ofs/tbofs/tbofs.html>`__
     - ROMS
     - Tampa Bay.
   * - Great Lakes: `LEOFS <https://tidesandcurrents.noaa.gov/ofs/leofs/leofs.html>`__,
       `LMHOFS <https://tidesandcurrents.noaa.gov/ofs/lmhofs/lmhofs.html>`__,
       `LOOFS <https://tidesandcurrents.noaa.gov/ofs/loofs/loofs.html>`__,
       `LSOFS <https://tidesandcurrents.noaa.gov/ofs/lsofs/lsofs.html>`__
     - FVCOM
     - Lakes Erie, Michigan and Huron, Ontario, Superior.

Earlier years in the Northeast
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 48 30

   * - Source
     - What it offers
     - Adds
   * - **Doppio reanalysis** (`Rutgers Ocean Modeling Group <https://rucool.marine.rutgers.edu/>`__
       for `MARACOOS <https://maracoos.org/>`__; `on AWS
       <https://registry.opendata.aws/noaa-ioos-roms-doppio/>`__)
     - ROMS with 4D-Var data assimilation, 7 km, 40 s-levels, hourly sea level, velocity,
       temperature and salinity from 2007-01-02 to 2024-12-31, Mid-Atlantic Bight and Gulf of
       Maine (López et al., 2020; Wilkin et al., 2022).
     - A shelf-scale parent for 2007 to 2024, before the NECOFS archive begins.
   * - **Doppio real time** (`Rutgers THREDDS
       <https://tds.marine.rutgers.edu/thredds/catalog/roms/doppio/catalog.html>`__)
     - The same model run forward with assimilation: hourly, from 2017-11-02 to two days ahead,
       as one OPeNDAP aggregation.
     - A second Northeast parent for current dates, and cover for NECOFS's missing days.

Atmosphere outside the contiguous US
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 48 30

   * - Source
     - What it offers
     - Adds
   * - **HRRR Alaska** (same bucket as HRRR, ``alaska/``)
     - 3 km, archived from 2018-07-12.
     - Alaska, with the reader forcingkit already has.
   * - **ECMWF IFS open data** (`ECMWF <https://www.ecmwf.int/en/forecasts/datasets/open-data>`__;
       `on AWS <https://registry.opendata.aws/ecmwf-forecasts/>`__)
     - 0.25 degree global forecasts, archived on AWS from 2023-01-18; CC BY 4.0 since 2025-10-01.
     - A global atmosphere closer to the present than ERA5's five-day latency.
   * - **RRFS** (NOAA; `on AWS <https://registry.opendata.aws/noaa-rrfs/>`__)
     - 3 km over North America, operational from 2026-10-06.
     - Waters beyond HRRR's domain at similar resolution, once an operational archive builds up.
   * - **GFS** (`on AWS <https://registry.opendata.aws/noaa-gfs-bdp-pds/>`__)
     - 0.25 degree global.
     - A NOAA global fallback.

Candidate sources: resolution
-----------------------------

Where a source already serves, these resolve the coast more finely.

.. list-table::
   :header-rows: 1
   :widths: 22 48 30

   * - Source
     - What it offers
     - Improves on
   * - **NYOFS fine grid** (``nyofs_fg``, same archive as NYOFS)
     - About 70 to 150 m (median 117 m) over the Kill van Kull and its approaches.
     - The NYOFS coarse grid (median 340 m) there.
   * - **WCOFS Monterey Bay nests** (UC Santa Cruz, on the `CeNCOOS THREDDS
       <http://thredds.cencoos.org/thredds/catalog/catalog.html>`__)
     - ROMS nests inside WCOFS at 772 m and 150 m, 40 s-levels, daily from 2024-01-15; the
       latest record on 2026-10-06 was dated 2026-02-17.
     - WCOFS's 4 km in Monterey Bay, if the runs continue.
   * - **SFBOFS** (above)
     - FVCOM, finer in the bay than WCOFS.
     - WCOFS inside San Francisco Bay.

Europe
------

US coverage comes first, but one European source pair would open every European coast. The
recommended first step is the **Copernicus Marine Service** for the ocean, through the same
cloud-native access as GLO12, with ERA5 (already reachable through NumericalEarth) or ECMWF IFS
for the atmosphere.

.. list-table::
   :header-rows: 1
   :widths: 22 48 30

   * - Source
     - What it offers
     - Adds
   * - **Copernicus IBI** (`NOW Systems <https://nowsystems.eu/>`__ for Copernicus Marine,
       `IBI_ANALYSISFORECAST_PHY_005_001
       <https://data.marine.copernicus.eu/product/IBI_ANALYSISFORECAST_PHY_005_001/description>`__)
     - NEMO with tides, 0.028 degree (about 3 km), 50 levels, sub-hourly to monthly fields,
       26 N to 56 N and 19 W to 5 E, from 2022-12-01 to 10 days ahead.
     - Iberia, Biscay, Ireland and the English Channel.
   * - **Copernicus North West Shelf** (`Met Office <https://www.metoffice.gov.uk/>`__ for
       Copernicus Marine, `NWSHELF_ANALYSISFORECAST_PHY_004_013
       <https://data.marine.copernicus.eu/product/NWSHELF_ANALYSISFORECAST_PHY_004_013/description>`__)
     - NEMO (AMM15) with tides and waves, about 1.5 km, 33 levels, quarter-hourly to daily
       fields, 46 N to 62.74 N and 16 W to 13 E, from 2024-08-02 to 7 days ahead.
     - The North Sea, Irish Sea and Channel at the finest Copernicus resolution.
   * - **PML WCOOF** (`Plymouth Marine Laboratory <https://www.pml.ac.uk/>`__)
     - FVCOM for south-west England, bounded by AMM15 and forced by WRF, 3-day forecasts
       (Bedington et al., 2022), shown at `Plymouth Marine Forecasts
       <https://plymouthmarineforecasts.org/forecasts/plymouth-sound/>`__; no public download
       found.
     - Plymouth Sound and the Tamar, by arrangement with PML.
   * - **MET Norway MEPS** (`thredds.met.no
       <https://thredds.met.no/thredds/catalog/meps25epsarchive/catalog.html>`__)
     - 2.5 km ensemble forecasts for the Nordic region, archived from 2016-12.
     - A regional atmosphere with a long archive, for Scandinavia and the Baltic.
   * - **Met Office UKV** (`on AWS <https://registry.opendata.aws/met-office-uk-deterministic/>`__)
     - 2 km for the UK and Ireland, hourly to 54 h, 2-year rolling archive. CC BY-SA 4.0, so
       stores derived from it would carry the same licence.
     - The UK at HRRR-like resolution.
   * - **ICON-EU** (`DWD open data <https://opendata.dwd.de/weather/nwp/icon-eu/>`__)
     - Regular-grid European forecasts; only the latest runs are online, with no archive.
     - Forecasts only.

Contributing
------------

The development workflow (``uv``, ``pre-commit``, ``ruff``, ``mypy``, ``pytest``) is in
`CONTRIBUTING.md <https://github.com/lhzn-io/forcingkit/blob/main/CONTRIBUTING.md>`__. Open an
`issue <https://github.com/lhzn-io/forcingkit/issues>`__ before a large change, especially a new
source, so the approach can be agreed first.

A new parent ocean
~~~~~~~~~~~~~~~~~~

Model it on ``forcingkit/fetchers/necofs.py``, the reference ``z-v3`` donor:

1. ``get_metadata()`` (id, name, approximate resolution, domain box) and ``supports_bbox()``.
   Declare the domain the model really covers; the dispatcher ranks donors by domain area, so
   an oversized box makes the donor win requests it cannot serve.
2. ``iter_parent(start_date, duration_hours, bbox, pad_cells, vertical_spacing_m)``, yielding
   one ``("static", Dataset)`` and then one ``("record", time, fields)`` per hour, on the same
   regular grid and z levels as NECOFS, with ``PARENT_RECORD_DIMS`` and ``PARENT_UNITS``.
   Raise rather than return a short or partial record.
3. Register the module in ``dispatcher.get_obc_fetchers``.
4. Unit tests for the source's own logic (file naming, time handling, interpolation), and a
   live integration test that fetches one archive date for every access path.
5. A documentation page in the style of :doc:`necofs`: what the system is and who runs it,
   facts read from its output, where the data lives and over what span, how forcingkit reads
   it, known limits, and an entry in :doc:`sources` with what to cite.

A new atmosphere
~~~~~~~~~~~~~~~~

Model it on ``forcingkit/fetchers/hrrr_atmosphere.py``: the eight surface fields listed in
:doc:`atmospheric_forcing`, in the same units, interpolated to a regular grid and streamed hour by
hour under a versioned schema. Say how accumulated and instantaneous fields are placed in time;
it differs between models.

For either kind, check every archive path against the live server before relying on it: the
existing pages record several cases where a provider's layout differed from its documentation.
Corrections to these pages are welcome too, especially from people who run or use the systems
described.

References
----------

- Bedington, M., L. M. García-García, M. Sourisseau and M. Ruiz-Villarreal (2022). Assessing the
  performance and application of operational Lagrangian transport HAB forecasting systems.
  *Frontiers in Marine Science*, 9, 749071.
  `doi:10.3389/fmars.2022.749071 <https://doi.org/10.3389/fmars.2022.749071>`__
- Lellouche, J.-M., E. Greiner, R. Bourdallé-Badie, et al. (2021). The Copernicus global 1/12°
  oceanic and sea ice GLORYS12 reanalysis. *Frontiers in Earth Science*, 9, 698876.
  `doi:10.3389/feart.2021.698876 <https://doi.org/10.3389/feart.2021.698876>`__
- López, A. G., J. L. Wilkin and J. C. Levin (2020). Doppio: a ROMS (v3.6)-based circulation
  model for the Mid-Atlantic Bight and Gulf of Maine: configuration and comparison to integrated
  coastal observing network observations. *Geoscientific Model Development*, 13(8), 3709-3729.
  `doi:10.5194/gmd-13-3709-2020 <https://doi.org/10.5194/gmd-13-3709-2020>`__
- Wilkin, J., J. Levin, A. Moore, H. Arango, A. López and E. Hunter (2022). A data-assimilative
  model reanalysis of the U.S. Mid Atlantic Bight and Gulf of Maine: Configuration and
  comparison to observations and global ocean models. *Progress in Oceanography*, 209, 102919.
  `doi:10.1016/j.pocean.2022.102919 <https://doi.org/10.1016/j.pocean.2022.102919>`__
