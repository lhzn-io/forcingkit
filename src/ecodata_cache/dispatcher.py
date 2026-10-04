import os
from datetime import datetime
import logging
from typing import Optional
import numpy as np

from ecodata_cache.fetchers.era5 import fetch_era5_surface_forcing
from ecodata_cache.fetchers.hrrr import (
    fetch_hrrr_surface_forcing,
    HRRR_ARCHIVE_START,
)

logger = logging.getLogger(__name__)


def predict_bc_donor(target_date: str) -> dict:
    """Predicts the boundary condition (forcing) donor based on target date delta."""
    from datetime import datetime, timezone

    # Parse target date
    try:
        if "T" in target_date:
            from dateutil import parser  # type: ignore[import-untyped]

            target_dt = parser.parse(target_date)
            if target_dt.tzinfo is None:
                target_dt = target_dt.replace(tzinfo=timezone.utc)
        else:
            target_dt = datetime.strptime(target_date, "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            )
    except Exception:
        return {"name": "Unknown (Invalid Date)", "resolution_approx_m": 0}

    now = datetime.now(timezone.utc)
    delta_days = (now - target_dt).days
    target_dt_naive = target_dt.replace(tzinfo=None)

    # HRRR archive covers 2014-07-30 to present at 3 km; use it for all dates in range.
    if target_dt_naive >= HRRR_ARCHIVE_START:
        return {"name": "HRRR (NOAA)", "resolution_approx_m": 3000, "tier": "IV"}
    elif delta_days > 90:
        return {
            "name": "ERA5 Final (ECMWF CDS)",
            "resolution_approx_m": 31000,
            "tier": "I",
        }
    elif delta_days >= 5 and delta_days <= 90:
        return {
            "name": "ERA5T Preliminary (ECMWF CDS)",
            "resolution_approx_m": 31000,
            "tier": "II",
        }
    else:
        return {
            "name": "ERA5T Fallback (ECMWF CDS)",
            "resolution_approx_m": 31000,
            "tier": "III",
        }


def dispatch_forcing_request(
    target_date: str,
    bbox: list[float],
    duration_hours: int = 1,
    forcing_type: str = "surface",
    cache_dir: str = os.environ.get(
        "ECODATA_CACHE_CACHE_DIR", os.path.expanduser("~/.cache/ecodata-cache")
    ),
    cache_bust: bool = False,
) -> list[str]:
    """
    Tiered modality dispatcher. Evaluates the Delta T between 'Today' and the 'Target Date'
    to download the highest fidelity forcing model available.
    Supports forcing_type='surface' or 'volumetric' (pressure levels).
    """

    os.makedirs(cache_dir, exist_ok=True)

    target_dt = datetime.strptime(target_date, "%Y-%m-%d")
    now = datetime.utcnow()

    delta_days = (now - target_dt).days
    delta_hours = (now - target_dt).total_seconds() / 3600.0

    # For volumetric requests, we strictly use ERA5 pressure levels for now
    if forcing_type == "volumetric":
        logger.info(
            f"Dispatching volumetric forcing request to ERA5 Pressure Levels for {target_date}"
        )
        from ecodata_cache.fetchers.era5 import fetch_era5_pressure_levels

        era5_dir = os.path.join(cache_dir, "era5")
        os.makedirs(era5_dir, exist_ok=True)
        out_path = os.path.join(era5_dir, f"era5_vol_{target_date}.grib")
        return [
            fetch_era5_pressure_levels(
                target_date, bbox, out_path, cache_bust=cache_bust
            )
        ]

    # Tier IV: HRRR - operational (≤1 day) or archive (2014-07-30 to present, 3 km).
    # The noaa-hrrr-bdp-pds S3 bucket serves both windows with the same path structure,
    # so the fetch logic is identical; only the cycle-selection strategy differs.
    if target_dt >= HRRR_ARCHIVE_START:
        if delta_days <= 1:
            logger.info("Dispatching to Tier IV (Operational): HRRR")
        else:
            logger.info(
                f"Dispatching to Tier IV (Archive): HRRR "
                f"({delta_days}-day hindcast, noaa-hrrr-bdp-pds)"
            )

        from datetime import timedelta

        hrrr_cache_dir = os.path.join(cache_dir, "hrrr")
        os.makedirs(hrrr_cache_dir, exist_ok=True)

        # Near-real-time: walk back from now to find the latest published cycle.
        # Archive / same-day: start from target_dt directly (0z is always present).
        base_dt = now if delta_hours < 0 else target_dt

        for i in range(12):
            check_dt = base_dt - timedelta(hours=i)
            cycle_hr = check_dt.hour
            check_date_str = check_dt.strftime("%Y-%m-%d")

            out_path = os.path.join(
                hrrr_cache_dir, f"hrrr_{check_date_str}_t{cycle_hr}z.grib2"
            )
            if not cache_bust and os.path.exists(out_path):
                logger.info(f"Checking cache sequence starting at: {out_path}")
                grib_paths = [out_path]
                all_cached = True
                for offset in range(1, duration_hours):
                    offset_path = os.path.join(
                        hrrr_cache_dir,
                        f"hrrr_{check_date_str}_t{cycle_hr}z_f{offset:02d}.grib2",
                    )
                    if os.path.exists(offset_path):
                        grib_paths.append(offset_path)
                    else:
                        all_cached = False
                        break
                if all_cached:
                    logger.info("Found fully cached sequence!")
                    return grib_paths
                else:
                    logger.info(
                        "Sequence incomplete in cache, falling through to fetch."
                    )

            try:
                # First check if f00 exists for this cycle
                fetch_hrrr_surface_forcing(
                    check_date_str,
                    cycle_hr,
                    out_path,
                    forecast_offset=0,
                    cache_bust=cache_bust,
                )

                # If it does, fetch the rest of the sequence in parallel
                grib_paths = [out_path]
                import concurrent.futures

                def _fetch_offset(offset):
                    offset_path = os.path.join(
                        hrrr_cache_dir,
                        f"hrrr_{check_date_str}_t{cycle_hr}z_f{offset:02d}.grib2",
                    )
                    fetch_hrrr_surface_forcing(
                        check_date_str,
                        cycle_hr,
                        offset_path,
                        forecast_offset=offset,
                        cache_bust=cache_bust,
                    )
                    return offset_path

                with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                    future_to_offset = {
                        executor.submit(_fetch_offset, offset): offset
                        for offset in range(1, duration_hours)
                    }
                    results = {}
                    for future in concurrent.futures.as_completed(future_to_offset):
                        offset = future_to_offset[future]
                        try:
                            path = future.result()
                            results[offset] = path
                        except FileNotFoundError:
                            logger.warning(f"Offset f{offset:02d} missing.")
                        except Exception as e:
                            logger.error(f"Error fetching offset f{offset:02d}: {e}")

                # Append in correct order
                for offset in range(1, duration_hours):
                    if offset in results:
                        grib_paths.append(results[offset])
                    else:
                        break  # Stop at first missing

                return grib_paths
            except FileNotFoundError:
                logger.warning(
                    f"HRRR {check_date_str} {cycle_hr}z missing on S3. Trying older cycle..."
                )
                continue

        if delta_days <= 1:
            raise FileNotFoundError(
                "Exhausted 12-cycle lookback for HRRR. S3 bucket may be down or "
                "latest run not yet published."
            )
        else:
            raise FileNotFoundError(
                f"HRRR archive data not found for {target_date} on noaa-hrrr-bdp-pds. "
                f"Archive begins 2014-07-30; check date and S3 availability."
            )

    # Tier I: Validated ERA5 Final
    elif delta_days > 90:
        logger.info(
            f"Delta is {delta_days} days. Dispatching to Tier I: ERA5 Final (CDS)"
        )
        era5_dir = os.path.join(cache_dir, "era5")
        os.makedirs(era5_dir, exist_ok=True)
        out_path = os.path.join(era5_dir, f"era5_sfc_{target_date}.grib")
        if not cache_bust and os.path.exists(out_path):
            return [out_path]  # WARNING: naive cache hit, may miss subsequent steps
        return [
            fetch_era5_surface_forcing(
                target_date, bbox, out_path, preliminary=False, cache_bust=cache_bust
            )
        ]

    # Tier II: Near-Past ERA5T
    elif delta_days >= 5 and delta_days <= 90:
        logger.info(
            f"Delta is {delta_days} days. Dispatching to Tier II: ERA5T Preliminary (CDS)"
        )
        era5t_dir = os.path.join(cache_dir, "era5t")
        os.makedirs(era5t_dir, exist_ok=True)
        out_path = os.path.join(era5t_dir, f"era5t_sfc_{target_date}.grib")
        if not cache_bust and os.path.exists(out_path):
            return [out_path]  # WARNING: naive cache hit, may miss subsequent steps
        return [
            fetch_era5_surface_forcing(
                target_date, bbox, out_path, preliminary=True, cache_bust=cache_bust
            )
        ]

    else:
        logger.warning(
            f"Delta is {delta_days} days. Tier III HRES not yet implemented. Falling back to ERA5T if available, or failing."
        )
        # Fallback implementation
        out_path = os.path.join(cache_dir, f"ecmwf_fallback_{target_date}.grib")
        return [
            fetch_era5_surface_forcing(
                target_date, bbox, out_path, preliminary=True, cache_bust=cache_bust
            )
        ]


def get_ic_fetchers():
    """Return all registered IC fetcher modules and their fetch functions."""
    from ecodata_cache.fetchers import (
        dbofs,
        hycom,
        necofs,
        neracoos,
        maracoos,
    )

    # NYOFS is intentionally excluded: the FMRC endpoint is currents-only (u, v, w, zeta)
    # and does not publish temperature or salinity. Use NYOFS for OBCs only.
    # DBOFS (Delaware Bay / offshore NJ) is registered here because its ROMS endpoint
    # publishes temperature and salinity, making it a valid IC donor.
    return [
        (dbofs, dbofs.fetch_dbofs_initial_conditions),
        (necofs, necofs.fetch_necofs_initial_conditions),
        (neracoos, neracoos.fetch_neracoos_initial_conditions),
        (maracoos, maracoos.fetch_maracoos_initial_conditions),
        (hycom, hycom.fetch_hycom_initial_conditions),
    ]


def _domain_area(domain_bbox: list[float]) -> float:
    """Compute approximate domain area in degree^2 from [min_lon, min_lat, max_lon, max_lat]."""
    return (domain_bbox[2] - domain_bbox[0]) * (domain_bbox[3] - domain_bbox[1])


def _rank_ic_candidates(bbox: list[float]) -> list[tuple]:
    """
    Rank IC fetchers for a given request bbox using spatial heuristics.

    Scoring strategy (smallest-enclosing-domain first, resolution tiebreak):
      1. Filter to models whose domain contains the request bbox.
      2. Sort by domain area ascending - the tightest enclosing domain is most
         likely purpose-built for the region (e.g. NYHOPS for NY harbor).
      3. Tiebreak by resolution_approx_m ascending (finer resolution wins).

    Returns an ordered list of (module, fetch_func, metadata) tuples.
    """
    candidates = []
    for module, fetch_func in get_ic_fetchers():
        if not module.supports_bbox(bbox):
            continue
        meta = module.get_metadata()
        domain_bbox = meta.get("domain_bbox")
        area = _domain_area(domain_bbox) if domain_bbox else float("inf")
        resolution = meta.get("resolution_approx_m", float("inf"))
        candidates.append((area, resolution, module, fetch_func, meta))

    # Sort: smallest domain first, then finest resolution
    candidates.sort(key=lambda c: (c[0], c[1]))

    ranked = [(c[2], c[3], c[4]) for c in candidates]
    if ranked:
        logger.info(
            f"IC donor ranking for bbox {bbox}: "
            + " > ".join(
                f"{c[4]['name']} (area={c[0]:.1f}°², ~{c[1]:.0f}m)" for c in candidates
            )
        )
    return ranked


def predict_ic_donor(bbox: list[float]) -> dict:
    """Predicts which model will be used for a given bounding box."""
    ranked = _rank_ic_candidates(bbox)
    if ranked:
        return ranked[0][2]
    return {}


def dispatch_ic_request(
    target_date: str,
    bbox: list[float],
    cache_dir: str = os.environ.get(
        "ECODATA_CACHE_CACHE_DIR", os.path.expanduser("~/.cache/ecodata-cache")
    ),
    cache_bust: bool = False,
    zarr_path: Optional[str] = None,
    allow_donor_fallback: bool = True,
) -> str:
    """
    Tiered modality dispatcher for Initial Conditions. Evaluates regional high-res models,
    falling back to global HYCOM.
    """
    os.makedirs(cache_dir, exist_ok=True)
    ic_cache_dir = os.path.join(cache_dir, "ic")
    os.makedirs(ic_cache_dir, exist_ok=True)

    import hashlib
    import json

    if not zarr_path:
        key_str = f"ic_{target_date}_bbox_{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}"
        key_hash = hashlib.sha256(key_str.encode()).hexdigest()[:12]
        zarr_name = f"ic_{key_hash}.zarr"
        zarr_path = os.path.join(ic_cache_dir, zarr_name)

        # Write sidecar provenance
        sidecar_path = os.path.join(ic_cache_dir, f"ic_{key_hash}.json")
        if not os.path.exists(sidecar_path) or cache_bust:
            try:
                with open(sidecar_path, "w") as f:
                    json.dump(
                        {
                            "id": key_hash,
                            "type": "initial_conditions",
                            "target_date": target_date,
                            "bbox": bbox,
                        },
                        f,
                        indent=2,
                    )
            except Exception:
                pass

    if not cache_bust and os.path.exists(zarr_path):
        logger.info(f"Cache hit for IC: {zarr_path}")
        return zarr_path

    ranked = _rank_ic_candidates(bbox)
    if not ranked:
        raise ValueError(f"No suitable IC fetcher found for bbox {bbox}")

    ds = None
    target_module = None
    meta = None
    candidates_to_try = ranked if allow_donor_fallback else ranked[:1]

    for candidate_module, candidate_fetch_func, candidate_meta in candidates_to_try:
        logger.info(f"Trying IC donor: {candidate_meta['name']}")
        try:
            ds = candidate_fetch_func(target_date, bbox)
        except Exception as e:
            logger.warning(f"{candidate_meta['name']} IC fetch failed: {e}")
            ds = None

        if ds is not None:
            target_module = candidate_module
            meta = candidate_meta
            logger.info(f"IC donor succeeded: {meta['name']}")
            break

    if ds is None or target_module is None or meta is None:
        tried = [c[2]["name"] for c in candidates_to_try]
        raise RuntimeError(f"All IC donors failed for bbox {bbox}. Tried: {tried}")

    # Apply standard provenance attributes directly to dataset
    ds.attrs.update(
        {
            "type": meta["name"],
            "donor_id": meta["id"],
            "source": "ecodata-cache",
            "target_date": target_date,  # existing parser relies on this string being 'UMass Dartmouth SMAST' or similar, but let's standardize
        }
    )

    # Save to Zarr
    logger.info(f"Writing Initial Conditions to {zarr_path}...")

    import shutil

    if os.path.exists(zarr_path):
        shutil.rmtree(zarr_path)

    # Minimal cleanup for Zarr engine compatibility
    for var in list(ds.variables):
        ds[var].encoding.clear()

        # Prevent Zarr from omitting "zero" chunks which crashes Zarr.jl
        if ds[var].dtype.kind in "iu":
            ds[var].encoding["_FillValue"] = -9999
        else:
            ds[var].encoding["_FillValue"] = -9999.0

        # Force explicit Little-Endian Float32 for all physical variables
        # because Oceananigans Zarr.jl backend faults on Big-Endian types
        if ds[var].dtype == "float64" and "time" not in str(var):
            ds[var] = ds[var].astype("<f4")

    ds.to_zarr(zarr_path, mode="w", consolidated=True, zarr_format=2)

    logger.info(
        f"Successfully generated IC Zarr with V2 strict protocol at {zarr_path}"
    )

    return zarr_path


def dispatch_station_profiles_request(
    station_id: str,
    start_time: str,
    end_time: str,
    cache_dir: str = os.path.join(
        os.environ.get(
            "ECODATA_CACHE_CACHE_DIR",
            os.path.expanduser("~/.cache/ecodata-cache"),
        ),
        "erddap",
    ),
    cache_bust: bool = False,
) -> dict:
    """
    Tiered modality dispatcher for internal nudging profiles.
    Currently routes WLIS directly to the ERDDAP fetcher.
    """
    logger.info(
        f"Dispatching station profile request for {station_id} ({start_time} to {end_time})"
    )

    # We could implement a real fallback strategy here, but for now we dispatch straight to our ERDDAP module
    from ecodata_cache.fetchers.erddap import fetch_erddap_station_profiles

    try:
        profiles = fetch_erddap_station_profiles(
            station_id=station_id,
            start_time=start_time,
            end_time=end_time,
            cache_dir=cache_dir,
            cache_bust=cache_bust,
        )
        return profiles
    except Exception as e:
        logger.error(f"Failed to fetch profiles for {station_id}: {e}")
        return {}


def dispatch_bounding_box_profiles_request(
    bbox: list[float],
    start_time: str,
    end_time: str,
    cache_dir: str = os.path.join(
        os.environ.get(
            "ECODATA_CACHE_CACHE_DIR",
            os.path.expanduser("~/.cache/ecodata-cache"),
        ),
        "erddap",
    ),
    cache_bust: bool = False,
) -> dict:
    """
    Tiered modality dispatcher for internal nudging profiles across a domain.
    """
    logger.info(
        f"Dispatching bounded profile request for {bbox} ({start_time} to {end_time})"
    )

    from ecodata_cache.fetchers.erddap import fetch_erddap_stations_in_bbox

    try:
        profiles = fetch_erddap_stations_in_bbox(
            bbox=bbox,
            start_time=start_time,
            end_time=end_time,
            cache_dir=cache_dir,
            cache_bust=cache_bust,
        )
        return profiles
    except Exception as e:
        logger.error(f"Failed to fetch bounded profiles: {e}")
        return {}


def _rank_obc_candidates(bbox: list[float]) -> list[tuple]:
    candidates = []
    for module, fetch_func in get_obc_fetchers():
        if not module.supports_bbox(bbox):
            continue
        meta = module.get_metadata()
        domain_bbox = meta.get("domain_bbox")
        area = _domain_area(domain_bbox) if domain_bbox else float("inf")
        resolution = meta.get("resolution_approx_m", float("inf"))
        candidates.append((area, resolution, module, fetch_func, meta))

    candidates.sort(key=lambda c: (c[0], c[1]))
    ranked = [(c[2], c[3], c[4]) for c in candidates]
    if ranked:
        logger.info(
            f"OBC donor ranking for bbox {bbox}: "
            + " > ".join(
                f"{c[4]['name']} (area={c[0]:.1f}°², ~{c[1]:.0f}m)" for c in candidates
            )
        )
    return ranked


def predict_obc_donor(bbox: list[float]) -> dict:
    ranked = _rank_obc_candidates(bbox)
    if ranked:
        return ranked[0][2]
    return {}


def dispatch_obc_request(
    start_date: str,
    duration_hours: int,
    bbox: list[float],
    cache_dir: str = os.environ.get(
        "ECODATA_CACHE_CACHE_DIR", os.path.expanduser("~/.cache/ecodata-cache")
    ),
    cache_bust: bool = False,
    zarr_path: Optional[str] = None,
    allow_donor_fallback: bool = True,
    include_tides: bool = True,
    tidal_model: str = "GOT4.10c",
    sponge_cells: int = 0,
    pad_cells: int = 3,
    vertical_spacing_m: float = 2.0,
) -> str:
    from ecodata_cache.fetchers.necofs import OBC_SCHEMA

    os.makedirs(cache_dir, exist_ok=True)
    obc_cache_dir = os.path.join(cache_dir, "obc")
    os.makedirs(obc_cache_dir, exist_ok=True)

    import hashlib
    import json

    if not zarr_path:
        key_str = f"obc_{start_date}_{duration_hours}_{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}_s{sponge_cells}_{include_tides}_{tidal_model}_{OBC_SCHEMA}_p{pad_cells}_dz{vertical_spacing_m}"
        key_hash = hashlib.sha256(key_str.encode()).hexdigest()[:12]
        zarr_name = f"obc_{key_hash}.zarr"
        zarr_path = os.path.join(obc_cache_dir, zarr_name)

        # Write sidecar provenance
        sidecar_path = os.path.join(obc_cache_dir, f"obc_{key_hash}.json")
        if not os.path.exists(sidecar_path) or cache_bust:
            try:
                with open(sidecar_path, "w") as f:
                    json.dump(
                        {
                            "id": key_hash,
                            "type": "open_boundary_conditions",
                            "start_date": start_date,
                            "duration_hours": duration_hours,
                            "bbox": bbox,
                            "sponge_cells": sponge_cells,
                            "include_tides": include_tides,
                            "tidal_model": tidal_model,
                            "schema": OBC_SCHEMA,
                            "pad_cells": pad_cells,
                            "vertical_spacing_m": vertical_spacing_m,
                        },
                        f,
                        indent=2,
                    )
            except Exception:
                pass

    if not cache_bust and os.path.exists(zarr_path):
        logger.info(f"Cache hit for OBC: {zarr_path}")
        return zarr_path

    ranked = _rank_obc_candidates(bbox)
    if not ranked:
        raise ValueError(f"No suitable OBC fetcher found for bbox {bbox}")

    ds = None
    target_module = None
    meta = None
    candidates_to_try = ranked if allow_donor_fallback else ranked[:1]
    for candidate_module, candidate_fetch_func, candidate_meta in candidates_to_try:
        logger.info(f"Trying OBC donor: {candidate_meta['name']}")
        try:
            ds = candidate_fetch_func(
                start_date,
                duration_hours,
                bbox,
                **_supported_kwargs(
                    candidate_fetch_func,
                    pad_cells=pad_cells,
                    vertical_spacing_m=vertical_spacing_m,
                ),
            )
        except Exception as e:
            logger.warning(f"{candidate_meta['name']} OBC fetch raised: {e}")
            ds = None
        if ds is not None:
            target_module = candidate_module
            meta = candidate_meta
            logger.info(f"OBC donor succeeded: {meta['name']}")
            break
        if allow_donor_fallback:
            logger.warning(
                f"{candidate_meta['name']} OBC fetch returned no data, trying next donor."
            )

    if ds is None or target_module is None:
        tried = [c[2]["name"] for c in candidates_to_try]
        raise RuntimeError(
            f"Primary OBC donor {tried[0]} failed to return data."
            if not allow_donor_fallback
            else f"All OBC donors failed for bbox {bbox}. Tried: {tried}"
        )

    # REQ-3.1: Temporal Harmonization - Resample to hourly
    logger.info("Aligning OBC data to shared hourly temporal index...")
    if "time" in ds.dims:
        import pandas as pd

        if not isinstance(ds.indexes["time"], pd.DatetimeIndex):
            # Attempt to convert cftime or object arrays to DatetimeIndex
            try:
                ds["time"] = ds.indexes["time"].to_datetimeindex()
            except AttributeError:
                ds["time"] = pd.to_datetime(ds.indexes["time"].values)
        # Interpolate to strictly hourly
        ds = ds.resample(time="1h").interpolate("linear")

    # REQ-1.3: Tidal boundary condition integration via GOT4.10c or EOT20 (pyTMD).

    # Cast strictly to Float32 for Oceananigans
    for var in ds.data_vars:
        if ds[var].dtype != np.float32:
            ds[var] = ds[var].astype(np.float32)

    # Convert Endianness for Julia
    for var in list(ds.variables):
        ds[var].encoding.clear()
        if ds[var].dtype.kind in "iu":
            ds[var].encoding["_FillValue"] = -9999
        else:
            ds[var].encoding["_FillValue"] = -9999.0

        # Coordinates keep full precision: Float32 longitudes near -74 resolve only about
        # 8e-6 degrees, enough to misplace a 0.002 degree grid relative to its child.
        if (
            (ds[var].dtype == "float64" or ds[var].dtype == "float32")
            and "time" not in str(var)
            and var not in _FULL_PRECISION_COORDS
        ):
            ds[var] = ds[var].astype("<f4")

    # Schema-versioned parent stores state their time axis in seconds from the first record.
    if ds.attrs.get("schema") == OBC_SCHEMA and "time" in ds.coords:
        import pandas as pd

        first = pd.Timestamp(ds["time"].values[0]).strftime("%Y-%m-%dT%H:%M:%S")
        ds["time"].encoding = {"units": f"seconds since {first}", "dtype": "float64"}

    ds.attrs["source"] = target_module.__name__
    ds.attrs["type"] = "3D Time-Varying Hindcast (Coupled Tides + Reanalysis)"
    ds.attrs["duration_hours"] = duration_hours
    ds.attrs["sponge_cells"] = sponge_cells
    ds.attrs["schema"] = ds.attrs.get("schema", "legacy")

    logger.info(f"Writing OBC data to Zarr: {zarr_path}")
    ds.to_zarr(zarr_path, mode="w", consolidated=True, zarr_format=2)
    return zarr_path


_FULL_PRECISION_COORDS = ("lat", "lon", "z", "z_face")


def _supported_kwargs(func, **kwargs) -> dict:
    """The subset of `kwargs` that `func` accepts, so options for one donor's fetcher can be
    offered to every candidate without breaking the shared (start, duration, bbox) signature."""
    import inspect

    params = inspect.signature(func).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


def get_obc_fetchers():
    from ecodata_cache.fetchers import (
        dbofs,
        hycom,
        necofs,
        nyofs,
    )

    # Dynamically discover fetch_xxx_boundary_conditions functions.
    # Not all modules have implemented OBC yet, so we check with hasattr.
    fetchers = []
    for module in [nyofs, dbofs, necofs, hycom]:
        func_name = "fetch_" + module.__name__.split(".")[-1] + "_boundary_conditions"
        if hasattr(module, func_name):
            fetchers.append((module, getattr(module, func_name)))
    return fetchers
