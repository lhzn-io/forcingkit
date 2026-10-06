# <img src="https://raw.githubusercontent.com/lhzn-io/forcingkit/main/static/logo.svg" height="36" valign="middle" alt="Logo" /> forcingkit

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

<div align="center">
  <img src="https://raw.githubusercontent.com/lhzn-io/forcingkit/main/docs/source/_static/forcingkit-noreaster-wind.gif" alt="Animation of the HRRR 10 m wind from the NY Bight to Cape Cod, every 3 hours from 2026-09-22 00:00 to 2026-09-29 12:00 UTC" width="720" />
  <br />
  <sub>HRRR 10 m wind served by forcingkit through the 26-27 September 2026 nor'easter, NY Bight to Cape Cod, from the calm of 22 September to the calm of 29 September, every 3 hours (peak 25.9 m/s). The forcingkit viewer's time slider, played.</sub>
</div>

Spatiotemporal forcing for computational Earth-system models: selects, regrids and serves model-ready time series with provenance.

`forcingkit` sits between operational and archive data providers (NOAA HRRR, NECOFS, NOAA OFS, HYCOM, NDBC, CO-OPS) and model codes such as `Oceananigans.jl`. It delivers a model's parent ocean on true z levels (`/api/v1/obc`) and its HRRR surface atmosphere on a regular grid (`/api/v1/atmosphere`), plus station observations for validation, as schema-versioned Zarr stores that record their sources.

Documentation: <https://forcingkit.docs.lhzn.io>.

## Goals

The primary goal of forcingkit is to provide clean, standardized ocean and atmospheric forcing boundaries for high-fidelity coastal hydrodynamic models. We achieve this by:

- Operating as an invisible, self-managing proxy service that orchestrates downloading and caching of remote data.
- Stripping away the quirks of individual forecasting nodes by harmonizing temporal alignment, grid structures, and variable naming conventions.
- Generating deterministic cache hashes to ensure simulation runs are 100% physically reproducible over time.

## Features

- **Decoupled System Architecture**: The microservice guarantees reproducible datasets by deterministic hashing, providing standardized local representations of external APIs.
- **Continuous Coupled Hindcasts**: Support for 45-to-60-day continuous hindcasts by dynamically stitching historical and operational HYCOM experiments.
- **Temporal Alignment**: Automated CF-compliant hourly interpolation across heterogeneous upstream timestamps and grid frequencies.

## Current Coverage

- **Oceanic Forcing**: NYOFS, NECOFS, DBOFS, HYCOM (global, 1994 to present).
- **Atmospheric Forcing**: HRRR (3 km, hourly, 2014-07-30 to present), regridded to 0.03 degrees.

## Roadmap

- **Coverage first**: a second global parent (Copernicus GLO12, GLORYS12 for 1993 on, NOAA RTOFS), the remaining NOAA forecast systems (West Coast, Gulf, Chesapeake, Great Lakes, Alaska), the Doppio reanalysis for 2007 to 2024 in the Northeast, HRRR Alaska and ECMWF IFS for the atmosphere, and a first European ocean source from Copernicus Marine (IBI or the North West Shelf).
- **Resolution second**: the NYOFS fine grid, the Monterey Bay nests of WCOFS, and SFBOFS inside San Francisco Bay.
- **NYOFS and DBOFS as z-v3 parents**: both on true depths with geographic axes.
- **Forecast mode and cache policy**: HRRR forecast cycles, and invalidation of forecast-built stores when a newer cycle is published.

The full list, with what each candidate offers, is at <https://forcingkit.docs.lhzn.io/roadmap.html>. Contributions are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md).

## Architecture overview

The service exposes HTTP APIs (via FastAPI/Uvicorn) which:

1. Deterministically hash bounding boxes and time windows to generate an automated local Zarr cache (`~/.cache/forcingkit`).
2. Dispatch fetch operations via tiered fallback chains for high-availability. (e.g. DBOFS or NYOFS -> NECOFS -> HYCOM for the parent ocean).
3. Standardize dimensions, bounds, floating-point precision, and timezone offsets, providing a direct ingestion path for downstream coupled engines.

## Setup

```bash
uv sync
pre-commit install

# Development (single worker, run from the repository root):
uv run uvicorn service.forcingkit_serve.main:app --port 9598

# Service launcher (supports --host/--port/--workers/--reload):
uv run python service/run_server.py --host 0.0.0.0 --port 9598 --workers 4
```

## Docker Environment

The `Dockerfile` defaults to `ubuntu:24.04`, which is multi-arch. This service is
CPU-only, so the default works unchanged on x86_64 and on arm64 edge hardware;
`BASE_IMAGE` exists for cases that need a different base.

```bash
# x86_64 and arm64 alike (default)
docker build -t forcingkit .

docker run --rm -p 9598:9598 forcingkit
```

On NVIDIA Jetson under JetPack 7 (L4T R39, Ubuntu 24.04), keep the default base.
An earlier revision of this file recommended
`--build-arg BASE_IMAGE=nvcr.io/nvidia/l4t-base:r36.2.0`; that was wrong on two
counts. The `l4t-*` image family was never published beyond `r36.4.0`
(JetPack 6.1), so there is no R39 equivalent to move to, and this service needs
no CUDA in the image regardless. Where a CUDA base genuinely is required, the
JetPack 7 path is the unified `nvcr.io/nvidia/cuda:<ver>-*-ubuntu24.04` images,
which carry real arm64 manifests, combined with CDI device injection
(`--device nvidia.com/gpu=all`). Note that `--gpus all` is rejected on Tegra.

## Licensing

This project is licensed under the Apache 2.0 License.
