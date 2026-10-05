# forcingkit Reference

## Mission

Python microservice providing real-time and historical forcing (the parent ocean and
the HRRR atmosphere) and validation observations to the `coastal-sim` Julia physics
engine.

## Environment

- **Python 3.11+** via `uv`. Always use `uv run` — never `pip install`.
- **FastAPI** microservice in `service/forcingkit_serve/`.
- Tests run with `uv run pytest`.

## Architecture

### Core Pipeline

```text
HTTP Request → service/forcingkit_serve/main.py
             → src/forcingkit/dispatcher.py
             → fetchers/*.py
             → Zarr (~/.cache/forcingkit/)
```

- **Entry point**: `service/forcingkit_serve/main.py` — FastAPI application.
- **Dispatcher**: `src/forcingkit/dispatcher.py` — interprets
  requests and routes to the appropriate fetcher via tiered fallback.
  Uses `_rank_obc_candidates(bbox)` to select the best parent by
  resolution and domain overlap.
- **Fetchers**: `src/forcingkit/fetchers/` — isolated modules per data source.

## Fetcher Tier (Parent Ocean Priority Order)

| Fetcher | Resolution | Domain | Notes |
|---|---|---|---|
| `nyofs.py` | ~100 m | NY/NJ Harbor | POM curvilinear grid, C-grid stagger |
| `necofs.py` | ~200 m | New England | FVCOM unstructured |
| `hycom.py` | ~9 km | Global | Final fallback (Operational + Historical) |

*Note: Because the architecture is modular, a future roadmap item includes the integration of TPXO (via `tpxo.py`) given usage rights.*

Atmospheric forcing: HRRR from 2014-07-30 (`hrrr_atmosphere.py`, `/api/v1/atmosphere`).
Earlier runs use ERA5 through NumericalEarth in coastal-sim, not this service.

## Testing

```bash
# Unit tests (no network I/O, fast)
uv run pytest tests/unit/ -v

# Single test file
uv run pytest tests/unit/test_nyofs.py -v

# Integration tests (live APIs, slower)
uv run pytest tests/integration/ -v

# Pre-commit checks (ruff + mypy) on modified files
uv run pre-commit run --files <file1> <file2> ...
```

## Commands

```bash
# Sync dependencies
uv sync

# Start data service
uv run python service/run_server.py

# Run one-off fetch
uv run python -c "from forcingkit.fetchers import nyofs; print(nyofs.get_metadata())"
```

## Gotchas

- **Telemetry performance**: Use `.csvp` endpoints for ERDDAP
  tabular time-series to avoid heavy NetCDF binary parsing overhead.
- **OPeNDAP engine**: Always pass `engine="pydap"` to `xr.open_dataset` for THREDDS
  endpoints. The default netcdf4 engine fails on remote DAP URLs.
- **NYOFS FMRC vs NCEI tiering**: FMRC aggregation covers rolling 7-day window
  (< 31 days). Older data requires per-hour NCEI file enumeration with naming
  conventions that changed on 2024-09-09.
- **C-grid interpolation**: NYOFS u/v are on Arakawa C-grid face
  points. After `.where(mask, drop=True)`, the subset is already
  zero-indexed — use the `_c_grid_to_rho()` helper that averages
  adjacent cells on the subset array, not the old absolute-index
  approach.
- **Mocking strategy**: Dispatcher unit tests check tuple/dict return
  boundaries carefully. When modifying mocked fetchers, track
  keyword-argument vs positional argument boundaries.
- **Grid normalization**: `coastal-sim` expects elevations positive
  up (LMSL/NAVD88). Normalize any inverted datasets in the fetcher
  tier before the dispatcher sees them.
- **Cache keys**: Cache Zarr keys are deterministic hashes of bbox
  - time window. Changing fetcher output schema (variable names, dims)
  will miss existing cache entries — purge
  `~/.cache/forcingkit/` when making breaking schema changes.
