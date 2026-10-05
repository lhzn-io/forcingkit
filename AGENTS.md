# Ecodata Cache Agent Guide

This file is the canonical, vendor-neutral entry point for agent
guidance in this repository.

## Scope

- Prefer standards and shared conventions over vendor-specific agent layouts.
- Treat `AGENTS.md` and `.agents/` as the primary instruction surface.
- Keep agent-specific files as thin compatibility shims when needed by a tool.

## Tool Discovery Map

- Canonical source of truth: `AGENTS.md` and `/.agents/`.
- Claude Code:
  - Compatibility shim: `/.claude/CLAUDE.md`.
- VS Code GitHub Copilot Agent:
  - Compatibility shim: `/.github/copilot-instructions.md`.
- Rule for all tool-specific files: redirect to this file and
  `/.agents/`, and only add minimal tool-local behavior.

## Repository Focus

Ecodata Cache is a Python microservice that fetches, harmonizes, regrids, and serves
real-time and historical forcing to the `coastal-sim` Julia physics engine: the parent
ocean (`/api/v1/obc`, schema z-v2), the HRRR atmosphere (`/api/v1/atmosphere`), tides,
and station telemetry and NDBC observations for validation.

## Core Rules

- **Dependency management**: Always use `uv run` - never `pip install` directly.
- **Linting/formatting**: Run `uv run pre-commit run --files <files>` (ruff + mypy)
  before staging any changes.
- **Testing**: Run tests with `uv run python -m pytest tests/unit/` (unit) or
  `uv run python -m pytest tests/integration/` (live API, roughly 10 minutes).
  Use `python -m pytest` rather than the bare `pytest` entry point: the latter can
  resolve to a system interpreter outside the venv. `tests/conftest.py` puts
  `src/` and `service/` on `sys.path`, so no `PYTHONPATH` export is needed.
- **Commits**: Conventional commits (`feat:`, `fix:`, `docs:`, `chore:`).
  Author and committer are always the human contributor (`dfry-lhzn <dfry@lhzn.io>`,
  from `git config`), so the log shows who was behind each change. No `Co-Authored-By:`
  trailers or non-human identities: GitHub parses those into the Contributors list,
  which is reserved for people. Noting the agent harness or model that collaborated
  is welcome as plain text in the commit body. Verify with `git var GIT_AUTHOR_IDENT`
  before the first commit in a shell.
- **Git staging**: Stage files explicitly (`git add <file>`). Never `git add .`.
- **Never commit without approval**: Present the proposed change set and commit message
  to the user and wait for explicit confirmation before running any `git commit`.

## Domain Constraints

- **Grid normalization**: Elevations and water levels are positive up. Normalize
  external dataset quirks at the fetcher tier - never let them propagate into the
  dispatcher.
- **OPeNDAP access**: Use the `pydap` engine (`engine="pydap"`) for all THREDDS/OPeNDAP
  endpoints. Prefer `.csvp` endpoints for tabular time-series to avoid NetCDF overhead.
- **C-grid stagger**: NYOFS (POM) uses an Arakawa C-grid. Interpolation to rho-points
  must be done in the fetcher before returning data to the dispatcher.
- **Precision**: Output arrays should default to `float32` (`<f4`) and use Little-Endian
  endianness for compatibility with `Zarr.jl` and `Oceananigans.jl`.

## Where To Read Next

- `.agents/ecodata-cache.md`: architecture pipeline, fetcher tier, testing strategy
  and gotchas

## Compatibility

If a tool only reads `.github/copilot-instructions.md`, `CLAUDE.md`, or another
vendor-specific file, that file redirects here and adds only the minimum compatibility
details required by that tool.
