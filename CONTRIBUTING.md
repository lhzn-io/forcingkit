# Contributing to forcingkit

Thank you for considering contributing to `forcingkit`. This repository is a community resource for providing clean, standardized ocean and atmospheric forcing boundaries for high-fidelity coastal hydrodynamic models. Whether you're fixing a bug, improving documentation, or adding a new data node, contributions are welcome.

## Development Workflow

### 1. Environment Setup

We use `uv` for Python package management.

1. Install `uv`.
2. Sync the environment: `uv sync`
3. We use `pre-commit` to ensure code format standardization. Run `pre-commit install` to set up your git hooks.

### 2. Making Changes

- All code must pass `ruff` (for formatting and linting) and `mypy` (for typing).
- Ensure your changes are covered by tests where applicable (we use `pytest`). Tests are located in the `tests/` directory.
- `forcingkit.fetchers` logic handles API integrations. If adding a new telemetry source (e.g., a new regional IOOS node), follow the patterns established in `ndbc.py` (observations) or `dbofs.py` (a parent ocean).
- Keep the Sphinx documentation up to date.

### 3. Pull Requests

1. Create a descriptive branch name: `git checkout -b feature/ioos-integration`
2. Make your commits clear and logical.
3. Open a Pull Request.

### 4. Code of Conduct

We encourage an inclusive, patient, and highly respectful environment for all collaborators.
