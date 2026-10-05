# Tested on: NVIDIA Jetson AGX Orin (JetPack 6.x / l4t r36) and x86_64 (Ubuntu 24.04)
# Override BASE_IMAGE at build time for your target platform:
#   Jetson Orin:  --build-arg BASE_IMAGE=nvcr.io/nvidia/l4t-base:r36.2.0
#   x86_64/WSL2:  --build-arg BASE_IMAGE=ubuntu:24.04  (default)
ARG BASE_IMAGE=ubuntu:24.04
FROM ${BASE_IMAGE}

WORKDIR /app

# Install python and dev dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    gcc \
    curl \
    libeccodes-dev \
    && rm -rf /var/lib/apt/lists/*

# Alias python to python3
RUN ln -s /usr/bin/python3 /usr/bin/python

# Install UV
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="/usr/local/bin" sh

# Configure environment
ENV UV_PROJECT_ENVIRONMENT="/app/.venv" \
    PYTHONPATH="/app/src:/app/service"

COPY pyproject.toml uv.lock ./
COPY README.md ./
COPY src/ /app/src/
COPY service/ /app/service/
COPY static/ /app/static/

# Downgrade python was done in pyproject.toml.
# We use --frozen to respect uv.lock
RUN uv sync --frozen --no-dev --compile-bytecode

# Fix ownership so non-root users can write to .venv
ARG UID=1000
ARG GID=1000
RUN chown -R ${UID}:${GID} /app

# Default port 9598
EXPOSE 9598

CMD ["uv", "run", "python", "-m", "forcingkit_serve.main", "--host", "0.0.0.0", "--port", "9598"]
