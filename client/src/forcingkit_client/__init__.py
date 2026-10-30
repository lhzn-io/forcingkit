"""Lightweight client for a forcingkit service."""

from .client import AsyncClient, Client, ForcingkitError
from .schema import SchemaError, normalise
from .trajectory import interpolate_points, interpolate_trajectory, trajectory_window

__all__ = [
    "AsyncClient",
    "Client",
    "ForcingkitError",
    "SchemaError",
    "interpolate_points",
    "interpolate_trajectory",
    "normalise",
    "trajectory_window",
]

__version__ = "0.1.0"
