"""CPU-only PGW-v3 Stage-2A token to workspace structural interface."""

from .model import (
    PGWV3Stage2A, PGWV3Stage2AConfig, batch_verified_examples,
    instantiated_parameter_count,
)

__all__ = [
    "PGWV3Stage2A", "PGWV3Stage2AConfig", "batch_verified_examples",
    "instantiated_parameter_count",
]
