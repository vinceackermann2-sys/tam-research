"""Stage-2B1: static PGW-v3 data & derivative audits, NOT training."""

from .manifest import static_fixture_manifest
from .capacity import gradient_capacity_preflight

__all__ = ["static_fixture_manifest", "gradient_capacity_preflight"]
