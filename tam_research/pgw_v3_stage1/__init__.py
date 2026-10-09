"""Frozen PGW-v3 Stage-1 deterministic data/oracle only; no training."""

from .oracle import (
    RetrievalExample, assert_panel_disjoint, fixture_panel, make_example,
    oracle_latest_value,
)

__all__ = [
    "RetrievalExample", "assert_panel_disjoint", "fixture_panel",
    "make_example", "oracle_latest_value",
]
