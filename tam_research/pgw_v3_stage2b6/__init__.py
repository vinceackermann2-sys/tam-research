"""PGW-v3 Stage-2B6 immutable corpus and two-arm fairness gate (no training)."""

from .readiness import (
    SEALED_ROOTS, OPEN_SCIENCE_BLOCKERS, ReadonlyScientificGate,
    verify_sealed_corpus, evaluate_verified_predictions, readonly_research_gate,
)

__all__ = [
    "SEALED_ROOTS", "OPEN_SCIENCE_BLOCKERS", "ReadonlyScientificGate",
    "verify_sealed_corpus", "evaluate_verified_predictions", "readonly_research_gate",
]
