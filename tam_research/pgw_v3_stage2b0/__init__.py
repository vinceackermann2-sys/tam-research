"""Stage-2B0 CPU-only predictive auxiliary loss: structural derivative probe."""

from .auxiliary import (
    causal_event_hidden,
    predictor_auxiliary_loss,
    predictor_step_errors,
)

__all__ = [
    "causal_event_hidden",
    "predictor_auxiliary_loss",
    "predictor_step_errors",
]
