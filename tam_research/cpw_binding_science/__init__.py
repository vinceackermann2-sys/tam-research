"""Pre-authority, zero-GPU corrected-binding science harness for CPW #1321.

Only copied/selected existing model constructors. No scientific seeds,
paid-run identities, Modal integration, scaling, or breakthrough authority.
"""
from .models import ARMS, PARAMETERS, build_model, query_logits
from .evaluation import (
    cpu_train_integrity_step,
    evaluate_counterfactual_groups,
    evaluate_distances,
)

__all__ = [
    "ARMS", "PARAMETERS", "build_model", "query_logits",
    "cpu_train_integrity_step", "evaluate_counterfactual_groups",
    "evaluate_distances",
]
