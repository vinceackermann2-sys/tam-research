"""CPU-only deconfounded key/value binding task (issue #1306)."""

from .task import BindingRequiredBatch, counterfactual_anchor_queries, make_binding_required_batch

__all__ = ["BindingRequiredBatch", "counterfactual_anchor_queries", "make_binding_required_batch"]
