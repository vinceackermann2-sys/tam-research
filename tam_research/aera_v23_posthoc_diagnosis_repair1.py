from __future__ import annotations

"""Repair1 binding for issue #344.

The first #341/#343 diagnostic failed before any measurement because the
orchestration function was decorated with ``torch.inference_mode``.  PyTorch
therefore constructed the model parameters as inference tensors, which do not
expose ``_version`` counters.  The underlying function body is otherwise the
frozen diagnostic we want.

For auditability we leave the failed #343 module untouched and export its exact
undecorated body via ``__wrapped__``.  All individual measurement routines called
by that body remain independently decorated with ``torch.inference_mode``.
"""

from . import aera_v23_posthoc_diagnosis as _base

if not hasattr(_base.run_posthoc_diagnosis, "__wrapped__"):
    raise RuntimeError("repair1 expected the historical inference-mode wrapper")

run_posthoc_diagnosis = _base.run_posthoc_diagnosis.__wrapped__
frozen_protocol = _base.frozen_protocol
parameter_versions = _base.parameter_versions
scaled_memory_reads = _base.scaled_memory_reads

REPAIR_ISSUE = 344
SOURCE_FAILED_TRIGGER = 343
SEMANTIC_CHANGE = "construct_and_load_model_outside_inference_mode_only"
