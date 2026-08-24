from __future__ import annotations

"""Implementation-only efficiency repair after the seed8391 timeout in issue #326.

This module does not change AERA-v22 architecture or scientific objectives.  It
corrects the #324 memory-auxiliary budget from an accidental 1024 samples per
*gradient-accumulation microbatch* to the preregistered maximum of 1024 samples
per *optimizer step*.  The inherited trainer uses four microbatches/step, so the
correct cap is 256 sampled adjacent events per microbatch.

It also exposes a torch.compile factory for the exact existing dual-delta update.
The eager v22 function remains the correctness oracle and production fallback.
"""

from typing import Any, Callable

import torch

from . import aera_hardware_core_v22 as core
from . import aera_real_language_v19 as v19
from . import aera_real_language_v22 as v22
from .aera_real_language import GRAD_ACCUM

# Frozen #324 budget is per OPTIMIZER step, not per microbatch.
MAX_MEMORY_AUX_EVENTS_PER_OPTIMIZER_STEP = 1024
if MAX_MEMORY_AUX_EVENTS_PER_OPTIMIZER_STEP % GRAD_ACCUM:
    raise RuntimeError("v22 memory auxiliary budget must divide GRAD_ACCUM exactly")
MAX_MEMORY_AUX_EVENTS_PER_MICROBATCH = (
    MAX_MEMORY_AUX_EVENTS_PER_OPTIMIZER_STEP // GRAD_ACCUM
)

# Re-export the unchanged real-language architecture/training schedule expected by
# the inherited harness.
CPU_DIAGNOSTIC_SEED = v22.CPU_DIAGNOSTIC_SEED
CHUNK_SIZE = v22.CHUNK_SIZE
DENSE_WARMUP_STEPS = v22.DENSE_WARMUP_STEPS
ROUTER_CALIBRATION_END = v22.ROUTER_CALIBRATION_END
SPARSE_CALIBRATION_EVERY = v22.SPARSE_CALIBRATION_EVERY
STAGE_DIFFICULTY_WEIGHT = v22.STAGE_DIFFICULTY_WEIGHT
STAGE_BUDGET_WEIGHT = v22.STAGE_BUDGET_WEIGHT
STAGE_POLARIZATION_WEIGHT = v22.STAGE_POLARIZATION_WEIGHT
ADDRESS_TEMPERATURE = v22.ADDRESS_TEMPERATURE
ADDRESS_CONTRASTIVE_WEIGHT = v22.ADDRESS_CONTRASTIVE_WEIGHT
PAYLOAD_TOKEN_WEIGHT = v22.PAYLOAD_TOKEN_WEIGHT
LATENT_PAYLOAD_WEIGHT = v22.LATENT_PAYLOAD_WEIGHT

phase_for_step = v22.phase_for_step
route_mode_for_step = v22.route_mode_for_step
router_policy_for_step = v22.router_policy_for_step
per_chunk_language_loss = v22.per_chunk_language_loss
build_aera = v22.build_aera
build_transformer = v22.build_transformer
set_optional_stage_router_trainable = v22.set_optional_stage_router_trainable


def memory_auxiliary_terms(
    model,
    tokens: torch.Tensor,
    *,
    step: int,
    max_events: int = MAX_MEMORY_AUX_EVENTS_PER_MICROBATCH,
) -> dict[str, torch.Tensor]:
    """Exact v22 auxiliary with the corrected per-microbatch sample cap."""
    if max_events > MAX_MEMORY_AUX_EVENTS_PER_MICROBATCH:
        raise ValueError(
            "memory auxiliary exceeds corrected per-microbatch budget: "
            f"{max_events} > {MAX_MEMORY_AUX_EVENTS_PER_MICROBATCH}"
        )
    return v22.memory_auxiliary_terms(
        model,
        tokens,
        step=step,
        max_events=max_events,
    )


def aera_matched_loss(model, x: torch.Tensor, y: torch.Tensor, *, step: int):
    """Inherited v19 loss plus unchanged conflict-free memory objective.

    This intentionally bypasses v22.aera_matched_loss because that function's
    default was captured at 1024 events per microbatch.  Every scientific weight,
    target, detach boundary and routing rule is otherwise identical.
    """
    inherited_total, terms, mode, phase = v19.aera_matched_loss(
        model,
        x,
        y,
        step=step,
    )
    memory_terms = memory_auxiliary_terms(model, x, step=step)
    total = (
        inherited_total
        + ADDRESS_CONTRASTIVE_WEIGHT
        * memory_terms["memory_address_contrastive_loss"]
        + PAYLOAD_TOKEN_WEIGHT * memory_terms["memory_payload_token_loss"]
    )
    merged = dict(terms)
    merged.update(memory_terms)
    merged["total"] = total
    return total, merged, mode, phase


def make_compiled_dual_delta_update(
    *,
    backend: str | Callable[..., Any] | None = None,
    mode: str | None = None,
):
    """Compile the exact eager v22 recurrence without changing its equations."""
    kwargs: dict[str, Any] = {
        "fullgraph": True,
        "dynamic": False,
    }
    if backend is not None:
        kwargs["backend"] = backend
    if mode is not None:
        kwargs["mode"] = mode
    return torch.compile(core.interference_corrected_dual_delta_update, **kwargs)


def cpu_preflight() -> dict[str, Any]:
    inherited = v22.cpu_preflight()
    return {
        **inherited,
        "version": "aera-v22-exact-semantics-training-efficiency-repair",
        "gpu_authorized": False,
        "scientific_architecture_changed": False,
        "scientific_objective_changed": False,
        "gradient_accumulation_microbatches": GRAD_ACCUM,
        "memory_aux_events_per_microbatch": MAX_MEMORY_AUX_EVENTS_PER_MICROBATCH,
        "memory_aux_events_per_optimizer_step": (
            MAX_MEMORY_AUX_EVENTS_PER_MICROBATCH * GRAD_ACCUM
        ),
        "frozen_memory_aux_optimizer_step_cap": (
            MAX_MEMORY_AUX_EVENTS_PER_OPTIMIZER_STEP
        ),
        "eager_dual_delta_remains_reference": True,
        "seed8391_rerun_authorized": False,
    }


def efficiency_protocol() -> dict[str, Any]:
    return {
        "source_failure": "issue #326 runtime timeout before scientific evaluation",
        "architecture_changed": False,
        "memory_equations_changed": False,
        "candidate_order_changed": False,
        "write_strengths_changed": False,
        "objective_weights_changed": False,
        "aux_budget_bug_repaired": True,
        "gradient_accumulation": GRAD_ACCUM,
        "memory_aux_events_per_microbatch": MAX_MEMORY_AUX_EVENTS_PER_MICROBATCH,
        "memory_aux_events_per_optimizer_step": (
            MAX_MEMORY_AUX_EVENTS_PER_MICROBATCH * GRAD_ACCUM
        ),
        "frozen_optimizer_step_cap": MAX_MEMORY_AUX_EVENTS_PER_OPTIMIZER_STEP,
        "compiled_candidate": "torch.compile of exact eager dual-delta recurrence",
        "eager_reference_preserved": True,
        "gpu_training_authorized": False,
        "100m_authorized": False,
    }
