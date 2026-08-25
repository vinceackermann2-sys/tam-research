from __future__ import annotations

"""Frozen AERA-v23 real-language/systems binding for issue #333.

This module changes only the model class used by the already-frozen v22
real-language efficiency harness: HardwareAwareAERATextLMV23 replaces v22 so
completed 256-token chunks physically execute 16/255 event-pair writes.  The
real-language geometry, routing schedule, loss weights, memory auxiliary budget,
M/P dual-delta equations and evaluation protocol remain unchanged.

Importing this module does not authorize corpus training.
"""

from typing import Any

import torch

from . import aera_real_language_v12 as v12
from . import aera_real_language_v22 as v22
from . import aera_real_language_v22_efficiency as v22eff
from .aera_hardware_core_v23 import (
    HardwareAwareAERATextLMV23,
    BudgetedSparseDualDeltaFastMemoryStage,
    sparse_dual_delta_memory_protocol,
    sparse_write_budget,
)
from .aera_real_language import GRAD_ACCUM, parameter_accounting

CPU_DIAGNOSTIC_SEED = 8421
CHUNK_SIZE = v22eff.CHUNK_SIZE
DENSE_WARMUP_STEPS = v22eff.DENSE_WARMUP_STEPS
ROUTER_CALIBRATION_END = v22eff.ROUTER_CALIBRATION_END
SPARSE_CALIBRATION_EVERY = v22eff.SPARSE_CALIBRATION_EVERY
STAGE_DIFFICULTY_WEIGHT = v22eff.STAGE_DIFFICULTY_WEIGHT
STAGE_BUDGET_WEIGHT = v22eff.STAGE_BUDGET_WEIGHT
STAGE_POLARIZATION_WEIGHT = v22eff.STAGE_POLARIZATION_WEIGHT
ADDRESS_TEMPERATURE = v22eff.ADDRESS_TEMPERATURE
ADDRESS_CONTRASTIVE_WEIGHT = v22eff.ADDRESS_CONTRASTIVE_WEIGHT
PAYLOAD_TOKEN_WEIGHT = v22eff.PAYLOAD_TOKEN_WEIGHT
LATENT_PAYLOAD_WEIGHT = v22eff.LATENT_PAYLOAD_WEIGHT
MAX_MEMORY_AUX_EVENTS_PER_OPTIMIZER_STEP = (
    v22eff.MAX_MEMORY_AUX_EVENTS_PER_OPTIMIZER_STEP
)
MAX_MEMORY_AUX_EVENTS_PER_MICROBATCH = v22eff.MAX_MEMORY_AUX_EVENTS_PER_MICROBATCH

phase_for_step = v22eff.phase_for_step
route_mode_for_step = v22eff.route_mode_for_step
router_policy_for_step = v22eff.router_policy_for_step
per_chunk_language_loss = v22eff.per_chunk_language_loss
build_transformer = v22eff.build_transformer


def build_aera(device: torch.device) -> HardwareAwareAERATextLMV23:
    return HardwareAwareAERATextLMV23(v12.aera_v12_config()).to(device)


def set_optional_stage_router_trainable(
    model: HardwareAwareAERATextLMV23,
    trainable: bool,
) -> None:
    model.set_optional_stage_routers_trainable(trainable)


def memory_auxiliary_terms(
    model: HardwareAwareAERATextLMV23,
    tokens: torch.Tensor,
    *,
    step: int,
    max_events: int = MAX_MEMORY_AUX_EVENTS_PER_MICROBATCH,
) -> dict[str, torch.Tensor]:
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


def aera_matched_loss(
    model: HardwareAwareAERATextLMV23,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    step: int,
):
    inherited_total, terms, mode, phase = v22eff.v19.aera_matched_loss(
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


def _production_sparse_smoke(model: HardwareAwareAERATextLMV23) -> dict[str, Any]:
    stage = model.stages[0]
    if not isinstance(stage, BudgetedSparseDualDeltaFastMemoryStage):
        raise RuntimeError("v23 production build did not install sparse memory stage")
    stage.memory.set_differentiable_pretraining(True)
    g = torch.Generator().manual_seed(CPU_DIAGNOSTIC_SEED + 2)
    events = torch.randn(
        1,
        CHUNK_SIZE,
        model.cfg.d_model,
        generator=g,
        requires_grad=True,
    )
    out, next_state, _ = stage.forward_chunk(
        events,
        None,
        hard=False,
        update_memory=True,
    )
    expected = sparse_write_budget(CHUNK_SIZE - 1)
    if stage.last_candidate_count != CHUNK_SIZE - 1:
        raise RuntimeError("v23 CPU smoke candidate count mismatch")
    if stage.last_selected_count != expected:
        raise RuntimeError(
            f"v23 CPU smoke selected {stage.last_selected_count}, expected {expected}"
        )
    loss = (
        out.float().square().mean()
        + next_state.memory.matrix.float().square().mean()
        + 0.01 * next_state.memory.inverse_key_covariance.float().square().mean()
    )
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("v23 CPU sparse-write smoke produced nonfinite loss")
    loss.backward()
    pair_grad = stage.pair_write_gate.weight.grad
    if pair_grad is None or not bool(torch.isfinite(pair_grad).all()):
        raise RuntimeError("v23 sparse selector did not receive finite gradient")
    memory_grad = stage.memory.k.weight.grad
    if memory_grad is None or not bool(torch.isfinite(memory_grad).all()):
        raise RuntimeError("v23 sparse memory did not receive finite gradient")
    result = {
        "candidates": stage.last_candidate_count,
        "selected_writes": stage.last_selected_count,
        "selected_fraction": stage.last_selected_count / stage.last_candidate_count,
        "pair_gate_grad_norm": float(torch.linalg.vector_norm(pair_grad.float())),
        "memory_k_grad_norm": float(torch.linalg.vector_norm(memory_grad.float())),
    }
    stage.memory.set_differentiable_pretraining(False)
    model.zero_grad(set_to_none=True)
    return result


def cpu_preflight() -> dict[str, Any]:
    torch.manual_seed(CPU_DIAGNOSTIC_SEED)
    reference = v22eff.build_aera(torch.device("cpu")).eval()
    torch.manual_seed(CPU_DIAGNOSTIC_SEED)
    candidate = build_aera(torch.device("cpu")).eval()

    if sum(p.numel() for p in reference.parameters()) != sum(
        p.numel() for p in candidate.parameters()
    ):
        raise RuntimeError("v23 changed learned parameter count relative to v22")

    ref_state = reference.state_dict()
    got_state = candidate.state_dict()
    if set(ref_state) != set(got_state):
        raise RuntimeError("v23 changed checkpoint/state-dict layout relative to v22")
    changed = [
        key for key, value in ref_state.items()
        if not torch.equal(got_state[key], value)
    ]
    unexpected = [key for key in changed if ".pair_write_gate.weight" not in key]
    if unexpected:
        raise RuntimeError(f"v23 changed non-selector initialization: {unexpected[:8]}")

    torch.manual_seed(CPU_DIAGNOSTIC_SEED)
    transformer = build_transformer(torch.device("cpu")).eval()
    counts = parameter_accounting(candidate, transformer)
    if abs(counts["stored_parameter_delta_fraction"]) > 0.05:
        raise RuntimeError(f"stored parameter mismatch exceeds 5%: {counts}")

    g = torch.Generator().manual_seed(CPU_DIAGNOSTIC_SEED + 1)
    tokens = torch.randint(
        0,
        candidate.cfg.vocab_size,
        (1, CHUNK_SIZE),
        generator=g,
    )
    with torch.no_grad():
        ref_out = reference(
            tokens,
            hard=True,
            route_mode="hard_sparse",
            update_memory=False,
        )
        got_out = candidate(
            tokens,
            hard=True,
            route_mode="hard_sparse",
            update_memory=False,
        )
    ref_logits = ref_out["logits"]
    got_logits = got_out["logits"]
    if not isinstance(ref_logits, torch.Tensor) or not isinstance(got_logits, torch.Tensor):
        raise RuntimeError("v23 preflight expected tensor logits")
    torch.testing.assert_close(got_logits, ref_logits, atol=0.0, rtol=0.0)
    ref_hw = ref_out["state"]
    got_hw = got_out["state"]
    for old, new in zip(ref_hw.stages, got_hw.stages):
        torch.testing.assert_close(new.stream, old.stream, atol=0.0, rtol=0.0)
        torch.testing.assert_close(
            new.memory.matrix,
            old.memory.matrix,
            atol=0.0,
            rtol=0.0,
        )
        torch.testing.assert_close(
            new.memory.inverse_key_covariance,
            old.memory.inverse_key_covariance,
            atol=0.0,
            rtol=0.0,
        )

    smoke = _production_sparse_smoke(candidate.train())
    if smoke["selected_writes"] != 16 or smoke["candidates"] != 255:
        raise RuntimeError(f"v23 production sparse budget is not 16/255: {smoke}")

    if GRAD_ACCUM != 4:
        raise RuntimeError("frozen gradient accumulation changed")
    if MAX_MEMORY_AUX_EVENTS_PER_MICROBATCH != 256:
        raise RuntimeError("corrected auxiliary microbatch cap changed")
    if MAX_MEMORY_AUX_EVENTS_PER_OPTIMIZER_STEP != 1024:
        raise RuntimeError("corrected auxiliary optimizer-step cap changed")
    try:
        memory_auxiliary_terms(
            candidate,
            tokens,
            step=0,
            max_events=MAX_MEMORY_AUX_EVENTS_PER_MICROBATCH + 1,
        )
    except ValueError:
        pass
    else:
        raise RuntimeError("v23 auxiliary accepted an over-budget microbatch")

    return {
        "version": "aera-v23-sparse-write-systems-binding",
        "cpu_diagnostic_seed": CPU_DIAGNOSTIC_SEED,
        "scientific_training_performed": False,
        "gpu_training_authorized": False,
        "architecture_delta_from_v22": "physical event-write selection/execution only",
        "chunk_size": CHUNK_SIZE,
        "memory_dim": candidate.cfg.memory_dim,
        "pair_candidates_per_chunk": CHUNK_SIZE - 1,
        "selected_writes_per_chunk": sparse_write_budget(CHUNK_SIZE - 1),
        "learned_parameter_count_changed_from_v22": False,
        "changed_initialization_keys": changed,
        "memory_off_logits_and_state_bit_exact_v22": True,
        "memory_aux_events_per_microbatch": MAX_MEMORY_AUX_EVENTS_PER_MICROBATCH,
        "memory_aux_events_per_optimizer_step": MAX_MEMORY_AUX_EVENTS_PER_OPTIMIZER_STEP,
        "parameter_accounting": counts,
        "sparse_smoke": smoke,
        "memory": sparse_dual_delta_memory_protocol(),
    }


def efficiency_protocol() -> dict[str, Any]:
    return {
        "preregistration": "issue #333",
        "source_controlled_pass": "issue #332",
        "source_model": "aera-v22 dual-delta",
        "architecture_delta": "budgeted physical sparse event writes",
        "memory_equations_changed": False,
        "routing_changed": False,
        "objective_weights_changed": False,
        "chunk_size": CHUNK_SIZE,
        "real_language_candidates": CHUNK_SIZE - 1,
        "real_language_selected_writes": sparse_write_budget(CHUNK_SIZE - 1),
        "memory_aux_events_per_microbatch": MAX_MEMORY_AUX_EVENTS_PER_MICROBATCH,
        "memory_aux_events_per_optimizer_step": MAX_MEMORY_AUX_EVENTS_PER_OPTIMIZER_STEP,
        "primary_runtime_requires_compile": False,
        "gpu_training_authorized": False,
        "fresh_real_language_seed_authorized": False,
        "architecture_freeze_authorized": False,
        "100m_authorized": False,
    }
