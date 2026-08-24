from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from . import aera_real_language_v12 as v12
from . import aera_real_language_v17 as v17
from .aera_hardware_core import hardware_parameter_accounting
from .aera_hardware_core_v18 import (
    HardwareAwareAERATextLMV18,
    memory_pretraining_protocol,
)
from .aera_real_language import parameter_accounting

CPU_DIAGNOSTIC_SEED = 8341
CHUNK_SIZE = v17.CHUNK_SIZE
DENSE_WARMUP_STEPS = v17.DENSE_WARMUP_STEPS
ROUTER_CALIBRATION_END = v17.ROUTER_CALIBRATION_END
SPARSE_CALIBRATION_EVERY = v17.SPARSE_CALIBRATION_EVERY
STAGE_DIFFICULTY_WEIGHT = v17.STAGE_DIFFICULTY_WEIGHT
STAGE_BUDGET_WEIGHT = v17.STAGE_BUDGET_WEIGHT
STAGE_POLARIZATION_WEIGHT = v17.STAGE_POLARIZATION_WEIGHT

phase_for_step = v17.phase_for_step
route_mode_for_step = v17.route_mode_for_step
router_policy_for_step = v17.router_policy_for_step
per_chunk_language_loss = v17.per_chunk_language_loss
build_transformer = v17.build_transformer


def build_aera(device: torch.device) -> HardwareAwareAERATextLMV18:
    return HardwareAwareAERATextLMV18(v12.aera_v12_config()).to(device)


def set_optional_stage_router_trainable(
    model: HardwareAwareAERATextLMV18,
    trainable: bool,
) -> None:
    model.set_optional_stage_routers_trainable(trainable)


def aera_matched_loss(
    model: HardwareAwareAERATextLMV18,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    step: int,
):
    """V17 matched objective with causal fast-memory writes enabled in pretraining.

    The delta write itself is differentiable only while this forward is built, so
    later-chunk language loss can train the memory projections and write-strength
    controller outputs. The deployment mode remains detached/local and uses the
    identical update equation.
    """
    policy = router_policy_for_step(step)
    phase = str(policy["phase"])
    mode = str(policy["route_mode"])
    hard = mode == "hard_sparse"
    supervised = bool(policy["supervised"])
    model.set_optional_stage_routers_trainable(bool(policy["trainable"]))
    model.set_router_task_gradient_isolation(supervised)

    weights = v12._loss_weights(step)
    model.set_memory_pretraining_mode(True)
    try:
        out = model(
            x,
            hard=hard,
            route_mode=mode,
            update_memory=True,
            return_block_logits=False,
        )
    finally:
        # Never leave a training-only differentiable local-update mode armed for
        # evaluation/inference by accident.
        model.set_memory_pretraining_mode(False)

    logits = out["logits"]
    assert isinstance(logits, torch.Tensor)
    primary = F.cross_entropy(
        logits.float().reshape(-1, model.cfg.vocab_size),
        y.reshape(-1),
    )
    if hard:
        aux = model.hard_sparse_task_loss(
            x,
            out,
            event_weight=weights["event"],
            block_weight=0.0,
            stream_forecast_weight=weights["stream"],
        )
    else:
        chunk_losses = per_chunk_language_loss(logits, y) if supervised else None
        aux = model.soft_objective(
            x,
            out,
            event_weight=weights["event"],
            compute_weight=weights["compute"],
            balance_weight=weights["balance"],
            block_weight=0.0,
            stream_forecast_weight=weights["stream"],
            chunk_losses=chunk_losses,
            stage_difficulty_weight=STAGE_DIFFICULTY_WEIGHT,
            stage_budget_weight=STAGE_BUDGET_WEIGHT,
            stage_polarization_weight=STAGE_POLARIZATION_WEIGHT,
        )
    total = primary + (aux["total"] - aux["next_token"])
    terms = dict(aux)
    terms["next_token"] = primary
    terms["total"] = total
    return total, terms, mode, phase


def cpu_preflight() -> dict[str, Any]:
    torch.manual_seed(CPU_DIAGNOSTIC_SEED)
    aera = build_aera(torch.device("cpu"))
    torch.manual_seed(CPU_DIAGNOSTIC_SEED)
    transformer = build_transformer(torch.device("cpu"))
    counts = parameter_accounting(aera, transformer)
    if abs(counts["stored_parameter_delta_fraction"]) > 0.05:
        raise RuntimeError(f"stored parameter mismatch exceeds 5%: {counts}")

    torch.manual_seed(CPU_DIAGNOSTIC_SEED)
    v17_aera = v17.build_aera(torch.device("cpu"))
    if set(v17_aera.state_dict()) != set(aera.state_dict()):
        raise RuntimeError("v18 changed checkpoint/state-dict layout relative to v17")
    if sum(p.numel() for p in v17_aera.parameters()) != sum(p.numel() for p in aera.parameters()):
        raise RuntimeError("v18 changed stored parameter count relative to v17")

    return {
        "cpu_diagnostic_seed": CPU_DIAGNOSTIC_SEED,
        "version": "aera-v18-v17-routing-with-differentiable-pretraining-fast-memory",
        "gpu_authorized": False,
        "routing_schedule_changed": False,
        "routing_teacher_changed": False,
        "optional_stage_targets_changed": False,
        "hard_threshold_changed": False,
        "chunk_size": CHUNK_SIZE,
        "real_language_update_memory": True,
        "deployment_memory_pretraining_mode_default": aera.memory_pretraining_mode(),
        "memory": memory_pretraining_protocol(),
        "parameter_accounting": counts,
        "top1_full_stage_active_estimate": hardware_parameter_accounting(
            aera, mean_active_experts=1.0
        ),
    }
