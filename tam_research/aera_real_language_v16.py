from __future__ import annotations

from typing import Any

import torch

from . import aera_real_language_v12 as v12
from . import aera_real_language_v15 as v15
from .aera_hardware_core import hardware_parameter_accounting
from .aera_hardware_core_v16 import HardwareAwareAERATextLMV16
from .aera_real_language import TOKEN_BUDGET, TOKENS_PER_STEP, TOTAL_STEPS, parameter_accounting

CPU_DIAGNOSTIC_SEED = 8301
CHUNK_SIZE = v15.CHUNK_SIZE
DENSE_WARMUP_STEPS = v15.DENSE_WARMUP_STEPS
ROUTER_CALIBRATION_END = v15.ROUTER_CALIBRATION_END
SPARSE_CALIBRATION_EVERY = v15.SPARSE_CALIBRATION_EVERY
STAGE_DIFFICULTY_WEIGHT = v15.STAGE_DIFFICULTY_WEIGHT
STAGE_BUDGET_WEIGHT = v15.STAGE_BUDGET_WEIGHT
STAGE_POLARIZATION_WEIGHT = v15.STAGE_POLARIZATION_WEIGHT

phase_for_step = v15.phase_for_step
route_mode_for_step = v15.route_mode_for_step
router_policy_for_step = v15.router_policy_for_step
per_chunk_language_loss = v15.per_chunk_language_loss
build_transformer = v15.build_transformer


def build_aera(device: torch.device) -> HardwareAwareAERATextLMV16:
    return HardwareAwareAERATextLMV16(v12.aera_v12_config()).to(device)


def set_optional_stage_router_trainable(model: HardwareAwareAERATextLMV16, trainable: bool) -> None:
    model.set_optional_stage_routers_trainable(trainable)


def aera_matched_loss(model: HardwareAwareAERATextLMV16, x: torch.Tensor, y: torch.Tensor, *, step: int):
    # V16 changes only routing_supervision, so reuse V15's already isolated training path
    # by applying the same objective with a V16 model.
    policy = router_policy_for_step(step)
    phase = str(policy["phase"])
    mode = str(policy["route_mode"])
    hard = mode == "hard_sparse"
    supervised = bool(policy["supervised"])
    model.set_optional_stage_routers_trainable(bool(policy["trainable"]))
    model.set_router_task_gradient_isolation(supervised)

    weights = v12._loss_weights(step)
    out = model(x, hard=hard, route_mode=mode, update_memory=False, return_block_logits=False)
    import torch.nn.functional as F

    logits = out["logits"]
    assert isinstance(logits, torch.Tensor)
    primary = F.cross_entropy(logits.float().reshape(-1, model.cfg.vocab_size), y.reshape(-1))
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
    active = hardware_parameter_accounting(aera, mean_active_experts=1.0)
    return {
        "cpu_diagnostic_seed": CPU_DIAGNOSTIC_SEED,
        "gpu_authorized": False,
        "version": "aera-v16-hard-execution-budget",
        "token_budget_if_later_authorized": TOKEN_BUDGET,
        "tokens_per_step": TOKENS_PER_STEP,
        "optimizer_steps": TOTAL_STEPS,
        "chunk_size": CHUNK_SIZE,
        "optional_stage_target_rates": list(HardwareAwareAERATextLMV16.OPTIONAL_STAGE_RUN_RATES),
        "budget_constraint": "per-stage straight-through hard-run fraction vs per-stage target fraction",
        "hard_threshold": 0.5,
        "architecture_changed": False,
        "data_changed": False,
        "target_rates_changed": False,
        "parameter_accounting": counts,
        "top1_full_stage_active_estimate": active,
    }
