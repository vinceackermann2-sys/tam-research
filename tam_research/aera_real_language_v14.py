from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from . import aera_real_language_v12 as v12
from . import aera_real_language_v13 as v13
from .aera_hardware_core import hardware_parameter_accounting
from .aera_hardware_core_v14 import HardwareAwareAERATextLMV14
from .aera_real_language import (
    TOKEN_BUDGET,
    TOKENS_PER_STEP,
    TOTAL_STEPS,
    VOCAB_SIZE,
    parameter_accounting,
)

CPU_DIAGNOSTIC_SEED = 8261
CHUNK_SIZE = v13.CHUNK_SIZE
DENSE_WARMUP_STEPS = v13.DENSE_WARMUP_STEPS
ROUTER_CALIBRATION_END = v13.ROUTER_CALIBRATION_END
SPARSE_CALIBRATION_EVERY = v13.SPARSE_CALIBRATION_EVERY
STAGE_DIFFICULTY_WEIGHT = v13.STAGE_DIFFICULTY_WEIGHT
STAGE_BUDGET_WEIGHT = v13.STAGE_BUDGET_WEIGHT
STAGE_POLARIZATION_WEIGHT = v13.STAGE_POLARIZATION_WEIGHT

phase_for_step = v13.phase_for_step
route_mode_for_step = v13.route_mode_for_step
router_policy_for_step = v13.router_policy_for_step
per_chunk_language_loss = v13.per_chunk_language_loss
build_transformer = v13.build_transformer


def build_aera(device: torch.device) -> HardwareAwareAERATextLMV14:
    return HardwareAwareAERATextLMV14(v12.aera_v12_config()).to(device)


def set_optional_stage_router_trainable(
    model: HardwareAwareAERATextLMV14,
    trainable: bool,
) -> None:
    model.set_optional_stage_routers_trainable(trainable)


def aera_matched_loss(
    model: HardwareAwareAERATextLMV14,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    step: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], str, str]:
    """Shared x->y objective with task-gradient-isolated router calibration."""
    policy = router_policy_for_step(step)
    phase = str(policy["phase"])
    mode = str(policy["route_mode"])
    hard = mode == "hard_sparse"
    supervised = bool(policy["supervised"])

    model.set_optional_stage_routers_trainable(bool(policy["trainable"]))
    model.set_router_task_gradient_isolation(supervised)

    weights = v12._loss_weights(step)
    out = model(
        x,
        hard=hard,
        route_mode=mode,
        update_memory=False,
        return_block_logits=False,
    )
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


def initial_cpu_fairness(seed: int = CPU_DIAGNOSTIC_SEED) -> dict[str, Any]:
    torch.manual_seed(seed)
    aera = build_aera(torch.device("cpu")).eval()
    torch.manual_seed(seed)
    transformer = build_transformer(torch.device("cpu")).eval()
    g = torch.Generator().manual_seed(seed + 10_000)
    x = torch.randint(0, VOCAB_SIZE, (2, 32), generator=g)
    y = torch.randint(0, VOCAB_SIZE, (2, 32), generator=g)
    with torch.no_grad():
        aout = aera(x, hard=False, route_mode="straight_through", update_memory=False)
        alogits = aout["logits"]
        assert isinstance(alogits, torch.Tensor)
        tlogits = transformer(x)
    anll = float(F.cross_entropy(alogits.float().reshape(-1, VOCAB_SIZE), y.reshape(-1)))
    tnll = float(F.cross_entropy(tlogits.float().reshape(-1, VOCAB_SIZE), y.reshape(-1)))
    chance = math.log(VOCAB_SIZE)
    result = {
        "aera_initial_nll": anll,
        "transformer_initial_nll": tnll,
        "nll_gap": anll - tnll,
        "chance_nll": chance,
    }
    if abs(anll - tnll) > 0.50:
        raise RuntimeError(f"initial NLL mismatch exceeds 0.50: {result}")
    if abs(anll - chance) > 0.75:
        raise RuntimeError(f"AERA initial NLL is not chance-scale: {result}")
    return result


def cpu_preflight() -> dict[str, Any]:
    torch.manual_seed(CPU_DIAGNOSTIC_SEED)
    aera = build_aera(torch.device("cpu"))
    torch.manual_seed(CPU_DIAGNOSTIC_SEED)
    transformer = build_transformer(torch.device("cpu"))
    counts = parameter_accounting(aera, transformer)
    if abs(counts["stored_parameter_delta_fraction"]) > 0.05:
        raise RuntimeError(f"stored parameter mismatch exceeds 5%: {counts}")
    active = hardware_parameter_accounting(aera, mean_active_experts=1.0)
    initial = initial_cpu_fairness(CPU_DIAGNOSTIC_SEED)
    return {
        "cpu_diagnostic_seed": CPU_DIAGNOSTIC_SEED,
        "gpu_authorized": False,
        "version": "aera-v14-router-gradient-isolation",
        "token_budget_if_later_authorized": TOKEN_BUDGET,
        "tokens_per_step": TOKENS_PER_STEP,
        "optimizer_steps": TOTAL_STEPS,
        "chunk_size": CHUNK_SIZE,
        "curriculum": {
            "representation_warmup_steps": DENSE_WARMUP_STEPS,
            "difficulty_router_calibration_end_step": ROUTER_CALIBRATION_END,
            "sparse_calibration_every": SPARSE_CALIBRATION_EVERY,
        },
        "routing": {
            "foundation_stage": HardwareAwareAERATextLMV14.FOUNDATION_STAGE,
            "optional_stage_run_rates": list(HardwareAwareAERATextLMV14.OPTIONAL_STAGE_RUN_RATES),
            "target_mean_total_stage_execution": 0.50,
            "difficulty_weight": STAGE_DIFFICULTY_WEIGHT,
            "budget_weight": STAGE_BUDGET_WEIGHT,
            "polarization_weight": STAGE_POLARIZATION_WEIGHT,
            "calibration_primary_task_router_gradient": "detached",
            "calibration_explicit_router_supervision_gradient": "enabled",
        },
        "parameter_accounting": counts,
        "top1_full_stage_active_estimate": active,
        "initialization_fairness": initial,
        "v14_change_scope": "router calibration gradient path only; architecture/data/init unchanged",
    }
