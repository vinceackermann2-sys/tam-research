from __future__ import annotations

"""AERA-v13 CPU-first routing curriculum.

V12 showed that explicit difficulty/budget supervision could move real-language
stage execution from 100% to 75%, but the supervision disappeared after the
router-calibration phase while the optional routers remained trainable.  V13 keeps
the model architecture, initialization, chunking, experts, state/memory and data
protocol unchanged and changes only the router update schedule:

* representation warm-up: optional routers frozen;
* difficulty calibration: optional routers train on task + explicit v12 budget;
* mostly-hard sparse phase: hard deployment steps freeze optional routers;
* periodic straight-through calibration steps unfreeze the routers and retain the
  explicit difficulty/budget/polarization supervision.

This separates learning the compute policy from training the expensive routed
representation under the deployed hard decisions.  No v13 GPU run is authorized
by this module.
"""

from typing import Any

import torch
import torch.nn.functional as F

from . import aera_real_language_v12 as v12
from .aera_hardware_core_v12 import HardwareAwareAERATextLMV12

CPU_DIAGNOSTIC_SEED = 8241
CHUNK_SIZE = v12.CHUNK_SIZE
DENSE_WARMUP_STEPS = v12.DENSE_WARMUP_STEPS
ROUTER_CALIBRATION_END = v12.ROUTER_CALIBRATION_END
SPARSE_CALIBRATION_EVERY = v12.SPARSE_CALIBRATION_EVERY
STAGE_DIFFICULTY_WEIGHT = v12.STAGE_DIFFICULTY_WEIGHT
STAGE_BUDGET_WEIGHT = v12.STAGE_BUDGET_WEIGHT
STAGE_POLARIZATION_WEIGHT = v12.STAGE_POLARIZATION_WEIGHT

build_aera = v12.build_aera
build_transformer = v12.build_transformer
phase_for_step = v12.phase_for_step
route_mode_for_step = v12.route_mode_for_step
per_chunk_language_loss = v12.per_chunk_language_loss


def router_policy_for_step(step: int) -> dict[str, object]:
    """Return the frozen v13 optional-router update policy for one optimizer step."""
    phase = phase_for_step(step)
    mode = route_mode_for_step(step)
    if phase == "representation_warmup":
        return {
            "trainable": False,
            "supervised": False,
            "phase": phase,
            "route_mode": mode,
        }
    calibration = mode == "straight_through"
    return {
        "trainable": calibration,
        "supervised": calibration,
        "phase": phase,
        "route_mode": mode,
    }


def set_optional_stage_router_trainable(
    model: HardwareAwareAERATextLMV12,
    trainable: bool,
) -> None:
    model.set_optional_stage_routers_trainable(trainable)


def aera_matched_loss(
    model: HardwareAwareAERATextLMV12,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    step: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], str, str]:
    """Shared x->y objective with persistent calibrated router budgeting."""
    policy = router_policy_for_step(step)
    phase = str(policy["phase"])
    mode = str(policy["route_mode"])
    hard = mode == "hard_sparse"

    # Critical v13 change: hard sparse deployment steps do not train the routers.
    # Periodic straight-through calibration steps do train them and keep the v12
    # difficulty/budget target active even after ROUTER_CALIBRATION_END.
    model.set_optional_stage_routers_trainable(bool(policy["trainable"]))

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
        chunk_losses = None
        if bool(policy["supervised"]):
            chunk_losses = per_chunk_language_loss(logits, y)
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
    base = v12.cpu_preflight()
    # Parameter counts/active accounting are seed invariant, but initialization
    # fairness must be recomputed with the seed that this preflight reports.
    initial = v12.initial_cpu_fairness(CPU_DIAGNOSTIC_SEED)
    policy = {
        "warmup": router_policy_for_step(DENSE_WARMUP_STEPS - 1),
        "first_router_calibration": router_policy_for_step(DENSE_WARMUP_STEPS),
        "last_dense_calibration": router_policy_for_step(ROUTER_CALIBRATION_END - 1),
        "first_sparse_calibration": router_policy_for_step(ROUTER_CALIBRATION_END),
        "first_hard_sparse": router_policy_for_step(ROUTER_CALIBRATION_END + 1),
        "next_sparse_calibration": router_policy_for_step(
            ROUTER_CALIBRATION_END + SPARSE_CALIBRATION_EVERY
        ),
    }
    return {
        **base,
        "cpu_diagnostic_seed": CPU_DIAGNOSTIC_SEED,
        "gpu_authorized": False,
        "version": "aera-v13-persistent-router-budget",
        "initialization_fairness": initial,
        "router_update_policy": policy,
        "v13_change_scope": "router update schedule only; architecture/data/init unchanged from v12",
    }
