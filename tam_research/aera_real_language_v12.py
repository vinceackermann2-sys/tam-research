from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .aera_hardware_core import HardwareAERAConfig, hardware_parameter_accounting
from .aera_hardware_core_v8 import MostlyHardRoutingSchedule
from .aera_hardware_core_v12 import HardwareAwareAERATextLMV12
from .aera_real_language import (
    VOCAB_SIZE,
    SEQ_LEN,
    TOKEN_BUDGET,
    MICRO_BATCH,
    GRAD_ACCUM,
    TOKENS_PER_STEP,
    TOTAL_STEPS,
    parameter_accounting,
    transformer_25m_config,
)
from .models import ResearchLM

# CPU/design seed only.  No v12 GPU run is authorized by this module.
CPU_DIAGNOSTIC_SEED = 8221
CHUNK_SIZE = 256
DENSE_WARMUP_STEPS = 192
ROUTER_CALIBRATION_END = 384
SPARSE_CALIBRATION_EVERY = 4
STAGE_DIFFICULTY_WEIGHT = 0.10
STAGE_BUDGET_WEIGHT = 0.05
STAGE_POLARIZATION_WEIGHT = 0.01


def aera_v12_config() -> HardwareAERAConfig:
    return HardwareAERAConfig(
        vocab_size=VOCAB_SIZE,
        d_model=200,
        n_stages=4,
        n_heads=8,
        chunk_size=CHUNK_SIZE,
        n_experts=8,
        max_active_experts=2,
        expert_mult=4,
        memory_dim=50,
        max_reason_steps=4,
        block_size=4,
    )


def build_aera(device: torch.device) -> HardwareAwareAERATextLMV12:
    return HardwareAwareAERATextLMV12(aera_v12_config()).to(device)


def build_transformer(device: torch.device) -> ResearchLM:
    return ResearchLM(transformer_25m_config()).to(device)


def phase_for_step(step: int) -> str:
    if step < 0 or step >= TOTAL_STEPS:
        raise ValueError(f"step must be in [0,{TOTAL_STEPS})")
    if step < DENSE_WARMUP_STEPS:
        return "representation_warmup"
    if step < ROUTER_CALIBRATION_END:
        return "difficulty_router_calibration"
    return "mostly_hard_sparse"


def route_mode_for_step(step: int) -> str:
    phase = phase_for_step(step)
    if phase in {"representation_warmup", "difficulty_router_calibration"}:
        return "straight_through"
    return MostlyHardRoutingSchedule(
        calibration_every=SPARSE_CALIBRATION_EVERY
    ).mode_for_step(step - ROUTER_CALIBRATION_END)


def set_optional_stage_router_trainable(
    model: HardwareAwareAERATextLMV12,
    trainable: bool,
) -> None:
    model.set_optional_stage_routers_trainable(trainable)


def _loss_weights(step: int) -> dict[str, float]:
    phase = phase_for_step(step)
    if phase == "representation_warmup":
        return {
            "event": 0.002,
            "stream": 0.010,
            "compute": 0.0,
            "balance": 0.005,
        }
    return {
        "event": 0.005,
        "stream": 0.020,
        "compute": 0.0005,
        "balance": 0.010,
    }


def per_chunk_language_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    chunk_size: int = CHUNK_SIZE,
) -> torch.Tensor:
    """Detached [batch,chunks] CE used only as router-difficulty supervision."""
    if logits.ndim != 3 or targets.ndim != 2:
        raise ValueError("logits/targets must be [batch,time,vocab] and [batch,time]")
    if logits.shape[:2] != targets.shape:
        raise ValueError("logits/targets shape mismatch")
    if targets.size(1) % chunk_size:
        raise ValueError("sequence length must divide exactly by chunk_size")
    token_loss = F.cross_entropy(
        logits.float().reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="none",
    ).view(targets.size(0), targets.size(1))
    return token_loss.view(targets.size(0), -1, chunk_size).mean(dim=-1).detach()


def aera_matched_loss(
    model: HardwareAwareAERATextLMV12,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    step: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], str, str]:
    """Exact shared x->y CE plus v12 routing supervision on calibration passes."""
    phase = phase_for_step(step)
    mode = route_mode_for_step(step)
    hard = mode == "hard_sparse"
    weights = _loss_weights(step)
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
        if phase == "difficulty_router_calibration":
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

    # As in v11, replace inherited x[t]->x[t+1] CE with the exact shared x->y target.
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
    """CPU-only architecture checks that require no production data mount."""
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
        "token_budget_if_later_authorized": TOKEN_BUDGET,
        "tokens_per_step": TOKENS_PER_STEP,
        "optimizer_steps": TOTAL_STEPS,
        "chunk_size": CHUNK_SIZE,
        "chunks_per_sequence": SEQ_LEN // CHUNK_SIZE,
        "curriculum": {
            "representation_warmup_steps": DENSE_WARMUP_STEPS,
            "difficulty_router_calibration_end_step": ROUTER_CALIBRATION_END,
            "sparse_calibration_every": SPARSE_CALIBRATION_EVERY,
        },
        "routing": {
            "foundation_stage": HardwareAwareAERATextLMV12.FOUNDATION_STAGE,
            "optional_stage_run_rates": list(HardwareAwareAERATextLMV12.OPTIONAL_STAGE_RUN_RATES),
            "target_mean_total_stage_execution": 0.50,
            "difficulty_weight": STAGE_DIFFICULTY_WEIGHT,
            "budget_weight": STAGE_BUDGET_WEIGHT,
            "polarization_weight": STAGE_POLARIZATION_WEIGHT,
        },
        "parameter_accounting": counts,
        "top1_full_stage_active_estimate": active,
        "initialization_fairness": initial,
    }


def validate_production_data(data_dir: str) -> dict[str, Any]:
    """Exact immutable data guard for a future run; this function does not train."""
    root = Path(data_dir)
    required = [root / "train.bin", root / "val.bin", root / "meta.json"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"real-language source data missing: {missing}")
    meta = json.loads((root / "meta.json").read_text())
    expected = {
        "assembly_version": 3,
        "train_tokens": 2_000_000_000,
        "val_tokens": 5_000_000,
        "seed": 8100,
        "tokenizer": "gpt2",
        "dtype": "uint16",
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            raise RuntimeError(f"data metadata mismatch {key}: {meta.get(key)!r} != {value!r}")
    if (root / "train.bin").stat().st_size != 4_000_000_000:
        raise RuntimeError("train.bin is not exact 2B uint16 token stream")
    if (root / "val.bin").stat().st_size != 10_000_000:
        raise RuntimeError("val.bin is not exact 5M uint16 validation stream")
    return expected
