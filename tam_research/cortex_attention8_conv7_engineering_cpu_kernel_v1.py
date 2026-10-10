"""CPU-only reusable optimizer semantics for unapproved Conv7 engineering.

This module has no Modal, CUDA runner, workflow, file writes, seed reservation,
checkpoint loading, or paid execution path. All updates must stay on CPU with
synthetic, caller-provided batches. Not a scientific or engineering result.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from scripts.cortex_attention8_conv7_readiness_v1 import check_readiness
from tam_research.train import cosine_lr

ROOT = Path(__file__).resolve().parents[1]
PROPOSAL = ROOT / "research/reduced_attention/attention8_conv7_engineering_proposal_v1.json"
MODEL_ORDER = (
    "fresh_transformer",
    "reduced_attention_dense_v2_attention8",
    "attention8_causal_depthwise_conv7",
)
EXPECTED_PARAMETERS = (101803520, 101795328, 101803536)
TOTAL_STEPS = 3052
FULL_HORIZON_STEPS = 30518
WARMUP_STEPS = 610
ACCUMULATION_STEPS = 2
LR_PEAK = 3e-4
GRAD_CLIP = 1.0
SYNTHETIC_CLASS = "SYNTHETIC_CPU_OPTIMIZER_SMOKE_NO_GPU_EVIDENCE"


def builder_for(name: str) -> Callable[[], nn.Module]:
    if name == MODEL_ORDER[0]:
        from architectures.cortex_s.reduced_attention_200m_panel_v2 import build_fresh_transformer
        return build_fresh_transformer
    if name == MODEL_ORDER[1]:
        from architectures.cortex_s.reduced_attention_100m_v2_attention8 import ReducedAttentionDenseV2Attention8LM
        return ReducedAttentionDenseV2Attention8LM
    if name == MODEL_ORDER[2]:
        from architectures.cortex_s.attention8_causal_conv7_v1 import Attention8CausalConv7LM
        return Attention8CausalConv7LM
    raise ValueError("unrecognized engineering comparator")


def verify_meta_model_geometry() -> dict[str, Any]:
    """Construct ONLY metadata tensors; no model weights or training batches."""
    readiness = check_readiness()
    if readiness["gpu_launch_permitted"] is not False:
        raise RuntimeError("source preflight unexpectedly permits GPU execution")
    p = json.loads(PROPOSAL.read_text(encoding="utf-8"))
    pilot = p["draft_engineering_pilot"]
    if p["stage"] != "ZERO_GPU_PROPOSAL_NOT_PREREGISTERED":
        raise RuntimeError("unapproved proposal stage changed")
    if any(value is not False for value in p["authority"].values()):
        raise RuntimeError("unexpected run authority")
    if pilot["engineering_seed"] is not None or pilot["strict_max_gpu_spend_usd"] is not None:
        raise RuntimeError("engineering seed or spend cap populated")
    if tuple(pilot["comparators"]) != MODEL_ORDER:
        raise RuntimeError("model order mismatch")
    if pilot["per_model_optimizer_steps"] != TOTAL_STEPS:
        raise RuntimeError("pilot steps mismatch")
    if pilot["full_horizon_optimizer_steps"] != FULL_HORIZON_STEPS:
        raise RuntimeError("full cosine horizon mismatch")
    if pilot["warmup_steps_full_horizon"] != WARMUP_STEPS:
        raise RuntimeError("full cosine warmup mismatch")
    if pilot["gradient_accumulation"] != ACCUMULATION_STEPS:
        raise RuntimeError("grad accumulation mismatch")
    if pilot["sequence_length"] * pilot["micro_batch"] * ACCUMULATION_STEPS != 65536:
        raise RuntimeError("token exposure per optimizer step mismatch")

    actual = []
    with torch.device("meta"):
        for model_name in MODEL_ORDER:
            model = builder_for(model_name)()
            actual.append(sum(q.numel() for q in model.parameters()))
            del model
    if tuple(actual) != EXPECTED_PARAMETERS:
        raise RuntimeError("meta-only parameter contract mismatch")
    return {
        "classification": "CONV7_META_MODEL_CONTRACT_PASS_PAID_EXECUTION_BLOCKED",
        "comparators": list(MODEL_ORDER),
        "parameter_counts": actual,
        "optimizer_steps": TOTAL_STEPS,
        "tokens_per_optimizer_step": 65536,
        "full_horizon_steps": FULL_HORIZON_STEPS,
        "engineering_seed_assigned": False,
        "strict_spend_cap_assigned": False,
        "paid_execution_authorized": False,
        "gpu_allocated": False,
        "scientific_evidence": False,
        "model_weights_allocated": False,
    }


def cpu_synthetic_optimizer_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    microbatches: Sequence[tuple[torch.Tensor, torch.Tensor]],
    step_index: int,
) -> dict[str, Any]:
    """Perform exactly one small CPU-only synthetic optimizer step.

    This intentionally does not use a GPU, disk data or real training seed.
    The future H100 harness will require separate design/authorization.
    """
    if not isinstance(step_index, int) or isinstance(step_index, bool) or not 0 <= step_index < TOTAL_STEPS:
        raise ValueError("step must be inside proposed 200M pilot, 0-based")
    if len(microbatches) != ACCUMULATION_STEPS:
        raise ValueError("exactly two synthetic microbatches required")
    parameters = list(model.parameters())
    if not parameters or any(p.device.type != "cpu" for p in parameters):
        raise RuntimeError("CPU-only test forbids meta/CUDA/offloaded model")
    if any(p.device.type != "cpu" for group in optimizer.param_groups for p in group["params"]):
        raise RuntimeError("CPU-only optimizer only")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    accumulated_loss = 0.0
    try:
        for x, y in microbatches:
            if x.device.type != "cpu" or y.device.type != "cpu":
                raise RuntimeError("CPU-only synthetic tokens required")
            if x.ndim != 2 or x.shape != y.shape or x.numel() == 0:
                raise ValueError("synthetic token and label shapes must match")
            logits = model(x)
            if logits.shape[:2] != x.shape or logits.ndim != 3:
                raise RuntimeError("model must yield [batch, sequence, vocabulary]")
            loss = F.cross_entropy(
                logits.float().reshape(-1, logits.size(-1)), y.reshape(-1)
            ) / ACCUMULATION_STEPS
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite synthetic CPU loss")
            loss.backward()
            accumulated_loss += float(loss.detach())
        if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in parameters):
            raise FloatingPointError("non-finite synthetic CPU gradient")
        norm = torch.nn.utils.clip_grad_norm_(
            parameters, GRAD_CLIP, error_if_nonfinite=True
        )
        if not math.isfinite(float(norm)):
            raise FloatingPointError("non-finite synthetic CPU gradient norm")
        lr = cosine_lr(step_index, FULL_HORIZON_STEPS, WARMUP_STEPS, LR_PEAK)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        return {
            "classification": SYNTHETIC_CLASS,
            "synthetic_loss": accumulated_loss,
            "gradient_norm_before_clip": float(norm),
            "lr_full_horizon_prefix": lr,
            "step_zero_based": step_index,
            "optimizer_steps_performed": 1,
            "gpu_allocated": False,
            "real_engineering_result": False,
            "scientific_evidence": False,
            "paid_execution_authorized": False,
            "seed_consumed": False,
            "retry_authorized": False,
        }
    except BaseException:
        optimizer.zero_grad(set_to_none=True)
        raise
