from __future__ import annotations

"""Pure execution contract for CHM-v3 DAEC Stage-C run-control #1323.

This module freezes identities, deterministic sample plans, and cost/resource
guards.  It deliberately contains no optimizer, training loop, Modal import,
GPU allocation, checkpoint writer, or scientific-launch authority.
"""

from typing import Any

import torch
import torch.nn.functional as F

from .chm_v1_100m_stage_c_execution import (
    build_start_plan,
    build_validation_start_plan as _build_validation_start_plan,
    start_plan_sha256,
)
from .chm_v3_100m_daec_stage_c import (
    CASES_PER_FAMILY,
    GENERATOR_VERSION,
    OPTIMIZER_STEPS_PER_MODEL,
    PROBE_SEED,
    SCIENTIFIC_SEED,
    TOKENS_PER_OPTIMIZER_STEP,
    TRAINING_TOKENS_PER_MODEL,
    VALIDATION_SEED,
    VALIDATION_TOKENS,
    WARMUP_STEPS,
    validate_protocol_manifest,
)
from .chm_v1_small_lm_protocol import GRAD_ACCUM, MICRO_BATCH, SESSION_LEN

CONTROL_ISSUE = 1323
PREREG_ISSUE = 1316
HYPOTHESIS_ISSUE = 1234
SYSTEMS_ISSUE = 1302

RESULT_ROOT = "/vol/chm-v3/100m-daec-stage-c/issue-1323/seed-2013161-v1"
TRIGGER_TITLE = "[modal-chm-v3-100m-daec-stage-c-1323-seed-2013161-v1]"
AUDIT_TITLE = "[modal-chm-v3-100m-daec-stage-c-1323-authority-audit-v1]"

TRAIN_STREAM_GENERATOR_SEED = 2_023_161
GPU_CLASS = "L4"
CPU_CORES = 4
RAM_GIB = 16
MAX_GPU_SECONDS = 9_600
MAX_BILLED_COMPUTE_USD = 3.00

DAEC_BLOB = "77e9ccbff4383be40e6e2865503e1c3926bbda74"
STAGE_C_PROTOCOL_BLOB = "b1a58e52b76ff7d06f14a569a7b6274afee8076c"
MODEL_BLOB = "b9b141c0e52d4fd0fff28b12a3588b2adc659b8f"
SMALL_PROTOCOL_BLOB = "d5c2e405b5306f556e7fbe70aacd552867e69d3e"
EVALUATOR_BLOB = "863bd038e60da5511503adb0c8e1046a680ed3bd"
DUAL_ACCOUNT_BLOB = "adb979e2ecaa7cdf1c0ee36e7a4d929783e078d6"
DUAL_ACCOUNT_CLI_BLOB = "440292942066d0b3d3a71d40ca8495092674ce6c"
RUNTIME_PROBE_BLOB = "04b1e9c610195b0896a209eb9d6ce3fd4f014fbc"

STABLE_NLL_BLOB = "482e2c930fa02448e45327747f10571315fd41dc"
PREV_MODEL_BLOB = "0b0a68f47186f154a48d1d75e3fe3b236188d69d"
TRIGGER_AUTHORIZED_BY_MODULE = False
GPU_ALLOCATION_AUTHORIZED_BY_MODULE = False
SCIENTIFIC_SEED_CONSUMED_BY_MODULE = False
REPLICATION_AUTHORIZED_BY_MODULE = False
STAGE_D_AUTHORIZED_BY_MODULE = False


def validate_execution_contract() -> dict[str, Any]:
    protocol = validate_protocol_manifest()
    if SCIENTIFIC_SEED != 2_013_161:
        raise RuntimeError("#1323 scientific seed drift")
    if TRAIN_STREAM_GENERATOR_SEED != SCIENTIFIC_SEED + 10_000:
        raise RuntimeError("#1323 training-plan seed derivation drift")
    if TRAINING_TOKENS_PER_MODEL != 33_554_432:
        raise RuntimeError("#1323 training token budget drift")
    if OPTIMIZER_STEPS_PER_MODEL != 2_048:
        raise RuntimeError("#1323 optimizer-step count drift")
    if WARMUP_STEPS != 40 or TOKENS_PER_OPTIMIZER_STEP != 16_384:
        raise RuntimeError("#1323 optimizer schedule drift")
    if (MICRO_BATCH, GRAD_ACCUM, SESSION_LEN) != (4, 4, 1024):
        raise RuntimeError("#1323 batching/session geometry drift")
    if PROBE_SEED != 977_301 or VALIDATION_SEED != 977_302:
        raise RuntimeError("#1323 inherited evaluator seed drift")
    if VALIDATION_TOKENS != 1_048_576 or CASES_PER_FAMILY != 128:
        raise RuntimeError("#1323 evaluator envelope drift")
    if GENERATOR_VERSION != "chm-v1-100m-heldout-aligned-v4":
        raise RuntimeError("#1323 aligned-v4 generator drift")
    if (GPU_CLASS, CPU_CORES, RAM_GIB) != ("L4", 4, 16):
        raise RuntimeError("#1323 resource geometry drift")
    if MAX_GPU_SECONDS != 9_600 or MAX_BILLED_COMPUTE_USD != 3.00:
        raise RuntimeError("#1323 runtime/cost envelope drift")
    if any(
        (
            TRIGGER_AUTHORIZED_BY_MODULE,
            GPU_ALLOCATION_AUTHORIZED_BY_MODULE,
            SCIENTIFIC_SEED_CONSUMED_BY_MODULE,
            REPLICATION_AUTHORIZED_BY_MODULE,
            STAGE_D_AUTHORIZED_BY_MODULE,
        )
    ):
        raise RuntimeError("#1323 pure execution contract must never self-authorize")
    return {
        "classification": "CHM_V3_100M_DAEC_STAGE_C_RUN_CONTROL_TRIGGER_WITHHELD",
        "control_issue": CONTROL_ISSUE,
        "preregistration_issue": PREREG_ISSUE,
        "hypothesis_issue": HYPOTHESIS_ISSUE,
        "systems_issue": SYSTEMS_ISSUE,
        "scientific_seed_reserved": SCIENTIFIC_SEED,
        "train_stream_generator_seed": TRAIN_STREAM_GENERATOR_SEED,
        "result_root": RESULT_ROOT,
        "trigger_title": TRIGGER_TITLE,
        "audit_title": AUDIT_TITLE,
        "training_tokens_per_model": TRAINING_TOKENS_PER_MODEL,
        "optimizer_steps_per_model": OPTIMIZER_STEPS_PER_MODEL,
        "warmup_steps": WARMUP_STEPS,
        "tokens_per_optimizer_step": TOKENS_PER_OPTIMIZER_STEP,
        "gpu_class": GPU_CLASS,
        "cpu_cores": CPU_CORES,
        "ram_gib": RAM_GIB,
        "max_gpu_seconds": MAX_GPU_SECONDS,
        "max_billed_compute_usd": MAX_BILLED_COMPUTE_USD,
        "protocol": protocol,
        "trigger_authorized_by_module": False,
        "gpu_allocation_authorized_by_module": False,
        "scientific_seed_consumed_by_module": False,
        "replication_authorized_by_module": False,
        "stage_d_authorized_by_module": False,
    }


def build_training_start_plan(train_tokens: int) -> torch.Tensor:
    plan = build_start_plan(
        shard_tokens=int(train_tokens),
        seq_len=SESSION_LEN,
        steps=OPTIMIZER_STEPS_PER_MODEL,
        batches_per_step=GRAD_ACCUM,
        batch_size=MICRO_BATCH,
        seed=TRAIN_STREAM_GENERATOR_SEED,
    )
    expected = (OPTIMIZER_STEPS_PER_MODEL, GRAD_ACCUM, MICRO_BATCH)
    if tuple(plan.shape) != expected or plan.numel() != 32_768:
        raise RuntimeError(f"#1323 training-plan shape drift: {tuple(plan.shape)}")
    return plan


def build_validation_start_plan(val_tokens: int) -> torch.Tensor:
    return _build_validation_start_plan(int(val_tokens))


def validate_live_rate_cap(hourly_resource_usd: float) -> dict[str, float]:
    hourly = float(hourly_resource_usd)
    if not (hourly > 0.0):
        raise RuntimeError("#1323 live resource rate must be positive")
    worst = hourly * (MAX_GPU_SECONDS / 3600.0)
    if worst > MAX_BILLED_COMPUTE_USD:
        raise RuntimeError(
            f"#1323 live timeout worst-case ${worst:.6f} exceeds "
            f"${MAX_BILLED_COMPUTE_USD:.2f} cap"
        )
    return {
        "hourly_resource_usd": hourly,
        "worst_case_usd": worst,
        "max_gpu_seconds": float(MAX_GPU_SECONDS),
        "max_billed_compute_usd": float(MAX_BILLED_COMPUTE_USD),
    }


__all__ = [
    "AUDIT_TITLE",
    "CONTROL_ISSUE",
    "DUAL_ACCOUNT_BLOB",
    "DUAL_ACCOUNT_CLI_BLOB",
    "EVALUATOR_BLOB",
    "GPU_CLASS",
    "MAX_BILLED_COMPUTE_USD",
    "MAX_GPU_SECONDS",
    "MODEL_BLOB",
    "DAEC_BLOB",
    "RESULT_ROOT",
    "RUNTIME_PROBE_BLOB",
    "SCIENTIFIC_SEED",
    "SMALL_PROTOCOL_BLOB",
    "STAGE_C_PROTOCOL_BLOB",
    "STABLE_NLL_BLOB",
    "TRAIN_STREAM_GENERATOR_SEED",
    "TRIGGER_TITLE",
    "build_training_start_plan",
    "build_validation_start_plan",
    "start_plan_sha256",
    "validate_execution_contract",
    "validate_live_rate_cap",
]


@torch.no_grad()
def daec_hard_flat_validation_nll(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Matched 2x512 ordinary validation with hard-flat two-hop DAEC readout.

    The first chunk is exact LOCAL. Completed first-chunk normalized memory
    keys and immutable token IDs are read for every query in the second chunk;
    two hard nearest-position hops produce the decoder-aligned copied token ID.
    The scoring result is mean target NLL, with no dense copy-vocabulary tensor.
    This is evaluation ONLY, never an optimizer/training path.
    """
    from .chm_v1_small_lm import LOCAL_WINDOW, _hidden
    if tokens.shape != targets.shape or tokens.ndim != 2:
        raise ValueError("#1323 validation tokens and targets must match [batch,seq]")
    if tokens.shape[1] != 2 * LOCAL_WINDOW:
        raise ValueError("#1323 hard-flat validation requires exactly 2x512 tokens")

    x1, x2 = tokens[:, :LOCAL_WINDOW], tokens[:, LOCAL_WINDOW:]
    y1, y2 = targets[:, :LOCAL_WINDOW], targets[:, LOCAL_WINDOW:]
    h1 = _hidden(model.backbone, x1)
    h2 = _hidden(model.backbone, x2)
    logits1 = model.backbone.lm_head(h1).float()
    logits2 = model.backbone.lm_head(h2).float()
    vocab = int(logits1.shape[-1])

    nll1 = F.cross_entropy(
        logits1.reshape(-1, vocab), y1.reshape(-1), reduction="mean",
    )

    keys = model.key_for(h1)
    q0 = model.query_for(h2)
    first_hop_idx = torch.argmax(torch.einsum("btd,bsd->bts", q0, keys), dim=-1)
    first_hop_tokens = torch.gather(x1, dim=1, index=first_hop_idx)
    first_hop_embeddings = F.embedding(
        first_hop_tokens, model.backbone.token_emb.weight,
    )
    q1 = model.daec.second_query(q0, first_hop_embeddings)
    second_hop_idx = torch.argmax(torch.einsum("btd,bsd->bts", q1, keys), dim=-1)
    copied_tokens = torch.gather(x1, dim=1, index=second_hop_idx)

    gate = model.daec.gate(h2).float().squeeze(-1)
    base_log_targets = -F.cross_entropy(
        logits2.reshape(-1, vocab), y2.reshape(-1), reduction="none",
    ).reshape_as(y2)
    base_branch = base_log_targets + torch.log1p(-gate)
    with_copy = torch.logaddexp(base_branch, torch.log(gate))
    target_log_probs = torch.where(copied_tokens.eq(y2), with_copy, base_branch)
    nll2 = -target_log_probs.mean()
    if not bool(torch.isfinite(nll1 + nll2).item()):
        raise FloatingPointError("#1323 hard-flat validation non-finite")
    return (nll1 + nll2) / 2
