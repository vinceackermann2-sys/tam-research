from __future__ import annotations

"""CHM-v1 ~100M Stage-C scientific execution primitives (#990).

This module contains deterministic training/evaluation mechanics used by the
separately guarded Modal runner. It does not create triggers, allocate compute,
or itself grant scientific execution authority.
"""

from contextlib import nullcontext
import hashlib
from typing import Any

import torch

from .chm_v1_100m_stage_c_eval import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    CASES_PER_FAMILY,
    GENERATOR_VERSION,
    PROBE_SEED,
    TOTAL_PROBES,
    VALIDATION_BATCHES,
    VALIDATION_BATCH_SIZE,
    VALIDATION_SEED,
    VALIDATION_SESSION_LEN,
    VALIDATION_TOKENS,
    validate_protocol_manifest,
)
from .chm_v1_100m_stage_c_run_control_prep import (
    BETAS,
    CHECKPOINT_STEPS,
    FINAL_EVALUATION_STEP,
    GRAD_ACCUM,
    GRAD_CLIP,
    MICRO_BATCH,
    OPTIMIZER_STEPS_PER_MODEL,
    PEAK_LR,
    SCIENTIFIC_SEED,
    SESSION_LEN,
    TOKENS_PER_OPTIMIZER_STEP,
    TRAINING_TOKENS_PER_MODEL,
    TRAIN_STREAM_GENERATOR_SEED,
    WARMUP_STEPS,
    WEIGHT_DECAY,
    validate_contract as validate_prep_contract,
)
from .chm_v1_small_lm import LOCAL_WINDOW, RETRIEVAL_HOPS, _hidden

CONTROL_ISSUE = 990
PREP_ISSUE = 988
IMPLEMENTATION_BASE_SHA = "0e21e48ce7b86ca6b385bcafc6eb556f617d3fbb"
IMPLEMENTATION_BASE_TREE = "8d09e0723160ce0eaae7166bad1c2ba414e52abc"

PHASE = "chm-v1-100m-stage-c-990-v1"
TRIGGER_TITLE = "[modal-chm-v1-100m-stage-c-988-seed-977001-v1]"
RESULT_ROOT = "/vol/chm-v1/100m-stage-c/issue-988/seed-977001-v1"

CLASSIFICATION = "CHM_V1_100M_STAGE_C_EXECUTION_IMPLEMENTATION_TRIGGER_WITHHELD"


def autocast_context(device: torch.device):
    return (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )


def validate_execution_contract() -> dict[str, Any]:
    prep = validate_prep_contract()
    evaluator = validate_protocol_manifest()
    if SCIENTIFIC_SEED != 977_001:
        raise RuntimeError("#990 scientific seed drift")
    if TRAINING_TOKENS_PER_MODEL != 33_554_432:
        raise RuntimeError("#990 per-model token budget drift")
    if OPTIMIZER_STEPS_PER_MODEL != 2_048 or WARMUP_STEPS != 40:
        raise RuntimeError("#990 optimizer schedule drift")
    if TOKENS_PER_OPTIMIZER_STEP != 16_384:
        raise RuntimeError("#990 token accounting drift")
    if tuple(CHECKPOINT_STEPS) != (512, 1024, 1536, 2048):
        raise RuntimeError("#990 checkpoint schedule drift")
    if FINAL_EVALUATION_STEP != OPTIMIZER_STEPS_PER_MODEL:
        raise RuntimeError("#990 only completed step 2048 may be scientifically evaluated")
    if TRAIN_STREAM_GENERATOR_SEED != 987_001:
        raise RuntimeError("#990 training stream generator seed drift")
    if (SESSION_LEN, LOCAL_WINDOW, RETRIEVAL_HOPS) != (1024, 512, 2):
        raise RuntimeError("#990 local/session/retrieval geometry drift")
    if (MICRO_BATCH, GRAD_ACCUM) != (4, 4):
        raise RuntimeError("#990 optimizer batching drift")
    if tuple(BETAS) != (0.9, 0.95):
        raise RuntimeError("#990 AdamW beta drift")
    if (WEIGHT_DECAY, PEAK_LR, GRAD_CLIP) != (0.1, 3e-4, 1.0):
        raise RuntimeError("#990 optimizer hyperparameter drift")
    if (
        GENERATOR_VERSION != "chm-v1-100m-heldout-aligned-v4"
        or PROBE_SEED != 977_301
        or VALIDATION_SEED != 977_302
        or BOOTSTRAP_SEED != 977_303
    ):
        raise RuntimeError("#990 evaluator identity drift")
    if (
        CASES_PER_FAMILY != 128
        or TOTAL_PROBES != 512
        or VALIDATION_TOKENS != 1_048_576
        or VALIDATION_BATCHES != 128
        or VALIDATION_BATCH_SIZE != 8
        or VALIDATION_SESSION_LEN != 1024
        or BOOTSTRAP_RESAMPLES != 10_000
    ):
        raise RuntimeError("#990 evaluator envelope drift")
    return {
        "classification": CLASSIFICATION,
        "control_issue": CONTROL_ISSUE,
        "prep_issue": PREP_ISSUE,
        "phase": PHASE,
        "trigger_title": TRIGGER_TITLE,
        "result_root": RESULT_ROOT,
        "prep_contract": prep,
        "evaluator_contract": evaluator,
        "trigger_authorized_by_module": False,
        "gpu_allocation_authorized_by_module": False,
        "scientific_seed_consumed_by_module": False,
    }


def build_start_plan(
    *,
    shard_tokens: int,
    seq_len: int,
    steps: int,
    batches_per_step: int,
    batch_size: int,
    seed: int,
) -> torch.Tensor:
    """Materialize immutable CPU start offsets for a sampled token stream."""
    shard_tokens = int(shard_tokens)
    seq_len = int(seq_len)
    steps = int(steps)
    batches_per_step = int(batches_per_step)
    batch_size = int(batch_size)
    if min(shard_tokens, seq_len, steps, batches_per_step, batch_size) <= 0:
        raise ValueError("sample-plan dimensions must be positive")
    hi = shard_tokens - seq_len - 1
    if hi <= 0:
        raise ValueError("token shard is too short for requested sequence length")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    plan = torch.randint(
        0,
        hi,
        (steps, batches_per_step, batch_size),
        generator=generator,
        dtype=torch.int64,
    )
    if int(plan.min()) < 0 or int(plan.max()) >= hi:
        raise RuntimeError("sample-plan offset escaped the legal shard range")
    return plan


def build_training_start_plan(train_tokens: int) -> torch.Tensor:
    return build_start_plan(
        shard_tokens=int(train_tokens),
        seq_len=SESSION_LEN,
        steps=OPTIMIZER_STEPS_PER_MODEL,
        batches_per_step=GRAD_ACCUM,
        batch_size=MICRO_BATCH,
        seed=TRAIN_STREAM_GENERATOR_SEED,
    )


def build_validation_start_plan(val_tokens: int) -> torch.Tensor:
    return build_start_plan(
        shard_tokens=int(val_tokens),
        seq_len=VALIDATION_SESSION_LEN,
        steps=VALIDATION_BATCHES,
        batches_per_step=1,
        batch_size=VALIDATION_BATCH_SIZE,
        seed=VALIDATION_SEED,
    )


def start_plan_sha256(plan: torch.Tensor) -> str:
    plan = plan.detach().to(device="cpu", dtype=torch.int64).contiguous()
    payload = plan.numpy().tobytes(order="C")
    return hashlib.sha256(payload).hexdigest()


def gather_batch_from_source(
    source: torch.Tensor,
    starts_cpu: torch.Tensor,
    *,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather exact x/y batches from an integer token source using frozen offsets."""
    if source.ndim != 1:
        raise ValueError("source must be a 1-D token tensor")
    if starts_cpu.ndim != 1:
        raise ValueError("starts_cpu must be a 1-D vector")
    seq_len = int(seq_len)
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    starts = starts_cpu.to(device=source.device, dtype=torch.long)
    offsets = torch.arange(seq_len + 1, device=source.device, dtype=torch.long)
    if int(starts.min()) < 0 or int(starts.max()) + seq_len >= source.numel():
        raise ValueError("sample start is outside token source")
    chunks = source[starts[:, None] + offsets[None, :]].long()
    return chunks[:, :-1], chunks[:, 1:]


def _exact_nearest_positions(
    keys: torch.Tensor,
    queries: torch.Tensor,
    *,
    query_block: int = 32,
) -> torch.Tensor:
    """Exhaustive float64 squared-distance argmin with insertion-order tie break."""
    if keys.ndim != 3 or queries.ndim != 3:
        raise ValueError("keys/queries must be rank-3")
    if keys.shape[0] != queries.shape[0] or keys.shape[2] != queries.shape[2]:
        raise ValueError("keys/queries batch or address dimensions differ")
    if keys.shape[1] < 1:
        raise ValueError("memory must be non-empty")
    if query_block < 1:
        raise ValueError("query_block must be positive")
    k64 = keys.float().double()
    out: list[torch.Tensor] = []
    for start in range(0, queries.shape[1], query_block):
        q64 = queries[:, start : start + query_block].float().double()
        delta = k64[:, None, :, :] - q64[:, :, None, :]
        distances = (delta * delta).sum(dim=-1)
        out.append(torch.argmin(distances, dim=-1))
    return torch.cat(out, dim=1)


@torch.no_grad()
def eiem_exact_flat_two_chunk_logits(
    model: torch.nn.Module,
    tokens: torch.Tensor,
) -> torch.Tensor:
    """Vectorized exact flat evaluator for 1024-token two-chunk sessions.

    The second 512-token chunk can retrieve only raw keys/hidden values from the
    completed first chunk. Both retrieval hops are exhaustive nearest-neighbour
    searches; there is no clipped/indexed path and no current-chunk write.
    """
    if tokens.ndim != 2 or tokens.shape[1] != 2 * LOCAL_WINDOW:
        raise ValueError(f"expected [batch,{2 * LOCAL_WINDOW}] sessions")
    first = tokens[:, :LOCAL_WINDOW]
    second = tokens[:, LOCAL_WINDOW:]

    first_hidden = _hidden(model.backbone, first)
    memory_keys = model.key_for(first_hidden).float()
    memory_values = first_hidden.float()
    first_logits = model.backbone.lm_head(first_hidden)

    second_hidden = _hidden(model.backbone, second)
    query_state = second_hidden
    for _ in range(RETRIEVAL_HOPS):
        queries = model.query_for(query_state)
        positions = _exact_nearest_positions(memory_keys, queries)
        gather_index = positions.unsqueeze(-1).expand(
            positions.shape[0], positions.shape[1], memory_values.shape[-1]
        )
        memory = torch.gather(memory_values, dim=1, index=gather_index)
        query_state = model._integrate(query_state, memory.to(query_state.dtype))

    second_logits = model.backbone.lm_head(query_state)
    return torch.cat((first_logits, second_logits), dim=1)
