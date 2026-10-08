"""Implementation-only #1321 harness for the deconfounded #1307 binding task.

Scientific execution is NOT authorized here. This code defines no H100 runner,
fresh scientific seeds or durable output namespace. It does not modify any of
the frozen architecture arms or the historical data generators.
"""
from __future__ import annotations

from contextlib import nullcontext
import math
import time
from typing import Any

import torch
import torch.nn.functional as F

from tam_research.cpw_binding_placement.model import EarlyAFMResearchLM
from tam_research.cpw_binding_v2.task import (
    SCORED_DELAYS,
    counterfactual_query,
    explicit_key_lookup,
    make_binding_batch,
)
from tam_research.cpw_r1.model import CPWR1ResearchLM
from tam_research.cpw_v1.model import CPWV1Config
from tam_research.cpw_v3_sparse_world.model import SparseWorldCPWResearchLM
from tam_research.cpw_v5_afm.model import AFMCPWResearchLM
from tam_research.cpw_v5_afm.train import query_logits as frozen_query_logits
from tam_research.models import ModelConfig, ResearchLM, parameter_count
from tam_research.train import cosine_lr, seed_all

ISSUE = 1321
ARMS = ("transformer", "sequence_only", "afm_last1", "afm_first1", "r1_final")
PARAMETERS = {
    "transformer": 24_940_288,
    "sequence_only": 21_745_408,
    "afm_last1": 21_721_344,
    "afm_first1": 21_721_344,
    "r1_final": 21_745_408,
}
LONG_DISTANCES = (64, 128, 256)
# Engineering defaults inherited from #1319's PREAUTHORITY proposal only.
# Values here are not scientific-run authorization or a frozen seed identity.
DEFAULT_LR = 3e-4
DEFAULT_WEIGHT_DECAY = 0.1


def build_arm(arm: str) -> torch.nn.Module:
    if arm == "transformer":
        model = ResearchLM(ModelConfig(architecture="transformer"))
    elif arm == "sequence_only":
        model = SparseWorldCPWResearchLM("sequence_only", CPWV1Config())
    elif arm == "afm_last1":
        model = AFMCPWResearchLM(CPWV1Config())
    elif arm == "afm_first1":
        model = EarlyAFMResearchLM(CPWV1Config())
    elif arm == "r1_final":
        model = CPWR1ResearchLM(CPWV1Config())
    else:
        raise ValueError(f"unknown binding arm: {arm!r}")
    actual = parameter_count(model)
    if actual != PARAMETERS[arm]:
        raise AssertionError(f"{arm} parameter drift: {actual} vs {PARAMETERS[arm]}")
    return model


def query_logits(model: torch.nn.Module, tokens: torch.Tensor) -> torch.Tensor:
    """Output only the final query logits; never pass task metadata to models."""
    if isinstance(model, CPWR1ResearchLM):
        return model.query_logits(tokens)
    return frozen_query_logits(model, tokens)


def _autocast(device: torch.device):
    return (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )


@torch.no_grad()
def evaluate_arm(
    model: torch.nn.Module,
    *,
    seed: int,
    examples_per_distance: int = 64,
    counterfactual_source_groups: int = 64,
    eval_batch_size: int = 16,
) -> dict[str, Any]:
    if examples_per_distance < 1 or counterfactual_source_groups < 1:
        raise ValueError("evaluation counts must be positive")
    if eval_batch_size < 1:
        raise ValueError("eval_batch_size must be positive")

    model.eval()
    device = next(model.parameters()).device
    by_distance: dict[str, dict[str, float | int]] = {}

    # Independent held-out deterministic examples, seeded solely from arm-
    # independent arguments. Each arm receives identical inputs and targets.
    for distance in SCORED_DELAYS:
        generator = torch.Generator(device="cpu").manual_seed(
            seed + 20_000 + int(distance)
        )
        correct = 0
        nll_sum = 0.0
        seen = 0
        while seen < examples_per_distance:
            size = min(eval_batch_size, examples_per_distance - seen)
            batch = make_binding_batch(
                batch_size=size, generator=generator, delay=int(distance)
            )
            if not torch.equal(explicit_key_lookup(batch.tokens), batch.targets):
                raise AssertionError("corrected binding task oracle failure")
            x, y = batch.tokens.to(device), batch.targets.to(device)
            with _autocast(device):
                logits = query_logits(model, x)
            if not torch.isfinite(logits).all():
                raise RuntimeError("non-finite query logits")
            correct += int((logits.argmax(dim=-1) == y).sum())
            nll_sum += float(F.cross_entropy(logits.float(), y, reduction="sum"))
            seen += size
        by_distance[str(distance)] = {
            "examples": seen,
            "accuracy": correct / seen,
            "nll": nll_sum / seen,
        }

    # Four distinct queried keys for the *same source prefix*. A group counts
    # as correct only when every one of its four queries is answered correctly.
    generator = torch.Generator(device="cpu").manual_seed(seed + 30_000)
    group_success = 0
    groups_seen = 0
    per_query_success = 0
    keyblind_fixed_success = 0
    while groups_seen < counterfactual_source_groups:
        size = min(eval_batch_size, counterfactual_source_groups - groups_seen)
        base = make_binding_batch(batch_size=size, generator=generator, delay=32)
        q_inputs: list[torch.Tensor] = []
        q_targets: list[torch.Tensor] = []
        fixed_guess = base.tokens[:, -1 - 32]
        for delay in SCORED_DELAYS:
            altered, targets = counterfactual_query(base, new_delay=int(delay))
            if not torch.equal(altered[:, :-1], base.tokens[:, :-1]):
                raise AssertionError("counterfactual source prefix changed")
            if not torch.equal(explicit_key_lookup(altered), targets):
                raise AssertionError("counterfactual lookup mismatch")
            q_inputs.append(altered)
            q_targets.append(targets)
            keyblind_fixed_success += int((fixed_guess == targets).sum())

        # Query-major stacked tensor permits a single forward pass and exact
        # per-source regrouping without changing the model or data.
        stacked = torch.cat(q_inputs, dim=0).to(device)
        targets = torch.cat(q_targets, dim=0).to(device)
        with _autocast(device):
            logits = query_logits(model, stacked)
        correctness = (logits.argmax(-1) == targets).view(len(SCORED_DELAYS), size)
        group_success += int(correctness.all(dim=0).sum())
        per_query_success += int(correctness.sum())
        groups_seen += size

    keyblind_rate = keyblind_fixed_success / (
        groups_seen * len(SCORED_DELAYS)
    )
    if not math.isclose(keyblind_rate, 0.25, abs_tol=1e-12):
        raise AssertionError("key-blind counterfactual control drift")
    return {
        "by_distance": by_distance,
        "long_mean_accuracy": sum(
            float(by_distance[str(d)]["accuracy"]) for d in LONG_DISTANCES
        ) / len(LONG_DISTANCES),
        "long_mean_nll": sum(
            float(by_distance[str(d)]["nll"]) for d in LONG_DISTANCES
        ) / len(LONG_DISTANCES),
        "counterfactual_groups": groups_seen,
        "counterfactual_all_four_accuracy": group_success / groups_seen,
        "counterfactual_individual_accuracy": per_query_success / (
            groups_seen * len(SCORED_DELAYS)
        ),
        "keyblind_fixed_query_accuracy": keyblind_rate,
    }


def train_arm(
    *,
    arm: str,
    seed: int,
    steps: int,
    micro_batch_size: int,
    grad_accum_steps: int,
    eval_examples_per_distance: int,
    counterfactual_source_groups: int,
    device: str = "cpu",
) -> dict[str, Any]:
    """Matched trainer definition, usable for CPU engineering smoke only.

    Do not invoke this as a paid scientific attempt without a separate approved
    immutable run-control, budget, fresh seed reservation and result namespace.
    """
    if arm not in ARMS:
        raise ValueError("invalid arm")
    if min(
        steps, micro_batch_size, grad_accum_steps,
        eval_examples_per_distance, counterfactual_source_groups,
    ) < 1:
        raise ValueError("steps/batches/evaluation sizes must be positive")

    seed_all(seed)
    device_obj = torch.device(device)
    model = build_arm(arm).to(device_obj)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=DEFAULT_LR,
        betas=(0.9, 0.95),
        weight_decay=DEFAULT_WEIGHT_DECAY,
        fused=(device_obj.type == "cuda"),
    )
    batch_gen = torch.Generator(device="cpu").manual_seed(seed + 10_000)
    warmup = max(1, int(steps * 0.02))
    train_seconds = 0.0
    if device_obj.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device_obj)
        torch.cuda.synchronize(device_obj)
    started = time.perf_counter()

    last_loss = float("nan")
    for step in range(steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        step_start = time.perf_counter()
        losses = 0.0
        for _ in range(grad_accum_steps):
            batch = make_binding_batch(
                batch_size=micro_batch_size,
                generator=batch_gen,
                delay=None,
            )
            with _autocast(device_obj):
                logits = query_logits(model, batch.tokens.to(device_obj))
                loss = F.cross_entropy(logits.float(), batch.targets.to(device_obj))
            (loss / grad_accum_steps).backward()
            losses += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = cosine_lr(step, steps, warmup, DEFAULT_LR)
        for g in optimizer.param_groups:
            g["lr"] = lr
        optimizer.step()
        if device_obj.type == "cuda":
            torch.cuda.synchronize(device_obj)
        train_seconds += time.perf_counter() - step_start
        last_loss = losses / grad_accum_steps

    evaluation = evaluate_arm(
        model,
        seed=seed,
        examples_per_distance=eval_examples_per_distance,
        counterfactual_source_groups=counterfactual_source_groups,
        eval_batch_size=min(16, micro_batch_size),
    )
    if device_obj.type == "cuda":
        torch.cuda.synchronize(device_obj)
    total_seconds = time.perf_counter() - started
    examples_seen = steps * micro_batch_size * grad_accum_steps
    return {
        "issue": ISSUE,
        "arm": arm,
        "seed": seed,
        "steps": steps,
        "parameters": parameter_count(model),
        "examples_seen": examples_seen,
        "last_train_query_loss": last_loss,
        "training_seconds": train_seconds,
        "total_compute_seconds": total_seconds,
        "examples_per_second": examples_seen / max(train_seconds, 1e-9),
        "peak_vram_gib": (
            torch.cuda.max_memory_allocated(device_obj) / 1024**3
            if device_obj.type == "cuda"
            else 0.0
        ),
        "evaluation": evaluation,
        "paid_scientific_execution_authorized": False,
        "breakthrough_claim_allowed": False,
    }
