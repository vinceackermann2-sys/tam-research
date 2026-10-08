"""Paired corrected-binding evaluation with mandatory counterfactual query groups.

Only the tokens tensor is ever passed to a prediction function. Generator,
delay bucket, source keys, targets and association labels remain outside it.
"""
from __future__ import annotations

from contextlib import nullcontext
import hashlib
import time
from typing import Callable

import torch
import torch.nn.functional as F

from tam_research.cpw_binding_v2.task import (
    SCORED_DELAYS,
    counterfactual_query,
    explicit_key_lookup,
    make_binding_batch,
)

Predict = Callable[[torch.Tensor], torch.Tensor]
LONG_DELAYS = (64, 128, 256)


def _observe(logits: torch.Tensor, targets: torch.Tensor) -> tuple[int, float]:
    if logits.ndim != 2 or logits.size(0) != targets.size(0):
        raise ValueError("predictor must return [batch, vocab] final-query logits")
    if not torch.isfinite(logits).all():
        raise ValueError("non-finite query logits")
    loss = F.cross_entropy(logits.float(), targets, reduction="sum")
    return int((logits.argmax(-1) == targets).sum().item()), float(loss.item())


def _context(device: torch.device):
    return (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )


@torch.no_grad()
def evaluate_distances(
    predict: Predict,
    *,
    eval_seed: int,
    examples_per_distance: int,
    device: torch.device = torch.device("cpu"),
    batch_size: int = 32,
) -> dict[str, object]:
    if examples_per_distance <= 0 or batch_size <= 0:
        raise ValueError("positive example and batch counts required")
    started = time.perf_counter()
    by_distance: dict[str, dict[str, float | int]] = {}
    fingerprints: dict[str, str] = {}
    total = 0

    for delay in SCORED_DELAYS:
        # Recreate a fresh generator for EACH arm and bucket. Passing the
        # externally chosen delay to the model is forbidden.
        rng = torch.Generator(device="cpu").manual_seed(
            int(eval_seed) + 1_000_000 + int(delay)
        )
        digest = hashlib.sha256()
        hits = 0
        loss_total = 0.0
        consumed = 0
        while consumed < examples_per_distance:
            count = min(batch_size, examples_per_distance - consumed)
            batch = make_binding_batch(
                batch_size=count, generator=rng, delay=int(delay)
            )
            digest.update(batch.tokens.contiguous().numpy().tobytes())
            digest.update(batch.targets.contiguous().numpy().tobytes())
            with _context(device):
                logits = predict(batch.tokens.to(device, non_blocking=True))
            correct, loss = _observe(
                logits, batch.targets.to(device, non_blocking=True)
            )
            hits += correct
            loss_total += loss
            consumed += count
        by_distance[str(delay)] = {
            "examples": consumed,
            "accuracy": hits / consumed,
            "nll": loss_total / consumed,
        }
        fingerprints[str(delay)] = digest.hexdigest()
        total += consumed

    long_acc = sum(float(by_distance[str(d)]["accuracy"]) for d in LONG_DELAYS) / 3
    long_nll = sum(float(by_distance[str(d)]["nll"]) for d in LONG_DELAYS) / 3
    return {
        "by_distance": by_distance,
        "long_mean_accuracy": long_acc,
        "long_mean_nll": long_nll,
        "data_sha256_by_distance": fingerprints,
        "evaluation_examples": total,
        "evaluation_wall_seconds": time.perf_counter() - started,
    }


@torch.no_grad()
def evaluate_counterfactual_groups(
    predict: Predict,
    *,
    eval_seed: int,
    source_groups: int,
    device: torch.device = torch.device("cpu"),
    group_batch_size: int = 8,
) -> dict[str, object]:
    """Score all four queries for identical source prefixes.

    A query-blind guess scores exactly 1/4 per-query and 0 all-four-groups.
    No model receives the query pair index, delay or a target tensor.
    """
    if source_groups <= 0 or group_batch_size <= 0:
        raise ValueError("positive source_groups and group_batch_size required")
    rng = torch.Generator(device="cpu").manual_seed(int(eval_seed) + 2_000_000)
    digest = hashlib.sha256()
    started = time.perf_counter()
    done = 0
    correct_queries = 0
    correct_groups = 0
    ce_total = 0.0
    while done < source_groups:
        count = min(group_batch_size, source_groups - done)
        base = make_binding_batch(batch_size=count, generator=rng)
        variants = [counterfactual_query(base, new_delay=d) for d in SCORED_DELAYS]
        for variant_tokens, variant_targets in variants:
            if not torch.equal(variant_tokens[:, :-1], base.tokens[:, :-1]):
                raise AssertionError("counterfactual unexpectedly mutated source prefix")
            if not torch.equal(explicit_key_lookup(variant_tokens), variant_targets):
                raise AssertionError("symbolic association oracle failed")
        # Four query variants per source; score and restore [delay, group].
        tokens = torch.cat([v[0] for v in variants], dim=0)
        targets = torch.cat([v[1] for v in variants], dim=0)
        target_groups = torch.stack([v[1] for v in variants], dim=1)
        if not torch.all(torch.stack([v[0][:, -1] for v in variants], 1).unique(dim=1).shape[1] == 4):
            raise AssertionError("query set is not unique")
        if any(int(torch.unique(row).numel()) != 4 for row in target_groups):
            raise AssertionError("four target values must differ")
        digest.update(tokens.contiguous().numpy().tobytes())
        digest.update(targets.contiguous().numpy().tobytes())
        with _context(device):
            logits = predict(tokens.to(device, non_blocking=True))
        n_correct, loss = _observe(logits, targets.to(device, non_blocking=True))
        correct_queries += n_correct
        ce_total += loss
        predicted_groups = logits.argmax(-1).view(4, count).transpose(0, 1)
        matched = predicted_groups == target_groups.to(device)
        correct_groups += int(matched.all(dim=1).sum().item())
        done += count

    return {
        "source_groups": done,
        "query_examples": done * 4,
        "per_query_accuracy": correct_queries / (done * 4),
        "all_four_group_accuracy": correct_groups / done,
        "per_query_nll": ce_total / (done * 4),
        "data_sha256": digest.hexdigest(),
        "evaluation_wall_seconds": time.perf_counter() - started,
    }


def cpu_train_integrity_step(
    model: torch.nn.Module,
    *,
    train_seed: int,
    batch_size: int = 1,
) -> dict[str, float]:
    """Single CPU-only optimizer-step smoke; NEVER scientific model evidence."""
    from .models import query_logits
    if next(model.parameters()).device.type != "cpu":
        raise RuntimeError("CPU integrity check forbids CUDA")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    rng = torch.Generator(device="cpu").manual_seed(int(train_seed) + 10_000)
    batch = make_binding_batch(batch_size=batch_size, generator=rng)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1
    )
    optimizer.zero_grad(set_to_none=True)
    logits = query_logits(model, batch.tokens)
    loss = F.cross_entropy(logits.float(), batch.targets)
    if not torch.isfinite(loss):
        raise ValueError("non-finite CPU query loss")
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    if not grads or any(g is None or not torch.isfinite(g).all() for g in grads):
        raise AssertionError("missing or non-finite model gradient")
    if not any(bool(g.abs().sum() > 0) for g in grads):
        raise AssertionError("all model gradients are zero")
    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
    optimizer.step()
    return {
        "cpu_only": 1.0,
        "query_loss": float(loss.detach()),
        "gradient_norm": grad_norm,
    }
