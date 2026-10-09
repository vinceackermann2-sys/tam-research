"""AT CPU-only 8->64 token curriculum for falsifiable state-retention probing.

Distinct from AR (step-capped and non-learning) and AS (8-token easy control).
All models have exactly 29,504 parameters and get 20s near-write learning +
90s mixed-distance learning on CPU. Balanced heldout results are diagnostic,
not general-language evidence, scientific replication or GPU authorization.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from experiments.rlt.cpu_lastwrite_probe_ar import (
    BATCH_SIZE, CANDIDATES, EXPECTED_PARAMETERS, QUERY, SEQ_LEN, SET0,
    last_write_labels, make_batch, make_models, query_binary_logits,
)
from experiments.rlt.model import parameter_count

JOB_ID = "rlt-cpu-lastwrite-curriculum-distance-20261009-at"
EASY_SEQ_LEN = 8
DISTANCES = (1, 4, 16, 40, 63)
EASY_SECONDS = 20.0
LONG_SECONDS = 90.0
EASY_LEARNING_RATE = 0.003
LONG_LEARNING_RATE = 0.001
EVAL_PER_CLASS = 128
EASY_TRAIN_SEED = 20_261_063
LONG_TRAIN_SEED = 20_271_063
EVAL_SEED = 20_291_063
MODEL_SEED = 20_301_063
RESULT_PATH = Path(
    "experiments/rlt/cpu/results/rlt-lastwrite-curriculum-distance-20261009-at.json"
)


def balanced_batch_for_distance(
    distance: int, seq_len: int, eval_seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    x, y, dist = make_batch(
        1024, (distance,), eval_seed, seq_len=seq_len,
    )
    zeros = torch.nonzero(y == 0).flatten()[:EVAL_PER_CLASS]
    ones = torch.nonzero(y == 1).flatten()[:EVAL_PER_CLASS]
    if len(zeros) != EVAL_PER_CLASS or len(ones) != EVAL_PER_CLASS:
        raise RuntimeError("balanced holdout underpopulated")
    take = torch.stack((zeros, ones), dim=1).flatten()
    x, y = x[take], y[take]
    if not bool((x[:, -1] == QUERY).all()):
        raise RuntimeError("query missing")
    if not torch.equal(last_write_labels(x), y):
        raise RuntimeError("target leakage/mismatch")
    if not bool((dist[take] == distance).all()):
        raise RuntimeError("incorrect gap")
    if not bool((x[:, seq_len-1-distance] == y + SET0).all()):
        raise RuntimeError("last write does not determine target")
    if int(y.sum()) != EVAL_PER_CLASS:
        raise RuntimeError("target imbalance")
    return x, y


@torch.no_grad()
def eval_distance(
    model: torch.nn.Module, distance: int, seq_len: int, seed: int,
) -> dict[str, float | int]:
    model.eval()
    x, y = balanced_batch_for_distance(distance, seq_len, seed)
    nll_sum, correct, correct_class = 0.0, 0, [0, 0]
    for lo in range(0, len(y), 16):
        hi = lo + 16
        logits = query_binary_logits(model, x[lo:hi])
        truth = y[lo:hi]
        prediction = logits.argmax(dim=-1)
        nll_sum += float(F.cross_entropy(logits, truth, reduction="sum"))
        correct += int((prediction == truth).sum())
        for cls in (0, 1):
            correct_class[cls] += int(
                ((prediction == cls) & (truth == cls)).sum()
            )
    return {
        "examples": len(y), "accuracy": correct / len(y),
        "binary_nll": nll_sum / len(y),
        "class0_accuracy": correct_class[0] / EVAL_PER_CLASS,
        "class1_accuracy": correct_class[1] / EVAL_PER_CLASS,
    }


def eval_all(model: torch.nn.Module) -> dict[str, dict]:
    return {
        str(d): eval_distance(model, d, SEQ_LEN, EVAL_SEED + d)
        for d in DISTANCES
    }


def timed_phase(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    seconds: float,
    seed: int,
    seq_len: int,
    distances: tuple[int, ...],
) -> dict[str, float | int]:
    model.train()
    start = time.perf_counter()
    steps = 0
    final_loss = math.nan
    while True:
        elapsed = time.perf_counter() - start
        if elapsed >= seconds and steps > 0:
            break
        x, y, used = make_batch(
            BATCH_SIZE, distances, seed + steps, seq_len=seq_len,
        )
        if not set(used.tolist()).issubset(distances):
            raise RuntimeError("unregistered training distance")
        optimizer.zero_grad(set_to_none=True)
        logits = query_binary_logits(model, x)
        loss = F.cross_entropy(logits, y)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("nonfinite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        final_loss = float(loss.detach())
        steps += 1
    actual_seconds = time.perf_counter() - start
    if actual_seconds < seconds or steps < 1:
        raise RuntimeError("under-consumed CPU phase")
    tokens = steps * BATCH_SIZE * seq_len
    return {
        "steps": steps, "tokens_seen": tokens,
        "train_seconds": actual_seconds,
        "tokens_per_second": tokens / actual_seconds,
        "last_train_binary_nll": final_loss,
    }


def train_candidate(name: str) -> dict:
    torch.manual_seed(MODEL_SEED)
    model = make_models()[name]
    assert parameter_count(model) == EXPECTED_PARAMETERS
    opt = torch.optim.AdamW(
        model.parameters(), lr=EASY_LEARNING_RATE, weight_decay=0.01,
    )
    initial_long = eval_all(model)
    easy = timed_phase(
        model, opt, seconds=EASY_SECONDS, seed=EASY_TRAIN_SEED,
        seq_len=EASY_SEQ_LEN, distances=(1,),
    )
    control = eval_distance(
        model, 1, EASY_SEQ_LEN, EVAL_SEED + 100,
    )
    before_long = eval_all(model)
    for group in opt.param_groups:
        group["lr"] = LONG_LEARNING_RATE
    long = timed_phase(
        model, opt, seconds=LONG_SECONDS, seed=LONG_TRAIN_SEED,
        seq_len=SEQ_LEN, distances=DISTANCES,
    )
    after_long = eval_all(model)
    easy_after = eval_distance(
        model, 1, EASY_SEQ_LEN, EVAL_SEED + 100,
    )
    return {
        "name": name, "parameters": parameter_count(model),
        "initial_long_by_distance": initial_long,
        "easy_training": easy,
        "easy_positive_control_after_phase1": control,
        "long_before_by_distance": before_long,
        "long_training": long,
        "long_after_by_distance": after_long,
        "easy_control_after_phase2": easy_after,
    }


def run() -> dict:
    torch.set_num_threads(2)
    candidates = [train_candidate(name) for name in CANDIDATES]
    return {
        "schema": 1, "job_id": JOB_ID, "status": "complete",
        "classification": "CPU_SYNTHETIC_LASTWRITE_CURRICULUM_EXPLORATORY_ONLY",
        "gpu_requested": False, "engineering_only": True,
        "scientific_execution": False, "breakthrough_claim_supported": False,
        "protocol": {
            "model_order": list(CANDIDATES),
            "expected_parameters_each": EXPECTED_PARAMETERS,
            "easy_seq_len": EASY_SEQ_LEN, "long_seq_len": SEQ_LEN,
            "easy_distance": 1, "long_distances": list(DISTANCES),
            "easy_seconds": EASY_SECONDS, "long_seconds": LONG_SECONDS,
            "no_max_step_cap": True, "batch_size": BATCH_SIZE,
            "easy_learning_rate": EASY_LEARNING_RATE,
            "long_learning_rate": LONG_LEARNING_RATE,
            "weight_decay": 0.01, "optimizer": "AdamW",
            "easy_train_seed": EASY_TRAIN_SEED,
            "long_train_seed": LONG_TRAIN_SEED,
            "eval_seed": EVAL_SEED, "model_seed": MODEL_SEED,
            "eval_examples_per_class_per_distance": EVAL_PER_CLASS,
        },
        "results": candidates,
        "runtime": {
            "torch_version": str(torch.__version__),
            "cpu_threads": torch.get_num_threads(),
        },
        "interpretation_ceiling": (
            "CPU-only, one seed, exactly matched tiny 29,504-parameter models, "
            "20-second easy near-write control and 90-second length-64 distance "
            "curriculum per architecture, balanced held-out targets. Synthetic "
            "task evidence only; no general language, breakthrough, GPU science, "
            "replication or scale authority."
        ),
    }


if __name__ == "__main__":
    payload = run()
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "job_id": JOB_ID,
        "scores": [{
            "name": row["name"],
            "easy": row["easy_positive_control_after_phase1"]["accuracy"],
            "long": {d: cell["accuracy"] for d, cell in row["long_after_by_distance"].items()},
        } for row in payload["results"]],
    }, sort_keys=True))
