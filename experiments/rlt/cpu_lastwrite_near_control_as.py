"""AS CPU-only easy last-write positive control, not a scientific result.

Models with exactly 29,504 trainable parameters must classify the last SET
immediately preceding QUERY in a short 8-token prompt. This diagnoses whether
the AR task/readout/optimizer pipeline can learn, before interpreting memory
over longer spans. All models receive their FULL 25-second CPU training budget.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from experiments.rlt.cpu_lastwrite_probe_ar import (
    CANDIDATES, BATCH_SIZE, EXPECTED_PARAMETERS, QUERY, SET0,
    last_write_labels, make_batch, make_models, query_binary_logits,
)
from experiments.rlt.model import parameter_count

JOB_ID = "rlt-cpu-lastwrite-near-positive-control-20261009-as"
SEQ_LEN = 8
DISTANCES = (1,)
TRAIN_SECONDS_PER_CANDIDATE = 25.0
TRAIN_SEED = 20_261_062
EVAL_SEED = 20_291_062
MODEL_SEED = 20_301_062
EVAL_PER_CLASS = 128
LR = 3e-3
RESULT_PATH = Path(
    "experiments/rlt/cpu/results/rlt-lastwrite-near-positive-control-20261009-as.json"
)


def balanced_holdout() -> tuple[torch.Tensor, torch.Tensor]:
    # Evaluate both labels equally. Never disclose an answer after QUERY.
    x, y, distances = make_batch(
        1024, DISTANCES, EVAL_SEED, seq_len=SEQ_LEN,
    )
    n = EVAL_PER_CLASS
    zeros = torch.nonzero(y == 0).flatten()[:n]
    ones = torch.nonzero(y == 1).flatten()[:n]
    if len(zeros) != n or len(ones) != n:
        raise RuntimeError("insufficient examples to construct balanced holdout")
    ids = torch.stack((zeros, ones), dim=1).reshape(-1)
    x, y = x[ids], y[ids]
    if not torch.equal(last_write_labels(x), y):
        raise RuntimeError("heldout labels differ from latest-write oracle")
    if not bool((x[:, -1] == QUERY).all()):
        raise RuntimeError("query token not at final position")
    if not bool((x[:, -2] == y + SET0).all()):
        raise RuntimeError("last SET is not immediately before QUERY")
    if int(y.sum()) != n:
        raise RuntimeError("heldout labels not precisely balanced")
    if not bool((distances[ids] == 1).all()):
        raise RuntimeError("heldout nearest-write gap drift")
    return x, y


@torch.no_grad()
def evaluate(model: torch.nn.Module) -> dict[str, float | int]:
    x, y = balanced_holdout()
    model.eval()
    total_nll, correct, class_correct = 0.0, 0, [0, 0]
    for lo in range(0, len(y), BATCH_SIZE):
        hi = lo + BATCH_SIZE
        logits = query_binary_logits(model, x[lo:hi])
        truth = y[lo:hi]
        predicted = logits.argmax(dim=-1)
        total_nll += float(F.cross_entropy(logits, truth, reduction="sum"))
        correct += int((predicted == truth).sum())
        for label in (0, 1):
            class_correct[label] += int(
                ((predicted == label) & (truth == label)).sum()
            )
    return {
        "examples": len(y),
        "accuracy": correct / len(y),
        "binary_nll": total_nll / len(y),
        "class0_accuracy": class_correct[0] / EVAL_PER_CLASS,
        "class1_accuracy": class_correct[1] / EVAL_PER_CLASS,
    }


def train_one(name: str) -> dict:
    torch.manual_seed(MODEL_SEED)
    model = make_models()[name]
    if parameter_count(model) != EXPECTED_PARAMETERS:
        raise RuntimeError("positive-control parameter mismatch")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=0.01,
    )
    initial = evaluate(model)
    model.train()
    start = time.perf_counter()
    steps, last_loss = 0, math.nan
    while True:
        elapsed = time.perf_counter() - start
        if elapsed >= TRAIN_SECONDS_PER_CANDIDATE and steps > 0:
            break
        x, y, distances = make_batch(
            BATCH_SIZE, DISTANCES, TRAIN_SEED + steps, seq_len=SEQ_LEN,
        )
        if not bool((distances == 1).all()):
            raise RuntimeError("training examples are not near-write")
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(query_binary_logits(model, x), y)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"nonfinite loss for {name} step {steps}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        last_loss = float(loss.detach())
        steps += 1
    seconds = time.perf_counter() - start
    if steps < 1 or seconds < TRAIN_SECONDS_PER_CANDIDATE:
        raise RuntimeError("failed to consume full positive-control train window")
    return {
        "name": name, "parameters": parameter_count(model),
        "steps": steps, "tokens_seen": steps * BATCH_SIZE * SEQ_LEN,
        "actual_training_seconds": seconds,
        "tokens_per_second": steps * BATCH_SIZE * SEQ_LEN / seconds,
        "last_train_binary_nll": last_loss,
        "initial": initial, "final": evaluate(model),
    }


def run() -> dict:
    torch.set_num_threads(2)
    results = [train_one(name) for name in CANDIDATES]
    return {
        "schema": 1, "job_id": JOB_ID, "status": "complete",
        "classification": "CPU_SYNTHETIC_NEAR_WRITE_POSITIVE_CONTROL_ONLY",
        "gpu_requested": False, "engineering_only": True,
        "scientific_execution": False, "breakthrough_claim_supported": False,
        "protocol": {
            "task": "nearest-token last-write easy learnability control",
            "seq_len": SEQ_LEN, "distance": 1,
            "batch_size": BATCH_SIZE,
            "train_seconds_each": TRAIN_SECONDS_PER_CANDIDATE,
            "no_max_step_cap": True,
            "training_seed": TRAIN_SEED, "eval_seed": EVAL_SEED,
            "model_seed": MODEL_SEED, "expected_parameters_each": EXPECTED_PARAMETERS,
            "model_order": list(CANDIDATES),
            "balanced_heldout_examples_per_class": EVAL_PER_CLASS,
            "learning_rate": LR, "optimizer": "AdamW",
        },
        "results": results,
        "runtime": {"torch_version": torch.__version__, "cpu_threads": torch.get_num_threads()},
        "interpretation_ceiling": (
            "Exactly matched tiny models, CPU only, one seed, easy nearest-write "
            "readout diagnostic. Accuracy >=85% merely validates the task can be "
            "learned; this is not language generalization, architectural superiority, "
            "scientific replication, or GPU/scale authorization."
        ),
    }


if __name__ == "__main__":
    data = run()
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n")
    print(json.dumps({
        "status": data["status"],
        "scores": [{k: row[k] for k in ("name", "steps", "actual_training_seconds", "final")}
                   for row in data["results"]],
    }, sort_keys=True))
