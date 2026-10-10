"""AW: first CPU-only full-model fixed-chunk memory vs reset-state control.

Exactly equal 27,712 trainable parameters and identical local encoder/decoder
computation in each model. The sole experimental difference is whether the
learned affine-scan state crosses an 8-token chunk boundary. No attention KV
or encoder/cross-attention memory persists beyond a chunk. Length-128 final
query tasks probe distances [1,4,16,64,127].

A no-carry model cannot infer a balanced target set by an earlier chunk, unless
the benchmark leaks it to the final chunk. We test that counterfactual at
generation time. This is a narrow synthetic-inductive-bias demonstration, not
a general-language Transformer-beating claim.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from experiments.rlt.cpu_lastwrite_probe_ar import (
    SET0, SET1, QUERY, last_write_labels, make_batch,
)
from experiments.rlt.model import parameter_count
from experiments.rlt.model_streaming_bounded import (
    BoundedChunkGatedRLT, tiny_streaming_config,
)

JOB_ID = "rlt-cpu-bounded-streaming-vs-reset-20261010-aw"
DISTANCES = (1, 4, 16, 64, 127)
SEQ_LEN = 128
CHUNK_SIZE = 8
BATCH_SIZE = 8
EVAL_PER_CLASS = 128
PARAMETERS = 27_712
EASY_SECONDS = 15.0
LONG_SECONDS = 90.0
EASY_SEED = 20_261_081
LONG_SEED = 20_271_081
EVAL_SEED = 20_291_081
MODEL_SEED = 20_301_081
RESULT_PATH = Path(
    "experiments/rlt/cpu/results/rlt-bounded-streaming-vs-reset-20261010-aw.json"
)


def models() -> tuple[BoundedChunkGatedRLT, BoundedChunkGatedRLT]:
    cfg = tiny_streaming_config()
    torch.manual_seed(MODEL_SEED)
    carry = BoundedChunkGatedRLT(cfg, carry_between_chunks=True)
    torch.manual_seed(MODEL_SEED)
    reset = BoundedChunkGatedRLT(cfg, carry_between_chunks=False)
    reset.load_state_dict(carry.state_dict(), strict=True)
    counts = [parameter_count(carry), parameter_count(reset)]
    if counts != [PARAMETERS, PARAMETERS]:
        raise RuntimeError(f"parameter mismatch: {counts}")
    return carry, reset


def balanced_holdout(
    distance: int, *, seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    x, y, d = make_batch(1024, (distance,), seed, seq_len=SEQ_LEN)
    zeros = torch.nonzero(y == 0).flatten()[:EVAL_PER_CLASS]
    ones = torch.nonzero(y == 1).flatten()[:EVAL_PER_CLASS]
    if len(zeros) != EVAL_PER_CLASS or len(ones) != EVAL_PER_CLASS:
        raise RuntimeError("balanced heldout sample too small")
    chosen = torch.stack((zeros, ones), dim=1).reshape(-1)
    x, y = x[chosen], y[chosen]
    if not torch.equal(last_write_labels(x), y):
        raise RuntimeError("target does not match last-write oracle")
    if not bool((x[:, -1] == QUERY).all()):
        raise RuntimeError("query not at final input")
    if not bool((x[:, SEQ_LEN - 1 - distance] == y + SET0).all()):
        raise RuntimeError("forced last write missing")
    if not bool((d[chosen] == distance).all()):
        raise RuntimeError("incorrect distance")
    if int(y.sum()) != EVAL_PER_CLASS:
        raise RuntimeError("unbalanced targets")
    return x, y


def assert_no_target_information_in_bounded_suffix() -> None:
    """A deterministic counterfactual pair differs only before last chunk."""
    for distance in DISTANCES:
        if distance < CHUNK_SIZE:
            continue
        x, y = balanced_holdout(distance, seed=EVAL_SEED + distance)
        flips = x.clone()
        idx = SEQ_LEN - 1 - distance
        flips[:, idx] = (1 - y) + SET0
        if not torch.equal(last_write_labels(flips), 1 - y):
            raise RuntimeError("counterfactual oracle failed")
        if not torch.equal(x[:, -CHUNK_SIZE:], flips[:, -CHUNK_SIZE:]):
            raise RuntimeError("bounded suffix leaked target information")


def binary_logits(model, tokens):
    all_logits = model(tokens)
    return all_logits[:, -1, SET0:SET1 + 1]


@torch.no_grad()
def evaluate(model) -> dict[str, dict[str, float | int]]:
    model.eval()
    output = {}
    for distance in DISTANCES:
        x, y = balanced_holdout(distance, seed=EVAL_SEED + distance)
        nll_total, correct, per_class = 0.0, 0, [0, 0]
        for lo in range(0, len(y), 8):
            hi = lo + 8
            logits = binary_logits(model, x[lo:hi])
            truth = y[lo:hi]
            prediction = logits.argmax(dim=-1)
            nll_total += float(F.cross_entropy(logits, truth, reduction="sum"))
            correct += int((prediction == truth).sum())
            for label in (0, 1):
                per_class[label] += int(
                    ((prediction == label) & (truth == label)).sum()
                )
        output[str(distance)] = {
            "examples": len(y), "accuracy": correct / len(y),
            "binary_nll": nll_total / len(y),
            "class0_accuracy": per_class[0] / EVAL_PER_CLASS,
            "class1_accuracy": per_class[1] / EVAL_PER_CLASS,
        }
    return output


def timed_training(
    model, optimizer, *, seconds: float,
    seed: int, length: int, distances: tuple[int, ...],
) -> dict[str, float | int]:
    model.train()
    start = time.perf_counter()
    steps, last_loss = 0, math.nan
    while True:
        elapsed = time.perf_counter() - start
        if elapsed >= seconds and steps > 0:
            break
        x, y, _ = make_batch(
            BATCH_SIZE, distances, seed + steps, seq_len=length
        )
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(binary_logits(model, x), y)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("nonfinite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        last_loss = float(loss.detach())
        steps += 1
    actual = time.perf_counter() - start
    if actual < seconds or steps == 0:
        raise RuntimeError("underconsumed CPU train budget")
    return {
        "steps": steps, "seconds": actual,
        "tokens_seen": steps * BATCH_SIZE * length,
        "tokens_per_second": steps * BATCH_SIZE * length / actual,
        "final_batch_binary_nll": last_loss,
    }


def evaluate_short_positive_control(model) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        x, y, _ = make_batch(
            256, (1,), EVAL_SEED + 200, seq_len=8
        )
        logits = torch.cat(
            [binary_logits(model, x[i:i + 8]) for i in range(0, 256, 8)]
        )
        return {
            "accuracy": float((logits.argmax(-1) == y).float().mean()),
            "binary_nll": float(F.cross_entropy(logits, y)),
        }


def train_one(model, name: str) -> dict:
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    initial = evaluate(model)
    easy = timed_training(
        model, opt, seconds=EASY_SECONDS, seed=EASY_SEED,
        length=8, distances=(1,),
    )
    easy_control = evaluate_short_positive_control(model)
    for group in opt.param_groups:
        group["lr"] = 1e-3
    long = timed_training(
        model, opt, seconds=LONG_SECONDS, seed=LONG_SEED,
        length=SEQ_LEN, distances=DISTANCES,
    )
    final = evaluate(model)
    return {
        "name": name, "parameters": parameter_count(model),
        "initial": initial, "easy_training": easy,
        "easy_positive_control": easy_control,
        "long_training": long, "final": final,
    }


def run() -> dict:
    torch.set_num_threads(2)
    assert_no_target_information_in_bounded_suffix()
    carry, reset = models()
    rows = [
        train_one(carry, "streaming_carry"),
        train_one(reset, "same_compute_reset_control"),
    ]
    return {
        "schema": 1, "job_id": JOB_ID, "status": "complete",
        "classification": "CPU_BOUNDED_STREAMING_MEMORY_ENGINEERING_ONLY",
        "gpu_requested": False, "scientific_execution": False,
        "breakthrough_claim_supported": False, "engineering_only": True,
        "protocol": {
            "models": ["streaming_carry", "same_compute_reset_control"],
            "same_architecture_weights_except_carry": True,
            "same_initialization_seed": True,
            "full_cross_chunk_attention": False,
            "only_cross_chunk_memory_is_affine_state": True,
            "chunk_size": CHUNK_SIZE, "seq_len": SEQ_LEN,
            "distances": list(DISTANCES), "batch_size": BATCH_SIZE,
            "params_each": PARAMETERS,
            "balanced_eval_examples_per_class_per_distance": EVAL_PER_CLASS,
            "easy_seconds_each": EASY_SECONDS, "long_seconds_each": LONG_SECONDS,
            "no_step_cap": True, "easy_lr": 3e-3, "long_lr": 1e-3,
            "optimizer": "AdamW", "weight_decay": 0.01,
            "easy_seed": EASY_SEED, "long_seed": LONG_SEED,
            "eval_seed": EVAL_SEED, "model_seed": MODEL_SEED,
            "counterfactual_suffix_leakage_test": True,
        },
        "results": rows,
        "runtime": {"torch": torch.__version__, "cpu_threads": torch.get_num_threads()},
        "interpretation_ceiling": (
            "CPU-only one-seed synthetic last-write task. The explicit binary "
            "SET/QUERY protocol favors architectures with learned persistent state "
            "when both architectures are denied old chunks. This does not establish "
            "Transformer inferiority at equal language context, compute, or capability. "
            "No GPU science, scaling, replication, or breakthrough claim."
        ),
    }


if __name__ == "__main__":
    result = run()
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "job_id": JOB_ID, "scores": [
            {"name": r["name"], "easy": r["easy_positive_control"],
             "accuracy_by_distance": {k: v["accuracy"] for k, v in r["final"].items()}}
            for r in result["results"]
        ]
    }, sort_keys=True))
