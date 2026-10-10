"""AY: one-shot CPU engineering comparison of three bounded-memory designs.

Do not rerun or reuse the job, seeds, or checkpoints after an attempt. This is
an engineering synthetic task, not scientific replication or a language win.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from experiments.rlt.cpu_bounded_streaming_aw import balanced_holdout
from experiments.rlt.cpu_bounded_ax_audit import prove_unreachable_counterfactuals, scan_retention_trace
from experiments.rlt.cpu_lastwrite_probe_ar import SET0, SET1, make_batch
from experiments.rlt.model import parameter_count
from experiments.rlt.model_sliding_window_ax import SlidingWindowTransformer
from experiments.rlt.model_streaming_bounded import BoundedChunkGatedRLT, tiny_streaming_config

JOB_ID = "rlt-cpu-bounded-threeway-20261010-ay"
CANDIDATES = ("carry", "reset", "sliding_kv")
DISTANCES = (1, 4, 16, 64, 127)
SEQ_LEN, BATCH_SIZE, EVAL_PER_CLASS = 128, 8, 128
EASY_SECONDS, LONG_SECONDS = 15.0, 90.0
EASY_SEED, LONG_SEED = 20411081, 20421081
EVAL_SEED, MODEL_SEED, DIAG_SEED = 20431081, 20441081, 20451081
EXPECTED_PARAMETERS = {"carry": 27712, "reset": 27712, "sliding_kv": 27680}
RESULT_PATH = Path("experiments/rlt/cpu/results/rlt-bounded-threeway-20261010-ay.json")
CHECKPOINT_DIR = Path("experiments/rlt/cpu/checkpoints/rlt-bounded-threeway-20261010-ay")


def build_models() -> dict[str, torch.nn.Module]:
    torch.manual_seed(MODEL_SEED)
    carry = BoundedChunkGatedRLT(tiny_streaming_config(), carry_between_chunks=True)
    torch.manual_seed(MODEL_SEED)
    reset = BoundedChunkGatedRLT(tiny_streaming_config(), carry_between_chunks=False)
    reset.load_state_dict(carry.state_dict(), strict=True)
    torch.manual_seed(MODEL_SEED)
    sliding = SlidingWindowTransformer()
    models = dict(carry=carry, reset=reset, sliding_kv=sliding)
    counts = {name: parameter_count(model) for name, model in models.items()}
    if counts != EXPECTED_PARAMETERS:
        raise RuntimeError(f"parameter budget drift {counts}")
    for key, value in carry.state_dict().items():
        torch.testing.assert_close(value, reset.state_dict()[key], atol=0, rtol=0)
    return models


def binary_logits(model, x: torch.Tensor) -> torch.Tensor:
    return model(x)[:, -1, SET0:SET1 + 1]


@torch.no_grad()
def evaluate(model) -> dict[str, dict[str, float | int]]:
    model.eval()
    rows = {}
    for distance in DISTANCES:
        x, y = balanced_holdout(distance, seed=EVAL_SEED + distance)
        if x.shape != (2 * EVAL_PER_CLASS, SEQ_LEN) or int(y.sum()) != EVAL_PER_CLASS:
            raise RuntimeError("held-out class balance mismatch")
        correct, class_correct, nll_total = 0, [0, 0], 0.0
        for start in range(0, len(y), BATCH_SIZE):
            a, b = x[start:start + BATCH_SIZE], y[start:start + BATCH_SIZE]
            logits = binary_logits(model, a)
            pred = logits.argmax(-1)
            correct += int((pred == b).sum())
            nll_total += float(F.cross_entropy(logits, b, reduction="sum"))
            for cls in (0, 1):
                class_correct[cls] += int(((pred == cls) & (b == cls)).sum())
        rows[str(distance)] = {
            "examples": len(y), "accuracy": correct / len(y),
            "binary_nll": nll_total / len(y),
            "class0_accuracy": class_correct[0] / EVAL_PER_CLASS,
            "class1_accuracy": class_correct[1] / EVAL_PER_CLASS,
        }
    return rows


@torch.no_grad()
def easy_control(model) -> dict[str, float]:
    model.eval()
    x, y, _ = make_batch(256, (1,), EVAL_SEED + 200, seq_len=8)
    logits = torch.cat([binary_logits(model, x[i:i + 8]) for i in range(0, 256, 8)])
    return {"accuracy": float((logits.argmax(-1) == y).float().mean()),
            "binary_nll": float(F.cross_entropy(logits, y))}


def timed_training(model, opt, seconds: float, seed: int, length: int, distances: tuple[int, ...]) -> dict:
    model.train()
    start = time.perf_counter()
    steps, last_loss = 0, math.nan
    while True:
        if time.perf_counter() - start >= seconds and steps > 0:
            break
        x, y, _ = make_batch(BATCH_SIZE, distances, seed + steps, seq_len=length)
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(binary_logits(model, x), y)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("nonfinite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        last_loss = float(loss.detach())
        steps += 1
    actual = time.perf_counter() - start
    if steps < 1 or actual < seconds:
        raise RuntimeError("training time budget not consumed")
    tokens = steps * BATCH_SIZE * length
    return {"steps": steps, "seconds": actual, "tokens_seen": tokens,
            "tokens_per_second": tokens / actual, "last_batch_binary_nll": last_loss}


@torch.no_grad()
def retention_diagnostics(carry: BoundedChunkGatedRLT) -> dict:
    output = {}
    for distance in (16, 64, 127):
        x, y = balanced_holdout(distance, seed=DIAG_SEED + distance)
        for cls in (0, 1):
            subset = x[y == cls][:8]
            traces = scan_retention_trace(carry, subset, last_write_distance=distance)
            output[f"{distance}-class{cls}"] = {
                "examples": len(subset),
                "mean_gate": traces["mean_gate"],
                "fraction_gate_above_0_99": traces["fraction_gate_above_0_99"],
                "fraction_gate_below_0_5": traces["fraction_gate_below_0_5"],
                "mean_direct_retention_over_gap": traces["mean_direct_retention_over_gap"],
                "state_norm_each_chunk": traces["state_norm_by_token"][7::8],
                "write_norm_each_chunk": traces["write_norm_by_token"][7::8],
            }
    return output


def save_once(model, name: str) -> dict[str, str | int]:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINT_DIR / f"{name}.pt"
    with path.open("xb") as f:
        torch.save({"job_id": JOB_ID, "name": name, "state_dict": model.state_dict()}, f)
    raw = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def run() -> dict:
    torch.set_num_threads(2)
    if RESULT_PATH.exists() or CHECKPOINT_DIR.exists():
        raise RuntimeError("AY attempt already has persisted output, refusing rerun")
    models = build_models()
    leak_proof = prove_unreachable_counterfactuals(models["sliding_kv"])
    rows = []
    for name in CANDIDATES:
        model = models[name]
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
        initial = evaluate(model)
        easy = timed_training(model, opt, EASY_SECONDS, EASY_SEED, 8, (1,))
        easy_check = easy_control(model)
        for group in opt.param_groups:
            group["lr"] = 1e-3
        longer = timed_training(model, opt, LONG_SECONDS, LONG_SEED, SEQ_LEN, DISTANCES)
        final = evaluate(model)
        rows.append({"name": name, "parameters": parameter_count(model),
                     "initial": initial, "easy_training": easy,
                     "easy_positive_control": easy_check, "long_training": longer,
                     "final": final, "checkpoint": save_once(model, name)})
    trace = retention_diagnostics(models["carry"])
    return {
        "schema": 1, "job_id": JOB_ID, "status": "complete",
        "classification": "CPU_SYNTHETIC_BOUNDED_MEMORY_ENGINEERING_ONLY",
        "gpu_requested": False, "engineering_only": True, "scientific_execution": False,
        "breakthrough_claim_supported": False,
        "protocol": {
            "candidate_order": list(CANDIDATES), "model_parameters": EXPECTED_PARAMETERS,
            "seq_len": SEQ_LEN, "window": 8, "sliding_layers": 2,
            "transformer_full_receptive_distance": 14,
            "distances": list(DISTANCES), "batch_size": BATCH_SIZE,
            "balanced_eval_per_class_per_distance": EVAL_PER_CLASS,
            "easy_seconds_each": EASY_SECONDS, "long_seconds_each": LONG_SECONDS,
            "easy_lr": 3e-3, "long_lr": 1e-3, "optimizer": "AdamW", "weight_decay": 0.01,
            "train_seeds": {"easy": EASY_SEED, "long": LONG_SEED, "model": MODEL_SEED},
            "eval_seed": EVAL_SEED, "diagnostic_seed": DIAG_SEED,
            "same_time_not_same_tokens_or_flops": True,
            "parameter_difference_relative_to_rlt": (27680 - 27712) / 27712,
            "independent_counterfactual_horizon_proof": True,
            "no_trained_aw_checkpoint_available": True,
        },
        "information_bottleneck_proofs": leak_proof,
        "carry_retention_diagnostics_after_ay_training": trace,
        "results": rows,
        "runtime": {"torch": torch.__version__, "cpu_threads": torch.get_num_threads()},
        "interpretation_ceiling": (
            "Three tiny models, one fresh CPU engineering seed, synthetic SET/QUERY task; "
            "AX transformer effective receptive distance 14 and cannot access distant SET at "
            "gaps 16/64/127. Close parameters and equal wall-time only, not equal FLOPs "
            "or equal tokens. No broad Transformer or language-model superiority."
        ),
    }


if __name__ == "__main__":
    output = run()
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULT_PATH.open("x") as f:
        json.dump(output, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps({"job_id": JOB_ID, "results": {
        r["name"]: {d: s["accuracy"] for d, s in r["final"].items()}
        for r in output["results"]}}, sort_keys=True))
