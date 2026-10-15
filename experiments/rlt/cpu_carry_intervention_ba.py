"""BA: one-shot CPU-only learned recurrent-state causal-memory diagnostic.

Do not rerun BA, even following an infrastructure failure. Fresh engineering seed.
Train exactly one independently initialized carry-RLT, save final weights, and
test paired opposite-label inputs with identical post-write suffixes. Report
state divergence and readout sensitivity separately, not as causal proof of a
general language-model breakthrough.
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
from experiments.rlt.cpu_bounded_ax_audit import scan_retention_trace
from experiments.rlt.cpu_lastwrite_probe_ar import SET0, SET1, make_batch, last_write_labels
from experiments.rlt.model import parameter_count
from experiments.rlt.model_streaming_bounded import BoundedChunkGatedRLT, tiny_streaming_config

JOB_ID = "rlt-cpu-carry-intervention-20261010-ba"
MODEL_SEED, EASY_SEED, LONG_SEED = 20541081, 20511081, 20521081
EVAL_SEED, DIAG_SEED = 20531081, 20551081
DISTANCES = (1, 4, 16, 64, 127)
SEQ_LEN, CHUNK_SIZE, BATCH_SIZE = 128, 8, 8
EASY_SECONDS, LONG_SECONDS = 15.0, 90.0
EVAL_PER_CLASS = 128
EXPECTED_PARAMETERS = 27712
RESULT = Path("experiments/rlt/cpu/results/rlt-carry-intervention-20261010-ba.json")
CHECKPOINT = Path("experiments/rlt/cpu/checkpoints/rlt-carry-intervention-20261010-ba/carry.pt")


def build() -> BoundedChunkGatedRLT:
    torch.manual_seed(MODEL_SEED)
    model = BoundedChunkGatedRLT(tiny_streaming_config(), carry_between_chunks=True)
    if parameter_count(model) != EXPECTED_PARAMETERS:
        raise RuntimeError("BA model parameter count drift")
    return model


def binary_logits(model, x: torch.Tensor) -> torch.Tensor:
    return model(x)[:, -1, SET0:SET1 + 1]


@torch.no_grad()
def evaluate(model) -> dict:
    model.eval()
    output = {}
    for gap in DISTANCES:
        x, y = balanced_holdout(gap, seed=EVAL_SEED + gap)
        total_nll, count, per_class = 0.0, 0, [0, 0]
        for start in range(0, len(y), BATCH_SIZE):
            end = start + BATCH_SIZE
            logits = binary_logits(model, x[start:end])
            truth = y[start:end]
            predictions = logits.argmax(-1)
            count += int((predictions == truth).sum())
            total_nll += float(F.cross_entropy(logits, truth, reduction="sum"))
            for cls in (0, 1):
                per_class[cls] += int(((predictions == cls) & (truth == cls)).sum())
        output[str(gap)] = {
            "examples": len(y), "accuracy": count / len(y),
            "binary_nll": total_nll / len(y),
            "class0_accuracy": per_class[0] / EVAL_PER_CLASS,
            "class1_accuracy": per_class[1] / EVAL_PER_CLASS,
        }
    return output


def timed_train(model, optim, *, seconds: float, seed: int, length: int, gaps: tuple[int, ...]) -> dict:
    model.train()
    start = time.perf_counter()
    steps = 0
    final_loss = math.nan
    while steps == 0 or (time.perf_counter() - start) < seconds:
        x, y, _ = make_batch(BATCH_SIZE, gaps, seed + steps, seq_len=length)
        optim.zero_grad(set_to_none=True)
        loss = F.cross_entropy(binary_logits(model, x), y)
        if not torch.isfinite(loss).item():
            raise RuntimeError("nonfinite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        final_loss = float(loss.detach())
        steps += 1
    duration = time.perf_counter() - start
    if duration < seconds:
        raise RuntimeError("training underconsumed duration")
    n_tokens = steps * BATCH_SIZE * length
    return {
        "steps": steps, "seconds": duration, "tokens_seen": n_tokens,
        "tokens_per_second": n_tokens / duration, "last_batch_binary_nll": final_loss,
    }


def counterfactual_pairs(gap: int, seed: int, n_each_class: int = 8):
    if gap not in DISTANCES:
        raise ValueError("gap unregistered")
    x, y = balanced_holdout(gap, seed=seed)
    a = torch.cat((x[y == 0][:n_each_class], x[y == 1][:n_each_class]), 0)
    target = torch.cat((y[y == 0][:n_each_class], y[y == 1][:n_each_class]), 0)
    b = a.clone()
    write_index = SEQ_LEN - 1 - gap
    b[:, write_index] = SET0 + (1 - target)
    if not torch.equal(last_write_labels(b), 1 - target):
        raise RuntimeError("counterfactual oracle mismatch")
    if not torch.equal(a[:, write_index + 1:], b[:, write_index + 1:]):
        raise RuntimeError("post-write suffix changed")
    return a, b, target


@torch.no_grad()
def compare_carry_states(model: BoundedChunkGatedRLT, gap: int, *, seed: int) -> dict:
    model.eval()
    a, b, targets = counterfactual_pairs(gap, seed)
    sa, sb = model.init_stream(len(a)), model.init_stream(len(b))
    dist, rel_dist = [], []
    states_before_final = None
    last_a, last_b = None, None
    for start in range(0, SEQ_LEN, CHUNK_SIZE):
        if start == SEQ_LEN - CHUNK_SIZE:
            states_before_final = (sa, sb)
        out_a, sa = model.forward_chunk(a[:, start:start+CHUNK_SIZE], sa)
        out_b, sb = model.forward_chunk(b[:, start:start+CHUNK_SIZE], sb)
        delta = (sa.recurrent - sb.recurrent).float().norm(dim=-1)
        normalizer = (
            0.5 * (sa.recurrent.float().norm(dim=-1) + sb.recurrent.float().norm(dim=-1))
            + 1e-8
        )
        dist.append(float(delta.mean()))
        rel_dist.append(float((delta / normalizer).mean()))
        last_a, last_b = out_a[:, -1, SET0:SET1+1], out_b[:, -1, SET0:SET1+1]
    if states_before_final is None:
        raise RuntimeError("missing last-chunk boundary")
    predicted_a, predicted_b = last_a.argmax(-1), last_b.argmax(-1)
    # Because the final chunk is identical when gap>=8, this isolates the
    # effect of the prior recurrent state at the final readout.
    last_chunk_equal = torch.equal(a[:, -CHUNK_SIZE:], b[:, -CHUNK_SIZE:])
    if gap >= CHUNK_SIZE and not last_chunk_equal:
        raise RuntimeError("invalid off-chunk comparison")
    flip_consistency = float(((predicted_a == targets) & (predicted_b == 1 - targets)).float().mean())
    mean_logits_l1 = float((last_a - last_b).abs().mean())
    if gap >= CHUNK_SIZE:
        pre_a, pre_b = states_before_final
        # Reuse *same* last chunk, with two legitimately produced states.
        swapped_a, _ = model.forward_chunk(a[:, -CHUNK_SIZE:], pre_a)
        swapped_b, _ = model.forward_chunk(a[:, -CHUNK_SIZE:], pre_b)
        torch.testing.assert_close(
            swapped_a[:, -1, SET0:SET1+1], last_a, atol=0, rtol=0
        )
        torch.testing.assert_close(
            swapped_b[:, -1, SET0:SET1+1], last_b, atol=0, rtol=0
        )
    retention = scan_retention_trace(model, a, last_write_distance=gap)
    return {
        "gap": gap, "examples": len(targets), "write_index": SEQ_LEN-1-gap,
        "identical_final_chunk": last_chunk_equal,
        "state_l2_by_chunk": dist,
        "state_relative_l2_by_chunk": rel_dist,
        "readout_logit_mean_absolute_difference": mean_logits_l1,
        "counterfactual_both_correct_fraction": flip_consistency,
        "mean_direct_retention_over_gap": retention["mean_direct_retention_over_gap"],
        "state_norm_by_chunk": retention["state_norm_by_token"][CHUNK_SIZE-1::CHUNK_SIZE],
        "write_norm_by_chunk": retention["write_norm_by_token"][CHUNK_SIZE-1::CHUNK_SIZE],
        "mean_gate": retention["mean_gate"],
        "fraction_gate_above_0_99": retention["fraction_gate_above_0_99"],
        "class_balance_each": len(targets)//2,
    }


def save_checkpoint_once(model) -> dict:
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    with CHECKPOINT.open("xb") as fd:
        torch.save({
            "job_id": JOB_ID, "model_seed": MODEL_SEED,
            "model_state_dict": model.state_dict(),
        }, fd)
    raw = CHECKPOINT.read_bytes()
    return {"path": CHECKPOINT.as_posix(), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def run() -> dict:
    torch.set_num_threads(2)
    if RESULT.exists() or CHECKPOINT.exists():
        raise RuntimeError("BA consumed/persisted output already exists; rerun forbidden")
    model = build()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    initial = evaluate(model)
    easy = timed_train(model, opt, seconds=EASY_SECONDS, seed=EASY_SEED, length=8, gaps=(1,))
    for group in opt.param_groups:
        group["lr"] = 1e-3
    long = timed_train(model, opt, seconds=LONG_SECONDS, seed=LONG_SEED, length=SEQ_LEN, gaps=DISTANCES)
    scores = evaluate(model)
    causal = {
        str(gap): compare_carry_states(model, gap, seed=DIAG_SEED + gap)
        for gap in DISTANCES
    }
    checkpoint = save_checkpoint_once(model)
    return {
        "schema": 1, "job_id": JOB_ID, "status": "complete",
        "classification": "CPU_CARRY_MEMORY_ATTRIBUTION_ENGINEERING_ONLY",
        "gpu_requested": False, "scientific_execution": False,
        "breakthrough_claim_supported": False,
        "protocol": {
            "model": "bounded_chunk_gated_carry", "parameters": EXPECTED_PARAMETERS,
            "window": CHUNK_SIZE, "seq_len": SEQ_LEN, "distances": list(DISTANCES),
            "batch_size": BATCH_SIZE, "balanced_eval_per_class": EVAL_PER_CLASS,
            "seeds": dict(model=MODEL_SEED, easy=EASY_SEED, long=LONG_SEED, eval=EVAL_SEED, diagnostic=DIAG_SEED),
            "optimizer": "AdamW", "weight_decay": 0.01,
            "easy_lr": 3e-3, "long_lr": 1e-3,
            "easy_seconds": EASY_SECONDS, "long_seconds": LONG_SECONDS,
            "counterfactual_pairs_per_class": 8,
            "same_suffix_held_constant": True, "automatic_retry": False,
        },
        "training": {"easy": easy, "long": long},
        "initial": initial, "final": scores, "counterfactual_diagnostics": causal,
        "checkpoint": checkpoint,
        "results": [{"name": "carry", "checkpoint": checkpoint}],
        "runtime": {"torch": torch.__version__, "cpu_threads": torch.get_num_threads()},
        "interpretation_ceiling": (
            "One new CPU engineering seed. Counterfactual state/readout observations "
            "measure this trained model only, and do not establish causal mechanism "
            "of AW or AY. Full-BPTT activation memory is not constant. "
            "No language superiority, scientific replication, GPU or scale-up."
        )
    }


if __name__ == "__main__":
    result = run()
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    with RESULT.open("x") as fd:
        json.dump(result, fd, indent=2, sort_keys=True)
        fd.write("\n")
    print(json.dumps({
        "job_id": JOB_ID,
        "accuracy": {k: v["accuracy"] for k, v in result["final"].items()},
        "counterfactual_pairs_correct": {
            k: v["counterfactual_both_correct_fraction"]
            for k, v in result["counterfactual_diagnostics"].items()
        },
        "checkpoint_sha256": result["checkpoint"]["sha256"],
    }, sort_keys=True))
