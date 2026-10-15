"""BC: preregistered CPU-only fixed-update carry-RLT seed-stability panel.

This is a NEW synthetic engineering attempt. Never retry/resume/replay it,
including after infrastructure failure; AW, AY and BA are immutable.
Exactly equal NUMBER of easy and long optimizer updates across three fresh
initialization seeds. Shared deterministic training/evaluation examples.
Checkpoints plus per-seed JSON are written once; reporting all failures.
No GPU, no scientific claim, no architecture tuning between seeds.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from experiments.rlt.cpu_bounded_streaming_aw import balanced_holdout
from experiments.rlt.cpu_lastwrite_probe_ar import SET0, SET1, make_batch
from experiments.rlt.model import parameter_count
from experiments.rlt.model_streaming_bounded import BoundedChunkGatedRLT, tiny_streaming_config

JOB_ID = "rlt-cpu-fixed-update-seed-stability-20261015-bc"
MODEL_SEEDS = (20671081, 20671082, 20671083)
EASY_DATA_SEED = 20681081
LONG_DATA_SEED = 20691081
EVAL_SEED = 20701081
DISTANCES = (1, 4, 16, 64, 127)
SEQ_LEN, BATCH, WINDOW, PARAMETERS = 128, 8, 8, 27712
EASY_UPDATES, LONG_UPDATES = 2100, 1200
MILESTONES = (400, 800, 1200)
RESULT_PATH = Path("experiments/rlt/cpu/results/rlt-fixed-update-seed-stability-20261015-bc.json")
CASE_DIR = Path("experiments/rlt/cpu/cases/rlt-fixed-update-seed-stability-20261015-bc")
CHECKPOINT_DIR = Path("experiments/rlt/cpu/checkpoints/rlt-fixed-update-seed-stability-20261015-bc")


def build(model_seed: int) -> BoundedChunkGatedRLT:
    if model_seed not in MODEL_SEEDS:
        raise ValueError("unregistered BC model seed")
    torch.manual_seed(model_seed)
    model = BoundedChunkGatedRLT(tiny_streaming_config(), carry_between_chunks=True)
    if parameter_count(model) != PARAMETERS or model.chunk_size != WINDOW:
        raise RuntimeError("BC architecture/parameter drift")
    return model


def binary_logits(model, x: torch.Tensor):
    return model(x)[:, -1, SET0:SET1 + 1]


def holdouts() -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    out = {gap: balanced_holdout(gap, seed=EVAL_SEED + gap) for gap in DISTANCES}
    for gap, (x, y) in out.items():
        if x.shape != (256, SEQ_LEN) or len(y) != 256 or int(y.sum()) != 128:
            raise RuntimeError(f"BC heldout mismatch at gap {gap}")
    return out


@torch.no_grad()
def evaluate(model, eval_data: dict) -> dict:
    model.eval()
    result = {}
    for gap in DISTANCES:
        x, y = eval_data[gap]
        correct, loss, cls_correct = 0, 0., [0, 0]
        for i in range(0, len(y), BATCH):
            xb, truth = x[i:i+BATCH], y[i:i+BATCH]
            logits = binary_logits(model, xb)
            pred = logits.argmax(-1)
            correct += int((pred == truth).sum())
            loss += float(F.cross_entropy(logits, truth, reduction="sum"))
            for cls in (0, 1):
                cls_correct[cls] += int(((pred == cls) & (truth == cls)).sum())
        result[str(gap)] = {
            "accuracy": correct / 256, "binary_nll": loss / 256,
            "examples": 256, "class0_accuracy": cls_correct[0] / 128,
            "class1_accuracy": cls_correct[1] / 128,
        }
    return result


@torch.no_grad()
def short_positive_control(model) -> dict:
    model.eval()
    x, y, _ = make_batch(256, (1,), EVAL_SEED + 200, seq_len=8)
    logits = torch.cat([binary_logits(model, x[i:i+BATCH]) for i in range(0, 256, BATCH)])
    return {
        "examples": 256, "accuracy": float((logits.argmax(-1) == y).float().mean()),
        "binary_nll": float(F.cross_entropy(logits, y)),
    }


def train_updates(model, optimizer, *, stage: str, n_updates: int, data_seed: int, seq_len: int, gaps: tuple[int, ...]) -> dict:
    model.train()
    start = time.perf_counter()
    last_loss = 0.
    for step in range(n_updates):
        x, y, _ = make_batch(BATCH, gaps, data_seed + step, seq_len=seq_len)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(binary_logits(model, x), y)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"nonfinite BC {stage} loss at update {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        last_loss = float(loss.detach())
    duration = time.perf_counter() - start
    return {
        "stage": stage, "steps": n_updates, "seconds": duration,
        "tokens_seen": n_updates * BATCH * seq_len,
        "tokens_per_second": (n_updates * BATCH * seq_len) / max(duration, 1e-12),
        "last_batch_binary_nll": last_loss,
    }


def save_once(model, idx: int) -> dict:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINT_DIR / f"case{idx}.pt"
    with path.open("xb") as f:
        torch.save(
            {"job_id": JOB_ID, "case": idx, "model_seed": MODEL_SEEDS[idx],
             "model_state_dict": model.state_dict()}, f
        )
    raw = path.read_bytes()
    return {"path": path.as_posix(), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def train_case(idx: int, eval_data: dict) -> dict:
    model_seed = MODEL_SEEDS[idx]
    model = build(model_seed)
    initial = evaluate(model, eval_data)
    opt = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
    easy = train_updates(
        model, opt, stage="easy", n_updates=EASY_UPDATES,
        data_seed=EASY_DATA_SEED, seq_len=8, gaps=(1,)
    )
    easy_control = short_positive_control(model)
    for g in opt.param_groups:
        g["lr"] = 0.001
    # Fixed-update intermediate snapshots; no early stopping, reranking or
    # checkpoint-based intervention. All seeds consume same 1200 updates.
    milestones = {}
    previous = 0
    segments = []
    for at_step in MILESTONES:
        # Continue shared synthetic examples at their GLOBAL step indices,
        # rather than starting each segment's stream at step zero.
        count = at_step - previous
        segment = train_updates(
            model, opt, stage=f"long_{previous}_to_{at_step}",
            n_updates=count, data_seed=LONG_DATA_SEED + previous,
            seq_len=SEQ_LEN, gaps=DISTANCES
        )
        segments.append(segment)
        milestones[str(at_step)] = evaluate(model, eval_data)
        previous = at_step
    if previous != LONG_UPDATES:
        raise RuntimeError("incomplete BC registered update schedule")
    checkpoint = save_once(model, idx)
    row = {
        "case": idx, "name": "carry", "parameters": PARAMETERS,
        "model_seed": model_seed,
        "initial": initial, "easy": easy, "easy_positive_control": easy_control,
        "long_segments": segments, "after_long_milestones": milestones,
        "final": milestones[str(LONG_UPDATES)], "checkpoint": checkpoint,
        "total_long_steps": sum(p["steps"] for p in segments),
        "total_long_tokens": sum(p["tokens_seen"] for p in segments),
    }
    CASE_DIR.mkdir(parents=True, exist_ok=True)
    with (CASE_DIR / f"case{idx}.json").open("x") as f:
        json.dump(row, f, indent=2, sort_keys=True)
        f.write("\n")
    return row


def run() -> dict:
    if RESULT_PATH.exists() or CASE_DIR.exists() or CHECKPOINT_DIR.exists():
        raise RuntimeError("BC has existing evidence: consumed/no resume/no rerun")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    data = holdouts()
    cases = [train_case(idx, data) for idx in range(len(MODEL_SEEDS))]
    success = [all(c["final"][str(g)]["class0_accuracy"] >= 0.95
                   and c["final"][str(g)]["class1_accuracy"] >= 0.95
                   for g in (16, 64, 127)) for c in cases]
    return {
        "schema": 1, "job_id": JOB_ID, "status": "complete",
        "classification": "CPU_FIXED_UPDATE_THREE_SEED_ENGINEERING_ONLY",
        "gpu_requested": False, "scientific_execution": False,
        "breakthrough_claim_supported": False,
        "protocol": {
            "model_seeds": list(MODEL_SEEDS), "easy_data_seed": EASY_DATA_SEED,
            "long_data_seed": LONG_DATA_SEED, "eval_seed": EVAL_SEED,
            "same_training_examples_per_update_across_seeds": True,
            "same_updates_not_same_wall_time": True,
            "early_updates": EASY_UPDATES, "long_updates": LONG_UPDATES,
            "milestones": list(MILESTONES), "batch": BATCH,
            "seq_len": SEQ_LEN, "chunk_size": WINDOW, "params_each": PARAMETERS,
            "distances": list(DISTANCES), "balanced_eval_per_class": 128,
            "optimizer": "AdamW", "easy_lr": .003, "long_lr": .001,
            "weight_decay": .01, "automatic_retry": False,
            "success_definition": ">=95% accuracy on each class separately at gaps 16,64,127 after 1200 long optimizer updates",
        },
        "results": cases,
        "pass_by_case": success,
        "success_count": sum(success),
        "failure_count": len(success) - sum(success),
        "runtime": {"torch": torch.__version__, "cpu_threads": torch.get_num_threads()},
        "interpretation_ceiling": (
            "Three fresh fixed-update synthetic CPU engineering seeds, same shared "
            "training examples; can reveal training instability but cannot estimate "
            "a reliable population failure rate, establish language-model superiority "
            "or justify GPU science, scaling or breakthrough claims. Preserve all seeds."
        ),
    }


if __name__ == "__main__":
    d = run()
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULT_PATH.open("x") as f:
        json.dump(d, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps({
        "job_id": JOB_ID,
        "pass_by_case": d["pass_by_case"],
        "final_accuracy": [{
            "seed": c["model_seed"],
            "accuracy": {k: v["accuracy"] for k, v in c["final"].items()},
        } for c in d["results"]],
    }, sort_keys=True))
