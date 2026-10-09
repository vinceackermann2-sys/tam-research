"""AR exploratory CPU-only last-write state-retention benchmark.

This is NOT a general-language model quality comparison and uses NO GPU.
Three exactly parameter-matched 29,504-parameter models share a 64-token task,
same training and held-out seed protocol, and 35 seconds of CPU training each.

A synthetic prompt contains SET0, SET1 and distractor tokens, then QUERY at
the last position. The classification target is the value of the most recent
SET, which is never appended to the input sequence as an answer. The position
of that last SET determines the last-write distance bin.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from experiments.rlt.model import RLTConfig, parameter_count
from experiments.rlt.model_gated_scan_residual import ResidualGatedScanRLT
from experiments.rlt.model_gated_scan_adaptive import AdaptiveResidualGatedScanRLT
from tam_research.models import ModelConfig, ResearchLM

JOB_ID = "rlt-cpu-lastwrite-distance-probe-20261009-ar"
VOCAB = 16
SET0, SET1, QUERY = 1, 2, 3
DISTRACTOR_MIN = 4
SEQ_LEN = 64
DISTANCES = (1, 4, 16, 40, 63)
BATCH_SIZE = 8
TRAIN_SECONDS_PER_MODEL = 35.0
MAX_TRAIN_STEPS = 1000
TRAIN_SEED = 20_261_061
EVAL_SEED = 20_291_061
MODEL_SEED = 20_301_061
EVAL_EXAMPLES_PER_DISTANCE = 128
EXPECTED_PARAMETERS = 29_504
RESULT_PATH = Path(
    "experiments/rlt/cpu/results/rlt-lastwrite-distance-probe-20261009-ar.json"
)
CANDIDATES = (
    "residual_scan", "adaptive_scan", "transformer_control",
)


def make_batch(
    size: int, distance_choices: tuple[int, ...], seed: int,
    seq_len: int = SEQ_LEN,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Produce disjoint synthetic source sequences and binary target labels.

    All positions after the final SET are distractors until the query;
    earlier SET instructions may overwrite a previous value. The final SET
    at position seq_len-1-distance forces the label, with no answer leakage.
    """
    if size < 1 or seq_len < 3:
        raise ValueError("size and seq_len must be positive and seq_len >= 3")
    if not distance_choices or any(not 1 <= x < seq_len for x in distance_choices):
        raise ValueError("distances must be within [1, seq_len-1]")
    gen = torch.Generator(device="cpu").manual_seed(seed)
    samples = torch.randint(
        DISTRACTOR_MIN, VOCAB, (size, seq_len), generator=gen,
        dtype=torch.long,
    )
    targets = torch.randint(0, 2, (size,), generator=gen, dtype=torch.long)
    choice = torch.randint(
        len(distance_choices), (size,), generator=gen,
    )
    distances = torch.tensor(distance_choices, dtype=torch.long)[choice]
    samples[:, -1] = QUERY
    for row in range(size):
        distance = int(distances[row])
        write_idx = seq_len - 1 - distance
        # An independent initial state ensures every prompt is well-defined.
        first_bit = int(torch.randint(0, 2, (1,), generator=gen))
        samples[row, 0] = SET0 + first_bit
        # Random earlier writes add overwrite interactions, never after write_idx.
        for pos in range(1, write_idx):
            if float(torch.rand(1, generator=gen)) < 0.08:
                samples[row, pos] = SET0 + int(
                    torch.randint(0, 2, (1,), generator=gen)
                )
        samples[row, write_idx] = SET0 + int(targets[row])
    return samples, targets, distances


def last_write_labels(tokens: torch.Tensor) -> torch.Tensor:
    """Independent oracle, never called inside the model or optimization."""
    if tokens.ndim != 2 or tokens.size(1) < 3:
        raise ValueError("expected [batch, length >= 3]")
    if not bool((tokens[:, -1] == QUERY).all()):
        raise ValueError("missing final QUERY")
    labels = []
    for row in tokens:
        writes = row[:-1]
        where = torch.nonzero((writes == SET0) | (writes == SET1)).flatten()
        if where.numel() == 0:
            raise ValueError("missing state write")
        labels.append(int(writes[int(where[-1])]) - SET0)
    return torch.tensor(labels, dtype=torch.long)


def make_models() -> dict[str, torch.nn.Module]:
    rlt_cfg = RLTConfig(
        vocab_size=VOCAB, d_model=32, n_heads=4, n_stages=2,
        max_seq_len=SEQ_LEN, ff_mult=2, swa_window=8,
    )
    transformer_cfg = ModelConfig(
        vocab_size=VOCAB, d_model=32, n_heads=4, n_layers=2,
        max_seq_len=SEQ_LEN, ff_inner=144, architecture="transformer",
    )
    models = {
        "residual_scan": ResidualGatedScanRLT(rlt_cfg),
        "adaptive_scan": AdaptiveResidualGatedScanRLT(rlt_cfg),
        "transformer_control": ResearchLM(transformer_cfg),
    }
    counts = {name: parameter_count(model) for name, model in models.items()}
    if any(count != EXPECTED_PARAMETERS for count in counts.values()):
        raise RuntimeError(f"Unequal parameter counts: {counts}")
    return models


def query_binary_logits(model: torch.nn.Module, tokens: torch.Tensor) -> torch.Tensor:
    # QUERY is the final *input* token. Predict next token SET0/SET1 as answer.
    logits = model(tokens)
    if logits.shape != (tokens.size(0), tokens.size(1), VOCAB):
        raise RuntimeError(f"invalid logits shape: {tuple(logits.shape)}")
    return logits[:, -1, SET0:SET1 + 1]


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
) -> dict[str, dict[str, float | int]]:
    model.eval()
    out = {}
    for distance in DISTANCES:
        x, y, d = make_batch(
            EVAL_EXAMPLES_PER_DISTANCE, (distance,),
            EVAL_SEED + distance,
        )
        assert bool((d == distance).all())
        assert torch.equal(last_write_labels(x), y)
        # Smaller evaluation chunks bound CPU peak memory consistently.
        nll_sum, correct = 0.0, 0
        for lo in range(0, len(y), BATCH_SIZE):
            hi = min(lo + BATCH_SIZE, len(y))
            logits = query_binary_logits(model, x[lo:hi])
            nll_sum += float(F.cross_entropy(logits, y[lo:hi], reduction="sum"))
            correct += int((logits.argmax(-1) == y[lo:hi]).sum())
        out[str(distance)] = {
            "examples": len(y),
            "accuracy": correct / len(y),
            "binary_nll": nll_sum / len(y),
            "target_ones": int(y.sum()),
        }
    return out


def train_one(name: str) -> dict:
    # Each model uses an independent reconstruction and deterministic init seed.
    torch.manual_seed(MODEL_SEED)
    model = make_models()[name]
    nparams = parameter_count(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-3, weight_decay=0.01,
    )
    initial = evaluate(model)
    model.train()
    start = time.perf_counter()
    steps, final_loss = 0, math.nan
    while steps < MAX_TRAIN_STEPS:
        elapsed = time.perf_counter() - start
        if elapsed >= TRAIN_SECONDS_PER_MODEL and steps > 0:
            break
        x, y, _ = make_batch(
            BATCH_SIZE, DISTANCES, TRAIN_SEED + steps,
        )
        optimizer.zero_grad(set_to_none=True)
        logits = query_binary_logits(model, x)
        loss = F.cross_entropy(logits, y)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"Nonfinite {name} loss on step {steps}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        final_loss = float(loss.detach())
        steps += 1
    seconds = time.perf_counter() - start
    return {
        "name": name, "parameters": nparams,
        "steps": steps, "tokens_seen": steps * BATCH_SIZE * SEQ_LEN,
        "training_seconds": seconds,
        "training_tokens_per_second": steps * BATCH_SIZE * SEQ_LEN / seconds,
        "last_train_binary_nll": final_loss,
        "initial_by_distance": initial,
        "final_by_distance": evaluate(model),
    }


def run() -> dict:
    torch.set_num_threads(2)
    results = [train_one(name) for name in CANDIDATES]
    if [r["name"] for r in results] != list(CANDIDATES):
        raise RuntimeError("candidate protocol order drift")
    return {
        "schema": 1, "job_id": JOB_ID, "status": "complete",
        "classification": "CPU_SYNTHETIC_STATE_RETENTION_EXPLORATORY_ONLY",
        "gpu_requested": False, "engineering_only": True,
        "scientific_execution": False, "breakthrough_claim_supported": False,
        "protocol": {
            "task": "binary last-write query after distractor sequence",
            "query_readout": "next-token binary logits at final QUERY",
            "vocab": VOCAB, "seq_len": SEQ_LEN,
            "distances": list(DISTANCES), "batch_size": BATCH_SIZE,
            "training_seconds_per_candidate": TRAIN_SECONDS_PER_MODEL,
            "max_train_steps": MAX_TRAIN_STEPS,
            "optimizer": "AdamW", "learning_rate": 3e-3,
            "weight_decay": 0.01,
            "model_init_seed": MODEL_SEED, "train_seed": TRAIN_SEED,
            "eval_seed": EVAL_SEED,
            "evaluation_examples_per_distance": EVAL_EXAMPLES_PER_DISTANCE,
            "parameter_count_each": EXPECTED_PARAMETERS,
            "model_order": list(CANDIDATES),
        },
        "runtime": {
            "torch_version": str(torch.__version__),
            "num_threads": torch.get_num_threads(),
        },
        "results": results,
        "interpretation_ceiling": (
            "Single CPU seed, tiny synthetic state-tracking training with exactly "
            "29,504 parameters each. Report per-distance held-out binary accuracy "
            "and NLL. No FineWeb/general language, scaling, scientific replication, "
            "or breakthrough inference is supported."
        ),
    }


if __name__ == "__main__":
    result = run()
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "job_id": result["job_id"],
        "candidate_results": [{
            "name": row["name"],
            "steps": row["steps"],
            "final": row["final_by_distance"],
        } for row in result["results"]],
    }, sort_keys=True))
