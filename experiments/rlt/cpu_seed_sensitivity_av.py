"""AV exploratory CPU-only fresh-seed sensitivity panel after AT/AU.

Three separate, preregistered case seeds. For each case, train static residual
and coupled adaptive RLT independently under identical 20s easy + 90s
64-token curriculum and balanced held-out tests; no GPU, no retries.
Cases are independently executed by a matrix GitHub Action and results are
aggregated/committed only after all three complete.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from experiments.rlt.cpu_lastwrite_curriculum_at import (
    eval_distance, timed_phase,
)
from experiments.rlt.cpu_lastwrite_probe_ar import (
    EXPECTED_PARAMETERS, make_batch, make_models,
)
from experiments.rlt.model import parameter_count
from experiments.rlt.model_gated_scan import GatedScanLightStateRLT
import torch.nn.functional as F

JOB_ID = "rlt-cpu-adaptive-seed-sensitivity-20261010-av"
CASES = (
    {"case": 0, "model_seed": 20301170, "easy_seed": 20261170,
     "long_seed": 20271170, "eval_seed": 20291170},
    {"case": 1, "model_seed": 20301171, "easy_seed": 20261171,
     "long_seed": 20271171, "eval_seed": 20291171},
    {"case": 2, "model_seed": 20301172, "easy_seed": 20261172,
     "long_seed": 20271172, "eval_seed": 20291172},
)
MODELS = ("residual_scan", "adaptive_scan")
DISTANCES = (1, 4, 16, 40, 63)
EASY_SECONDS = 20.0
LONG_SECONDS = 90.0
EASY_LR = 0.003
LONG_LR = 0.001
EVAL_PER_CLASS = 128
RESULT_DIR = Path("experiments/rlt/cpu/results")


def balanced_evaluate(model: torch.nn.Module, seed: int) -> dict[str, dict]:
    return {
        str(distance): eval_distance(model, distance, 64, seed + distance)
        for distance in DISTANCES
    }


@torch.no_grad()
def gate_diagnostics(model: torch.nn.Module, seed: int) -> dict[str, float]:
    x, _, _ = make_batch(32, DISTANCES, seed + 400)
    memory = model.encode(x)
    width = memory.size(-1)
    gate_projection = F.linear(memory, model.merge.weight[:, width:]).float()
    gain = 2.0 * torch.sigmoid(gate_projection)
    retention = torch.sigmoid(gate_projection + model.GATE_LOGIT_BIAS)
    recurrent = GatedScanLightStateRLT.gated_states(model, memory)
    full = model.gated_states(memory)
    return {
        "retention_mean": float(retention.mean()),
        "retention_near_one_frac": float((retention > 0.99).float().mean()),
        "gain_mean": float(gain.mean()),
        "gain_std": float(gain.std()),
        "gain_near_zero_frac": float((gain < 0.1).float().mean()),
        "gain_near_two_frac": float((gain > 1.9).float().mean()),
        "encoder_rms": float(memory.square().mean().sqrt()),
        "recurrent_rms": float(recurrent.square().mean().sqrt()),
        "hidden_rms": float(full.square().mean().sqrt()),
    }


def train_one(name: str, case: dict[str, int]) -> dict:
    torch.manual_seed(case["model_seed"])
    model = make_models()[name]
    if parameter_count(model) != EXPECTED_PARAMETERS:
        raise RuntimeError("parameter-count drift")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=EASY_LR, weight_decay=0.01,
    )
    easy = timed_phase(
        model, optimizer,
        seconds=EASY_SECONDS,
        seed=case["easy_seed"],
        seq_len=8,
        distances=(1,),
    )
    easy_control = eval_distance(
        model, 1, 8, case["eval_seed"] + 100,
    )
    before = balanced_evaluate(model, case["eval_seed"])
    for group in optimizer.param_groups:
        group["lr"] = LONG_LR
    long = timed_phase(
        model, optimizer,
        seconds=LONG_SECONDS,
        seed=case["long_seed"],
        seq_len=64,
        distances=DISTANCES,
    )
    after = balanced_evaluate(model, case["eval_seed"])
    easy_after = eval_distance(
        model, 1, 8, case["eval_seed"] + 100,
    )
    stats = gate_diagnostics(model, case["eval_seed"])
    # Metrics reflect both classes, not a constant prediction.
    return {
        "name": name, "parameters": parameter_count(model),
        "easy_training": easy, "easy_positive_control": easy_control,
        "long_before_by_distance": before,
        "long_training": long,
        "long_after_by_distance": after,
        "easy_after_long": easy_after,
        "after_long_gate_stats": stats,
        "long_all_bins_90pct": all(
            cell["accuracy"] >= 0.9 for cell in after.values()
        ),
        "long_all_bins_95pct": all(
            cell["accuracy"] >= 0.95 for cell in after.values()
        ),
        "long_constant_class_collapse": all(
            (cell["class0_accuracy"] == 1.0 and cell["class1_accuracy"] == 0.0)
            or (cell["class0_accuracy"] == 0.0 and cell["class1_accuracy"] == 1.0)
            for cell in after.values()
        ),
    }


def run_case(index: int) -> dict:
    if not 0 <= index < len(CASES):
        raise ValueError(f"unregistered AV case {index}")
    torch.set_num_threads(2)
    case = CASES[index]
    start = time.time()
    results = [train_one(name, case) for name in MODELS]
    return {
        "schema": 1, "job_id": JOB_ID, "case": index, "status": "complete",
        "classification": "CPU_SYNTHETIC_ADAPTIVE_SEED_SENSITIVITY_EXPLORATORY_ONLY",
        "scientific_execution": False, "gpu_requested": False,
        "breakthrough_claim_supported": False, "engineering_only": True,
        "protocol": {
            "models": list(MODELS), "case_seeds": dict(case),
            "case_count": len(CASES), "easy_seconds": EASY_SECONDS,
            "long_seconds": LONG_SECONDS, "no_max_step_cap": True,
            "easy_lr": EASY_LR, "long_lr": LONG_LR,
            "easy_seq_len": 8, "long_seq_len": 64,
            "distances": list(DISTANCES),
            "batch_size": 8, "eval_examples_per_class_per_distance": EVAL_PER_CLASS,
            "expected_parameters_each": EXPECTED_PARAMETERS,
            "same_initialization_seed_across_models": True,
            "same_data_and_eval_seeds_across_models": True,
            "weight_decay": 0.01, "optimizer": "AdamW",
        },
        "results": results,
        "started_unix": start, "finished_unix": time.time(),
        "runtime": {"torch_version": torch.__version__, "cpu_threads": torch.get_num_threads()},
        "interpretation_ceiling": (
            "Three new *CPU-only exploratory* task seeds, not registered "
            "scientific replication. Small binary synthetic curriculum, no "
            "language quality claims, no GPU or 250M/5B authority."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, required=True)
    args = parser.parse_args()
    result = run_case(args.case)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULT_DIR / f"rlt-adaptive-seed-sensitivity-20261010-av-case{args.case}.json"
    if path.exists():
        raise RuntimeError(f"refuse overwriting completed case: {path}")
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "job_id": JOB_ID, "case": args.case,
        "models": [{
            "name": row["name"],
            "easy_accuracy": row["easy_positive_control"]["accuracy"],
            "long": {
                distance: cell["accuracy"]
                for distance, cell in row["long_after_by_distance"].items()
            },
            "long_all_bins_95pct": row["long_all_bins_95pct"],
            "long_constant_class_collapse": row["long_constant_class_collapse"],
        } for row in result["results"]],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
