"""AU: paired CPU-only recurrent-path attribution after AT adaptive collapse.

Fresh CPU seed, 29,504 parameters, frozen AS/AT-style 20s easy + 90s
mixed-distance curriculum for residual and adaptive scan. At evaluation only,
replace decoder initializer with encoder-only, recurrent-only, or fixed-gain
scan. All other decoder layers remain unchanged. This evaluates dependency on
the trained recurrent output path, NOT intrinsic architectural causality.

No GPU or scientific replication. No tuning on held-out evaluation.
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
import time
from types import MethodType

import torch
import torch.nn.functional as F

from experiments.rlt.cpu_lastwrite_curriculum_at import eval_distance, timed_phase
from experiments.rlt.cpu_lastwrite_probe_ar import CANDIDATES, EXPECTED_PARAMETERS, make_models
from experiments.rlt.model import parameter_count
from experiments.rlt.model_gated_scan import GatedScanLightStateRLT

JOB_ID = "rlt-cpu-recurrence-path-attribution-20261009-au"
CANDIDATE_NAMES = ("residual_scan", "adaptive_scan")
DISTANCES = (1, 4, 16, 40, 63)
EASY_SECONDS = 20.0
LONG_SECONDS = 90.0
EASY_SEED = 20_261_064
LONG_SEED = 20_271_064
EVAL_SEED = 20_291_064
MODEL_SEED = 20_301_064
RESULT_PATH = Path("experiments/rlt/cpu/results/rlt-recurrence-path-attribution-20261009-au.json")
ABLATIONS = ("full", "encoder_only", "recurrent_only", "static_gain")


@contextmanager
def apply_ablation(model: torch.nn.Module, name: str):
    """Temporarily replace only the decoder's scan-to-hidden state interface."""
    if name not in ABLATIONS:
        raise ValueError(f"unknown ablation {name}")
    if name == "full":
        yield
        return
    original = model.gated_states
    if name == "encoder_only":
        def new_states(self, memory):
            return memory
    elif name == "recurrent_only":
        def new_states(self, memory):
            return GatedScanLightStateRLT.gated_states(self, memory)
    else:
        def new_states(self, memory):
            return memory + GatedScanLightStateRLT.gated_states(self, memory)
    model.gated_states = MethodType(new_states, model)
    try:
        yield
    finally:
        model.gated_states = original


def probe_accuracy(
    model: torch.nn.Module,
    *,
    ablation: str,
) -> dict[str, dict]:
    with apply_ablation(model, ablation):
        return {
            str(d): eval_distance(model, d, 64, EVAL_SEED + d)
            for d in DISTANCES
        }


@torch.no_grad()
def gate_statistics(model: torch.nn.Module) -> dict[str, float]:
    # Disjoint probe inputs, for diagnostics only. Not used in training.
    from experiments.rlt.cpu_lastwrite_probe_ar import make_batch
    inputs, _, _ = make_batch(16, DISTANCES, EVAL_SEED + 400)
    memory = model.encode(inputs)
    width = memory.size(-1)
    gate_projection = F.linear(memory, model.merge.weight[:, width:]).float()
    gain = 2.0 * torch.sigmoid(gate_projection)
    retention = torch.sigmoid(gate_projection + model.GATE_LOGIT_BIAS)
    recurrent = GatedScanLightStateRLT.gated_states(model, memory)
    full = model.gated_states(memory)
    return {
        "gain_mean": float(gain.mean()),
        "gain_std": float(gain.std()),
        "gain_min": float(gain.min()),
        "gain_max": float(gain.max()),
        "gain_near_zero_frac": float((gain < 0.1).float().mean()),
        "gain_near_two_frac": float((gain > 1.9).float().mean()),
        "retention_mean": float(retention.mean()),
        "retention_near_one_frac": float((retention > 0.99).float().mean()),
        "recurrent_rms": float(recurrent.square().mean().sqrt()),
        "encoder_rms": float(memory.square().mean().sqrt()),
        "full_hidden_rms": float(full.square().mean().sqrt()),
        "gate_weight_norm": float(model.merge.weight[:, width:].norm()),
    }


def run_candidate(name: str) -> dict:
    torch.manual_seed(MODEL_SEED)
    model = make_models()[name]
    if parameter_count(model) != EXPECTED_PARAMETERS:
        raise RuntimeError("unequal AU parameter count")
    opt = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
    initial = probe_accuracy(model, ablation="full")
    easy = timed_phase(
        model, opt, seconds=EASY_SECONDS,
        seed=EASY_SEED, seq_len=8, distances=(1,),
    )
    easy_control = eval_distance(model, 1, 8, EVAL_SEED + 100)
    after_easy = probe_accuracy(model, ablation="full")
    for group in opt.param_groups:
        group["lr"] = 0.001
    long = timed_phase(
        model, opt, seconds=LONG_SECONDS, seed=LONG_SEED,
        seq_len=64, distances=DISTANCES,
    )
    gate_stats = gate_statistics(model)
    evaluations = {
        key: probe_accuracy(model, ablation=key) for key in ABLATIONS
    }
    return {
        "name": name, "parameters": parameter_count(model),
        "easy_training": easy, "long_training": long,
        "initial_long": initial, "easy_control": easy_control,
        "after_easy_long": after_easy,
        "after_long_gate_stats": gate_stats,
        "ablated_long_by_distance": evaluations,
    }


def run() -> dict:
    torch.set_num_threads(2)
    rows = [run_candidate(name) for name in CANDIDATE_NAMES]
    return {
        "schema": 1, "job_id": JOB_ID, "status": "complete",
        "classification": "CPU_RECURRENT_PATH_ABLATION_EXPLORATORY_ONLY",
        "gpu_requested": False, "engineering_only": True,
        "scientific_execution": False, "breakthrough_claim_supported": False,
        "protocol": {
            "candidate_order": list(CANDIDATE_NAMES),
            "ablation_order": list(ABLATIONS),
            "expected_parameters_each": EXPECTED_PARAMETERS,
            "easy_seconds": EASY_SECONDS, "long_seconds": LONG_SECONDS,
            "easy_seq_len": 8, "long_seq_len": 64,
            "distances": list(DISTANCES),
            "train_batch_size": 8,
            "easy_lr": 0.003, "long_lr": 0.001,
            "easy_seed": EASY_SEED, "long_seed": LONG_SEED,
            "eval_seed": EVAL_SEED, "model_seed": MODEL_SEED,
            "balanced_eval_per_class_distance": 128,
            "optimizer": "AdamW", "weight_decay": 0.01,
            "no_max_step_cap": True,
            "inference_only_ablation_no_retraining": True,
        },
        "results": rows,
        "runtime": {
            "torch_version": torch.__version__,
            "cpu_threads": torch.get_num_threads(),
        },
        "interpretation_ceiling": (
            "One-seed exploratory causal-interface intervention on trained tiny "
            "synthetic state classifiers, not a controlled architecture effect "
            "or scientific replication. Zeroing the recurrent path at inference "
            "can introduce distribution shift; a drop alone does not prove the "
            "recurrent path is mathematically necessary. No GPU authorization."
        ),
    }


if __name__ == "__main__":
    result = run()
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps({
        "job_id": JOB_ID,
        "scores": {
            row["name"]: {
                "easy_control": row["easy_control"]["accuracy"],
                "ablations": {
                    name: {d: round(v["accuracy"], 4) for d, v in scores.items()}
                    for name, scores in row["ablated_long_by_distance"].items()
                },
            } for row in result["results"]
        },
    }, sort_keys=True))
