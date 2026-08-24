from __future__ import annotations

"""Guarded AERA-v17 real-language development harness.

Fresh seed8331 tests whether the pairwise difficulty teacher plus exact hard-run
budget keeps all optional depth levels alive while preserving quality, state,
sparse execution, and held-out difficulty-dependent compute. This is adaptive
development evidence only and cannot count as independent replication.
"""

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from . import aera_real_language_v11 as v11
from . import aera_real_language_v12 as v12
from . import aera_real_language_v17 as v17
from .aera_real_language import SEQ_LEN, VOCAB_SIZE
from .aera_v14_adaptivity import (
    BATCHES as ADAPTIVITY_BATCHES,
    BATCH_SIZE as ADAPTIVITY_BATCH_SIZE,
    CHUNK_SIZE as ADAPTIVITY_CHUNK_SIZE,
    CHUNKS_PER_SEQUENCE,
    EXPECTED_CHUNKS,
    _optional_stage_counts,
    summarize_adaptivity,
)
from .data import TokenBin

SEED = 8331
EVAL_SEED = 98_331
TARGET_RATES = torch.tensor([0.50, 1.0 / 3.0, 1.0 / 6.0], dtype=torch.float32)


def validate_protocol(data_dir: str) -> dict[str, Any]:
    cpu = v17.cpu_preflight()
    data = v12.validate_production_data(data_dir)
    if ADAPTIVITY_CHUNK_SIZE != v17.CHUNK_SIZE:
        raise RuntimeError("v17 held-out adaptivity chunk size mismatch")
    return {
        **cpu,
        "gpu_authorized": True,
        "gpu_authorization_scope": "one guarded AERA-v17 development seed8331 L4 run only",
        "data": data,
        "development_seed": SEED,
        "counts_toward_breakthrough_evidence": False,
        "frozen_optional_stage_target_rates": TARGET_RATES.tolist(),
        "difficulty_teacher": "pairwise ranking over detached per-chunk language difficulty",
        "budget_measurement": "straight-through hard execution at p>=0.5 against exact nominal rates",
        "heldout_adaptivity_eval_seed": EVAL_SEED,
        "heldout_adaptivity_samples": EXPECTED_CHUNKS,
    }


def _install_v17_harness() -> None:
    v11.build_aera = v17.build_aera
    v11.aera_matched_loss = v17.aera_matched_loss
    v11.DENSE_WARMUP_STEPS = v17.DENSE_WARMUP_STEPS
    v11.ROUTER_CALIBRATION_END = v17.ROUTER_CALIBRATION_END
    v11.SPARSE_CALIBRATION_EVERY = v17.SPARSE_CALIBRATION_EVERY
    v11.set_stage_router_trainable = v17.set_optional_stage_router_trainable
    v11.validate_protocol = validate_protocol


@torch.no_grad()
def _heldout_adaptivity(*, data_dir: str, run_dir: str, seed: int) -> dict[str, Any]:
    device = torch.device("cuda")
    root = Path(run_dir)
    aera_payload = torch.load(root / "aera.pt", map_location="cpu", weights_only=False)
    transformer_payload = torch.load(root / "transformer.pt", map_location="cpu", weights_only=False)
    if aera_payload.get("seed") != seed or transformer_payload.get("seed") != seed:
        raise RuntimeError("v17 held-out checkpoint seed mismatch")

    torch.manual_seed(seed)
    aera = v17.build_aera(device).eval()
    torch.manual_seed(seed)
    transformer = v17.build_transformer(device).eval()
    aera.load_state_dict(aera_payload["model"], strict=True)
    transformer.load_state_dict(transformer_payload["model"], strict=True)

    val = TokenBin(str(Path(data_dir) / "val.bin"))
    g = torch.Generator(device="cpu").manual_seed(EVAL_SEED)
    all_difficulty: list[torch.Tensor] = []
    all_compute: list[torch.Tensor] = []
    all_positions: list[torch.Tensor] = []
    stage_runs = torch.zeros(3, dtype=torch.float64)

    for _ in range(ADAPTIVITY_BATCHES):
        x, y = val.batch(ADAPTIVITY_BATCH_SIZE, SEQ_LEN, g, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            t_logits = transformer(x)
            a_out = aera(x, hard=True, route_mode="hard_sparse", update_memory=False)

        t_loss = F.cross_entropy(
            t_logits.float().reshape(-1, VOCAB_SIZE),
            y.reshape(-1),
            reduction="none",
        ).reshape(ADAPTIVITY_BATCH_SIZE, SEQ_LEN)
        counts = _optional_stage_counts(a_out, ADAPTIVITY_BATCH_SIZE)
        routes = a_out.get("stage_routes")
        if not isinstance(routes, list):
            raise RuntimeError("v17 held-out eval missing stage routes")

        for chunk_index, start in enumerate(range(0, SEQ_LEN, ADAPTIVITY_CHUNK_SIZE)):
            all_difficulty.append(
                t_loss[:, start : start + ADAPTIVITY_CHUNK_SIZE].mean(dim=1).cpu()
            )
            all_compute.append(counts[chunk_index].cpu())
            all_positions.append(
                torch.full((ADAPTIVITY_BATCH_SIZE,), chunk_index, dtype=torch.long)
            )
            chunk = routes[chunk_index]
            if not isinstance(chunk, list) or len(chunk) != 4:
                raise RuntimeError("v17 held-out eval invalid stage routes")
            for stage_index, item in enumerate(chunk[1:]):
                if not isinstance(item, dict):
                    raise RuntimeError("v17 held-out eval invalid route item")
                gate = item.get("stage_route_gate")
                if not isinstance(gate, torch.Tensor):
                    raise RuntimeError("v17 held-out eval missing stage gate")
                stage_runs[stage_index] += float((gate[:, 0] >= 0.5).sum())

    difficulty = torch.cat(all_difficulty)
    compute = torch.cat(all_compute)
    positions = torch.cat(all_positions)
    if difficulty.numel() != EXPECTED_CHUNKS:
        raise RuntimeError(
            f"v17 held-out sample count {difficulty.numel()} != {EXPECTED_CHUNKS}"
        )

    summary = summarize_adaptivity(difficulty, compute, positions)
    summary["seed"] = seed
    summary["eval_seed"] = EVAL_SEED
    summary["difficulty_definition"] = "matched Transformer per-chunk held-out next-token NLL"
    summary["compute_definition"] = "AERA hard-sparse optional whole-stage executions per chunk"
    summary["optional_stage_run_fractions"] = [
        float(v / EXPECTED_CHUNKS) for v in stage_runs
    ]
    summary["training_performed"] = False
    summary["checkpoint_mutated"] = False
    summary["counts_toward_independent_replication"] = False
    return summary


def train_matched_pair(*, data_dir: str, run_dir: str, seed: int = SEED) -> dict[str, Any]:
    if seed != SEED:
        raise ValueError(f"v17 development run is frozen to seed {SEED}")
    _install_v17_harness()
    result = v11.train_matched_pair(data_dir=data_dir, run_dir=run_dir, seed=seed)

    diagnostics = result["diagnostics"]
    stage_execution = float(diagnostics["stage_execution_fraction"])
    quality_gap = float(diagnostics["quality_gap_nll"])
    speed_ratio = float(diagnostics["batch8_speed_ratio"])
    state_advantage = float(diagnostics["state_advantage_nll"])

    adaptivity = _heldout_adaptivity(data_dir=data_dir, run_dir=run_dir, seed=seed)
    optional_rates = [float(v) for v in adaptivity["optional_stage_run_fractions"]]
    target_rates = TARGET_RATES.tolist()
    rate_mae = sum(abs(a - b) for a, b in zip(optional_rates, target_rates)) / len(target_rates)
    ordered = optional_rates[0] >= optional_rates[1] >= optional_rates[2]

    diagnostics["optional_stage_run_fractions"] = optional_rates
    diagnostics["optional_stage_target_rate_mae"] = rate_mae
    diagnostics["heldout_adaptivity"] = adaptivity
    checks = {
        "quality_gap_nll_le_0_50": quality_gap <= 0.50,
        "stage_execution_between_0_35_and_0_70": 0.35 <= stage_execution <= 0.70,
        "batch8_speed_ratio_ge_0_30": speed_ratio >= 0.30,
        "state_advantage_nonnegative": state_advantage >= 0.0,
        "all_optional_stages_run_ge_0_05": all(rate >= 0.05 for rate in optional_rates),
        "optional_stage_rates_shallow_to_deep": ordered,
        "optional_stage_target_mae_le_0_12": rate_mae <= 0.12,
        "heldout_difficulty_adaptivity_pass": bool(adaptivity["pass"]),
    }
    result["v17_development_checks"] = checks
    result["v17_development_pass"] = all(checks.values())
    result["claims"] = {
        "development_seed_only": True,
        "counts_toward_breakthrough_evidence": False,
        "100m_authorized": False,
        "breakthrough_proven": False,
    }
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "result.json").write_text(json.dumps(result, indent=2))
    return result
