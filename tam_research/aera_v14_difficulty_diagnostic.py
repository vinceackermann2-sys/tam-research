from __future__ import annotations

"""Held-out difficulty-dependence diagnostic for the frozen AERA-v14 seed8271.

This module is evaluation-only.  It never trains, updates, or mutates either saved
checkpoint.  Difficulty is defined externally by the matched Transformer's
per-example/per-chunk NLL; AERA compute is the number of optional stages actually
executed by hard-sparse inference.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from . import aera_real_language_v11 as v11
from . import aera_real_language_v14 as v14

SEED = 8271
EVAL_SEED = 18271
BATCHES = 64
BATCH_SIZE = 8
EXPECTED_CHUNKS = BATCHES * BATCH_SIZE * 2
MIN_SPEARMAN = 0.15
MIN_HARD_EASY_GAP = 0.25
MIN_COMPUTE_VARIANCE = 0.05
MAX_PERMUTATION_ABS_SPEARMAN = 0.08


def _average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("values must be a 1D array with at least two elements")
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    i = 0
    while i < values.size:
        j = i + 1
        while j < values.size and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1)
        i = j
    return ranks


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1 or x.size < 2:
        raise ValueError("x/y must be equal-length 1D arrays")
    rx = _average_ranks(x)
    ry = _average_ranks(y)
    if float(rx.std()) == 0.0 or float(ry.std()) == 0.0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def summarize_adaptivity(difficulty: np.ndarray, optional_stages: np.ndarray) -> dict[str, Any]:
    difficulty = np.asarray(difficulty, dtype=np.float64)
    optional_stages = np.asarray(optional_stages, dtype=np.float64)
    if difficulty.shape != optional_stages.shape or difficulty.ndim != 1:
        raise ValueError("difficulty and optional_stages must be equal-length 1D arrays")
    if difficulty.size < 5:
        raise ValueError("at least five chunks are required")

    rho = spearman_rho(difficulty, optional_stages)
    order = np.argsort(difficulty, kind="mergesort")
    quintile_indices = np.array_split(order, 5)
    quintile_means = [float(optional_stages[idx].mean()) for idx in quintile_indices]
    hard_easy_gap = quintile_means[-1] - quintile_means[0]
    monotonic = all(
        quintile_means[i + 1] >= quintile_means[i] - 1e-12
        for i in range(len(quintile_means) - 1)
    )
    compute_variance = float(optional_stages.var())

    rng = np.random.default_rng(9917)
    permuted = rng.permutation(optional_stages)
    permutation_rho = spearman_rho(difficulty, permuted)

    checks = {
        "evaluated_chunks_ge_1024": int(difficulty.size) >= EXPECTED_CHUNKS,
        "spearman_rho_ge_0_15": rho >= MIN_SPEARMAN,
        "hardest_minus_easiest_optional_stages_ge_0_25": hard_easy_gap >= MIN_HARD_EASY_GAP,
        "difficulty_quintile_compute_monotonic": monotonic,
        "optional_stage_count_variance_ge_0_05": compute_variance >= MIN_COMPUTE_VARIANCE,
        "permutation_abs_spearman_le_0_08": abs(permutation_rho) <= MAX_PERMUTATION_ABS_SPEARMAN,
    }
    return {
        "evaluated_chunks": int(difficulty.size),
        "spearman_rho": rho,
        "difficulty_quintile_mean_optional_stages": quintile_means,
        "hardest_minus_easiest_optional_stages": hard_easy_gap,
        "optional_stage_count_variance": compute_variance,
        "permutation_control_spearman_rho": permutation_rho,
        "checks": checks,
        "adaptivity_pass": all(checks.values()),
    }


def _load_frozen_models(run_dir: Path, device: torch.device):
    aera_path = run_dir / "aera.pt"
    transformer_path = run_dir / "transformer.pt"
    for path in (aera_path, transformer_path):
        if not path.exists():
            raise FileNotFoundError(f"missing frozen v14 checkpoint: {path}")

    aera_blob = torch.load(aera_path, map_location="cpu", weights_only=False)
    transformer_blob = torch.load(transformer_path, map_location="cpu", weights_only=False)
    if int(aera_blob.get("seed", -1)) != SEED or int(transformer_blob.get("seed", -1)) != SEED:
        raise RuntimeError("checkpoint seed mismatch; refusing non-v14-seed8271 evidence")

    aera = v14.build_aera(device)
    transformer = v11.build_transformer(device)
    aera.load_state_dict(aera_blob["model"], strict=True)
    transformer.load_state_dict(transformer_blob["model"], strict=True)
    aera.eval()
    transformer.eval()
    return aera, transformer


@torch.no_grad()
def run_diagnostic(*, data_dir: str, run_dir: str, result_path: str) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("v14 difficulty diagnostic requires CUDA for frozen-checkpoint inference")
    device = torch.device("cuda")
    root = Path(run_dir)
    output = Path(result_path)
    if output.exists():
        raise RuntimeError(f"refusing duplicate diagnostic: {output} already exists")

    # Reuse the production-data integrity checks; no data bytes are changed.
    from . import aera_real_language_v12 as v12

    data_meta = v12.validate_production_data(data_dir)
    aera, transformer = _load_frozen_models(root, device)
    val = v11.TokenBin(str(Path(data_dir) / "val.bin"))
    generator = torch.Generator(device="cpu").manual_seed(EVAL_SEED)

    difficulties: list[float] = []
    optional_counts: list[float] = []
    per_stage_runs = [0.0, 0.0, 0.0]
    per_stage_total = 0

    for _ in range(BATCHES):
        x, y = val.batch(BATCH_SIZE, v11.SEQ_LEN, generator, device)
        with v11._autocast(device):
            transformer_logits = transformer(x)
            aera_out = aera(x, hard=True, route_mode="hard_sparse", update_memory=False)

        token_losses = F.cross_entropy(
            transformer_logits.float().reshape(-1, v11.VOCAB_SIZE),
            y.reshape(-1),
            reduction="none",
        ).reshape(BATCH_SIZE, v11.SEQ_LEN)
        chunk_losses = token_losses.reshape(BATCH_SIZE, 2, v11.CHUNK_SIZE).mean(dim=-1)

        routes = aera_out.get("stage_routes")
        if not isinstance(routes, list) or len(routes) != 2:
            raise RuntimeError("unexpected AERA route history geometry")
        for chunk_index, chunk_routes in enumerate(routes):
            if not isinstance(chunk_routes, list) or len(chunk_routes) != 4:
                raise RuntimeError("unexpected AERA stage route geometry")
            optional = torch.zeros(BATCH_SIZE, device=device, dtype=torch.float32)
            for stage_offset, stage_index in enumerate((1, 2, 3)):
                item = chunk_routes[stage_index]
                if not isinstance(item, dict):
                    raise RuntimeError("invalid stage route record")
                gate = item.get("stage_route_gate")
                if not isinstance(gate, torch.Tensor) or gate.shape != (BATCH_SIZE, 1):
                    raise RuntimeError("invalid hard-sparse stage gate")
                ran = (gate[:, 0].float() >= 0.5).float()
                optional += ran
                per_stage_runs[stage_offset] += float(ran.sum())
            per_stage_total += BATCH_SIZE
            difficulties.extend(chunk_losses[:, chunk_index].float().cpu().tolist())
            optional_counts.extend(optional.cpu().tolist())

    difficulty_np = np.asarray(difficulties, dtype=np.float64)
    optional_np = np.asarray(optional_counts, dtype=np.float64)
    summary = summarize_adaptivity(difficulty_np, optional_np)
    summary.update(
        {
            "seed": SEED,
            "evaluation_seed": EVAL_SEED,
            "data": data_meta,
            "batches": BATCHES,
            "batch_size": BATCH_SIZE,
            "sequence_length": v11.SEQ_LEN,
            "chunk_size": v11.CHUNK_SIZE,
            "difficulty_definition": "matched_transformer_per_example_per_256_token_chunk_nll",
            "compute_definition": "aera_hard_sparse_optional_stages_1_to_3_executed_per_example_chunk",
            "optional_stage_run_rates": [r / max(per_stage_total, 1) for r in per_stage_runs],
            "training_tokens_added": 0,
            "weights_updated": False,
            "counts_toward_independent_replication": False,
            "100m_authorized": False,
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2))
    return summary
