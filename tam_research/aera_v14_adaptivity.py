from __future__ import annotations

"""Held-out adaptivity diagnostic for the frozen AERA-v14 seed8271 checkpoint.

This module performs evaluation only. It never trains, mutates, or rewrites the
seed8271 model. Difficulty is measured independently with the matched Transformer
checkpoint on the exact same held-out chunks; AERA compute is the number of
optional whole stages physically selected by hard-sparse inference.
"""

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .aera_real_language import SEQ_LEN, VOCAB_SIZE
from .aera_real_language_v11 import build_transformer
from .aera_real_language_v14 import build_aera
from .data import TokenBin

SEED = 8271
EVAL_SEED = 98_271
CHUNK_SIZE = 256
BATCH_SIZE = 8
BATCHES = 64
EXPECTED_CHUNKS = BATCHES * BATCH_SIZE * (SEQ_LEN // CHUNK_SIZE)

# Frozen development-only thresholds. These are intentionally set before the
# checkpoint adaptivity result is inspected.
MIN_SPEARMAN_RHO = 0.20
MIN_HARD_MINUS_EASY_STAGES = 0.25
QUARTILE_MONOTONIC_TOLERANCE = 0.05
MIN_DIVERSE_BUDGET_FRACTION = 0.05


def _rankdata(values: torch.Tensor) -> torch.Tensor:
    if values.ndim != 1 or values.numel() < 2:
        raise ValueError("rankdata requires a 1D tensor with at least two values")
    order = torch.argsort(values, stable=True)
    ranks = torch.empty(values.numel(), dtype=torch.float64)
    ranks[order.cpu()] = torch.arange(values.numel(), dtype=torch.float64)
    return ranks


def spearman_rho(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.shape != y.shape:
        raise ValueError("Spearman inputs must have identical shape")
    rx = _rankdata(x.detach().float().cpu())
    ry = _rankdata(y.detach().float().cpu())
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = torch.sqrt((rx.square().sum()) * (ry.square().sum()))
    if float(denom) == 0.0:
        return 0.0
    return float((rx * ry).sum() / denom)


def summarize_adaptivity(difficulty: torch.Tensor, optional_stages: torch.Tensor) -> dict[str, Any]:
    difficulty = difficulty.detach().float().cpu().reshape(-1)
    optional_stages = optional_stages.detach().float().cpu().reshape(-1)
    if difficulty.numel() != optional_stages.numel() or difficulty.numel() < 16:
        raise ValueError("need matching nontrivial difficulty/compute samples")

    rho = spearman_rho(difficulty, optional_stages)
    order = torch.argsort(difficulty, stable=True)
    quartiles = torch.tensor_split(order, 4)
    quartile_means = [float(optional_stages[idx].mean()) for idx in quartiles]
    hard_minus_easy = quartile_means[-1] - quartile_means[0]
    monotonic = all(
        quartile_means[i + 1] + QUARTILE_MONOTONIC_TOLERANCE >= quartile_means[i]
        for i in range(3)
    )

    histogram = {str(k): int((optional_stages == float(k)).sum()) for k in range(4)}
    n = float(optional_stages.numel())
    diverse_bins = sum((count / n) >= MIN_DIVERSE_BUDGET_FRACTION for count in histogram.values())

    checks = {
        "spearman_rho_ge_0_20": rho >= MIN_SPEARMAN_RHO,
        "hardest_minus_easiest_quartile_ge_0_25_stage": hard_minus_easy >= MIN_HARD_MINUS_EASY_STAGES,
        "quartile_compute_monotonic_with_tolerance": monotonic,
        "at_least_two_budget_bins_ge_5pct": diverse_bins >= 2,
    }
    return {
        "samples": int(optional_stages.numel()),
        "difficulty_compute_spearman_rho": rho,
        "optional_stage_quartile_means_easy_to_hard": quartile_means,
        "hardest_minus_easiest_optional_stages": hard_minus_easy,
        "optional_stage_budget_histogram": histogram,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _optional_stage_counts(output: dict[str, object], batch_size: int) -> list[torch.Tensor]:
    routes = output.get("stage_routes")
    if not isinstance(routes, list) or len(routes) != SEQ_LEN // CHUNK_SIZE:
        raise RuntimeError("unexpected AERA route history")
    counts: list[torch.Tensor] = []
    for chunk in routes:
        if not isinstance(chunk, list) or len(chunk) != 4:
            raise RuntimeError("unexpected AERA stage route history")
        count = torch.zeros(batch_size, device=next(iter(chunk[0].values())).device if isinstance(chunk[0], dict) else "cpu")
        for item in chunk[1:]:
            if not isinstance(item, dict):
                raise RuntimeError("invalid stage route item")
            gate = item.get("stage_route_gate")
            if not isinstance(gate, torch.Tensor) or gate.shape != (batch_size, 1):
                raise RuntimeError("invalid stage_route_gate")
            count = count + (gate[:, 0] >= 0.5).float()
        counts.append(count)
    return counts


@torch.no_grad()
def evaluate_checkpoint(*, data_dir: str, run_dir: str, device: torch.device) -> dict[str, Any]:
    root = Path(run_dir)
    aera_path = root / "aera.pt"
    transformer_path = root / "transformer.pt"
    if not aera_path.exists() or not transformer_path.exists():
        raise FileNotFoundError("seed8271 checkpoints are missing from durable run directory")

    aera_payload = torch.load(aera_path, map_location="cpu", weights_only=False)
    transformer_payload = torch.load(transformer_path, map_location="cpu", weights_only=False)
    if aera_payload.get("seed") != SEED or transformer_payload.get("seed") != SEED:
        raise RuntimeError("checkpoint seed mismatch")

    torch.manual_seed(SEED)
    aera = build_aera(device).eval()
    torch.manual_seed(SEED)
    transformer = build_transformer(device).eval()
    aera.load_state_dict(aera_payload["model"], strict=True)
    transformer.load_state_dict(transformer_payload["model"], strict=True)

    val = TokenBin(str(Path(data_dir) / "val.bin"))
    g = torch.Generator(device="cpu").manual_seed(EVAL_SEED)
    all_difficulty: list[torch.Tensor] = []
    all_compute: list[torch.Tensor] = []
    stage_runs = torch.zeros(3, dtype=torch.float64)

    for _ in range(BATCHES):
        x, y = val.batch(BATCH_SIZE, SEQ_LEN, g, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            t_logits = transformer(x)
            a_out = aera(x, hard=True, route_mode="hard_sparse", update_memory=False)

        t_loss = F.cross_entropy(
            t_logits.float().reshape(-1, VOCAB_SIZE), y.reshape(-1), reduction="none"
        ).reshape(BATCH_SIZE, SEQ_LEN)
        counts = _optional_stage_counts(a_out, BATCH_SIZE)
        routes = a_out["stage_routes"]
        assert isinstance(routes, list)

        for chunk_index, start in enumerate(range(0, SEQ_LEN, CHUNK_SIZE)):
            all_difficulty.append(t_loss[:, start : start + CHUNK_SIZE].mean(dim=1).cpu())
            all_compute.append(counts[chunk_index].cpu())
            chunk = routes[chunk_index]
            assert isinstance(chunk, list)
            for stage_index, item in enumerate(chunk[1:]):
                assert isinstance(item, dict)
                gate = item["stage_route_gate"]
                assert isinstance(gate, torch.Tensor)
                stage_runs[stage_index] += float((gate[:, 0] >= 0.5).sum())

    difficulty = torch.cat(all_difficulty)
    compute = torch.cat(all_compute)
    if difficulty.numel() != EXPECTED_CHUNKS:
        raise RuntimeError(f"unexpected held-out sample count {difficulty.numel()} != {EXPECTED_CHUNKS}")

    result = summarize_adaptivity(difficulty, compute)
    result["seed"] = SEED
    result["eval_seed"] = EVAL_SEED
    result["checkpoint_run_dir"] = run_dir
    result["difficulty_definition"] = "matched Transformer per-chunk held-out next-token NLL"
    result["compute_definition"] = "AERA hard-sparse optional whole-stage executions per chunk"
    result["optional_stage_run_fractions"] = [float(v / EXPECTED_CHUNKS) for v in stage_runs]
    result["training_performed"] = False
    result["checkpoint_mutated"] = False
    result["counts_toward_independent_replication"] = False
    result["100m_authorized"] = False
    return result


def write_result(result: dict[str, Any], path: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2))
