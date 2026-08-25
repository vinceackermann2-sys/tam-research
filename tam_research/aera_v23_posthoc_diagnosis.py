from __future__ import annotations

"""Post-hoc localization for the failed AERA-v23 seed8461 development gate.

Frozen by research issue #341.  This module never trains a model.  It reuses the
saved seed8461 checkpoint to localize whether the remaining real-language memory
failure comes from read calibration, stage placement, cross-chunk addressing,
sparse-write usefulness, or runtime overhead.
"""

from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import MethodType
from typing import Any, Iterable, Iterator, Sequence
import math
import time

import torch
import torch.nn.functional as F

from . import aera_real_language_v23_efficiency as v23
from .aera_hardware_core import HardwareAERAState
from .aera_hardware_core_v22 import DualDeltaFastMemoryState
from .aera_hardware_core_v23 import (
    BudgetedSparseDualDeltaFastMemoryStage,
    budgeted_topk_indices,
    sparse_write_budget,
)
from .aera_real_language import SEQ_LEN, VOCAB_SIZE
from .data import TokenBin

ISSUE = 341
CHECKPOINT_SEED = 8461
DIAGNOSTIC_SEED = 128_461
ALPHAS: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
DIAGNOSTIC_BATCHES = 8
DIAGNOSTIC_BATCH_SIZE = 8
RUNTIME_BATCH_SIZES: tuple[int, ...] = (8, 64)
RUNTIME_WARMUP_ITERS = 2
RUNTIME_TIMED_ITERS = 5
EXPECTED_CHUNK_SIZE = 256
EXPECTED_CANDIDATES = 255
EXPECTED_SELECTED_WRITES = 16


def frozen_protocol() -> dict[str, Any]:
    return {
        "research_issue": ISSUE,
        "source_scientific_result": 340,
        "checkpoint_seed": CHECKPOINT_SEED,
        "diagnostic_seed": DIAGNOSTIC_SEED,
        "posthoc_only": True,
        "training_performed": False,
        "optimizer_created": False,
        "checkpoint_mutation_authorized": False,
        "alphas": list(ALPHAS),
        "diagnostic_batches": DIAGNOSTIC_BATCHES,
        "diagnostic_batch_size": DIAGNOSTIC_BATCH_SIZE,
        "sequence_length": SEQ_LEN,
        "chunk_size": EXPECTED_CHUNK_SIZE,
        "candidates_per_completed_stage_chunk": EXPECTED_CANDIDATES,
        "selected_writes_per_completed_stage_chunk": EXPECTED_SELECTED_WRITES,
        "runtime_batch_sizes": list(RUNTIME_BATCH_SIZES),
        "runtime_warmup_iterations": RUNTIME_WARMUP_ITERS,
        "runtime_timed_iterations": RUNTIME_TIMED_ITERS,
        "claims": {
            "v24_authorized": False,
            "architecture_freeze_authorized": False,
            "independent_replication": False,
            "100m_authorized": False,
            "breakthrough_proven": False,
        },
    }


def _normalize_scales(scales: float | Sequence[float], n_stages: int) -> tuple[float, ...]:
    if isinstance(scales, (int, float)):
        result = (float(scales),) * n_stages
    else:
        result = tuple(float(x) for x in scales)
    if len(result) != n_stages:
        raise ValueError(f"expected {n_stages} read scales, got {len(result)}")
    if any(not math.isfinite(x) or x < 0.0 for x in result):
        raise ValueError("read scales must be finite and nonnegative")
    return result


@contextmanager
def scaled_memory_reads(model: torch.nn.Module, scales: float | Sequence[float]) -> Iterator[None]:
    """Post-hoc scale memory.read outputs without touching writes or parameters."""
    stages = getattr(model, "stages", None)
    if stages is None:
        raise TypeError("model has no stages")
    normalized = _normalize_scales(scales, len(stages))
    installed: list[tuple[torch.nn.Module, bool, object | None]] = []
    try:
        for stage, alpha in zip(stages, normalized):
            memory = getattr(stage, "memory", None)
            if memory is None or not hasattr(memory, "read"):
                raise TypeError("stage has no readable fast memory")
            had_instance_read = "read" in memory.__dict__
            prior_instance_read = memory.__dict__.get("read")
            original = memory.read

            def scaled_read(this, x, state, *, _original=original, _alpha=alpha):
                return _original(x, state) * _alpha

            object.__setattr__(memory, "read", MethodType(scaled_read, memory))
            installed.append((memory, had_instance_read, prior_instance_read))
        yield
    finally:
        for memory, had_instance_read, prior_instance_read in reversed(installed):
            if had_instance_read:
                object.__setattr__(memory, "read", prior_instance_read)
            elif "read" in memory.__dict__:
                object.__delattr__(memory, "read")


def parameter_versions(model: torch.nn.Module) -> tuple[int, ...]:
    return tuple(int(p._version) for p in model.parameters())


def _autocast(device: torch.device):
    return (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )


def _mean(xs: Iterable[float]) -> float:
    values = list(xs)
    if not values:
        return float("nan")
    return sum(values) / len(values)


def _nll(logits: torch.Tensor, y: torch.Tensor) -> float:
    return float(F.cross_entropy(logits.float().reshape(-1, VOCAB_SIZE), y.reshape(-1)))


def _second_chunk_nll(logits: torch.Tensor, y: torch.Tensor) -> float:
    return float(
        F.cross_entropy(
            logits[:, EXPECTED_CHUNK_SIZE:].float().reshape(-1, VOCAB_SIZE),
            y[:, EXPECTED_CHUNK_SIZE:].reshape(-1),
        )
    )


def _fixed_eval_batches(data_dir: str, device: torch.device) -> list[tuple[torch.Tensor, torch.Tensor]]:
    val = TokenBin(str(Path(data_dir) / "val.bin"))
    g = torch.Generator(device="cpu").manual_seed(DIAGNOSTIC_SEED)
    return [
        val.batch(DIAGNOSTIC_BATCH_SIZE, SEQ_LEN, g, device)
        for _ in range(DIAGNOSTIC_BATCHES)
    ]


@torch.inference_mode()
def evaluate_read_scales(
    model,
    batches: Sequence[tuple[torch.Tensor, torch.Tensor]],
    scales: Sequence[float],
) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    device = next(model.parameters()).device
    for alpha in scales:
        overall: list[float] = []
        second: list[float] = []
        with scaled_memory_reads(model, float(alpha)):
            for x, y in batches:
                with _autocast(device):
                    out = model(
                        x,
                        hard=True,
                        route_mode="hard_sparse",
                        update_memory=True,
                    )
                logits = out["logits"]
                if not isinstance(logits, torch.Tensor):
                    raise RuntimeError("read-sweep output missing logits")
                overall.append(_nll(logits, y))
                second.append(_second_chunk_nll(logits, y))
        rows[f"{float(alpha):.2f}"] = {
            "overall_nll": _mean(overall),
            "second_chunk_nll": _mean(second),
        }
    baseline = rows["0.00"]["second_chunk_nll"]
    production = rows["1.00"]["second_chunk_nll"]
    for row in rows.values():
        row["second_chunk_advantage_vs_alpha0"] = baseline - row["second_chunk_nll"]
        row["second_chunk_advantage_vs_production_alpha1"] = production - row["second_chunk_nll"]
    return rows


@torch.inference_mode()
def evaluate_stage_masks(
    model,
    batches: Sequence[tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, dict[str, float | list[float]]]:
    n = len(model.stages)
    masks: dict[str, tuple[float, ...]] = {}
    for i in range(n):
        one = [0.0] * n
        one[i] = 1.0
        masks[f"only_stage_{i}"] = tuple(one)
        leave = [1.0] * n
        leave[i] = 0.0
        masks[f"leave_out_stage_{i}"] = tuple(leave)
    masks["all_zero"] = (0.0,) * n
    masks["all_one"] = (1.0,) * n

    rows: dict[str, dict[str, float | list[float]]] = {}
    device = next(model.parameters()).device
    for name, mask in masks.items():
        second: list[float] = []
        with scaled_memory_reads(model, mask):
            for x, y in batches:
                with _autocast(device):
                    out = model(
                        x,
                        hard=True,
                        route_mode="hard_sparse",
                        update_memory=True,
                    )
                logits = out["logits"]
                if not isinstance(logits, torch.Tensor):
                    raise RuntimeError("stage-mask output missing logits")
                second.append(_second_chunk_nll(logits, y))
        rows[name] = {"mask": list(mask), "second_chunk_nll": _mean(second)}

    zero = float(rows["all_zero"]["second_chunk_nll"])
    one = float(rows["all_one"]["second_chunk_nll"])
    for row in rows.values():
        nll = float(row["second_chunk_nll"])
        row["advantage_vs_all_zero"] = zero - nll
        row["advantage_vs_all_one"] = one - nll
    return rows


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return float("inf") if numerator > 0.0 else 1.0
    return numerator / denominator


def repeat_usefulness_metrics(
    chunk0: torch.Tensor,
    chunk1: torch.Tensor,
    selected_indices: torch.Tensor,
) -> dict[str, float]:
    """Token-level future-usefulness audit for selected adjacent event pairs."""
    if chunk0.ndim != 2 or chunk1.shape != chunk0.shape:
        raise ValueError("chunk0/chunk1 must match [batch,chunk]")
    if chunk0.size(1) != EXPECTED_CHUNK_SIZE:
        raise ValueError("repeat audit requires 256-token chunks")
    if selected_indices.ndim != 2 or selected_indices.size(0) != chunk0.size(0):
        raise ValueError("selected_indices must be [batch,k]")
    if selected_indices.numel() and (
        int(selected_indices.min()) < 0 or int(selected_indices.max()) >= EXPECTED_CANDIDATES
    ):
        raise ValueError("selected index outside causal pair range")

    all_address = chunk0[:, :-1]
    all_payload = chunk0[:, 1:]
    repeated_all = all_address[:, :, None].eq(chunk1[:, None, :]).any(dim=-1)

    selected_address = all_address.gather(1, selected_indices)
    selected_payload = all_payload.gather(1, selected_indices)
    repeated_selected = selected_address[:, :, None].eq(chunk1[:, None, :]).any(dim=-1)

    next_address = chunk1[:, :-1]
    next_payload = chunk1[:, 1:]
    pair_match_selected = (
        selected_address[:, :, None].eq(next_address[:, None, :])
        & selected_payload[:, :, None].eq(next_payload[:, None, :])
    ).any(dim=-1)
    pair_match_all = (
        all_address[:, :, None].eq(next_address[:, None, :])
        & all_payload[:, :, None].eq(next_payload[:, None, :])
    ).any(dim=-1)

    all_repeat_rate = float(repeated_all.float().mean())
    selected_repeat_rate = float(repeated_selected.float().mean()) if selected_indices.numel() else 0.0
    selected_repeated_count = int(repeated_selected.sum())
    all_repeated_count = int(repeated_all.sum())
    selected_payload_match_given_repeat = (
        float((pair_match_selected & repeated_selected).sum()) / selected_repeated_count
        if selected_repeated_count
        else 0.0
    )
    all_payload_match_given_repeat = (
        float((pair_match_all & repeated_all).sum()) / all_repeated_count
        if all_repeated_count
        else 0.0
    )
    return {
        "selected_repeat_rate": selected_repeat_rate,
        "all_candidate_repeat_rate": all_repeat_rate,
        "repeat_enrichment_ratio": _safe_ratio(selected_repeat_rate, all_repeat_rate),
        "selected_payload_match_given_repeat": selected_payload_match_given_repeat,
        "all_candidate_payload_match_given_repeat": all_payload_match_given_repeat,
        "selected_pairs": float(selected_indices.numel()),
        "all_candidate_pairs": float(all_address.numel()),
    }


@torch.inference_mode()
def selector_future_usefulness(model, x: torch.Tensor) -> list[dict[str, Any]]:
    first = x[:, :EXPECTED_CHUNK_SIZE]
    second = x[:, EXPECTED_CHUNK_SIZE:]
    device = x.device
    with scaled_memory_reads(model, 1.0), _autocast(device):
        out = model(
            first,
            hard=True,
            route_mode="hard_sparse",
            update_memory=True,
        )
    routes = out.get("stage_routes")
    if not isinstance(routes, list) or len(routes) != 1:
        raise RuntimeError("selector audit expected exactly one chunk of routes")
    rows: list[dict[str, Any]] = []
    for stage_index, (stage, route) in enumerate(zip(model.stages, routes[0])):
        if not isinstance(stage, BudgetedSparseDualDeltaFastMemoryStage):
            raise RuntimeError("selector audit requires v23 sparse stage")
        gate = route.get("stage_route_gate") if isinstance(route, dict) else None
        if not isinstance(gate, torch.Tensor):
            raise RuntimeError("selector audit missing stage route gate")
        run_idx = (gate[:, 0] >= 0.5).nonzero(as_tuple=False).squeeze(-1)
        indices = stage.last_selected_indices
        if run_idx.numel() == 0:
            rows.append({"stage": stage_index, "executed_examples": 0, "metrics": None})
            continue
        if indices is None or indices.size(0) != run_idx.numel():
            raise RuntimeError("selector rows do not align with hard-routed examples")
        metrics = repeat_usefulness_metrics(
            first.index_select(0, run_idx),
            second.index_select(0, run_idx),
            indices,
        )
        rows.append(
            {
                "stage": stage_index,
                "executed_examples": int(run_idx.numel()),
                "selected_writes_per_example": int(indices.size(1)),
                "metrics": metrics,
            }
        )
    return rows


@torch.inference_mode()
def stage0_cross_chunk_address_metrics(model, x: torch.Tensor) -> dict[str, float]:
    if x.ndim != 2 or x.size(1) != 2 * EXPECTED_CHUNK_SIZE:
        raise ValueError("address audit requires [batch,512]")
    first = x[:, :EXPECTED_CHUNK_SIZE]
    second = x[:, EXPECTED_CHUNK_SIZE:]
    pos = torch.arange(EXPECTED_CHUNK_SIZE, device=x.device)
    stage = model.stages[0]
    if not isinstance(stage, BudgetedSparseDualDeltaFastMemoryStage):
        raise RuntimeError("address audit requires v23 stage0")
    with _autocast(x.device):
        h0 = stage.norm(model.token_emb(first) + model.local_pos(pos)[None])
        h1 = stage.norm(model.token_emb(second) + model.local_pos(pos)[None])
        q = F.normalize(stage.memory.q(h1), dim=-1).float()
        k = F.normalize(stage.memory.k(h0[:, :-1]), dim=-1).float()
        similarity = torch.einsum("bqd,bkd->bqk", q, k)
        pair_features = torch.cat((h0[:, :-1], h0[:, 1:]), dim=-1)
        selected = budgeted_topk_indices(stage.pair_write_gate(pair_features))

    candidate_tokens = first[:, :-1]
    query_tokens = second
    token_equal = query_tokens[:, :, None].eq(candidate_tokens[:, None, :])
    repeated_query = token_equal.any(dim=-1)
    top = similarity.argmax(dim=-1)
    top_tokens = candidate_tokens.gather(1, top)
    full_hit = top_tokens.eq(query_tokens) & repeated_query

    selected_sim = similarity.gather(
        2, selected[:, None, :].expand(-1, similarity.size(1), -1)
    )
    selected_tokens = candidate_tokens.gather(1, selected)
    selected_top = selected_sim.argmax(dim=-1)
    selected_top_tokens = selected_tokens.gather(1, selected_top)
    selected_hit = selected_top_tokens.eq(query_tokens) & repeated_query

    qpos = torch.arange(EXPECTED_CHUNK_SIZE, device=x.device)[:, None]
    kpos = torch.arange(EXPECTED_CANDIDATES, device=x.device)[None, :]
    different_local_position = qpos.ne(kpos)[None, :, :]
    same_token_different_pos = token_equal & different_local_position
    different_token = ~token_equal

    def masked_mean(mask: torch.Tensor) -> float:
        count = int(mask.sum())
        if count == 0:
            return float("nan")
        return float(similarity.masked_select(mask).mean())

    repeated_count = int(repeated_query.sum())
    return {
        "repeated_query_positions": float(repeated_count),
        "repeated_query_fraction": float(repeated_query.float().mean()),
        "full_255_top1_same_token_hit_rate_on_repeated_queries": (
            float(full_hit.sum()) / repeated_count if repeated_count else 0.0
        ),
        "selected_16_top1_same_token_hit_rate_on_repeated_queries": (
            float(selected_hit.sum()) / repeated_count if repeated_count else 0.0
        ),
        "same_token_different_local_position_mean_similarity": masked_mean(
            same_token_different_pos
        ),
        "different_token_mean_similarity": masked_mean(different_token),
        "same_token_different_pos_similarity_margin": (
            masked_mean(same_token_different_pos) - masked_mean(different_token)
        ),
    }


def _cuda_time(callable_, *, tokens: int, warmup: int, timed: int) -> dict[str, float]:
    for _ in range(warmup):
        callable_()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(timed):
        callable_()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    seconds = elapsed / timed
    return {
        "milliseconds": seconds * 1000.0,
        "tokens_per_second": tokens / seconds,
    }


@torch.inference_mode()
def runtime_decomposition(model, data_dir: str) -> dict[str, Any]:
    if next(model.parameters()).device.type != "cuda":
        raise RuntimeError("runtime decomposition requires CUDA")
    val = TokenBin(str(Path(data_dir) / "val.bin"))
    rows: dict[str, Any] = {}
    for batch_size in RUNTIME_BATCH_SIZES:
        g = torch.Generator(device="cpu").manual_seed(DIAGNOSTIC_SEED + batch_size)
        x, _ = val.batch(batch_size, SEQ_LEN, g, torch.device("cuda"))
        first = x[:, :EXPECTED_CHUNK_SIZE]
        second = x[:, EXPECTED_CHUNK_SIZE:]

        def full(update_memory: bool, alpha: float):
            def call():
                with scaled_memory_reads(model, alpha), _autocast(torch.device("cuda")):
                    return model(
                        x,
                        hard=True,
                        route_mode="hard_sparse",
                        update_memory=update_memory,
                    )
            return call

        memory_off = _cuda_time(
            full(False, 1.0),
            tokens=batch_size * SEQ_LEN,
            warmup=RUNTIME_WARMUP_ITERS,
            timed=RUNTIME_TIMED_ITERS,
        )
        production = _cuda_time(
            full(True, 1.0),
            tokens=batch_size * SEQ_LEN,
            warmup=RUNTIME_WARMUP_ITERS,
            timed=RUNTIME_TIMED_ITERS,
        )
        writes_no_reads = _cuda_time(
            full(True, 0.0),
            tokens=batch_size * SEQ_LEN,
            warmup=RUNTIME_WARMUP_ITERS,
            timed=RUNTIME_TIMED_ITERS,
        )

        with scaled_memory_reads(model, 1.0), _autocast(torch.device("cuda")):
            first_out = model(
                first,
                hard=True,
                route_mode="hard_sparse",
                update_memory=True,
            )
        state = first_out.get("state")
        if not isinstance(state, HardwareAERAState):
            raise RuntimeError("runtime audit failed to prepopulate state")

        def second_call(alpha: float):
            def call():
                with scaled_memory_reads(model, alpha), _autocast(torch.device("cuda")):
                    return model(
                        second,
                        state=state,
                        hard=True,
                        route_mode="hard_sparse",
                        update_memory=False,
                    )
            return call

        read_only = _cuda_time(
            second_call(1.0),
            tokens=batch_size * EXPECTED_CHUNK_SIZE,
            warmup=RUNTIME_WARMUP_ITERS,
            timed=RUNTIME_TIMED_ITERS,
        )
        second_reads_zero = _cuda_time(
            second_call(0.0),
            tokens=batch_size * EXPECTED_CHUNK_SIZE,
            warmup=RUNTIME_WARMUP_ITERS,
            timed=RUNTIME_TIMED_ITERS,
        )
        rows[str(batch_size)] = {
            "memory_off_full": memory_off,
            "production_full_memory": production,
            "writes_enabled_reads_zero": writes_no_reads,
            "second_chunk_read_only_prepopulated": read_only,
            "second_chunk_prepopulated_reads_zero": second_reads_zero,
            "production_over_memory_off_time_ratio": (
                production["milliseconds"] / memory_off["milliseconds"]
            ),
            "write_path_overhead_ms_estimate": (
                writes_no_reads["milliseconds"] - memory_off["milliseconds"]
            ),
            "read_path_overhead_ms_estimate_second_chunk": (
                read_only["milliseconds"] - second_reads_zero["milliseconds"]
            ),
        }
    return rows


def _assert_checkpoint_model(model) -> None:
    if model.cfg.memory_dim != 50 or model.cfg.chunk_size != EXPECTED_CHUNK_SIZE:
        raise RuntimeError("diagnostic checkpoint geometry mismatch")
    if len(model.stages) != 4:
        raise RuntimeError("diagnostic expected four v23 stages")
    for stage in model.stages:
        if not isinstance(stage, BudgetedSparseDualDeltaFastMemoryStage):
            raise RuntimeError("diagnostic model is not exact v23 sparse memory")
        if sparse_write_budget(EXPECTED_CANDIDATES) != EXPECTED_SELECTED_WRITES:
            raise RuntimeError("diagnostic sparse budget changed")


@torch.inference_mode()
def run_posthoc_diagnosis(*, data_dir: str, checkpoint_path: str) -> dict[str, Any]:
    device = torch.device("cuda")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("seed") != CHECKPOINT_SEED:
        raise RuntimeError("post-hoc diagnostic checkpoint seed mismatch")
    torch.manual_seed(CHECKPOINT_SEED)
    model = v23.build_aera(device).eval()
    model.load_state_dict(payload["model"], strict=True)
    model.set_memory_pretraining_mode(False)
    _assert_checkpoint_model(model)
    before = parameter_versions(model)

    batches = _fixed_eval_batches(data_dir, device)
    read_sweep = evaluate_read_scales(model, batches, ALPHAS)
    stage_masks = evaluate_stage_masks(model, batches)
    address_rows = [stage0_cross_chunk_address_metrics(model, x) for x, _ in batches]
    address_metrics = {
        key: _mean(float(row[key]) for row in address_rows)
        for key in address_rows[0]
    }
    selector_rows = [selector_future_usefulness(model, x) for x, _ in batches]
    selector_by_stage: list[dict[str, Any]] = []
    for stage_index in range(len(model.stages)):
        stage_samples = [rows[stage_index] for rows in selector_rows]
        metric_rows = [row["metrics"] for row in stage_samples if row["metrics"] is not None]
        if metric_rows:
            keys = metric_rows[0].keys()
            averaged = {key: _mean(float(row[key]) for row in metric_rows) for key in keys}
        else:
            averaged = None
        selector_by_stage.append(
            {
                "stage": stage_index,
                "mean_executed_examples": _mean(
                    float(row["executed_examples"]) for row in stage_samples
                ),
                "metrics": averaged,
            }
        )

    runtime = runtime_decomposition(model, data_dir)
    after = parameter_versions(model)
    parameters_unchanged = before == after
    if not parameters_unchanged:
        raise RuntimeError("post-hoc diagnostic mutated base parameters")

    sweep_best_key = min(read_sweep, key=lambda key: read_sweep[key]["second_chunk_nll"])
    sweep_best = read_sweep[sweep_best_key]
    mask_best_key = min(stage_masks, key=lambda key: float(stage_masks[key]["second_chunk_nll"]))
    mask_best = stage_masks[mask_best_key]
    hidden_signal = max(
        float(sweep_best["second_chunk_advantage_vs_alpha0"]),
        float(mask_best["advantage_vs_all_zero"]),
    )
    result = {
        "protocol": frozen_protocol(),
        "checkpoint": {
            "path": checkpoint_path,
            "seed": CHECKPOINT_SEED,
            "geometry_exact_v23": True,
            "base_parameter_versions_unchanged": parameters_unchanged,
            "checkpoint_written": False,
            "training_performed": False,
            "optimizer_created": False,
        },
        "read_strength_sweep": read_sweep,
        "read_strength_best": {"alpha": float(sweep_best_key), **sweep_best},
        "stage_read_masks": stage_masks,
        "stage_mask_best": {"name": mask_best_key, **mask_best},
        "hidden_signal_best_second_chunk_advantage_vs_no_reads": hidden_signal,
        "hidden_signal_ge_0_005": hidden_signal >= 0.005,
        "stage0_cross_chunk_address": address_metrics,
        "selector_future_usefulness": selector_by_stage,
        "runtime_decomposition": runtime,
        "claims": {
            "posthoc_only": True,
            "v24_authorized": False,
            "architecture_freeze_authorized": False,
            "independent_replication": False,
            "100m_authorized": False,
            "breakthrough_proven": False,
        },
    }
    return result
