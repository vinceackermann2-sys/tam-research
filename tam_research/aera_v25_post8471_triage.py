from __future__ import annotations

"""Checkpoint-only post-seed8471 triage frozen by research issue #369."""

from contextlib import contextmanager, nullcontext
import json
import math
from pathlib import Path
from types import MethodType
from typing import Any, Callable, Iterator, Sequence

import torch
import torch.nn.functional as F

from . import aera_real_language_v25 as v25
from .aera_hardware_core import HardwareAERAState
from .aera_hardware_core_v24 import ContextualEpisodicMemoryState
from .aera_hardware_core_v25 import FactorizedIdentityContextEpisodicMemoryStage
from .aera_real_language import SEQ_LEN, VOCAB_SIZE
from .aera_real_language_v12 import validate_production_data
from .aera_v14_adaptivity import QUARTILE_MONOTONIC_TOLERANCE, spearman_rho
from .data import TokenBin

RESEARCH_ISSUE = 369
SOURCE_RESULT_ISSUE = 368
SOURCE_SEED = 8471
DIAGNOSTIC_SEED = 138_471
MEMORY_BATCHES = 64
MEMORY_BATCH_SIZE = 8
ADAPTIVITY_BATCHES = 256
ADAPTIVITY_BATCH_SIZE = 8
CHUNK_SIZE = 256
EXPECTED_CHUNKS = ADAPTIVITY_BATCHES * ADAPTIVITY_BATCH_SIZE * (SEQ_LEN // CHUNK_SIZE)
READ_ALPHAS: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0)
BOOTSTRAP_RESAMPLES = 2_000
SYSTEM_BATCH_SIZES: tuple[int, ...] = (8, 64)
SYSTEM_WARMUP_CALLS = 3
SYSTEM_TIMED_CALLS_PER_ROUND = 20
SYSTEM_ROUNDS = 5
SYSTEMS_REFERENCE_UNTRAINED_MS = {
    8: {"core_memory_bypassed": 45.70, "writes_only": 66.42, "full": 75.97},
    64: {"core_memory_bypassed": 90.11, "writes_only": 112.88, "full": 121.83},
}


def frozen_protocol() -> dict[str, Any]:
    return {
        "research_issue": RESEARCH_ISSUE,
        "source_result_issue": SOURCE_RESULT_ISSUE,
        "source_checkpoint_seed": SOURCE_SEED,
        "diagnostic_sampling_seed": DIAGNOSTIC_SEED,
        "checkpoint_only": True,
        "training_performed": False,
        "checkpoint_mutation_authorized": False,
        "memory_batches": MEMORY_BATCHES,
        "memory_batch_size": MEMORY_BATCH_SIZE,
        "adaptivity_batches": ADAPTIVITY_BATCHES,
        "adaptivity_batch_size": ADAPTIVITY_BATCH_SIZE,
        "read_alphas": list(READ_ALPHAS),
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "systems_batch_sizes": list(SYSTEM_BATCH_SIZES),
        "systems_warmup_calls": SYSTEM_WARMUP_CALLS,
        "systems_timed_calls_per_round": SYSTEM_TIMED_CALLS_PER_ROUND,
        "systems_rounds": SYSTEM_ROUNDS,
        "claims": {
            "v25_primary_result_changed": False,
            "architecture_freeze_authorized": False,
            "s2_authorized": False,
            "independent_replication_credit": False,
            "100m_authorized": False,
            "breakthrough_proven": False,
        },
    }


def parameter_versions(model: torch.nn.Module) -> tuple[int, ...]:
    return tuple(int(parameter._version) for parameter in model.parameters())


def _autocast(device: torch.device):
    return (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )


def _normalize_scales(scales: float | Sequence[float], n_stages: int) -> tuple[float, ...]:
    if isinstance(scales, (float, int)):
        values = (float(scales),) * n_stages
    else:
        values = tuple(float(value) for value in scales)
    if len(values) != n_stages:
        raise ValueError(f"expected {n_stages} read scales, got {len(values)}")
    if any((not math.isfinite(value)) or value < 0.0 for value in values):
        raise ValueError("read scales must be finite and nonnegative")
    return values


@contextmanager
def scaled_ficem_reads(model: torch.nn.Module, scales: float | Sequence[float]) -> Iterator[None]:
    stages = getattr(model, "stages", None)
    if stages is None:
        raise TypeError("model has no stages")
    values = _normalize_scales(scales, len(stages))
    installed: list[tuple[torch.nn.Module, bool, object | None]] = []
    try:
        for stage, alpha in zip(stages, values):
            if not isinstance(stage, FactorizedIdentityContextEpisodicMemoryStage):
                raise TypeError("v25 triage requires FICEM stages")
            memory = stage.memory
            had_instance_read = "read" in memory.__dict__
            previous_instance_read = memory.__dict__.get("read")
            original = memory.read

            def scaled_read(
                this,
                identity_source,
                context_source,
                state,
                *,
                _original=original,
                _alpha=alpha,
            ):
                return _original(identity_source, context_source, state) * _alpha

            object.__setattr__(memory, "read", MethodType(scaled_read, memory))
            installed.append((memory, had_instance_read, previous_instance_read))
        yield
    finally:
        for memory, had_instance_read, previous_instance_read in reversed(installed):
            if had_instance_read:
                object.__setattr__(memory, "read", previous_instance_read)
            elif "read" in memory.__dict__:
                object.__delattr__(memory, "read")


@contextmanager
def bypass_ficem_reads(model: torch.nn.Module) -> Iterator[None]:
    stages = getattr(model, "stages", None)
    if stages is None:
        raise TypeError("model has no stages")
    installed: list[tuple[torch.nn.Module, bool, object | None]] = []
    try:
        for stage in stages:
            if not isinstance(stage, FactorizedIdentityContextEpisodicMemoryStage):
                raise TypeError("v25 triage requires FICEM stages")
            memory = stage.memory
            had_instance_read = "read" in memory.__dict__
            previous_instance_read = memory.__dict__.get("read")

            def zero_read(this, identity_source, context_source, state):
                return torch.zeros(
                    identity_source.size(0),
                    identity_source.size(1),
                    this.out.out_features,
                    device=identity_source.device,
                    dtype=identity_source.dtype,
                )

            object.__setattr__(memory, "read", MethodType(zero_read, memory))
            installed.append((memory, had_instance_read, previous_instance_read))
        yield
    finally:
        for memory, had_instance_read, previous_instance_read in reversed(installed):
            if had_instance_read:
                object.__setattr__(memory, "read", previous_instance_read)
            elif "read" in memory.__dict__:
                object.__delattr__(memory, "read")


def _flatten_cpu(values: Sequence[torch.Tensor]) -> torch.Tensor:
    if not values:
        return torch.empty(0, dtype=torch.float32)
    return torch.cat([value.detach().float().reshape(-1).cpu() for value in values])


def _summary(values: Sequence[torch.Tensor]) -> dict[str, float]:
    flat = _flatten_cpu(values)
    if flat.numel() == 0:
        return {
            "count": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "p10": 0.0,
            "p90": 0.0,
        }
    return {
        "count": float(flat.numel()),
        "mean": float(flat.mean()),
        "median": float(flat.median()),
        "std": float(flat.std(unbiased=False)),
        "p10": float(torch.quantile(flat, 0.10)),
        "p90": float(torch.quantile(flat, 0.90)),
    }


def bootstrap_mean_ci(
    values: torch.Tensor,
    *,
    seed: int,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, float]:
    x = values.detach().double().reshape(-1).cpu()
    if x.numel() < 2:
        raise ValueError("bootstrap requires at least two values")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.randint(
        0,
        x.numel(),
        (resamples, x.numel()),
        generator=generator,
    )
    means = x[indices].mean(dim=1)
    return {
        "resamples": float(resamples),
        "mean": float(x.mean()),
        "ci95_low": float(torch.quantile(means, 0.025)),
        "ci95_high": float(torch.quantile(means, 0.975)),
    }


def paired_difference_stats(
    production: torch.Tensor,
    control: torch.Tensor,
    *,
    seed: int,
) -> dict[str, float]:
    p = production.detach().double().reshape(-1).cpu()
    c = control.detach().double().reshape(-1).cpu()
    if p.shape != c.shape or p.numel() < 2:
        raise ValueError("paired statistics require matching nontrivial inputs")
    delta = p - c
    bootstrap = bootstrap_mean_ci(delta, seed=seed)
    std = float(delta.std(unbiased=False))
    return {
        "examples": float(delta.numel()),
        "production_minus_control_mean": float(delta.mean()),
        "production_minus_control_median": float(delta.median()),
        "production_minus_control_std": std,
        "production_minus_control_se": std / math.sqrt(delta.numel()),
        "production_minus_control_ci95_low": bootstrap["ci95_low"],
        "production_minus_control_ci95_high": bootstrap["ci95_high"],
        "control_minus_production_advantage_mean": float(-delta.mean()),
    }


def _per_example_nll(logits: torch.Tensor, y: torch.Tensor, *, start: int = 0) -> torch.Tensor:
    sliced_logits = logits[:, start:].float()
    sliced_y = y[:, start:]
    losses = F.cross_entropy(
        sliced_logits.reshape(-1, sliced_logits.size(-1)),
        sliced_y.reshape(-1),
        reduction="none",
    ).reshape(sliced_y.size(0), sliced_y.size(1))
    return losses.mean(dim=1)


def _fixed_batches(
    *,
    data_dir: str,
    batches: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    validation = TokenBin(str(Path(data_dir) / "val.bin"))
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return [
        validation.batch(batch_size, SEQ_LEN, generator, device)
        for _ in range(batches)
    ]


def _model_forward(
    model,
    x: torch.Tensor,
    *,
    update_memory: bool,
) -> dict[str, object]:
    with _autocast(x.device):
        output = model(
            x,
            hard=True,
            route_mode="hard_sparse",
            update_memory=update_memory,
        )
    if not isinstance(output, dict):
        raise RuntimeError("AERA forward did not return a result mapping")
    return output


def _condition_losses(
    model,
    batches: Sequence[tuple[torch.Tensor, torch.Tensor]],
    *,
    update_memory: bool,
    read_scales: float | Sequence[float] | None,
) -> dict[str, torch.Tensor]:
    overall: list[torch.Tensor] = []
    second: list[torch.Tensor] = []
    context = (
        scaled_ficem_reads(model, read_scales)
        if read_scales is not None
        else nullcontext()
    )
    with context:
        for x, y in batches:
            output = _model_forward(model, x, update_memory=update_memory)
            logits = output.get("logits")
            if not isinstance(logits, torch.Tensor):
                raise RuntimeError("AERA output missing logits")
            overall.append(_per_example_nll(logits, y))
            second.append(_per_example_nll(logits, y, start=CHUNK_SIZE))
    return {
        "overall": torch.cat(overall).cpu(),
        "second_chunk": torch.cat(second).cpu(),
    }


def _mean_loss_row(losses: dict[str, torch.Tensor]) -> dict[str, float]:
    return {
        "overall_nll": float(losses["overall"].mean()),
        "second_chunk_nll": float(losses["second_chunk"].mean()),
    }


@contextmanager
def capture_second_chunk_context_metrics(
    model,
    records: dict[int, dict[str, list[torch.Tensor]]],
) -> Iterator[None]:
    installed: list[tuple[FactorizedIdentityContextEpisodicMemoryStage, object]] = []
    try:
        for stage_index, stage in enumerate(model.stages):
            if not isinstance(stage, FactorizedIdentityContextEpisodicMemoryStage):
                raise TypeError("v25 triage requires FICEM stages")
            original = stage._tokenwise_context
            stage_records = records.setdefault(
                stage_index,
                {
                    "memory_read_norm": [],
                    "stream_norm": [],
                    "applied_memory_norm": [],
                    "memory_read_gate": [],
                    "raw_read_to_stream_norm_ratio": [],
                    "applied_read_to_stream_norm_ratio": [],
                },
            )

            def wrapped(
                this,
                h,
                state,
                start_control,
                *,
                _original=original,
                _records=stage_records,
            ):
                combined, memory_read = _original(h, state, start_control)
                carried = this.state_to_chunk(state.stream)
                gate = start_control["memory_read"]
                raw_norm = torch.linalg.vector_norm(memory_read.float(), dim=-1).mean(dim=1)
                stream_norm = torch.linalg.vector_norm(carried.float(), dim=-1)
                applied = gate[:, None, :] * memory_read
                applied_norm = torch.linalg.vector_norm(applied.float(), dim=-1).mean(dim=1)
                denominator = stream_norm.clamp_min(1e-8)
                _records["memory_read_norm"].append(raw_norm.detach().cpu())
                _records["stream_norm"].append(stream_norm.detach().cpu())
                _records["applied_memory_norm"].append(applied_norm.detach().cpu())
                _records["memory_read_gate"].append(gate.detach().float().reshape(-1).cpu())
                _records["raw_read_to_stream_norm_ratio"].append(
                    (raw_norm / denominator).detach().cpu()
                )
                _records["applied_read_to_stream_norm_ratio"].append(
                    (applied_norm / denominator).detach().cpu()
                )
                return combined, memory_read

            object.__setattr__(stage, "_tokenwise_context", MethodType(wrapped, stage))
            installed.append((stage, original))
        yield
    finally:
        for stage, original in reversed(installed):
            object.__setattr__(stage, "_tokenwise_context", original)


def _stage_state_metrics(
    model,
    batches: Sequence[tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, Any]:
    records: dict[int, dict[str, list[torch.Tensor]]] = {}
    utilization: dict[int, list[torch.Tensor]] = {index: [] for index in range(len(model.stages))}
    strengths: dict[int, list[torch.Tensor]] = {index: [] for index in range(len(model.stages))}
    for x, _ in batches:
        first = x[:, :CHUNK_SIZE]
        second = x[:, CHUNK_SIZE:]
        for stage in model.stages:
            if isinstance(stage, FactorizedIdentityContextEpisodicMemoryStage):
                stage.last_pair_strength = None
        first_output = _model_forward(model, first, update_memory=True)
        state = first_output.get("state")
        if not isinstance(state, HardwareAERAState):
            raise RuntimeError("first-chunk output missing HardwareAERAState")
        for stage_index, stage_state in enumerate(state.stages):
            memory = stage_state.memory
            if not isinstance(memory, ContextualEpisodicMemoryState):
                raise RuntimeError("v25 triage lost episodic memory state")
            utilization[stage_index].append(memory.valid.float().mean(dim=1).cpu())
            stage = model.stages[stage_index]
            if (
                isinstance(stage, FactorizedIdentityContextEpisodicMemoryStage)
                and stage.last_pair_strength is not None
            ):
                strengths[stage_index].append(stage.last_pair_strength.detach().float().cpu())
        with capture_second_chunk_context_metrics(model, records):
            with _autocast(x.device):
                model(
                    second,
                    state=state,
                    hard=True,
                    route_mode="hard_sparse",
                    update_memory=False,
                )
    result: dict[str, Any] = {}
    for stage_index in range(len(model.stages)):
        row = {
            name: _summary(values)
            for name, values in records.get(stage_index, {}).items()
        }
        row["valid_slot_utilization"] = _summary(utilization[stage_index])
        row["selected_write_strength"] = _summary(strengths[stage_index])
        result[str(stage_index)] = row
    return result


def memory_diagnostic(
    model,
    *,
    data_dir: str,
    device: torch.device,
) -> dict[str, Any]:
    versions_before = parameter_versions(model)
    batches = _fixed_batches(
        data_dir=data_dir,
        batches=MEMORY_BATCHES,
        batch_size=MEMORY_BATCH_SIZE,
        seed=DIAGNOSTIC_SEED,
        device=device,
    )
    production = _condition_losses(
        model,
        batches,
        update_memory=True,
        read_scales=1.0,
    )
    stream_only = _condition_losses(
        model,
        batches,
        update_memory=False,
        read_scales=None,
    )
    read_zero = _condition_losses(
        model,
        batches,
        update_memory=True,
        read_scales=0.0,
    )

    paired = {
        "production_vs_stream_only_overall": paired_difference_stats(
            production["overall"],
            stream_only["overall"],
            seed=DIAGNOSTIC_SEED + 10,
        ),
        "production_vs_stream_only_second_chunk": paired_difference_stats(
            production["second_chunk"],
            stream_only["second_chunk"],
            seed=DIAGNOSTIC_SEED + 11,
        ),
        "production_vs_write_preserving_read_zero_overall": paired_difference_stats(
            production["overall"],
            read_zero["overall"],
            seed=DIAGNOSTIC_SEED + 12,
        ),
        "production_vs_write_preserving_read_zero_second_chunk": paired_difference_stats(
            production["second_chunk"],
            read_zero["second_chunk"],
            seed=DIAGNOSTIC_SEED + 13,
        ),
    }

    sweep: dict[str, dict[str, float]] = {
        "0.00": _mean_loss_row(read_zero),
        "1.00": _mean_loss_row(production),
    }
    for alpha in READ_ALPHAS:
        key = f"{alpha:.2f}"
        if key in sweep:
            continue
        losses = _condition_losses(
            model,
            batches,
            update_memory=True,
            read_scales=alpha,
        )
        sweep[key] = _mean_loss_row(losses)
    alpha0 = sweep["0.00"]["second_chunk_nll"]
    alpha1 = sweep["1.00"]["second_chunk_nll"]
    for row in sweep.values():
        row["second_chunk_advantage_vs_alpha0"] = alpha0 - row["second_chunk_nll"]
        row["second_chunk_advantage_vs_production_alpha1"] = (
            alpha1 - row["second_chunk_nll"]
        )

    n_stages = len(model.stages)
    masks: dict[str, tuple[float, ...]] = {}
    for stage_index in range(n_stages):
        only = [0.0] * n_stages
        only[stage_index] = 1.0
        masks[f"only_stage_{stage_index}"] = tuple(only)
        leave = [1.0] * n_stages
        leave[stage_index] = 0.0
        masks[f"leave_out_stage_{stage_index}"] = tuple(leave)
    stage_rows: dict[str, dict[str, Any]] = {
        "all_zero": {
            "mask": [0.0] * n_stages,
            **_mean_loss_row(read_zero),
        },
        "all_one": {
            "mask": [1.0] * n_stages,
            **_mean_loss_row(production),
        },
    }
    for name, mask in masks.items():
        losses = _condition_losses(
            model,
            batches,
            update_memory=True,
            read_scales=mask,
        )
        stage_rows[name] = {"mask": list(mask), **_mean_loss_row(losses)}
    zero_nll = stage_rows["all_zero"]["second_chunk_nll"]
    one_nll = stage_rows["all_one"]["second_chunk_nll"]
    for row in stage_rows.values():
        row["second_chunk_advantage_vs_all_zero"] = (
            zero_nll - row["second_chunk_nll"]
        )
        row["second_chunk_advantage_vs_all_one"] = (
            one_nll - row["second_chunk_nll"]
        )

    mechanism = _stage_state_metrics(model, batches)
    versions_after = parameter_versions(model)
    return {
        "examples": MEMORY_BATCHES * MEMORY_BATCH_SIZE,
        "production": _mean_loss_row(production),
        "stream_only": _mean_loss_row(stream_only),
        "write_preserving_read_zero": _mean_loss_row(read_zero),
        "paired_statistics": paired,
        "read_scale_sweep": dict(sorted(sweep.items())),
        "stage_read_localization": stage_rows,
        "per_stage_second_chunk_mechanism": mechanism,
        "parameter_versions_unchanged": versions_before == versions_after,
    }


def _route_arrays(
    output: dict[str, object],
    batch_size: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    routes = output.get("stage_routes")
    if not isinstance(routes, list) or len(routes) != SEQ_LEN // CHUNK_SIZE:
        raise RuntimeError("unexpected AERA route history")
    counts: list[torch.Tensor] = []
    stages: list[torch.Tensor] = []
    for chunk in routes:
        if not isinstance(chunk, list) or len(chunk) != 4:
            raise RuntimeError("unexpected AERA stage route chunk")
        indicators: list[torch.Tensor] = []
        for item in chunk[1:]:
            if not isinstance(item, dict):
                raise RuntimeError("invalid stage route item")
            gate = item.get("stage_route_gate")
            if not isinstance(gate, torch.Tensor) or gate.shape != (batch_size, 1):
                raise RuntimeError("invalid optional-stage route gate")
            indicators.append((gate[:, 0] >= 0.5).float())
        matrix = torch.stack(indicators, dim=1)
        stages.append(matrix)
        counts.append(matrix.sum(dim=1))
    return counts, stages


def _binned_summary(
    difficulty: torch.Tensor,
    compute: torch.Tensor,
    *,
    bins: int,
) -> tuple[list[float], list[torch.Tensor]]:
    order = torch.argsort(difficulty, stable=True)
    groups = list(torch.tensor_split(order, bins))
    means = [float(compute[group].float().mean()) for group in groups]
    return means, groups


def bootstrap_group_mean_difference(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    seed: int,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, float]:
    a = left.detach().double().reshape(-1).cpu()
    b = right.detach().double().reshape(-1).cpu()
    if a.numel() < 2 or b.numel() < 2:
        raise ValueError("bootstrap groups must be nontrivial")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    ia = torch.randint(0, a.numel(), (resamples, a.numel()), generator=generator)
    ib = torch.randint(0, b.numel(), (resamples, b.numel()), generator=generator)
    diffs = b[ib].mean(dim=1) - a[ia].mean(dim=1)
    observed = float(b.mean() - a.mean())
    return {
        "resamples": float(resamples),
        "right_minus_left_observed": observed,
        "ci95_low": float(torch.quantile(diffs, 0.025)),
        "ci95_high": float(torch.quantile(diffs, 0.975)),
        "bootstrap_mean": float(diffs.mean()),
    }


def adaptivity_summary(
    difficulty: torch.Tensor,
    compute: torch.Tensor,
    positions: torch.Tensor,
    stage_indicators: torch.Tensor,
    *,
    bootstrap_seed: int,
) -> dict[str, Any]:
    difficulty = difficulty.detach().float().reshape(-1).cpu()
    compute = compute.detach().float().reshape(-1).cpu()
    positions = positions.detach().long().reshape(-1).cpu()
    stage_indicators = stage_indicators.detach().float().cpu()
    if not (
        difficulty.shape == compute.shape == positions.shape
        and stage_indicators.shape == (difficulty.numel(), 3)
    ):
        raise ValueError("adaptivity arrays have incompatible shapes")

    quartiles, q_groups = _binned_summary(difficulty, compute, bins=4)
    deciles, d_groups = _binned_summary(difficulty, compute, bins=10)
    monotonic = all(
        quartiles[index + 1] + QUARTILE_MONOTONIC_TOLERANCE >= quartiles[index]
        for index in range(3)
    )
    stage_by_decile = [
        [float(value) for value in stage_indicators[group].mean(dim=0)]
        for group in d_groups
    ]
    q4_minus_q3 = bootstrap_group_mean_difference(
        compute[q_groups[2]],
        compute[q_groups[3]],
        seed=bootstrap_seed,
    )
    by_position: list[dict[str, Any]] = []
    for position in range(SEQ_LEN // CHUNK_SIZE):
        mask = positions == position
        p_difficulty = difficulty[mask]
        p_compute = compute[mask]
        p_quartiles, _ = _binned_summary(p_difficulty, p_compute, bins=4)
        p_deciles, _ = _binned_summary(p_difficulty, p_compute, bins=10)
        by_position.append(
            {
                "chunk_position": position,
                "samples": int(mask.sum()),
                "spearman_rho": spearman_rho(p_difficulty, p_compute),
                "quartile_compute_means_easy_to_hard": p_quartiles,
                "decile_compute_means_easy_to_hard": p_deciles,
                "hardest_minus_easiest_quartile": p_quartiles[-1] - p_quartiles[0],
                "hardest_minus_easiest_decile": p_deciles[-1] - p_deciles[0],
            }
        )

    return {
        "samples": int(difficulty.numel()),
        "spearman_rho": spearman_rho(difficulty, compute),
        "quartile_compute_means_easy_to_hard": quartiles,
        "quartile_monotonic_with_original_0_05_tolerance": monotonic,
        "q4_minus_q3_compute": quartiles[3] - quartiles[2],
        "q4_minus_q3_bootstrap": q4_minus_q3,
        "decile_compute_means_easy_to_hard": deciles,
        "hardest_minus_easiest_quartile": quartiles[-1] - quartiles[0],
        "hardest_minus_easiest_decile": deciles[-1] - deciles[0],
        "optional_stage_run_fraction_by_difficulty_decile": stage_by_decile,
        "chunk_position_summaries": by_position,
    }


def adaptivity_diagnostic(
    model,
    transformer,
    *,
    data_dir: str,
    device: torch.device,
) -> dict[str, Any]:
    versions_before = parameter_versions(model)
    batches = _fixed_batches(
        data_dir=data_dir,
        batches=ADAPTIVITY_BATCHES,
        batch_size=ADAPTIVITY_BATCH_SIZE,
        seed=DIAGNOSTIC_SEED + 1,
        device=device,
    )
    difficulty_rows: list[torch.Tensor] = []
    position_rows: list[torch.Tensor] = []
    enabled_compute: list[torch.Tensor] = []
    enabled_stages: list[torch.Tensor] = []
    disabled_compute: list[torch.Tensor] = []
    disabled_stages: list[torch.Tensor] = []

    for x, y in batches:
        with _autocast(device):
            transformer_logits = transformer(x)
        token_loss = F.cross_entropy(
            transformer_logits.float().reshape(-1, VOCAB_SIZE),
            y.reshape(-1),
            reduction="none",
        ).reshape(ADAPTIVITY_BATCH_SIZE, SEQ_LEN)

        enabled = _model_forward(model, x, update_memory=True)
        disabled = _model_forward(model, x, update_memory=False)
        enabled_counts, enabled_matrix = _route_arrays(enabled, ADAPTIVITY_BATCH_SIZE)
        disabled_counts, disabled_matrix = _route_arrays(disabled, ADAPTIVITY_BATCH_SIZE)
        for chunk_index, start in enumerate(range(0, SEQ_LEN, CHUNK_SIZE)):
            difficulty_rows.append(
                token_loss[:, start : start + CHUNK_SIZE].mean(dim=1).detach().cpu()
            )
            position_rows.append(
                torch.full((ADAPTIVITY_BATCH_SIZE,), chunk_index, dtype=torch.long)
            )
            enabled_compute.append(enabled_counts[chunk_index].detach().cpu())
            enabled_stages.append(enabled_matrix[chunk_index].detach().cpu())
            disabled_compute.append(disabled_counts[chunk_index].detach().cpu())
            disabled_stages.append(disabled_matrix[chunk_index].detach().cpu())

    difficulty = torch.cat(difficulty_rows)
    positions = torch.cat(position_rows)
    if difficulty.numel() != EXPECTED_CHUNKS:
        raise RuntimeError(
            f"unexpected adaptivity sample count {difficulty.numel()} != {EXPECTED_CHUNKS}"
        )
    enabled_result = adaptivity_summary(
        difficulty,
        torch.cat(enabled_compute),
        positions,
        torch.cat(enabled_stages),
        bootstrap_seed=DIAGNOSTIC_SEED + 101,
    )
    disabled_result = adaptivity_summary(
        difficulty,
        torch.cat(disabled_compute),
        positions,
        torch.cat(disabled_stages),
        bootstrap_seed=DIAGNOSTIC_SEED + 102,
    )
    versions_after = parameter_versions(model)
    return {
        "difficulty_definition": "matched Transformer per-chunk held-out next-token NLL",
        "memory_enabled_hard_sparse": enabled_result,
        "memory_disabled_hard_sparse": disabled_result,
        "parameter_versions_unchanged": versions_before == versions_after,
    }


def _timing_stats(milliseconds: Sequence[float], *, batch_size: int) -> dict[str, float]:
    tensor = torch.tensor(list(milliseconds), dtype=torch.float64)
    if tensor.numel() < 2:
        raise ValueError("timing summary requires multiple samples")
    median_ms = float(torch.quantile(tensor, 0.50))
    return {
        "samples": float(tensor.numel()),
        "median_ms": median_ms,
        "p10_ms": float(torch.quantile(tensor, 0.10)),
        "p90_ms": float(torch.quantile(tensor, 0.90)),
        "mean_ms": float(tensor.mean()),
        "tokens_per_second_from_median": batch_size * SEQ_LEN * 1000.0 / median_ms,
    }


def _cuda_timed_call(call: Callable[[], object]) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    call()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end))


def _routing_accounting(output: dict[str, object], batch_size: int) -> dict[str, Any]:
    counts, matrices = _route_arrays(output, batch_size)
    flat = torch.cat(counts)
    stage_matrix = torch.cat(matrices, dim=0)
    return {
        "chunk_examples": int(flat.numel()),
        "mean_optional_stages": float(flat.float().mean()),
        "optional_stage_run_fractions": [
            float(value) for value in stage_matrix.float().mean(dim=0)
        ],
        "total_stage_execution_fraction": float((1.0 + flat.float().mean()) / 4.0),
    }


def _profile_top_cuda(call: Callable[[], object], *, limit: int = 12) -> list[dict[str, Any]]:
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
        profile_memory=False,
    ) as profile:
        call()
    rows = []
    for item in profile.key_averages():
        self_cuda_us = float(
            getattr(
                item,
                "self_cuda_time_total",
                getattr(item, "self_device_time_total", 0.0),
            )
        )
        rows.append(
            {
                "operator": str(item.key),
                "self_cuda_time_us": self_cuda_us,
                "cuda_time_us": float(
                    getattr(
                        item,
                        "cuda_time_total",
                        getattr(item, "device_time_total", 0.0),
                    )
                ),
                "calls": int(item.count),
            }
        )
    rows.sort(key=lambda row: row["self_cuda_time_us"], reverse=True)
    return rows[:limit]


def systems_condition_names() -> tuple[str, ...]:
    return (
        "transformer",
        "aera_core_memory_bypassed",
        "aera_read_path_empty_state",
        "aera_writes_only_reads_bypassed",
        "aera_full_memory",
    )


def systems_diagnostic(
    model,
    transformer,
    *,
    device: torch.device,
) -> dict[str, Any]:
    versions_before = parameter_versions(model)
    results: dict[str, Any] = {}
    for batch_size in SYSTEM_BATCH_SIZES:
        generator = torch.Generator(device="cpu").manual_seed(
            DIAGNOSTIC_SEED + 10_000 + batch_size
        )
        tokens = torch.randint(
            0,
            VOCAB_SIZE,
            (batch_size, SEQ_LEN),
            generator=generator,
        ).to(device)

        def transformer_call():
            with _autocast(device):
                return transformer(tokens)

        def core_call():
            with bypass_ficem_reads(model):
                return _model_forward(model, tokens, update_memory=False)

        def read_path_call():
            return _model_forward(model, tokens, update_memory=False)

        def writes_only_call():
            with bypass_ficem_reads(model):
                return _model_forward(model, tokens, update_memory=True)

        def full_call():
            return _model_forward(model, tokens, update_memory=True)

        calls: dict[str, Callable[[], object]] = {
            "transformer": transformer_call,
            "aera_core_memory_bypassed": core_call,
            "aera_read_path_empty_state": read_path_call,
            "aera_writes_only_reads_bypassed": writes_only_call,
            "aera_full_memory": full_call,
        }
        if tuple(calls) != systems_condition_names():
            raise RuntimeError("systems condition wiring drifted from frozen issue369 protocol")
        for call in calls.values():
            for _ in range(SYSTEM_WARMUP_CALLS):
                call()
        samples: dict[str, list[float]] = {name: [] for name in calls}
        names = list(calls)
        for round_index in range(SYSTEM_ROUNDS):
            rotated = names[round_index:] + names[:round_index]
            for name in rotated:
                for _ in range(SYSTEM_TIMED_CALLS_PER_ROUND):
                    samples[name].append(_cuda_timed_call(calls[name]))
        summaries = {
            name: _timing_stats(values, batch_size=batch_size)
            for name, values in samples.items()
        }

        accounting_output = full_call()
        if not isinstance(accounting_output, dict):
            raise RuntimeError("full AERA timing call did not return routing output")
        routing = _routing_accounting(accounting_output, batch_size)
        transformer_tps = summaries["transformer"]["tokens_per_second_from_median"]
        full_tps = summaries["aera_full_memory"]["tokens_per_second_from_median"]
        core_tps = summaries["aera_core_memory_bypassed"]["tokens_per_second_from_median"]
        core_ms = summaries["aera_core_memory_bypassed"]["median_ms"]
        read_ms = summaries["aera_read_path_empty_state"]["median_ms"]
        writes_ms = summaries["aera_writes_only_reads_bypassed"]["median_ms"]
        full_ms = summaries["aera_full_memory"]["median_ms"]
        reference = SYSTEMS_REFERENCE_UNTRAINED_MS[batch_size]
        results[str(batch_size)] = {
            "timings": summaries,
            "routing": routing,
            "full_aera_vs_transformer_speed_ratio": full_tps / transformer_tps,
            "memory_bypassed_aera_vs_transformer_speed_ratio": core_tps / transformer_tps,
            "read_path_overhead_ms": read_ms - core_ms,
            "write_path_overhead_ms": writes_ms - core_ms,
            "full_incremental_memory_overhead_ms": full_ms - core_ms,
            "trained_full_minus_untrained_reference_full_ms": full_ms - reference["full"],
            "untrained_issue365_reference_ms": reference,
            "profiler_top_cuda_full_aera": _profile_top_cuda(full_call),
        }
        del tokens
        torch.cuda.empty_cache()

    versions_after = parameter_versions(model)
    return {
        "timing_method": "CUDA events; 3 warmups; 20 calls/condition x5 rotated interleaved rounds",
        "rows": results,
        "parameter_versions_unchanged": versions_before == versions_after,
    }


def _load_models(
    *,
    run_dir: str,
    device: torch.device,
):
    root = Path(run_dir)
    aera_path = root / "aera.pt"
    transformer_path = root / "transformer.pt"
    if not aera_path.exists() or not transformer_path.exists():
        raise FileNotFoundError("seed8471 source checkpoints are missing")
    aera_payload = torch.load(aera_path, map_location="cpu", weights_only=False)
    transformer_payload = torch.load(
        transformer_path,
        map_location="cpu",
        weights_only=False,
    )
    if (
        aera_payload.get("seed") != SOURCE_SEED
        or transformer_payload.get("seed") != SOURCE_SEED
    ):
        raise RuntimeError("source checkpoint seed mismatch")
    torch.manual_seed(SOURCE_SEED)
    aera = v25.build_aera(device).eval()
    torch.manual_seed(SOURCE_SEED)
    transformer = v25.build_transformer(device).eval()
    aera.load_state_dict(aera_payload["model"], strict=True)
    transformer.load_state_dict(transformer_payload["model"], strict=True)
    aera.set_memory_pretraining_mode(False)
    return aera, transformer


def source_preflight(*, data_dir: str, run_dir: str) -> dict[str, Any]:
    data = validate_production_data(data_dir)
    root = Path(run_dir)
    aera_path = root / "aera.pt"
    transformer_path = root / "transformer.pt"
    if not aera_path.exists() or not transformer_path.exists():
        raise FileNotFoundError("seed8471 source checkpoints are missing")
    aera_payload = torch.load(aera_path, map_location="cpu", weights_only=False)
    transformer_payload = torch.load(
        transformer_path,
        map_location="cpu",
        weights_only=False,
    )
    if (
        aera_payload.get("seed") != SOURCE_SEED
        or transformer_payload.get("seed") != SOURCE_SEED
    ):
        raise RuntimeError("source checkpoint seed mismatch")
    return {
        "protocol": frozen_protocol(),
        "data": data,
        "source_checkpoint_seed": SOURCE_SEED,
        "aera_checkpoint_exists": True,
        "transformer_checkpoint_exists": True,
        "source_result_path": str(root / "result.json"),
        "gpu_authorized_by_preflight": False,
        "training_performed": False,
        "checkpoint_mutated": False,
    }


def _recommendations(
    memory: dict[str, Any],
    adaptivity: dict[str, Any],
    systems: dict[str, Any],
) -> list[str]:
    recommendations: list[str] = []
    sweep = memory["read_scale_sweep"]
    stage_rows = memory["stage_read_localization"]
    best_sweep_advantage = max(
        float(row["second_chunk_advantage_vs_alpha0"]) for row in sweep.values()
    )
    production_advantage = float(
        sweep["1.00"]["second_chunk_advantage_vs_alpha0"]
    )
    best_stage_advantage = max(
        float(row["second_chunk_advantage_vs_all_zero"])
        for row in stage_rows.values()
    )
    if (
        best_sweep_advantage >= production_advantage + 0.001
        or best_stage_advantage >= production_advantage + 0.001
        or best_sweep_advantage >= 0.005
        or best_stage_advantage >= 0.005
    ):
        recommendations.append("memory_calibration_or_stage_interaction_candidate")
    else:
        recommendations.append("memory_mechanism_effect_too_small_requires_v26")

    enabled = adaptivity["memory_enabled_hard_sparse"]
    if (
        not enabled["quartile_monotonic_with_original_0_05_tolerance"]
        and enabled["q4_minus_q3_bootstrap"]["ci95_high"] < 0.0
    ):
        recommendations.append("router_hard_tail_repair_required")

    core_miss = False
    ficem_miss = False
    for batch_size in SYSTEM_BATCH_SIZES:
        row = systems["rows"][str(batch_size)]
        full_ratio = float(row["full_aera_vs_transformer_speed_ratio"])
        core_ratio = float(row["memory_bypassed_aera_vs_transformer_speed_ratio"])
        threshold = 0.25 if batch_size == 8 else 1.25
        if core_ratio < threshold:
            core_miss = True
        if full_ratio < threshold and float(
            row["full_incremental_memory_overhead_ms"]
        ) > 0.15 * float(
            row["timings"]["aera_core_memory_bypassed"]["median_ms"]
        ):
            ficem_miss = True
    if core_miss:
        recommendations.append("systems_core_execution_repair_required")
    if ficem_miss:
        recommendations.append("ficem_runtime_repair_required")
    return list(dict.fromkeys(recommendations))


def run_checkpoint_triage(
    *,
    data_dir: str,
    run_dir: str,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("issue369 checkpoint triage requires one CUDA L4")
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    validate_production_data(data_dir)
    aera, transformer = _load_models(run_dir=run_dir, device=device)
    versions_initial = parameter_versions(aera)

    memory = memory_diagnostic(aera, data_dir=data_dir, device=device)
    if not memory["parameter_versions_unchanged"]:
        raise RuntimeError("memory diagnostic changed parameter versions")
    torch.cuda.empty_cache()

    adaptivity = adaptivity_diagnostic(
        aera,
        transformer,
        data_dir=data_dir,
        device=device,
    )
    if not adaptivity["parameter_versions_unchanged"]:
        raise RuntimeError("adaptivity diagnostic changed parameter versions")
    torch.cuda.empty_cache()

    systems = systems_diagnostic(aera, transformer, device=device)
    if not systems["parameter_versions_unchanged"]:
        raise RuntimeError("systems diagnostic changed parameter versions")

    versions_final = parameter_versions(aera)
    recommendations = _recommendations(memory, adaptivity, systems)
    return {
        "scope": "aera_v25_post_seed8471_checkpoint_only_triage_issue369",
        "protocol": frozen_protocol(),
        "gpu": torch.cuda.get_device_name(device),
        "source_checkpoint": {
            "seed": SOURCE_SEED,
            "run_dir": run_dir,
            "strict_load": True,
            "parameter_versions_unchanged": versions_initial == versions_final,
        },
        "memory_diagnostic": memory,
        "adaptivity_diagnostic": adaptivity,
        "systems_diagnostic": systems,
        "recommendations": recommendations,
        "training_performed": False,
        "checkpoint_mutated": False,
        "optimizer_created": False,
        "backward_performed": False,
        "new_model_checkpoint_written": False,
        "primary_issue368_result_changed": False,
        "claims": {
            "v25_pass": False,
            "architecture_freeze_authorized": False,
            "s2_authorized": False,
            "independent_replication_credit": False,
            "100m_authorized": False,
            "breakthrough_proven": False,
        },
    }


def main() -> None:
    raise SystemExit(
        "issue369 triage is launched only through its guarded Modal entrypoint"
    )


if __name__ == "__main__":
    main()
