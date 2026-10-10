"""PGW-v3 Stage-2B4 CPU derivative-capacity and resource preflight.

NO scientific training, optimizer/step, paid GPU, checkpoint, or science seed.
Uses frozen Stage-1B data/Stage-2A/2B0/2B2/2B3 implementations unchanged.
A one-batch nonzero-gradient count is NOT analytically active model capacity.
Wall-clock CPU times are machine-specific, NOT normalized training FLOPs.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import platform
import statistics
import sys
import time

import torch
import torch.nn.functional as F

from tam_research.pgw_v3_stage1b.oracle import make_example
from tam_research.pgw_v3_stage2a.model import (
    PGWV3Stage2A, PGWV3Stage2AConfig, batch_verified_examples,
)
from tam_research.pgw_v3_stage2b0.auxiliary import predictor_auxiliary_loss
from tam_research.pgw_v3_stage2b2.reference import (
    CausalReference, CausalReferenceConfig,
)
from tam_research.pgw_v3_stage2b3.position import GlobalPositionCausalReference
from tam_research.pgw_v3_stage2b3.accounting import (
    AttentionOperationCounts, attention_operation_counts,
)


ROUTE_MODES = (
    "hybrid", "utility_only", "surprise_only", "fixed_random",
    "recency", "no_workspace",
)
REFERENCE_ARMS = ("full_no_global", "full_global", "chunk_only")
ARM_NAMES = tuple(f"pgw_{mode}" for mode in ROUTE_MODES) + REFERENCE_ARMS
DELAYS = (1, 2, 4)
LOCAL_FIXTURE_INITIALIZATION = 1422  # structural CPU fixture, not a scientific seed


@dataclass(frozen=True)
class GradientActivity:
    grad_seen_elements: int
    nonzero_elements: int
    predictor_nonzero: int
    utility_nonzero: int
    other_workspace_nonzero: int
    nonworkspace_nonzero: int


@dataclass(frozen=True)
class ArmDerivativeReport:
    arm: str
    delay: int
    batch: int
    tokens: int
    instantiated: int
    answer_ce: GradientActivity
    auxiliary: GradientActivity
    union_nonzero: int
    state_unchanged: bool
    attention: AttentionOperationCounts


@dataclass(frozen=True)
class CpuDerivativeTiming:
    arm: str
    delay: int
    batch: int
    tokens: int
    warmups: int
    repeats: int
    requested_cpu_threads: int
    threads_restored: bool
    torch_version: str
    python_version: str
    platform_name: str
    median_forward_ms: float
    p95_forward_ms: float
    median_forward_and_ce_backward_ms: float
    p95_forward_and_ce_backward_ms: float
    unchanged_after_timing: bool
    attention: AttentionOperationCounts
    native_peak_tensor_memory: str = "unknown — not measured"
    timings_scope: str = (
        "CPU walltime; forward=no_grad answer; backward=answer forward+CE+backward; "
        "no optimizer; not comparable training FLOPs"
    )


def _validate_arm_delay(arm: str, delay: int) -> None:
    if arm not in ARM_NAMES:
        raise ValueError("unknown frozen Stage-2B4 arm")
    if type(delay) is not int or delay not in DELAYS:
        raise ValueError("delay must be 1, 2 or 4 completed chunks")


def _fixture(delay: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if type(delay) is not int or delay not in DELAYS:
        raise ValueError("invalid Stage-1B delay")
    samples = [
        make_example(
            split="validation",
            index=index,
            last_position=5,
            overwrite_count=2,
            interference=True,
            delay_chunks=delay,
            missing=missing,
        )
        for index, missing in ((0, False), (1, True))
    ]
    return batch_verified_examples(samples)


def _build_paired_arms() -> dict[str, torch.nn.Module]:
    """Same state within each architectural family, no global RNG change."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(LOCAL_FIXTURE_INITIALIZATION)
        pgw = PGWV3Stage2A(PGWV3Stage2AConfig(route_mode="hybrid")).cpu().eval()
        pgw_state = {k: v.detach().clone() for k, v in pgw.state_dict().items()}
        torch.manual_seed(LOCAL_FIXTURE_INITIALIZATION + 1)
        reference = CausalReference(
            CausalReferenceConfig(attention_scope="full")
        ).cpu().eval()
        reference_state = {
            k: v.detach().clone() for k, v in reference.state_dict().items()
        }
        models: dict[str, torch.nn.Module] = {}
        for mode in ROUTE_MODES:
            m = PGWV3Stage2A(PGWV3Stage2AConfig(route_mode=mode)).cpu().eval()
            m.load_state_dict(pgw_state, strict=True)
            models[f"pgw_{mode}"] = m
        for arm in REFERENCE_ARMS:
            if arm == "full_global":
                m = GlobalPositionCausalReference().cpu().eval()
            else:
                scope = "chunk" if arm == "chunk_only" else "full"
                m = CausalReference(
                    CausalReferenceConfig(attention_scope=scope)
                ).cpu().eval()
            m.load_state_dict(reference_state, strict=True)
            models[arm] = m
        return models


def _predictor_aux(
    model: torch.nn.Module, tokens: torch.Tensor, anchors: torch.Tensor
) -> torch.Tensor:
    if isinstance(model, PGWV3Stage2A):
        return predictor_auxiliary_loss(model, tokens, anchors)
    if isinstance(model, CausalReference):
        return model.forward_with_aux(tokens, anchors)[1]
    raise TypeError("unrecognized preflight model")


def _group(name: str) -> str:
    if name.startswith("workspace.predict_") or name.startswith("predict_"):
        return "predictor"
    if name.startswith("workspace.utility."):
        return "utility"
    if name.startswith("workspace."):
        return "other_workspace"
    return "nonworkspace"


def _activity(
    model: torch.nn.Module,
) -> tuple[GradientActivity, dict[str, torch.Tensor]]:
    seen = 0
    counts = {
        "predictor": 0, "utility": 0,
        "other_workspace": 0, "nonworkspace": 0,
    }
    masks: dict[str, torch.Tensor] = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            masks[name] = torch.zeros_like(p, dtype=torch.bool)
            continue
        if not bool(torch.isfinite(p.grad).all().item()):
            raise ValueError("nonfinite CPU derivative")
        seen += p.numel()
        mask = p.grad.detach().ne(0)
        masks[name] = mask
        counts[_group(name)] += int(mask.count_nonzero().item())
    return (
        GradientActivity(
            grad_seen_elements=seen,
            nonzero_elements=sum(counts.values()),
            predictor_nonzero=counts["predictor"],
            utility_nonzero=counts["utility"],
            other_workspace_nonzero=counts["other_workspace"],
            nonworkspace_nonzero=counts["nonworkspace"],
        ),
        masks,
    )


def _frozen_snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: x.detach().clone() for name, x in model.state_dict().items()}


def _unchanged(model: torch.nn.Module, before: dict[str, torch.Tensor]) -> bool:
    return all(
        torch.equal(x.detach(), before[name])
        for name, x in model.state_dict().items()
    )


def gradient_activity_preflight(
    *, arms: tuple[str, ...] = ARM_NAMES, delays: tuple[int, ...] = DELAYS
) -> tuple[ArmDerivativeReport, ...]:
    """One independent answer and auxiliary backward per arm/delay; no updates.

    For structural CI this is deliberately a small complete 9x3 panel, *not*
    a model training experiment, matched-active-capacity result or benchmark.
    """
    if not arms or not delays or len(set(arms)) != len(arms):
        raise ValueError("arm/delay selection must be nonempty and unique")
    for arm in arms:
        _validate_arm_delay(arm, 1)
    for delay in delays:
        if type(delay) is not int or delay not in DELAYS:
            raise ValueError("unrecognized Stage-1B delay")
    models = _build_paired_arms()
    results = []
    for delay in delays:
        tokens, anchors, targets = _fixture(delay)
        for arm in arms:
            model = models[arm]
            if any(p.device.type != "cpu" for p in model.parameters()):
                raise ValueError("CPU preflight only")
            frozen = _frozen_snapshot(model)
            model.zero_grad(set_to_none=True)
            F.cross_entropy(model(tokens, anchors), targets).backward()
            answer, answer_masks = _activity(model)
            model.zero_grad(set_to_none=True)
            _predictor_aux(model, tokens, anchors).backward()
            auxiliary, aux_masks = _activity(model)
            union = sum(
                int((answer_masks[name] | aux_masks[name]).count_nonzero().item())
                for name, _ in model.named_parameters()
            )
            instantiated = sum(p.numel() for p in model.parameters() if p.requires_grad)
            results.append(
                ArmDerivativeReport(
                    arm=arm,
                    delay=delay,
                    batch=tokens.shape[0],
                    tokens=tokens.shape[1],
                    instantiated=instantiated,
                    answer_ce=answer,
                    auxiliary=auxiliary,
                    union_nonzero=union,
                    state_unchanged=_unchanged(model, frozen),
                    attention=attention_operation_counts(
                        batch=tokens.shape[0], tokens=tokens.shape[1]
                    ),
                )
            )
            model.zero_grad(set_to_none=True)
    return tuple(results)


def _millis(values_ns: list[int]) -> tuple[float, float]:
    if not values_ns or any(x <= 0 for x in values_ns):
        raise ValueError("CPU timer samples must be positive")
    ordered = sorted(values_ns)
    median_ms = statistics.median(ordered) / 1e6
    nearest_p95_ms = ordered[math.ceil(len(ordered) * 0.95) - 1] / 1e6
    return median_ms, nearest_p95_ms


def cpu_derivative_timing_preflight(
    *, arm: str, delay: int, warmups: int = 3, repeats: int = 10
) -> CpuDerivativeTiming:
    """CPU single-thread walltime, not FLOP-normalized scientific throughput.

    Derivative timer includes BOTH its fresh answer forward and CE backward,
    explicitly not a backward-only measurement. Zero-grad occurs outside the
    measured range. Restores caller thread count in finally even on failure.
    """
    _validate_arm_delay(arm, delay)
    if (
        type(warmups) is not int or not 0 <= warmups <= 5
        or type(repeats) is not int or not 1 <= repeats <= 20
    ):
        raise ValueError("preflight warmups must be 0..5 and repeats 1..20")
    original_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        model = _build_paired_arms()[arm]
        tokens, anchors, targets = _fixture(delay)
        original = _frozen_snapshot(model)
        forward_ns: list[int] = []
        derivative_ns: list[int] = []
        for i in range(warmups + repeats):
            with torch.no_grad():
                t0 = time.perf_counter_ns()
                logits = model(tokens, anchors)
                t1 = time.perf_counter_ns()
            if logits.shape != (2, 33) or not bool(torch.isfinite(logits).all().item()):
                raise ValueError("invalid timed forward")
            model.zero_grad(set_to_none=True)
            t2 = time.perf_counter_ns()
            loss = F.cross_entropy(model(tokens, anchors), targets)
            loss.backward()
            t3 = time.perf_counter_ns()
            if not bool(torch.isfinite(loss).item()):
                raise ValueError("nonfinite timed derivative loss")
            if i >= warmups:
                forward_ns.append(t1 - t0)
                derivative_ns.append(t3 - t2)
        still_frozen = _unchanged(model, original)
        forward_med, forward_p95 = _millis(forward_ns)
        deriv_med, deriv_p95 = _millis(derivative_ns)
        data = (tokens.shape[0], tokens.shape[1])
    finally:
        torch.set_num_threads(original_threads)
    return CpuDerivativeTiming(
        arm=arm,
        delay=delay,
        batch=data[0],
        tokens=data[1],
        warmups=warmups,
        repeats=repeats,
        requested_cpu_threads=1,
        threads_restored=torch.get_num_threads() == original_threads,
        torch_version=str(torch.__version__),
        python_version=sys.version.split()[0],
        platform_name=platform.platform(),
        median_forward_ms=forward_med,
        p95_forward_ms=forward_p95,
        median_forward_and_ce_backward_ms=deriv_med,
        p95_forward_and_ce_backward_ms=deriv_p95,
        unchanged_after_timing=still_frozen,
        attention=attention_operation_counts(batch=data[0], tokens=data[1]),
    )
