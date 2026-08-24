from __future__ import annotations

"""Measurement-only accounting for AERA sparse routing systems behavior.

This module deliberately does not change routing, architecture, training, kernels,
or inference semantics. It separates two quantities that legacy diagnostics can
conflate:

1. per-example hard stage execution: how much conditional compute individual
   examples actually request;
2. per-batch stage invocation: how often a stage must be entered at all because at
   least one example in the batch requests it.

For divergent per-example routing, low per-example prevalence can still imply a
high batch invocation rate, limiting wall-clock savings even when active compute
per example is sparse.
"""

from typing import Any

import torch


def independent_batch_invocation_probability(run_rate: float, batch_size: int) -> float:
    """Probability at least one example invokes a stage under independent routing."""
    if not 0.0 <= run_rate <= 1.0:
        raise ValueError("run_rate must be in [0,1]")
    if batch_size < 1:
        raise ValueError("batch_size must be >=1")
    return 1.0 - (1.0 - float(run_rate)) ** int(batch_size)


def expected_total_batch_stage_invocation_fraction(
    optional_run_rates: list[float] | tuple[float, ...],
    *,
    batch_size: int,
    foundation_stages: int = 1,
) -> float:
    """Independent-routing approximation including always-on foundation stages."""
    if foundation_stages < 0:
        raise ValueError("foundation_stages must be nonnegative")
    optional = [
        independent_batch_invocation_probability(rate, batch_size)
        for rate in optional_run_rates
    ]
    total_stages = foundation_stages + len(optional)
    if total_stages < 1:
        raise ValueError("need at least one stage")
    return (float(foundation_stages) + sum(optional)) / float(total_stages)


def routing_execution_accounting(output: dict[str, object]) -> dict[str, Any]:
    """Summarize actual hard gates at example and batch granularity.

    `stage_routes` must be [chunks][stages], with every route item exposing a
    `stage_route_gate` tensor shaped [batch,1]. The function does not infer
    execution from optional metadata such as `start`/`end`; the hard gate is the
    authoritative deployment decision.
    """
    routes = output.get("stage_routes")
    if not isinstance(routes, list) or not routes:
        raise ValueError("output missing non-empty stage_routes")

    n_chunks = len(routes)
    first_chunk = routes[0]
    if not isinstance(first_chunk, list) or not first_chunk:
        raise ValueError("invalid stage route history")
    n_stages = len(first_chunk)

    stage_example_runs = torch.zeros(n_stages, dtype=torch.float64)
    stage_batch_invocations = torch.zeros(n_stages, dtype=torch.float64)
    stage_selected_counts = torch.zeros(n_stages, dtype=torch.float64)
    batch_size: int | None = None

    for chunk in routes:
        if not isinstance(chunk, list) or len(chunk) != n_stages:
            raise ValueError("inconsistent stage route history")
        for stage_index, item in enumerate(chunk):
            if not isinstance(item, dict):
                raise ValueError("invalid stage route item")
            gate = item.get("stage_route_gate")
            if not isinstance(gate, torch.Tensor) or gate.ndim != 2 or gate.size(1) != 1:
                raise ValueError("stage_route_gate must be [batch,1]")
            if batch_size is None:
                batch_size = int(gate.size(0))
            elif int(gate.size(0)) != batch_size:
                raise ValueError("inconsistent route batch size")
            selected = (gate[:, 0] >= 0.5)
            count = int(selected.sum())
            stage_example_runs[stage_index] += count
            stage_selected_counts[stage_index] += count
            stage_batch_invocations[stage_index] += float(count > 0)

    assert batch_size is not None
    example_opportunities_per_stage = batch_size * n_chunks
    batch_opportunities_per_stage = n_chunks
    total_example_opportunities = example_opportunities_per_stage * n_stages
    total_batch_opportunities = batch_opportunities_per_stage * n_stages

    per_stage_example_fraction = stage_example_runs / float(example_opportunities_per_stage)
    per_stage_batch_invocation_fraction = stage_batch_invocations / float(batch_opportunities_per_stage)
    selected_per_invocation = torch.where(
        stage_batch_invocations > 0,
        stage_selected_counts / stage_batch_invocations,
        torch.zeros_like(stage_selected_counts),
    )

    return {
        "batch_size": batch_size,
        "chunks": n_chunks,
        "stages": n_stages,
        "per_example_stage_execution_fraction": float(stage_example_runs.sum() / total_example_opportunities),
        "per_batch_stage_invocation_fraction": float(stage_batch_invocations.sum() / total_batch_opportunities),
        "per_stage_example_run_fractions": [float(v) for v in per_stage_example_fraction],
        "per_stage_batch_invocation_fractions": [float(v) for v in per_stage_batch_invocation_fraction],
        "mean_selected_examples_per_invoked_stage": [float(v) for v in selected_per_invocation],
        "optional_per_example_stage_execution_fraction": (
            float(stage_example_runs[1:].sum() / (example_opportunities_per_stage * max(n_stages - 1, 1)))
            if n_stages > 1
            else 0.0
        ),
        "optional_per_batch_stage_invocation_fraction": (
            float(stage_batch_invocations[1:].sum() / (batch_opportunities_per_stage * max(n_stages - 1, 1)))
            if n_stages > 1
            else 0.0
        ),
        "measurement_only": True,
    }
