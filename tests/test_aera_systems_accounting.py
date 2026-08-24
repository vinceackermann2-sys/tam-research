from __future__ import annotations

import pytest
import torch

from tam_research.aera_systems_accounting import (
    expected_total_batch_stage_invocation_fraction,
    independent_batch_invocation_probability,
    routing_execution_accounting,
)


def _synthetic_half_compute_full_batch_invocation() -> dict[str, object]:
    # Two chunks, batch8, four stages. Foundation always runs. Per chunk the
    # optional stages run on 4/8, 3/8, and 1/8 examples respectively. That is
    # exactly 16 selected stage-example pairs out of 32 total -> 50% per-example
    # stage execution, while every stage is invoked for every batch/chunk because
    # each has at least one selected example.
    routes = []
    patterns = [
        torch.ones(8),
        torch.tensor([1, 1, 1, 1, 0, 0, 0, 0], dtype=torch.float32),
        torch.tensor([1, 1, 1, 0, 0, 0, 0, 0], dtype=torch.float32),
        torch.tensor([1, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32),
    ]
    for _ in range(2):
        routes.append(
            [
                {"stage_route_gate": pattern.unsqueeze(1).clone()}
                for pattern in patterns
            ]
        )
    return {"stage_routes": routes}


def test_half_per_example_compute_can_still_invoke_every_stage_for_batch() -> None:
    summary = routing_execution_accounting(_synthetic_half_compute_full_batch_invocation())
    assert summary["per_example_stage_execution_fraction"] == pytest.approx(0.5)
    assert summary["per_batch_stage_invocation_fraction"] == pytest.approx(1.0)
    assert summary["per_stage_example_run_fractions"] == pytest.approx([1.0, 0.5, 0.375, 0.125])
    assert summary["per_stage_batch_invocation_fractions"] == pytest.approx([1.0, 1.0, 1.0, 1.0])
    assert summary["mean_selected_examples_per_invoked_stage"] == pytest.approx([8.0, 4.0, 3.0, 1.0])


def test_v17_target_rates_imply_high_batch8_stage_invocation_if_independent() -> None:
    rates = [0.5, 1.0 / 3.0, 1.0 / 6.0]
    invocations = [independent_batch_invocation_probability(rate, 8) for rate in rates]
    assert invocations == pytest.approx([
        0.99609375,
        0.9609815576893765,
        0.7674319606386221,
    ])
    total = expected_total_batch_stage_invocation_fraction(rates, batch_size=8)
    assert total == pytest.approx(0.9311268170819996)


def test_batch_invocation_pressure_grows_with_batch_size() -> None:
    rate = 1.0 / 6.0
    assert independent_batch_invocation_probability(rate, 1) == pytest.approx(rate)
    assert independent_batch_invocation_probability(rate, 8) > 0.75
    assert independent_batch_invocation_probability(rate, 32) > 0.99


def test_accounting_rejects_missing_authoritative_gates() -> None:
    with pytest.raises(ValueError, match="stage_route_gate"):
        routing_execution_accounting({"stage_routes": [[{}]]})
