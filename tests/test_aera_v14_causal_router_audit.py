from __future__ import annotations

import torch

from tam_research.aera_v14_causal_router_audit import (
    balanced_accuracy,
    binary_auc,
    fit_linear_probe,
    rebatching_flip_metrics,
)


def test_binary_metrics_are_exact_on_separable_scores():
    scores = torch.tensor([0.1, 0.2, 0.8, 0.9])
    labels = torch.tensor([0.0, 0.0, 1.0, 1.0])
    assert binary_auc(scores, labels) == 1.0
    assert balanced_accuracy(scores, labels) == 1.0


def test_fresh_linear_probe_recovers_causal_signal():
    g = torch.Generator().manual_seed(14001)
    x = torch.randn(240, 6, generator=g)
    y = (1.7 * x[:, 0] - 0.9 * x[:, 1] > 0).float()
    result = fit_linear_probe(
        x[:180],
        y[:180],
        x[180:],
        y[180:],
        seed=14002,
    )
    assert result["auc"] > 0.95
    assert result["balanced_accuracy"] > 0.85


def test_batch_relative_targets_are_peer_context_dependent():
    # Fixed absolute losses are regrouped with different peers. Mid-ranked examples
    # should sometimes cross the per-batch rank thresholds even though their own
    # loss never changes.
    losses = torch.linspace(0.0, 1.0, 64)
    metrics = rebatching_flip_metrics(
        losses,
        run_rates=(0.50, 1.0 / 3.0, 1.0 / 6.0),
        batch_size=8,
        repeats=48,
        seed=14003,
    )
    assert len(metrics) == 3
    assert all(row["unstable_example_fraction"] > 0.20 for row in metrics)
    assert all(row["expected_pairwise_label_flip_rate"] > 0.03 for row in metrics)
