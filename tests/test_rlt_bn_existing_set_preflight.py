"""Zero GPU / zero checkpoint / zero training correctness tests for BE/BN."""
import pytest
import torch

import experiments.rlt.analysis.bn_existing_set_preflight as bn
from experiments.rlt.analysis.bn_existing_set_preflight import (
    PREFLIGHT_ONLY_SEED, matched_existing_write_triplets,
    model_free_report, is_set,
)
from experiments.rlt.cpu_lastwrite_probe_ar import SET0, last_write_labels


def test_only_known_trained_gaps_and_unconsumed_preflight_seed():
    with pytest.raises(ValueError):
        matched_existing_write_triplets(120, seed=PREFLIGHT_ONLY_SEED)
    with pytest.raises(ValueError):
        matched_existing_write_triplets(16, seed=20671081)
    with pytest.raises(ValueError):
        matched_existing_write_triplets(16, seed=58232)


@pytest.fixture(scope="module")
def generated():
    return {gap: matched_existing_write_triplets(gap, seed=PREFLIGHT_ONLY_SEED + gap)
            for gap in (16, 64)}


@pytest.mark.parametrize("gap", (16, 64))
def test_same_row_counterfactuals_are_oracle_preserving(gap, generated):
    p = generated[gap]
    assert p.population_total == 256 and p.population_per_class == (128, 128)
    assert 0 < len(p.labels) < 256  # Explicit conditioning on eligible stratum
    assert len(p.labels) == len(p.local_positions) == len(p.remote_positions)
    write = 127 - gap
    rows = torch.arange(len(p.labels))
    assert bool((p.local_positions < write).all())
    assert bool((p.remote_positions < p.local_positions).all())
    assert torch.equal(p.baseline[rows, p.local_positions],
                       p.baseline[rows, p.remote_positions])
    assert torch.equal(last_write_labels(p.baseline), p.labels)
    assert torch.equal(last_write_labels(p.local_flip), p.labels)
    assert torch.equal(last_write_labels(p.remote_flip), p.labels)
    assert torch.equal(is_set(p.baseline), is_set(p.local_flip))
    assert torch.equal(is_set(p.baseline), is_set(p.remote_flip))
    assert torch.equal(p.baseline[:, write:], p.local_flip[:, write:])
    assert torch.equal(p.baseline[:, write:], p.remote_flip[:, write:])
    assert bool((p.baseline[:, write] == p.labels + SET0).all())
    assert bool(((p.baseline != p.local_flip).sum(1) == 1).all())
    assert bool(((p.baseline != p.remote_flip).sum(1) == 1).all())
    assert bool(((p.local_flip != p.remote_flip).sum(1) == 2).all())


def test_pure_preflight_deterministic_and_no_checkpoint_execution(monkeypatch, generated):
    # Reuse validated generated examples, not four more slow full-source draws.
    monkeypatch.setattr(bn, "matched_existing_write_triplets",
                        lambda gap, seed: generated[gap])
    a = model_free_report()
    b = model_free_report()
    assert a == b
    assert a["type"] == "MODEL_FREE_INPUT_PREFLIGHT_ONLY"
    assert not a["training"] and not a["checkpoint_loaded"]
    for gap in ("16", "64"):
        row = a["rows"][gap]
        assert row["selected"] == row["selected_class0"] + row["selected_class1"]
        assert 0 < row["selected"] < row["population"]
        assert row["selected_indices_sha256"]
