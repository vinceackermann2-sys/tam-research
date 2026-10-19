"""Causal world-boundary regression for original screenshot SI V2.

No optimizer, heldout, scientific seed reservation, GPU or breakthrough claim.
The parent V1 evidence is preserved; this only fixes world adapter isolation.
"""
from copy import deepcopy

import pytest
import torch

from tam_research.si_original_world_scoped_v2 import (
    SCHEMA, SIOriginalWorldScopedV2,
)


def _model(*, max_worlds=32):
    torch.set_num_threads(2)
    torch.manual_seed(34113)  # deterministic developer fixture, NOT science
    return SIOriginalWorldScopedV2(
        width=16, layers=4, experts=4, max_steps=3,
        max_worlds=max_worlds,
    ).eval()


def _public():
    history = [
        ("text", torch.tensor([11], dtype=torch.long)),
        ("motor", torch.tensor([[1., 0., 0., 0.]])),
    ]
    return history, history + [("text", torch.tensor([12], dtype=torch.long))]


@torch.no_grad()
def test_world_a_reward_changes_own_fast_state_but_never_world_b():
    model = _model()
    before, after = _public()
    b0 = model(before, world="B", enable_search=False,
               enable_memory=False, enable_fast=False)["state"].clone()
    model.ingest_executed(before, 0, 1., after, world="A")
    a_with = model(before, world="A", enable_search=False,
                   enable_memory=False)["state"]
    a_off = model(before, world="A", enable_search=False,
                  enable_memory=False, enable_fast=False)["state"]
    b_with = model(before, world="B", enable_search=False,
                   enable_memory=False)["state"]
    torch.testing.assert_close(b_with, b0, rtol=0, atol=0)
    assert not torch.allclose(a_with, a_off, rtol=1e-8, atol=1e-8)
    assert model.world_fast_update_counts() == {"A": 1}
    assert model.fast_updates == 1


@torch.no_grad()
def test_world_b_feedback_does_not_mutate_world_a_adapter_or_neutral_buffers():
    model = _model()
    before, after = _public()
    neutral_u, neutral_v = model.fast_u.clone(), model.fast_v.clone()
    model.ingest_executed(before, 1, 1., after, world="A")
    a_before = model(before, world="A", enable_memory=False,
                     enable_search=False)["state"].clone()
    model.ingest_executed(before, 2, -1., after, world="B")
    a_after = model(before, world="A", enable_memory=False,
                    enable_search=False)["state"]
    torch.testing.assert_close(a_before, a_after, rtol=0, atol=0)
    torch.testing.assert_close(model.fast_u, neutral_u, rtol=0, atol=0)
    torch.testing.assert_close(model.fast_v, neutral_v, rtol=0, atol=0)
    assert model.world_fast_update_counts() == {"A": 1, "B": 1}


@torch.no_grad()
def test_world_scoped_checkpoint_roundtrip_preserves_each_adaptation():
    original = _model()
    before, after = _public()
    original.ingest_executed(before, 0, 1., after, world="A")
    original.ingest_executed(before, 1, -1., after, world="B")
    record = original.export_persistent_state()
    assert record["schema"] == SCHEMA
    assert record["fast_updates"] == 2
    fresh = _model()
    fresh.import_persistent_state(record)
    assert fresh.world_fast_update_counts() == {"A": 1, "B": 1}
    for world in ("A", "B", "C"):
        expected = original(before, world=world,
                            enable_search=False)["state"]
        actual = fresh(before, world=world,
                       enable_search=False)["state"]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@torch.no_grad()
def test_forged_or_invalid_feedback_never_creates_world_fast_updates():
    model = _model()
    before, after = _public()
    with pytest.raises(ValueError):
        model.ingest_executed(
            before, 0, 1., after, world="A", provenance="model_guess")
    with pytest.raises(ValueError):
        model.ingest_executed(before, 0, 5., after, world="A")
    with pytest.raises(ValueError):
        model.ingest_executed(before, 0, 1., before, world="A")
    assert model.world_fast_update_counts() == {}
    assert model.fast_updates == 0


@torch.no_grad()
def test_capacity_and_checkpoint_fail_closed():
    model = _model(max_worlds=1)
    before, after = _public()
    model.ingest_executed(before, 0, 1., after, world="A")
    with pytest.raises(ValueError, match="capacity"):
        model.ingest_executed(before, 0, 1., after, world="B")
    with pytest.raises(ValueError):
        model(before, world="", enable_search=False)
    assert model.world_fast_update_counts() == {"A": 1}

    record = model.export_persistent_state()
    wrong = deepcopy(record)
    wrong["fast_updates"] = 2
    with pytest.raises(ValueError, match="ledger"):
        _model(max_worlds=1).import_persistent_state(wrong)
    wrong = deepcopy(record)
    wrong["fast_worlds"]["A"]["u"][0][0] = float("nan")
    with pytest.raises(ValueError, match="contents"):
        _model(max_worlds=1).import_persistent_state(wrong)
    wrong = deepcopy(record)
    wrong["schema"] = "si-original-integrated-v1/1"
    with pytest.raises(ValueError):
        _model(max_worlds=1).import_persistent_state(wrong)
    assert model.world_fast_update_counts() == {"A": 1}
