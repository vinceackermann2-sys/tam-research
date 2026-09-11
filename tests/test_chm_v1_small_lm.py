from __future__ import annotations

import gc

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from tam_research.chm_v1_exact_index import ExactEpisodicIndex
from tam_research.chm_v1_small_lm import (
    ADDRESS_DIM,
    LOCAL_WINDOW,
    NON_SCIENTIFIC_SMOKE_SEED,
    RETRIEVAL_HOPS,
    SCIENTIFIC_SEEDS,
    CHMV1EIEMLM,
    EpisodicState,
    build_matched_pair,
    parameter_accounting,
    parameter_digest,
)


def test_exact_index_matches_flat_including_duplicate_distance_ties() -> None:
    # Duplicate vectors are an intentional tie case. Stable insertion IDs, not
    # tree traversal order, must choose the same winner on both paths.
    points = np.asarray(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
        ],
        dtype=np.float32,
    )
    item_ids = np.asarray([9, 3, 7, 1, 5], dtype=np.int64)
    index = ExactEpisodicIndex(points, item_ids, leaf_size=1)
    flat, indexed = index.assert_exact(np.asarray([1.0, 0.0], dtype=np.float32))
    assert flat.item_id == 3
    assert indexed.item_id == 3

    rng = np.random.default_rng(NON_SCIENTIFIC_SMOKE_SEED)
    points = rng.normal(size=(257, 16)).astype(np.float32)
    ids = rng.permutation(np.arange(1000, 1257, dtype=np.int64))
    index = ExactEpisodicIndex(points, ids, leaf_size=8)
    for query in rng.normal(size=(96, 16)).astype(np.float32):
        flat, indexed = index.assert_exact(query)
        assert indexed.item_id == flat.item_id
        assert indexed.position == flat.position
        assert indexed.squared_distance == flat.squared_distance


def test_preregistered_parameter_match_is_below_one_percent_without_dummy_params() -> None:
    counts = parameter_accounting()
    assert counts["local_trainable_parameters"] == 24_940_288
    assert counts["eiem_extra_parameters"] == 2 * 256 * ADDRESS_DIM + 256
    assert counts["within_preregistered_one_percent"] is True
    assert counts["delta_fraction"] < 0.01
    assert counts["local_window"] == 512
    assert counts["retrieval_hops"] == RETRIEVAL_HOPS == 2
    del counts
    gc.collect()


def test_paired_backbones_are_identical_and_scientific_seeds_are_guarded() -> None:
    local, eiem = build_matched_pair(NON_SCIENTIFIC_SMOKE_SEED, torch.device("cpu"))
    for name, value in local.backbone.state_dict().items():
        assert torch.equal(value, eiem.backbone.state_dict()[name])
    del local, eiem
    gc.collect()

    for seed in SCIENTIFIC_SEEDS:
        with pytest.raises(RuntimeError, match="scientific seed refused"):
            build_matched_pair(seed, torch.device("cpu"))


def test_differentiable_flat_memory_is_causal_and_trains_address_path() -> None:
    torch.manual_seed(NON_SCIENTIFIC_SMOKE_SEED)
    model = CHMV1EIEMLM()
    tokens = torch.randint(0, 50_257, (1, 8))
    targets = torch.randint(0, 50_257, (1, 8))
    logits = model.forward_flat_differentiable(tokens)
    assert logits.shape == (1, 8, 50_257)
    assert torch.isfinite(logits).all()
    loss = F.cross_entropy(logits.reshape(-1, 50_257), targets.reshape(-1))
    loss.backward()
    for parameter in (
        model.query_address.weight,
        model.key_address.weight,
        model.memory_gate_logit,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert float(parameter.grad.abs().sum()) > 0.0
    del model, logits, loss
    gc.collect()


def test_session_writes_are_post_query_two_hop_isolated_and_parameter_safe() -> None:
    torch.manual_seed(NON_SCIENTIFIC_SMOKE_SEED)
    model = CHMV1EIEMLM().eval()
    state_a = EpisodicState("a")
    state_b = EpisodicState("b")
    before = parameter_digest(model)

    first = torch.randint(0, 50_257, (1, 6))
    logits1, stats1 = model.forward_session_chunk(
        first,
        [state_a],
        mode="indexed",
        update_memory=True,
    )
    assert torch.isfinite(logits1).all()
    assert stats1.calls == 0  # current chunk cannot read itself on either hop
    assert len(state_a) == 6
    assert len(state_b) == 0
    assert parameter_digest(model) == before

    second = torch.randint(0, 50_257, (1, 5))
    logits2, stats2 = model.forward_session_chunk(
        second,
        [state_a],
        mode="indexed",
        update_memory=True,
        verify_indexed_exactness=True,
    )
    assert torch.isfinite(logits2).all()
    assert stats2.calls == 5 * RETRIEVAL_HOPS
    assert stats2.exact_match_rate == 1.0
    assert len(state_a) == 11
    assert len(state_b) == 0
    assert parameter_digest(model) == before

    state_a.reset()
    assert len(state_a) == 0
    assert state_a.next_item_id == 0
    del model
    gc.collect()


def test_local_window_hard_limit() -> None:
    torch.manual_seed(NON_SCIENTIFIC_SMOKE_SEED)
    model = CHMV1EIEMLM().eval()
    too_long = torch.zeros((1, LOCAL_WINDOW + 1), dtype=torch.long)
    with pytest.raises(ValueError, match="local window"):
        model.forward_local(too_long)
