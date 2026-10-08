from __future__ import annotations

import inspect

import pytest
import torch

import tam_research.chm_v3_daec_a_address_target_toy_1350 as toy


def test_stage_a_is_cpu_only_frozen_and_does_not_authorize_science() -> None:
    assert toy.PAID_GPU_AUTHORIZED is False
    assert toy.GENERATING_SCIENTIFIC_ALIGNED_V4_PROBES is False
    assert toy.HISTORICAL_SCIENTIFIC_SEED_CONSUMED == 2_013_161
    assert toy.ALLOWED_MEMORY_LENGTHS == (128, 256, 512, 1024)
    assert len(set(toy.SPLIT_SEEDS.values())) == 3
    assert set(toy.SPLIT_SEEDS) == {"train", "development", "test"}
    src = inspect.getsource(toy)
    for token in ("modal.App(", "modal.run(", "torch.cuda", "checkpoint.pt", "2013161"):
        assert token not in src


@pytest.mark.parametrize("length", (128, 256, 512, 1024))
def test_episodes_have_completed_memory_and_exact_pointer_labels(length: int) -> None:
    for idx in (1, 2, 5, 9, 13):
        ep = toy.generate_toy_episode("development", idx, memory_length=length)
        assert ep.query_view.shape == (toy.DIM,)
        assert ep.key_views.shape == (length, toy.DIM)
        assert ep.relation_payload.shape == (length, toy.DIM)
        assert ep.role.shape == (length,)
        assert ep.memory_answer_ids.shape == (length,)
        assert ep.overwritten == (idx % 4 == 1)
        if idx % 5 == 0:
            assert ep.target_relation_pos is None
            assert ep.target_answer_pos is None
            assert ep.correct_answer_id is None
        else:
            first = ep.target_relation_pos
            second = ep.target_answer_pos
            assert first is not None and second is not None
            assert ep.role[first].item() == toy.ROLE_RELATION
            assert ep.role[second].item() == toy.ROLE_RECORD
            assert ep.correct_answer_id == int(ep.memory_answer_ids[second])
            assert ep.correct_answer_id >= 8000
        assert not ep.query_view.is_cuda


def test_negative_and_overwrite_oracles_have_distinct_semantics() -> None:
    ep = toy.generate_toy_episode("train", 1)
    assert ep.overwritten is True
    assert ep.target_relation_pos is not None
    chosen_doc = ep.relation_payload[ep.target_relation_pos]
    # The overwrite record is an exact key duplicate with a stale distinct ID.
    record_indices = torch.where(ep.role == toy.ROLE_RECORD)[0].tolist()
    matching = [
        j for j in record_indices
        if torch.allclose(ep.key_views[j], toy._key_view(chosen_doc), atol=1e-6)
    ]
    assert len(matching) == 2
    older, newer = sorted(matching)
    assert ep.target_answer_pos == newer
    assert ep.memory_answer_ids[older] != ep.memory_answer_ids[newer]
    assert ep.memory_answer_ids[newer] == ep.correct_answer_id

    negative = toy.generate_toy_episode("train", 5)
    assert negative.target_relation_pos is None
    assert negative.target_answer_pos is None
    assert negative.correct_answer_id is None


def _oracle_projection_model() -> toy.PointerAddressToy:
    # Orthogonal inverse of the frozen view transformations, constructed from
    # each basis vector rather than manually duplicating production math.
    eye = torch.eye(toy.DIM)
    matrix_query = torch.stack(
        [toy._query_view(eye[:, i]) for i in range(toy.DIM)], dim=1
    )
    matrix_key = torch.stack(
        [toy._key_view(eye[:, i]) for i in range(toy.DIM)], dim=1
    )
    model = toy.PointerAddressToy()
    with torch.no_grad():
        model.query_projection.weight.copy_(matrix_query.T)
        model.key_projection.weight.copy_(matrix_key.T)
    return model.eval()


@pytest.mark.parametrize("index", (1, 2, 3, 4, 6, 7, 9, 13))
def test_independent_oracle_address_can_reach_stored_value(index: int) -> None:
    model = _oracle_projection_model()
    for length in (128, 256, 1024):
        ep = toy.generate_toy_episode("test", index, memory_length=length)
        result = toy.evaluate_hard_read(model, ep)
        assert result["first_correct"] is True
        assert result["answer_correct"] is True
        assert result["first_selected"] == ep.target_relation_pos
        assert result["second_selected"] == ep.target_answer_pos
        assert result["predicted_answer_id"] == ep.correct_answer_id
        if ep.overwritten:
            assert ep.memory_answer_ids[result["second_selected"]] >= 8000


def test_split_disjointness_and_reproducibility() -> None:
    a = toy.generate_toy_episode("train", 13)
    b = toy.generate_toy_episode("train", 13)
    dev = toy.generate_toy_episode("development", 13)
    test = toy.generate_toy_episode("test", 13)
    assert torch.equal(a.query_view, b.query_view)
    assert torch.equal(a.key_views, b.key_views)
    assert not torch.equal(a.key_views, dev.key_views)
    assert not torch.equal(dev.key_views, test.key_views)
    with pytest.raises(ValueError):
        toy.generate_toy_episode("test", 64)
    with pytest.raises(ValueError):
        toy.generate_toy_episode("train", 0, memory_length=2048)


def test_supervised_two_hop_loss_is_finite_gradients_reach_address_weights() -> None:
    ep = toy.generate_toy_episode("train", 7)
    model = toy.PointerAddressToy()
    loss = toy.pointer_supervision_loss(model, ep)
    loss.backward()
    assert torch.isfinite(loss).item()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert bool(torch.isfinite(parameter.grad).all().item())
        assert parameter.grad.abs().sum() > 0


def test_zero_gpu_toy_training_preserves_matched_initial_weights() -> None:
    torch.set_num_threads(1)
    baseline, trained = toy.train_cpu_pointer_toy(steps=64)
    # Baseline is the unchanged, equal-init control; not a Transformer/LM baseline.
    reference = toy.PointerAddressToy()
    for parameter, initial in zip(baseline.parameters(),reference.parameters()):
        assert torch.equal(parameter, initial)
    assert any(
        not torch.equal(a,b) for a,b in zip(baseline.parameters(),trained.parameters())
    )
    for model in (baseline,trained):
        report = toy.score_split(model, "test", cases=16)
        assert report["cases"] == 16
        assert report["negative_cases"] > 0
        assert report["positive_cases"] > 0
        assert report["no_gpu"] is True
        assert report["scientific_evidence"] is False
        assert 0 <= report["hard_answer_top1"] <= 1
        assert 0 <= report["hard_first_pointer_top1"] <= 1
    with pytest.raises(ValueError):
        toy.train_cpu_pointer_toy(steps=257)


def test_hard_evaluation_never_uses_gold_pointer_as_routing_input() -> None:
    source = inspect.getsource(toy.evaluate_hard_read)
    assert "ep.relation_payload[first]" in source
    assert "model.second_logits(ep, payload)" in source
    assert "ep.relation_payload[ep.target_relation_pos]" not in source
    assert "ep.target_answer_pos" not in source
    assert "torch.argmax(model.first_logits(ep))" in source
