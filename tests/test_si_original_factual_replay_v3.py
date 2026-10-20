"""Original screenshot SI V3 factual sensory replay developmental contract tests.

These never run protected heldout, scientific seeds, GPU, or performance
confirmation. Caller-attested feedback is NOT an authenticated real world.
"""
from copy import deepcopy
import math

import pytest
import torch

from tam_research.si_original_factual_replay_v3 import SIOriginalFactualReplayV3


def _new():
    torch.set_num_threads(2)
    torch.manual_seed(34020)  # CPU-only deterministic engineering fixture
    return SIOriginalFactualReplayV3(
        width=16, layers=4, experts=4, max_steps=3,
        sensory_replay_capacity=4).eval()


def _real_public_history():
    before = [
        ("text", torch.tensor([11], dtype=torch.long)),
        ("image", torch.full((1, 3, 8, 8), 0.3)),
        ("audio", torch.full((1, 32), 0.4)),
        ("video", torch.full((1, 2, 3, 8, 8), 0.5)),
        ("motor", torch.tensor([[0., 1., 0., 0.]])),
    ]
    after = before + [("text", torch.tensor([12], dtype=torch.long))]
    return before, after


def test_real_before_after_sensor_replay_trains_original_backbone_and_all_encoders():
    model = _new()
    before, after = _real_public_history()
    old_text = model.text_encoder.weight.detach().clone()
    old_image = model.image_encoder[0].weight.detach().clone()
    old_backbone = model.blocks[0].norm1.weight.detach().clone()
    old_reason = model.latent_step.weight_ih.detach().clone()
    old_direct_action = model.action_head.weight.detach().clone()
    model.ingest_executed(before, action=2, reward=0.75,
                          after=after, world="A")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    report = model.train_sensor_world_replay(optimizer, world="A", count=1)
    assert report["status"] == "DEVELOPMENT_ONLY_FACTUAL_WORLD_REPLAY"
    assert report["executed_experiences"] == 1
    assert report["world"] == "A"
    assert math.isfinite(report["objective"])
    assert math.isfinite(report["next_representation_mse"])
    assert report["backbone_gradient_l1"] > 0
    assert report["latent_reasoner_gradient_l1"] > 0
    assert all(report["encoder_gradient_l1"][mod] > 0 for mod in
               ("text", "image", "audio", "video", "motor"))
    assert not torch.equal(old_text, model.text_encoder.weight)
    assert not torch.equal(old_image, model.image_encoder[0].weight)
    assert not torch.equal(old_backbone, model.blocks[0].norm1.weight)
    assert not torch.equal(old_reason, model.latent_step.weight_ih)
    # Executed actions are NOT optimal-action teaching labels.
    assert torch.equal(old_direct_action, model.action_head.weight)
    assert report["direct_policy_trained"] is False
    assert report["long_horizon_search_trained"] is False
    assert report["scientific_gate_passed"] is False


def test_stop_gradient_future_target_does_not_train_its_unique_token_embedding():
    model = _new()
    before, after = _real_public_history()
    model.ingest_executed(before, action=0, reward=0.,
                          after=after, world="A")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    model.train_sensor_world_replay(optimizer, world="A")
    # Token 12 exists ONLY in the next observation / stop-gradient target.
    assert torch.equal(model.text_encoder.weight.grad[12],
                       torch.zeros_like(model.text_encoder.weight.grad[12]))
    assert float(model.text_encoder.weight.grad[11].abs().sum()) > 0


def test_unexecuted_or_forged_feedback_is_not_saved_as_sensor_replay():
    model = _new()
    before, after = _real_public_history()
    with pytest.raises(ValueError):
        model.ingest_executed(before, 3, 0.5, after, world="A",
                              provenance="unexecuted_counterfactual")
    with pytest.raises(ValueError):
        model.ingest_executed(before, 3, float("nan"), after, world="A")
    with pytest.raises(ValueError):
        model.ingest_executed(before, 3, 0.5, before, world="A")
    assert not model.sensory_replay
    with pytest.raises(ValueError, match="insufficient"):
        model.train_sensor_world_replay(
            torch.optim.SGD(model.parameters(), lr=0.001), world="A")


def test_replay_history_is_immutable_snapshot_and_owned_by_world():
    model = _new()
    before, after = _real_public_history()
    model.ingest_executed(before, 1, 1., after, world="A")
    assert len(model.sensory_replay) == 1
    copied = model.sensory_replay[0]
    assert copied.action == 1 and copied.reward == 1.
    before[1][1].fill_(0.8)
    assert float(copied.before[1][1].mean()) == pytest.approx(0.3, abs=1e-6)
    with pytest.raises(ValueError, match="insufficient"):
        model.train_sensor_world_replay(
            torch.optim.SGD(model.parameters(), lr=0.001), world="B")
    assert model.world_fast_update_counts() == {"A": 1}


def test_capacity_bounded_and_no_unclaimed_implicit_disk_resume():
    model = _new()
    before, after = _real_public_history()
    for n in range(6):
        model.ingest_executed(before, n % 4, float(n % 2),
                              after=after, world="A")
    assert len(model.sensory_replay) == 4
    assert model.fast_updates == 6
    state = model.export_persistent_state()
    # V2 versioned checkpoint stores scoped factual LTM and fast state,
    # NOT the new raw sensory trajectory buffer.
    assert "sensory_replay" not in state
    restored = _new()
    restored.import_persistent_state(deepcopy(state))
    assert restored.world_fast_update_counts() == {"A": 6}
    assert not restored.sensory_replay
