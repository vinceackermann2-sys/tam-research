"""Engineering-only tests for factual durability. No heldout trials or scores."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from tam_research.si_pcma_v1 import PublicObservation
from tam_research.si_pcma_v1_ood_world import OODRuleWorld, OODWorldConfig
from tam_research.si_pcma_v1_fair_comparison import PCMAGRURewardPredictor
from tam_research.si_pcma_v1_neural_probe import FrozenNeuralTwoStepPolicy
from tam_research.si_pcma_v1_persistence import (
    MAX_CHECKPOINT_BYTES, checkpoint_bytes, load_checkpoint,
    model_weight_fingerprint, restore_checkpoint_bytes, save_checkpoint_atomic,
)


def _policy(kind="gru"):
    if kind == "gru":
        from tam_research.si_pcma_v1_fair_comparison import PCMAGRURewardPredictor
        model = PCMAGRURewardPredictor()
    else:
        from tam_research.si_pcma_v1_factual_learning import FactualRewardPredictor
        model = FactualRewardPredictor(kind)
    return FrozenNeuralTwoStepPolicy(model)


def _closed_policy(seed=6):
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    policy = _policy()
    world = OODRuleWorld(OODWorldConfig(seed + 2, "development", "stable"))
    for _ in range(6):
        event = world.step(policy.choose(world.observe()))
        policy.observe(event)
    assert len(policy.state.memory) == 3
    policy.restart_session()
    return policy, world


def test_factual_memory_checkpoint_recovers_same_values_and_clock(tmp_path):
    original, world = _closed_policy()
    checkpoint = tmp_path / "pcma.json"
    save_checkpoint_atomic(original, checkpoint)
    assert checkpoint.exists()
    assert len(checkpoint.read_bytes()) < MAX_CHECKPOINT_BYTES
    assert original.state._last_time == 6
    assert original.state.session_updates == 0
    assert original.state.total_updates == 3
    fresh = _policy()
    fresh.predictor.load_state_dict(original.predictor.state_dict())
    load_checkpoint(fresh, checkpoint)
    assert list(fresh.state.memory) == list(original.state.memory)
    assert fresh.state.u == original.state.u
    assert fresh.state.v == original.state.v
    assert fresh.state._last_time == original.state._last_time == 6
    assert fresh.state.total_updates == 3
    world.restart_session()
    obs = world.observe()
    assert fresh.choose(obs) == original.choose(obs)
    # Both policies can receive the same factual event in a test fixture;
    # the evaluator must only run one policy in a real experiment.
    actual = world.step(fresh._pending[1])
    fresh.observe(actual)
    original.observe(actual)
    assert fresh._last_time == original._last_time == 7


def test_real_new_python_process_restores_completed_public_memory(tmp_path):
    seed = 5
    original, _ = _closed_policy(seed)
    target = tmp_path / "restart.json"
    save_checkpoint_atomic(original, target)
    code = (
        "import torch; "
        "from tam_research.si_pcma_v1_fair_comparison import PCMAGRURewardPredictor; "
        "from tam_research.si_pcma_v1_neural_probe import FrozenNeuralTwoStepPolicy; "
        "from tam_research.si_pcma_v1_persistence import load_checkpoint; "
        f"torch.manual_seed({seed}); "
        "p=FrozenNeuralTwoStepPolicy(PCMAGRURewardPredictor()); "
        f"load_checkpoint(p, {str(target)!r}); "
        "print(len(p.state.memory), p.state._last_time, p.state.total_updates)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], check=True, text=True,
        capture_output=True, timeout=30,
    )
    assert completed.stdout.strip() == "3 6 3"


def test_checkpoint_rejects_unresolved_execution_and_mid_session(tmp_path):
    actor, world = _closed_policy()
    world.restart_session()
    first = actor.choose(world.observe())
    with pytest.raises(RuntimeError, match="boundary"):
        checkpoint_bytes(actor)
    actor.observe(world.step(first))
    with pytest.raises(RuntimeError, match="boundary"):
        checkpoint_bytes(actor)
    actor.choose(world.observe())
    with pytest.raises(RuntimeError, match="boundary"):
        save_checkpoint_atomic(actor, tmp_path / "unsafe.json")
    assert not (tmp_path / "unsafe.json").exists()


def test_checksum_failure_cannot_modify_existing_policy():
    actor, _ = _closed_policy()
    serialized = checkpoint_bytes(actor)
    fresh = _policy()
    fresh.predictor.load_state_dict(actor.predictor.state_dict())
    prior = tuple(fresh.state.memory)
    corrupted = serialized.replace(b"observed_net", b"observed_xet")
    with pytest.raises(ValueError, match="checksum"):
        restore_checkpoint_bytes(fresh, corrupted)
    assert tuple(fresh.state.memory) == prior and fresh._last_time == 0


def test_frozen_weight_mismatch_rejected_even_with_valid_checkpoint():
    actor, _ = _closed_policy()
    payload = checkpoint_bytes(actor)
    torch.manual_seed(99)
    other = _policy()
    assert model_weight_fingerprint(actor) != model_weight_fingerprint(other)
    with pytest.raises(ValueError, match="predictor_sha256"):
        restore_checkpoint_bytes(other, payload)
    assert len(other.state.memory) == 0 and other._last_time == 0


def test_cross_architecture_restore_is_forbidden():
    actor, _ = _closed_policy()
    payload = checkpoint_bytes(actor)
    other = _policy("transformer")
    with pytest.raises(ValueError, match="predictor_kind"):
        restore_checkpoint_bytes(other, payload)


def test_disabling_persistence_ablation_cannot_inherit_old_memory():
    actor, _ = _closed_policy()
    payload = checkpoint_bytes(actor)
    other = FrozenNeuralTwoStepPolicy(PCMAGRURewardPredictor(),
                                      memory_enabled=False)
    other.predictor.load_state_dict(actor.predictor.state_dict())
    with pytest.raises(ValueError, match="memory_enabled"):
        restore_checkpoint_bytes(other, payload)


def test_recomputed_checksum_still_cannot_introduce_nan_or_over_capacity():
    actor, _ = _closed_policy()
    payload = json.loads(checkpoint_bytes(actor))
    original_hash = payload["sha256"]

    def recalc(modified):
        raw = json.dumps(modified, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode()
        return json.dumps(
            {"sha256": hashlib.sha256(raw).hexdigest(), "payload": modified},
            sort_keys=True, separators=(",", ":")).encode()

    fresh = _policy()
    fresh.predictor.load_state_dict(actor.predictor.state_dict())
    payload["payload"]["capacity"] = 2
    with pytest.raises(ValueError, match="capacity"):
        restore_checkpoint_bytes(fresh, recalc(payload["payload"]))

    payload = json.loads(checkpoint_bytes(actor))
    payload["payload"]["u"][0][0] = 1e50
    with pytest.raises(ValueError, match="adapter row"):
        restore_checkpoint_bytes(fresh, recalc(payload["payload"]))
    assert fresh._last_time == 0 and not fresh.state.memory


def test_nonmonotonic_record_rejected_and_does_not_change_fresh_state():
    actor, _ = _closed_policy()
    payload = json.loads(checkpoint_bytes(actor))["payload"]
    payload["memory"][1]["first_time"] = payload["memory"][0]["first_time"]
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    forged = json.dumps({"sha256": hashlib.sha256(raw).hexdigest(),
                         "payload": payload}).encode()
    fresh = _policy()
    fresh.predictor.load_state_dict(actor.predictor.state_dict())
    with pytest.raises(ValueError, match="unordered"):
        restore_checkpoint_bytes(fresh, forged)
    assert fresh._last_time == 0 and not fresh.state.memory


def test_incompatible_continuation_and_all_invalid_oversized_inputs():
    actor, _ = _closed_policy()
    payload = checkpoint_bytes(actor)
    from tam_research.si_pcma_v1_probe_randomization import FrozenPublicContinuation
    other = FrozenNeuralTwoStepPolicy(
        PCMAGRURewardPredictor(),
        continuation=FrozenPublicContinuation((2, 0, 1)),
    )
    other.predictor.load_state_dict(actor.predictor.state_dict())
    with pytest.raises(ValueError, match="continuation"):
        restore_checkpoint_bytes(other, payload)
    for damaged in (b"not-json", b"{}", b" " * (MAX_CHECKPOINT_BYTES + 1)):
        with pytest.raises(ValueError):
            restore_checkpoint_bytes(other, damaged)
