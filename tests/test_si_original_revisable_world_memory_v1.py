"""RWM1 engineering regression: revision, action keys, isolation, persistence.

Deterministic developer fixtures only: untrained width-16 models, no trained
checkpoint, heldout, scientific seed or breakthrough claim. IDs follow
outputs/research_si_rwm1_20261021/TEST_PLAN.md (T1-T22).
"""
from copy import deepcopy
import hashlib

import pytest
import torch

from tam_research.si_original_integrated_v1 import PROVENANCE
from tam_research.si_original_revisable_world_memory_v1 import (
    SCHEMA, RevisableWorldMemory, SIOriginalRevisableWorldMemoryV1,
    state_digest,
)
from tam_research.si_original_world_scoped_v2 import SIOriginalWorldScopedV2

OUTPUTS = ("logits", "state", "retrieval", "uncertainty", "direct_action",
           "action")


def _pair(**kwargs):
    torch.set_num_threads(2)
    torch.manual_seed(34113)  # deterministic developer fixture, NOT science
    base = SIOriginalWorldScopedV2(width=16, layers=4, experts=4,
                                   max_steps=3).eval()
    model = SIOriginalRevisableWorldMemoryV1(
        width=16, layers=4, experts=4, max_steps=3, **kwargs).eval()
    model.load_state_dict(base.state_dict(), strict=True)
    return base, model


def _model(**kwargs):
    return _pair(**kwargs)[1]


def _context(token):
    return [("text", torch.tensor([token], dtype=torch.long)),
            ("motor", torch.tensor([[1., 0., 0., 0.]]))]


@torch.no_grad()
def _execute(model, world, token, action=0, reward=0, skill=None):
    before = _context(token)
    after = before + [("text", torch.tensor([100 + reward], dtype=torch.long))]
    model.ingest_executed(before, action, float(reward), after, world=world,
                          skill=skill)
    prior = model._backbone(before)[0][0]
    return prior, model._backbone(after)[0][0] - prior


def _vec(*values):
    return torch.tensor(values, dtype=torch.float32)


def _put(memory, world, action, key, value, reward=0.):
    return memory.put(world, action, key, value, reward=reward,
                      provenance=PROVENANCE)


def _assert_same_outputs(expected, actual):
    for key in OUTPUTS:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    assert torch.equal(actual["memory_hit"], expected["memory_hit"])


# A. Memory semantics ------------------------------------------------------

def test_t1_same_identity_replaces_value_and_audits_only_real_changes():
    memory = RevisableWorldMemory(4)
    key = _vec(1, 0, 0, 0)
    assert _put(memory, "A", 0, key, _vec(1, 1, 1, 1)) == 0
    assert _put(memory, "A", 0, key, _vec(2, 2, 2, 2), 1.) == 1
    _put(memory, "A", 0, key, _vec(2, 2, 2, 2), 1.)
    record = memory.find("A", key, 0)
    assert len(memory.rows["A"]) == 1
    assert (record.writes, record.changes, record.first_write,
            record.last_write) == (3, 1, 0, 2)
    assert torch.equal(memory.read("A", key), _vec(2, 2, 2, 2))
    [entry] = memory.audit["A"]
    assert entry["old_value_sha256"] == hashlib.sha256(
        _vec(1, 1, 1, 1).numpy().tobytes()).hexdigest()
    assert (entry["old_last_write"], entry["superseded_by"]) == (0, 1)


def test_t2_identity_tolerance_boundary():
    memory = RevisableWorldMemory(4)
    key = _vec(1, 0, 0, 0)
    _put(memory, "A", 0, key, _vec(1, 1, 1, 1))
    _put(memory, "A", 0, key + 1.5e-6, _vec(2, 2, 2, 2))
    assert len(memory.rows["A"]) == 1
    _put(memory, "A", 0, key + 3e-6, _vec(3, 3, 3, 3))
    assert len(memory.rows["A"]) == 2


def test_t3_actions_are_distinct_records_and_actionless_read_is_latest():
    memory = RevisableWorldMemory(4)
    key = _vec(0, 1, 0, 0)
    for action in range(4):
        _put(memory, "A", action, key, _vec(action, action, action, action))
    assert len(memory.rows["A"]) == 4
    for action in range(4):
        assert torch.equal(memory.read("A", key, action),
                           _vec(action, action, action, action))
    assert torch.equal(memory.read("A", key), _vec(3, 3, 3, 3))
    _put(memory, "A", 1, key, _vec(9, 9, 9, 9), 1.)
    assert torch.equal(memory.read("A", key), _vec(9, 9, 9, 9))


def test_t4_identity_mode_rejects_near_and_unseen_queries():
    memory = RevisableWorldMemory(4)
    key = _vec(1, 0, 0, 0)
    _put(memory, "A", 0, key, _vec(1, 1, 1, 1))
    near = _vec(1, .01, 0, 0)
    assert float(torch.nn.functional.cosine_similarity(near, key, dim=0)) > .9999
    for world, query in (("A", near), ("B", key)):
        value, hit = memory.lookup(world, query)
        assert not hit and torch.equal(value, torch.zeros(4))


def test_t5_cosine_mode_threshold_and_ties_go_to_latest_write():
    memory = RevisableWorldMemory(4, match_threshold=.9)
    _put(memory, "A", 0, _vec(1, .1, 0, 0), _vec(1, 1, 1, 1))
    _put(memory, "A", 0, _vec(1, -.1, 0, 0), _vec(2, 2, 2, 2))
    assert not memory.lookup("A", _vec(0, 0, 1, 0))[1]
    value, hit = memory.lookup("A", _vec(1, 0, 0, 0))
    assert hit and torch.equal(value, _vec(2, 2, 2, 2))
    value, hit = memory.lookup("A", _vec(1, .05, 0, 0))
    assert hit and torch.equal(value, _vec(1, 1, 1, 1))


def test_t6_reads_never_mutate_memory_state():
    torch.manual_seed(7)
    memory = RevisableWorldMemory(4, match_threshold=.5)
    for i in range(6):
        _put(memory, "A" if i % 2 else "B", i % 4, torch.randn(4),
             torch.randn(4))
    snapshot = state_digest(memory.dump())
    for _ in range(100):
        memory.read("A", torch.randn(4))
        memory.lookup("B", torch.randn(4), action=1)
    assert state_digest(memory.dump()) == snapshot


def test_t7_world_quota_evicts_only_own_oldest_written_record():
    memory = RevisableWorldMemory(4, world_capacity=2)
    a0, a1, a2 = _vec(1, 0, 0, 0), _vec(0, 1, 0, 0), _vec(0, 0, 1, 0)
    _put(memory, "A", 0, a0, _vec(1, 1, 1, 1))
    _put(memory, "A", 0, a1, _vec(1, 1, 1, 1))
    for i in range(10):
        _put(memory, "C", 0, _vec(0, 0, 0, i + 1), _vec(i, i, i, i))
    assert memory.lookup("A", a0)[1] and memory.lookup("A", a1)[1]
    assert len(memory.rows["C"]) == 2
    _put(memory, "A", 0, a0, _vec(5, 5, 5, 5))  # refreshes a0's recency
    _put(memory, "A", 0, a2, _vec(1, 1, 1, 1))  # evicts a1, never a0 or C
    assert memory.lookup("A", a0)[1] and memory.lookup("A", a2)[1]
    assert not memory.lookup("A", a1)[1]
    assert len(memory.rows["C"]) == 2


def test_t8_audit_log_is_bounded_and_counts_dropped_entries():
    memory = RevisableWorldMemory(4, audit_capacity=2)
    key = _vec(1, 0, 0, 0)
    for i in range(5):
        _put(memory, "A", 0, key, _vec(i, i, i, i))
    assert len(memory.audit["A"]) == 2 and memory.audit_dropped["A"] == 2
    assert [e["superseded_by"] for e in memory.audit["A"]] == [3, 4]


def test_t9_invalid_writes_raise_without_mutation():
    memory = RevisableWorldMemory(4)
    _put(memory, "A", 0, _vec(1, 0, 0, 0), _vec(1, 1, 1, 1))
    snapshot = state_digest(memory.dump())
    good = dict(world="A", action=0, key=_vec(0, 1, 0, 0),
                value=_vec(1, 1, 1, 1), reward=0., provenance=PROVENANCE)
    for change in (dict(world=""), dict(action=4), dict(action=True),
                   dict(key=_vec(float("nan"), 0, 0, 0)),
                   dict(value=torch.ones(5)), dict(reward=2.),
                   dict(reward=float("nan")), dict(provenance="model_guess")):
        arguments = {**good, **change}
        with pytest.raises(ValueError):
            memory.put(arguments.pop("world"), arguments.pop("action"),
                       arguments.pop("key"), arguments.pop("value"),
                       **arguments)
    assert state_digest(memory.dump()) == snapshot


# B. Model level ------------------------------------------------------------

def test_t10_no_added_parameters_and_state_dicts_interchange():
    base, model = _pair()
    assert ([(n, p.shape) for n, p in base.named_parameters()]
            == [(n, p.shape) for n, p in model.named_parameters()])
    assert list(base.state_dict()) == list(model.state_dict())
    base.load_state_dict(model.state_dict(), strict=True)
    with pytest.raises(ValueError):
        SIOriginalRevisableWorldMemoryV1(width=16, memory_capacity=8)


@torch.no_grad()
def test_t11_memory_disabled_outputs_and_fast_state_match_v2_bitwise():
    base, model = _pair()
    for event in (("A", 11, 0, 1), ("B", 12, 1, 0), ("A", 13, 2, 1),
                  ("C", 11, 3, 0)):
        for m in (base, model):
            _execute(m, *event)
    assert base.world_fast_update_counts() == model.world_fast_update_counts()
    assert base.fast_updates == model.fast_updates == 4
    for world, (u, v, n) in base._world_fast.items():
        mu, mv, mn = model._world_fast[world]
        assert torch.equal(u, mu) and torch.equal(v, mv) and n == mn
    for world in ("A", "B", "C", "D"):
        for token in (11, 12, 13, 14):
            expected = base(_context(token), world=world, enable_memory=False)
            actual = model(_context(token), world=world, enable_memory=False)
            for key, value in expected.items():
                if isinstance(value, torch.Tensor):
                    torch.testing.assert_close(actual[key], value,
                                               rtol=0, atol=0)
                elif key == "expert_routes":
                    for a, b in zip(actual[key], value, strict=True):
                        torch.testing.assert_close(a, b, rtol=0, atol=0)
                else:
                    assert actual[key] == value


@torch.no_grad()
def test_t12_revised_fact_returns_latest_delta_where_v2_returns_first():
    base, model = _pair()
    for m in (base, model):
        key, old = _execute(m, "A", 11, 0, 0)
        _, new = _execute(m, "A", 11, 0, 1)
    assert not torch.allclose(old, new)
    torch.testing.assert_close(model.memory.read("A", key), new, rtol=0, atol=0)
    # Documented V2 defect D1, recorded here rather than treated as a fault.
    torch.testing.assert_close(base.memory.read("A", key), old, rtol=0, atol=0)
    out = model(_context(11), world="A", enable_search=False)
    torch.testing.assert_close(out["retrieval"][0], new, rtol=0, atol=0)


@torch.no_grad()
def test_t13_rejected_feedback_and_world_cap_leave_all_state_unchanged():
    model = _model(max_worlds=1)
    _execute(model, "A", 11)
    snapshot = state_digest(model.export_lifetime_state())
    before = _context(12)
    after = before + [("text", torch.tensor([100]))]
    with pytest.raises(ValueError, match="capacity"):
        model.ingest_executed(before, 0, 0., after, world="B")
    for args, kwargs in (((0, 0., after), dict(provenance="model_guess")),
                         ((0, 5., after), {}), ((0, 0., before), {}),
                         ((4, 0., after), {})):
        with pytest.raises(ValueError):
            model.ingest_executed(before, *args, world="A", **kwargs)
    assert state_digest(model.export_lifetime_state()) == snapshot
    assert model.memory._staged is None


def test_t14_replay_quota_and_consolidation_match_v2_without_eviction():
    model = _model(replay_world_capacity=2)
    for token in (11, 12, 13):
        _execute(model, "A", token)
    for token in (14, 15, 16, 17, 18):
        _execute(model, "B", token, reward=1)
    assert len(model.replay.rows["A"]) == len(model.replay.rows["B"]) == 2
    assert [r.write_index for r in model.replay.rows["A"]] == [1, 2]
    indices = [r.write_index for r in model.replay]
    assert indices == sorted(indices) == [1, 2, 6, 7]

    base, model = _pair()
    for event in (("A", 11, 0, 0), ("B", 12, 1, 1), ("A", 13, 2, 1),
                  ("B", 14, 3, 0)):
        for m in (base, model):
            _execute(m, *event)
    fingerprint = model.representation_sha256()
    losses = []
    for m in (base, model):
        optimizer = torch.optim.AdamW(m.parameters(), lr=.01)
        losses.append(m.consolidate_verified_replay(optimizer, count=3))
    assert losses[0] == losses[1]
    for (name, a), (_, b) in zip(base.named_parameters(),
                                 model.named_parameters(), strict=True):
        assert torch.equal(a, b), name
    assert model.representation_sha256() == fingerprint


@torch.no_grad()
def test_t15_forward_reports_memory_hits_and_zero_retrieval_on_miss():
    model = _model()
    _execute(model, "A", 11)
    hit = model(_context(11), world="A", enable_search=False)
    miss = model(_context(12), world="A", enable_search=False)
    off = model(_context(11), world="A", enable_memory=False,
                enable_search=False)
    assert hit["memory_hit"].tolist() == [True]
    assert miss["memory_hit"].tolist() == [False]
    assert off["memory_hit"].tolist() == [False]
    assert torch.equal(miss["retrieval"], torch.zeros(1, 16))


@torch.no_grad()
def test_t16_other_world_pressure_never_changes_world_a_outputs():
    model = _model(memory_world_capacity=2, replay_world_capacity=2)
    _execute(model, "A", 11)
    _execute(model, "A", 12)
    expected = [model(_context(t), world="A", enable_search=False)
                for t in (11, 12, 13)]
    for token in range(20, 26):
        _execute(model, "B", token, reward=1)
    for token, before in zip((11, 12, 13), expected):
        _assert_same_outputs(before, model(_context(token), world="A",
                                           enable_search=False))
    assert len(model.memory.rows["A"]) == len(model.replay.rows["A"]) == 2


# C. Lifetime persistence ---------------------------------------------------

EVENTS = (("A", 11, 0, 0), ("B", 12, 1, 1), ("A", 11, 0, 1), ("C", 13, 2, 0),
          ("B", 14, 3, 1), ("A", 15, 0, 0))


def test_t17_lifetime_roundtrip_restores_everything(tmp_path):
    model = _model()
    for event in EVENTS:
        _execute(model, *event)
    path = tmp_path / "life.pt"
    model.save_lifetime(path)
    assert set(torch.load(path, weights_only=True)) == {"weights", "lifetime"}
    restored = SIOriginalRevisableWorldMemoryV1.load_lifetime(path).eval()
    record = restored.export_lifetime_state()
    assert record["schema"] == SCHEMA
    assert state_digest(record) == state_digest(model.export_lifetime_state())
    assert len(restored.replay) == len(model.replay) == len(EVENTS)
    with torch.no_grad():
        for world in ("A", "B", "C", "D"):
            for token in (11, 12, 13, 14, 15):
                _assert_same_outputs(
                    model(_context(token), world=world, enable_search=False),
                    restored(_context(token), world=world,
                             enable_search=False))


def test_t18_save_lifetime_never_overwrites(tmp_path):
    path = tmp_path / "life.pt"
    path.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        _model().save_lifetime(path)
    assert path.read_bytes() == b"keep"


def test_t19_representation_fingerprint_guards_stored_keys():
    model = _model()
    _execute(model, "A", 11)
    record = model.export_lifetime_state()
    stale = _model()
    with torch.no_grad():
        stale.text_encoder.weight[11, 0] += 1e-3
    snapshot = state_digest(stale.export_lifetime_state())
    with pytest.raises(ValueError, match="fingerprint"):
        stale.import_lifetime_state(record)
    assert state_digest(stale.export_lifetime_state()) == snapshot
    head = _model()
    with torch.no_grad():
        head.action_head.weight[0, 0] += 1e-3
    head.import_lifetime_state(record)
    assert (state_digest(head.export_lifetime_state()["memory"])
            == state_digest(record["memory"]))


def test_t20_import_rejects_old_schemas_and_corrupt_ledgers():
    base, model = _pair()
    for event in (("A", 11, 0, 0), ("A", 11, 0, 1), ("B", 12, 0, 1)):
        for m in (base, model):
            _execute(m, *event)
    record = model.export_lifetime_state()
    v1_style = deepcopy(base.export_persistent_state())
    v1_style.pop("fast_worlds")
    v1_style["schema"] = "si-original-integrated-v1/1"
    corrupted = [base.export_persistent_state(), v1_style]

    def corrupt(change):
        bad = deepcopy(record)
        change(bad)
        corrupted.append(bad)

    def duplicate_identity(bad):
        copy = deepcopy(bad["memory"]["rows"]["A"][0])
        copy["first_write"] = copy["last_write"] = 3
        bad["memory"]["write_count"] = 4
        bad["memory"]["rows"]["A"].append(copy)

    corrupt(lambda bad: bad.__setitem__("fast_updates", 4))
    corrupt(duplicate_identity)
    corrupt(lambda bad: bad["replay"].__setitem__(
        "A", bad["replay"]["A"] * 17))
    corrupt(lambda bad: bad["memory"]["rows"]["A"][0].__setitem__(
        "first_write", 2))
    corrupt(lambda bad: bad["memory"]["rows"]["A"][0].__setitem__(
        "changes", 5))
    corrupt(lambda bad: bad["replay"]["B"][0].__setitem__("write_index", 99))
    corrupt(lambda bad: bad["memory"]["rows"]["B"][0]["value"].__setitem__(
        0, float("nan")))
    corrupt(lambda bad: bad["replay"]["A"][0].__setitem__("skill", "ghost"))
    corrupt(lambda bad: bad["config"].__setitem__("memory_world_capacity", 8))
    corrupt(lambda bad: bad["replay"].pop("B"))
    fresh = _model()
    empty = state_digest(fresh.export_lifetime_state())
    for bad in corrupted:
        with pytest.raises(ValueError):
            fresh.import_lifetime_state(bad)
        assert state_digest(fresh.export_lifetime_state()) == empty


def test_t21_interrupted_and_resumed_run_equals_uninterrupted(tmp_path):
    whole = _model()
    for event in EVENTS:
        _execute(whole, *event)
    part = _model()
    for event in EVENTS[:3]:
        _execute(part, *event)
    part.save_lifetime(tmp_path / "half.pt")
    resumed = SIOriginalRevisableWorldMemoryV1.load_lifetime(
        tmp_path / "half.pt").eval()
    for event in EVENTS[3:]:
        _execute(resumed, *event)
    assert (state_digest(resumed.export_lifetime_state())
            == state_digest(whole.export_lifetime_state()))
    assert state_digest(resumed.state_dict()) == state_digest(whole.state_dict())
    with torch.no_grad():
        for world in ("A", "B", "C"):
            for token in (11, 12, 13, 14, 15):
                _assert_same_outputs(
                    whole(_context(token), world=world, enable_search=False),
                    resumed(_context(token), world=world,
                            enable_search=False))


def test_t22_lifetime_restores_skills_and_refuses_skill_mismatch(tmp_path):
    model = _model()
    model.add_skill("grasp")
    _execute(model, "A", 11, skill="grasp")
    path = tmp_path / "skill.pt"
    model.save_lifetime(path)
    restored = SIOriginalRevisableWorldMemoryV1.load_lifetime(path)
    assert list(restored.skills) == ["grasp"]
    assert restored.replay.rows["A"][0].skill == "grasp"
    with pytest.raises(ValueError, match="skills"):
        _model().import_lifetime_state(model.export_lifetime_state())
    payload = torch.load(path, weights_only=True)
    payload["lifetime"]["skills"] = []
    torch.save(payload, tmp_path / "mismatch.pt")
    with pytest.raises(ValueError):
        SIOriginalRevisableWorldMemoryV1.load_lifetime(tmp_path / "mismatch.pt")
