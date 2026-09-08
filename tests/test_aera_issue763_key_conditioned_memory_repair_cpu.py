from __future__ import annotations

import json
from pathlib import Path
import subprocess

import torch

from tam_research import aera_issue763_key_conditioned_memory_repair_cpu as repair


def _reps(tokens: list[int], d_model: int = 4) -> torch.Tensor:
    mapping = {
        1000: torch.tensor([1.0, 0.0, 0.0, 0.0]),
        1001: torch.tensor([0.0, 1.0, 0.0, 0.0]),
        2000: torch.tensor([0.0, 0.0, 0.5, 0.0]),
        2001: torch.tensor([0.0, 0.0, 0.0, 0.5]),
        2002: torch.tensor([0.0, 0.0, -0.5, 0.0]),
    }
    marker = torch.tensor([0.25, 0.25, 0.25, 0.25])
    noise = torch.tensor([-0.1, 0.2, -0.3, 0.4])
    rows = []
    for token in tokens:
        if token in mapping:
            rows.append(mapping[token])
        elif token in {repair.WRITE, repair.UPDATE, repair.QUERY, repair.ANSWER, repair.RESET, repair.SEP}:
            rows.append(marker)
        else:
            rows.append(noise + float(token % 7) * 0.001)
    out = torch.stack(rows)
    assert out.shape == (len(tokens), d_model)
    return out


def test_authority_is_cpu_only_and_scientific_seeds_are_not_authorized() -> None:
    snapshot = repair.authority_snapshot()
    assert snapshot["research_issue"] == 763
    assert snapshot["cpu_test_seed"] == 763_999
    assert snapshot["consumed_scientific_seed"] == 17_641
    for key in (
        "gpu_authorized",
        "scientific_training_authorized",
        "scientific_seed_authorized",
        "seeds_2_3_authorized",
        "systems_optimization_authorized",
        "architecture_freeze_authorized",
        "scaling_authorized",
        "breakthrough_proven",
    ):
        assert snapshot[key] is False


def test_legacy_read_is_key_invariant_but_repaired_read_is_key_conditioned() -> None:
    memory = repair.EventAlignedDeltaMemory(4)
    state = memory.empty_state()
    write_tokens = [repair.WRITE, 1000, 2000, repair.SEP, repair.WRITE, 1001, 2001, repair.SEP]
    state, _ = repair.apply_capability_chunk(write_tokens, _reps(write_tokens), memory, state)

    a = [repair.QUERY, 1000, repair.ANSWER, 99, repair.SEP]
    b = [repair.QUERY, 1001, repair.ANSWER, 77, repair.SEP]
    a_reps, b_reps = _reps(a), _reps(b)
    # Keep the frozen token-0 representation identical; only the later key differs.
    b_reps[0] = a_reps[0]
    legacy_a = repair.legacy_chunk_start_read(a_reps, memory, state)
    legacy_b = repair.legacy_chunk_start_read(b_reps, memory, state)
    assert torch.equal(legacy_a, legacy_b)

    repaired_a = repair.repaired_query_read(a, a_reps, memory, state).recall
    repaired_b = repair.repaired_query_read(b, b_reps, memory, state).recall
    assert repaired_a is not None and repaired_b is not None
    assert not torch.equal(repaired_a, repaired_b)


def test_repaired_query_read_cannot_depend_on_future_tokens() -> None:
    memory = repair.EventAlignedDeltaMemory(4)
    state = memory.empty_state()
    write = [repair.WRITE, 1000, 2000, repair.SEP]
    state, _ = repair.apply_capability_chunk(write, _reps(write), memory, state)

    a = [31, repair.QUERY, 1000, repair.ANSWER, 41, repair.SEP, 51]
    b = [31, repair.QUERY, 1000, repair.ANSWER, 9999, repair.SEP, 8888]
    reps_a, reps_b = _reps(a), _reps(b)
    key_index = repair.find_query_key_index(a)
    reps_b[: key_index + 1] = reps_a[: key_index + 1]
    read_a = repair.repaired_query_read(a, reps_a, memory, state).recall
    read_b = repair.repaired_query_read(b, reps_b, memory, state).recall
    assert read_a is not None and read_b is not None
    assert torch.equal(read_a, read_b)


def test_event_aligned_write_is_invariant_to_trailing_noise() -> None:
    memory = repair.EventAlignedDeltaMemory(4)
    base = [repair.WRITE, 1000, 2000, repair.SEP]
    a = base + [101, 102, 103, 104]
    b = base + [901, 902, 903, 904]
    state_a, events_a = repair.apply_capability_chunk(a, _reps(a), memory, memory.empty_state())
    state_b, events_b = repair.apply_capability_chunk(b, _reps(b), memory, memory.empty_state())
    assert [e.kind for e in events_a] == ["write"]
    assert [e.kind for e in events_b] == ["write"]
    assert torch.equal(state_a.matrix, state_b.matrix)


def test_two_keys_are_distinguishable_and_update_replaces_stale_mapping() -> None:
    memory = repair.EventAlignedDeltaMemory(4)
    tokens = [
        repair.WRITE, 1000, 2000, repair.SEP,
        repair.WRITE, 1001, 2001, repair.SEP,
    ]
    state, _ = repair.apply_capability_chunk(tokens, _reps(tokens), memory, memory.empty_state())

    q0 = [repair.QUERY, 1000, repair.ANSWER]
    q1 = [repair.QUERY, 1001, repair.ANSWER]
    r0 = repair.repaired_query_read(q0, _reps(q0), memory, state).recall
    r1 = repair.repaired_query_read(q1, _reps(q1), memory, state).recall
    assert r0 is not None and r1 is not None
    assert not torch.equal(r0, r1)
    expected0 = torch.tanh(_reps([2000])[0])
    expected1 = torch.tanh(_reps([2001])[0])
    assert torch.allclose(r0, expected0, atol=1e-6)
    assert torch.allclose(r1, expected1, atol=1e-6)

    update = [repair.UPDATE, 1000, 2002, repair.SEP]
    updated, _ = repair.apply_capability_chunk(update, _reps(update), memory, state)
    new_r0 = repair.repaired_query_read(q0, _reps(q0), memory, updated).recall
    expected_new = torch.tanh(_reps([2002])[0])
    assert new_r0 is not None
    assert torch.allclose(new_r0, expected_new, atol=1e-6)
    assert not torch.allclose(new_r0, expected0, atol=1e-6)


def test_reset_clears_repaired_memory_before_later_query() -> None:
    memory = repair.EventAlignedDeltaMemory(4)
    write = [repair.WRITE, 1000, 2000, repair.SEP]
    state, _ = repair.apply_capability_chunk(write, _reps(write), memory, memory.empty_state())
    reset = [17, repair.RESET, repair.SEP, 18]
    state, events = repair.apply_capability_chunk(reset, _reps(reset), memory, state)
    assert "reset" in [e.kind for e in events]
    query = [repair.QUERY, 1000, repair.ANSWER]
    recall = repair.repaired_query_read(query, _reps(query), memory, state).recall
    assert recall is not None
    assert torch.equal(recall, torch.zeros_like(recall))


def test_frozen_fixture_layout_and_marker_contract() -> None:
    from tam_research import aera_memory_capability_gate_v1 as gate

    assert (gate.WRITE, gate.UPDATE, gate.QUERY, gate.ANSWER, gate.RESET, gate.SEP) == (
        repair.WRITE, repair.UPDATE, repair.QUERY, repair.ANSWER, repair.RESET, repair.SEP
    )
    for sample_index in range(32):
        case = gate.generate_case(
            split="eval",
            seed=gate.HELDOUT_DATA_SEED,
            sample_index=7_630_000 + sample_index,
            retention_distance_chunks=8,
            concurrent_facts=1,
            distractor_records=0,
            correction=bool(sample_index % 2),
            reset_before_query=False,
        )
        final = case.chunks[-1]
        qpos = final.index(gate.QUERY)
        assert 0 <= qpos <= 7
        assert final[qpos + 1] == case.target_key
        assert qpos + 1 >= 1
        assert final[qpos + 2] == gate.ANSWER


def test_frozen_scientific_files_remain_byte_identical() -> None:
    root = Path(__file__).resolve().parents[1]
    expected = {
        "tam_research/aera_issue748_memory_capability_seed1_harness.py": repair.FROZEN_WRAPPER_BLOB,
        "tam_research/aera_issue748_memory_capability_seed1_harness_base.py": repair.FROZEN_BASE_BLOB,
        "tam_research/aera_memory_capability_gate_v1.py": repair.FROZEN_FIXTURE_BLOB,
    }
    for relative, blob in expected.items():
        actual = subprocess.check_output(["git", "hash-object", str(root / relative)], text=True).strip()
        assert actual == blob


def test_protocol_authority_matches_module() -> None:
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads((root / "docs/aera_issue763_key_conditioned_memory_repair_protocol.json").read_text())
    assert protocol["research_issue"] == repair.RESEARCH_ISSUE
    assert protocol["source_main"] == repair.SOURCE_MAIN
    assert protocol["authority"]["gpu"] is False
    assert protocol["authority"]["scientific_seed"] is False
    assert protocol["authority"]["merge"] is False
