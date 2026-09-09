from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from tam_research import aera_issue785_eval_materialization_bridge_cpu as bridge
from tam_research import aera_issue748_memory_capability_seed1_harness as corrected
from tam_research import aera_issue776_event_memory_scientific_adapter as repaired
from tam_research import aera_memory_capability_gate_v1 as gate


ROOT = Path(__file__).resolve().parents[1]


def _blob(path: str) -> str:
    return subprocess.check_output(
        ["git", "hash-object", str(ROOT / path)],
        text=True,
    ).strip()


def test_issue785_frozen_lineage_blobs_are_exact() -> None:
    assert _blob("tam_research/aera_issue748_memory_capability_seed1_harness.py") == bridge.CORRECTED_HARNESS_BLOB
    assert _blob("tam_research/aera_issue776_event_memory_scientific_adapter.py") == bridge.REPAIRED_ADAPTER_BLOB
    assert _blob("tam_research/aera_issue748_memory_capability_seed1_harness_base.py") == bridge.FROZEN_BASE_BLOB
    assert _blob("tam_research/aera_memory_capability_gate_v1.py") == bridge.FROZEN_GATE_BLOB
    assert _blob("tam_research/aera_issue770_integrated_event_memory_cpu.py") == bridge.INTEGRATED_REPAIRED_MODEL_BLOB
    assert _blob("tam_research/aera_hardware_core_v18.py") == bridge.V18_CORE_BLOB
    assert _blob("tam_research/aera_delta_memory.py") == bridge.DELTA_MEMORY_BLOB


def test_issue785_bridge_binds_authoritative_corrected_materializer() -> None:
    assert bridge.evaluation_cases is corrected.evaluation_cases
    assert repaired.base.evaluation_cases is corrected.evaluation_cases


def test_issue785_eval_materialization_is_deterministic_balanced_unique_and_heldout() -> None:
    first = bridge.evaluation_cases()
    second = bridge.evaluation_cases()
    authoritative = corrected.evaluation_cases()

    assert first == second == authoritative
    assert len(first) == 432

    ids = [case.sample_index for case in first]
    assert len(set(ids)) == 432

    nonreset = [case for case in first if not case.reset_before_query]
    reset = [case for case in first if case.reset_before_query]
    assert len(nonreset) == 324
    assert len(reset) == 108

    for distance in (2, 8, 32):
        assert sum(case.retention_distance_chunks == distance for case in nonreset) == 108
    for distance in (8, 32):
        assert sum(case.retention_distance_chunks == distance for case in reset) == 54

    assert all(case.split == "eval" for case in first)
    assert all(gate.pair_split(case.target_key, case.original_value) == "eval" for case in first)
    for case in first:
        if case.correction:
            assert gate.pair_split(case.target_key, case.latest_value) == "eval"


def test_issue785_bridge_does_not_depend_on_raw_base_materializer(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden_raw_base_materializer():
        raise AssertionError("raw #748 base evaluation_cases executed")

    monkeypatch.setattr(repaired.base, "evaluation_cases", forbidden_raw_base_materializer)
    cases = bridge.evaluation_cases()
    assert len(cases) == 432


def test_issue785_scientific_seeds_remain_guarded_and_no_future_seed_authority() -> None:
    for seed in (17_641, 27_641, 37_641):
        with pytest.raises(PermissionError):
            bridge.build_repaired_model(
                "A_backbone",
                seed,
                scientific_seed_authorized=False,
                device="cpu",
            )
    assert bridge.CONSUMED_SCIENTIFIC_SEEDS == (17_641, 27_641)
    assert bridge.FUTURE_SCIENTIFIC_SEED_LOCKED == 37_641
    assert all(value is False for value in bridge.authority_snapshot().values())


def test_issue785_source_is_cpu_bridge_only() -> None:
    source = (ROOT / "tam_research/aera_issue785_eval_materialization_bridge_cpu.py").read_text()
    lowered = source.lower()
    assert "modal." not in lowered
    assert "gpu=" not in lowered
    assert ".cuda(" not in lowered
    assert "torch.cuda" not in lowered
    assert "base.evaluation_cases(" not in source
    assert "evaluation_cases = corrected_harness.evaluation_cases" in source


def test_issue785_protocol_json_matches_python_snapshot() -> None:
    expected = json.loads(
        (ROOT / "docs/aera_issue785_eval_materialization_bridge_protocol.json").read_text()
    )
    assert expected == bridge.bridge_protocol_snapshot()
