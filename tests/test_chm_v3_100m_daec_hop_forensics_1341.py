from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path

import pytest

from tam_research.chm_v3_100m_daec_hop_forensics_1341 import analyze_archived_hops

FIXTURES = Path(__file__).parent / "fixtures"


def archives() -> tuple[dict, dict]:
    old = json.loads(
        (FIXTURES / "chm_v3_1337_archived_copy_rows.json").read_text(encoding="utf-8")
    )
    hop = json.loads(
        (FIXTURES / "chm_v3_1341_archived_hop_rows.json").read_text(encoding="utf-8")
    )
    return old, hop


def test_authoritative_archived_first_and_second_hop_address_counts() -> None:
    old, hop = archives()
    original = json.dumps((old, hop), sort_keys=True)
    result = analyze_archived_hops(old, hop)
    assert json.dumps((old, hop), sort_keys=True) == original
    assert result["copy_fixture_rows"] == 512
    assert result["joined_memory_probe_rows"] == 384
    assert result["local_negative_memory_rows"] == 0
    assert result["overall"]["same_first_second_hop"] == 268
    assert result["overall"]["different_first_second_hop"] == 116
    assert result["overall"]["same_hop_fraction"] == pytest.approx(268 / 384)
    assert result["overall"]["second_hop_first_32"] == 171
    assert result["overall"]["second_hop_last_32"] == 75
    assert result["overall"]["mean_copy_gate"] == pytest.approx(
        0.28666941324869794, abs=1e-12
    )
    assert result["overall"]["copied_token_counts"] == {
        317: 193, 383: 5, 8125: 79, 20816: 107
    }
    assert result["overall"]["top_second_hop_indices"][:3] == [
        {"index": 14, "count": 70},
        {"index": 26, "count": 48},
        {"index": 28, "count": 37},
    ]
    expected = {
        "rare_fact": (70, 92, 15),
        "overwrite": (105, 79, 30),
        "two_hop": (93, 0, 30),
    }
    for family, (same, head32, tail32) in expected.items():
        item = result["per_family"][family]
        assert item["n"] == 128
        assert item["same_first_second_hop"] == same
        assert item["same_hop_fraction"] == pytest.approx(same / 128)
        assert item["second_hop_first_32"] == head32
        assert item["second_hop_last_32"] == tail32
    assert result["original_scientific_classification"] == "CHM_V3_100M_DAEC_STAGE_C_STOP"
    assert result["original_seed_consumed"] is True
    assert result["scientific_result_reclassified"] is False
    assert result["historical_hard_flat_parity_proof"] == "missing"
    assert result["gpu_or_checkpoint_inference_performed"] is False
    assert result["new_scientific_attempt_authorized"] is False


@pytest.mark.parametrize("field,value", [
    ("source_sha", "0" * 40),
    ("scientific_seed", 2013162),
    ("scientific_seed_consumed", False),
    ("passed", True),
    ("classification", "CHM_V3_POSITIVE"),
    ("archived_run_id", 37760374219),
    ("archived_job_id", 113255052798),
])
def test_provenance_fails_closed(field: str, value: object) -> None:
    old, hop = archives()
    old[field] = value
    with pytest.raises(ValueError, match="provenance"):
        analyze_archived_hops(old, hop)
    old, hop = archives()
    hop[field] = value
    with pytest.raises(ValueError, match="provenance"):
        analyze_archived_hops(old, hop)


def test_missing_parity_evidence_cannot_be_retroactively_added() -> None:
    old, hop = archives()
    old["integrity"]["hard_flat_evaluation_parity_verified"] = True
    with pytest.raises(ValueError, match="parity"):
        analyze_archived_hops(old, hop)
    old, hop = archives()
    hop["original_integrity_parity_flag_present"] = True
    with pytest.raises(ValueError, match="parity"):
        analyze_archived_hops(old, hop)


def test_archive_join_fails_closed_on_duplicates_missing_or_misbound_rows() -> None:
    old, hop = archives()
    hop["trace_rows"][0] = copy.deepcopy(hop["trace_rows"][1])
    with pytest.raises(ValueError, match="duplicate"):
        analyze_archived_hops(old, hop)

    old, hop = archives()
    hop["trace_rows"].pop()
    with pytest.raises(ValueError, match="384"):
        analyze_archived_hops(old, hop)

    old, hop = archives()
    hop["trace_rows"][0]["case_id"] = 300000
    with pytest.raises(ValueError, match="not part"):
        analyze_archived_hops(old, hop)

    old, hop = archives()
    hop["trace_rows"][0]["copied_token_id"] = 99
    with pytest.raises(ValueError, match="token mismatch"):
        analyze_archived_hops(old, hop)

    old, hop = archives()
    old["probe_rows"][0]["daec_copy_matches_answer"] = True
    with pytest.raises(ValueError, match="zero answer hits"):
        analyze_archived_hops(old, hop)


@pytest.mark.parametrize("field,value,error", [
    ("first_hop_index", -1, "bounds"),
    ("second_hop_index", 512, "bounds"),
    ("memory_count", 0, "memory count"),
    ("first_hop_index", None, "bounds"),
    ("gate", 0.0, "gate"),
    ("gate", 1.0, "gate"),
    ("gate", float("nan"), "gate"),
])
def test_hop_geometry_and_gate_integrity(field: str, value: object, error: str) -> None:
    old, hop = archives()
    hop["trace_rows"][0][field] = value
    with pytest.raises(ValueError, match=error):
        analyze_archived_hops(old, hop)


def test_distinct_hop_synthetic_counterexample_changes_only_observed_metric() -> None:
    old, hop = archives()
    source = next(r for r in hop["trace_rows"]
                  if r["first_hop_index"] == r["second_hop_index"])
    old_position = source["second_hop_index"]
    source["second_hop_index"] = (old_position + 1) % source["memory_count"]
    result = analyze_archived_hops(old, hop)
    assert result["overall"]["same_first_second_hop"] == 267
    assert result["overall"]["different_first_second_hop"] == 117
    assert result["scientific_result_reclassified"] is False


def test_forensic_code_has_no_training_or_modal_path() -> None:
    import tam_research.chm_v3_100m_daec_hop_forensics_1341 as helper
    src = inspect.getsource(helper)
    for forbidden in ("modal.App(", "modal.run(", "torch.cuda", "optimizer.step(",
                      "run_scientific(", "checkpoint.load(", "torch.load("):
        assert forbidden not in src
