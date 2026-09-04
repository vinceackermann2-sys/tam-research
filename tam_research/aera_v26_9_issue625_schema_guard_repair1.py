from __future__ import annotations

"""Issue #625 schema-only successor for the consumed #624 pre-L4 failure.

The scientific row evaluator remains the frozen #622 implementation.  This
module changes only immutable #602 result identity validation from a nonexistent
top-level ``research_issue`` field to ``protocol.research_issue`` and restores
the frozen loader after the one call.
"""

from pathlib import Path
from typing import Any
import hashlib
import json

from . import aera_v26_9_issue622_corrected_autocast_dtype_gate as frozen622

RESEARCH_ISSUE = 625
SOURCE_MAIN = "7c4dd6ac038943e882035ff92a6336a764369c61"
SOURCE_TREE = "9f2977e3f4698593b4c030352aa1561e3b4ad33d"
ISSUE602_RESULT_PATH = frozen622.ISSUE602_RESULT_PATH
ISSUE602_RESULT_SHA256 = frozen622.ISSUE602_RESULT_SHA256
ISSUE622_PROBE_BLOB = "4e08ac9af18f666f09009e4d2c5822b11e91c2c1"
ISSUE622_LAUNCHER_BLOB = "e0fa3b0856b9750402209c6487f407b189672436"
V26_9_BACKEND_BLOB = "b81cc209f5d95abbe1fb8bd620c78e87c067bc19"
ISSUE602_PROBE_BLOB = "456203f515d67d1c92b0a9c3e0e59ce4137ac10a"
DESIGN_SEED = 891_475_817


def issue625_protocol() -> dict[str, Any]:
    base = dict(frozen622.issue622_protocol())
    base.update(
        {
            "probe_version": "aera-v26.9-issue625-schema-guard-repair1",
            "research_issue": RESEARCH_ISSUE,
            "source_main_issue625": SOURCE_MAIN,
            "source_tree_issue625": SOURCE_TREE,
            "issue622_probe_blob": ISSUE622_PROBE_BLOB,
            "issue622_launcher_blob": ISSUE622_LAUNCHER_BLOB,
            "schema_guard_repair_only": True,
            "issue602_identity_field": "protocol.research_issue",
            "issue624_consumed_pre_l4": True,
            "issue624_fresh_rows_executed": False,
            "gpu_authorized_by_probe_module": False,
            "end_to_end_systems_authorized": False,
            "architecture_freeze_authorized": False,
            "s2_authorized": False,
            "fresh_scientific_seed_authorized": False,
            "independent_replication_credit": False,
            "100m_authorized": False,
            "breakthrough_proven": False,
            "scientific_seed_consumed": False,
        }
    )
    return base


def cpu_contract_preflight_issue625() -> dict[str, Any]:
    frozen = frozen622.cpu_contract_preflight_issue622()
    if DESIGN_SEED != 891_475_817 or frozen622.DESIGN_SEED != DESIGN_SEED:
        raise RuntimeError("issue625 design-only seed drifted")
    if frozen622.BATCH_SIZES != (8, 64) or frozen622.VALIDITY_KINDS != ("mixed", "full"):
        raise RuntimeError("issue625 row grid drifted")
    if (frozen622.BF16_ATOL, frozen622.BF16_RTOL) != (1e-2, 1e-2):
        raise RuntimeError("issue625 tolerance drifted")
    protocol = issue625_protocol()
    false_keys = (
        "gpu_authorized_by_probe_module",
        "end_to_end_systems_authorized",
        "architecture_freeze_authorized",
        "s2_authorized",
        "fresh_scientific_seed_authorized",
        "independent_replication_credit",
        "100m_authorized",
        "breakthrough_proven",
        "scientific_seed_consumed",
    )
    if any(protocol[key] for key in false_keys):
        raise RuntimeError("issue625 CPU contract unexpectedly authorizes higher work")
    return {
        "frozen_issue622_contract": frozen,
        "protocol": protocol,
        "gpu_authorized_by_cpu_preflight": False,
        "synthetic_only": True,
        "model_constructed": False,
        "checkpoint_loaded": False,
        "corpus_accessed": False,
        "training_performed": False,
        "optimizer_created": False,
        "backward_performed": False,
        "scientific_seed_consumed": False,
    }


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_issue602_preserved_authority_issue625(
    path: str | Path = ISSUE602_RESULT_PATH,
) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise RuntimeError(f"issue625 missing immutable issue602 result: {source}")
    digest = _sha256_path(source)
    if digest != ISSUE602_RESULT_SHA256:
        raise RuntimeError(
            f"issue625 issue602 result SHA drift: got={digest} expected={ISSUE602_RESULT_SHA256}"
        )
    payload = json.loads(source.read_text())
    protocol = payload.get("protocol")
    if not isinstance(protocol, dict) or protocol.get("research_issue") != 602:
        raise RuntimeError("issue625 source result protocol is not issue602")
    if payload.get("decision") != "FAIL" or payload.get("overall_pass") is not False:
        raise RuntimeError("issue625 must preserve issue602 authoritative FAIL")
    gate_meta = payload.get("issue602_gate_metadata", {})
    if gate_meta.get("v26_9_backend_blob") != V26_9_BACKEND_BLOB:
        raise RuntimeError("issue625 issue602 backend authority drifted")

    preserved = payload.get("preserved_issue558")
    if not isinstance(preserved, dict):
        raise RuntimeError("issue625 missing preserved issue558 authority")
    historical = preserved.get("historical", {})
    mixed = preserved.get("mixed", {})
    required_historical = (
        "overall_pass",
        "correctness_pass",
        "known_empty_pass",
        "near_tie_pass",
        "row_latency_pass",
        "full_event_ratio_pass",
        "single_tail_kernel_pass",
        "candidate_no_reference_tail_ops_pass",
    )
    if preserved.get("decision") != "PASS" or preserved.get("overall_pass") is not True:
        raise RuntimeError("issue625 preserved issue558 overall authority is not PASS")
    if historical.get("decision") != "PASS":
        raise RuntimeError("issue625 preserved issue558 historical decision drifted")
    if any(historical.get(key) is not True for key in required_historical):
        raise RuntimeError("issue625 preserved issue558 historical gate drifted")
    if any(
        mixed.get(key) is not True
        for key in ("overall_pass", "rows_pass", "near_tie_pass", "known_empty_pass")
    ):
        raise RuntimeError("issue625 preserved issue558 mixed gate drifted")
    if mixed.get("timing_decision_bearing") is not False:
        raise RuntimeError("issue625 preserved mixed timing contract drifted")

    return {
        "source_result_sha256": digest,
        "issue602_decision": "FAIL",
        "issue602_overall_pass": False,
        "preserved_issue558": {
            "decision": preserved["decision"],
            "overall_pass": preserved["overall_pass"],
            "historical": {
                key: historical[key]
                for key in (
                    "decision",
                    *required_historical,
                    "geomean_latency_ratio_by_dtype",
                    "geomean_latency_pass_by_dtype",
                )
            },
            "mixed": {
                key: mixed[key]
                for key in (
                    "overall_pass",
                    "rows_pass",
                    "near_tie_pass",
                    "known_empty_pass",
                    "timing_decision_bearing",
                )
            },
        },
    }


def run_schema_guard_repair1_gate_v26_9_issue625(
    *,
    issue602_result_path: str | Path = ISSUE602_RESULT_PATH,
) -> dict[str, Any]:
    cpu_contract_preflight_issue625()
    original_loader = frozen622.load_issue602_preserved_authority
    try:
        frozen622.load_issue602_preserved_authority = load_issue602_preserved_authority_issue625
        result = frozen622.run_corrected_autocast_dtype_gate_v26_9_issue622(
            issue602_result_path=issue602_result_path
        )
    finally:
        frozen622.load_issue602_preserved_authority = original_loader

    result["protocol"] = issue625_protocol()
    result["issue625_schema_guard_metadata"] = {
        "source_main": SOURCE_MAIN,
        "source_tree": SOURCE_TREE,
        "issue622_probe_blob": ISSUE622_PROBE_BLOB,
        "issue622_launcher_blob": ISSUE622_LAUNCHER_BLOB,
        "v26_9_backend_blob": V26_9_BACKEND_BLOB,
        "issue602_probe_blob": ISSUE602_PROBE_BLOB,
        "issue602_result_sha256": ISSUE602_RESULT_SHA256,
        "identity_field": "protocol.research_issue",
        "schema_guard_repair_only": True,
        "scientific_seed_consumed": False,
    }
    return result