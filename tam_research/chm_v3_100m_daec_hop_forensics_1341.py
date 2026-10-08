from __future__ import annotations

"""#1341: read-only first/second-hop forensic reconstruction of frozen Stage-C.

Joins two subsets of ONE immutable scientific RESULT: the 512-row copy fixture
from #1337 and the 384-row hop-position/gate fixture from #1341. No model
loading, trained-checkpoint replay, GPU, Modal, scientific attempt, or result
reclassification. All values are observations, not explanations of causality.
"""

from collections import Counter
from collections.abc import Mapping
import math
from typing import Any

ORIGINAL_RUN_ID = 37_760_374_218
ORIGINAL_JOB_ID = 113_255_052_797
ORIGINAL_COMMENT_ID = 6_058_454_735
ORIGINAL_SOURCE_SHA = "eb4f0e36ee6c4095e550f26cb7a52c20c5116aa9"
CONSUMED_SCIENTIFIC_SEED = 2_013_161
ORIGINAL_CLASSIFICATION = "CHM_V3_100M_DAEC_STAGE_C_STOP"
FAMILIES = ("rare_fact", "overwrite", "two_hop", "local_negative")
MEMORY_FAMILIES = FAMILIES[:3]
EXPECTED_MEMORY_TOKENS = {"rare_fact": 512, "overwrite": 512, "two_hop": 1024}


def _verify_envelope(fixture: Mapping[str, Any], *, hop_fixture: bool) -> None:
    expected = {
        "archived_run_id": ORIGINAL_RUN_ID,
        "archived_job_id": ORIGINAL_JOB_ID,
        "terminal_comment_id": ORIGINAL_COMMENT_ID,
        "source_sha": ORIGINAL_SOURCE_SHA,
        "scientific_seed": CONSUMED_SCIENTIFIC_SEED,
        "scientific_seed_consumed": True,
        "classification": ORIGINAL_CLASSIFICATION,
        "passed": False,
        "status": "COMPLETE",
    }
    for field, value in expected.items():
        if type(fixture.get(field)) is not type(value) or fixture.get(field) != value:
            raise ValueError(f"archived {field} provenance mismatch")
    if hop_fixture:
        if fixture.get("evidence_kind") != "READ_ONLY_VERBATIM_HOP_FIELDS_OF_AUTHORITATIVE_STAGE_C_RESULT":
            raise ValueError("hop fixture evidence-kind mismatch")
        if fixture.get("original_probe_count") != 512:
            raise ValueError("original probe count drifted")
        if fixture.get("original_integrity_parity_flag_present") is not False:
            raise ValueError("cannot add historical parity proof")
    else:
        if fixture.get("evidence_kind") != "READ_ONLY_SUBSET_OF_AUTHORITATIVE_STAGE_C_RESULT_JSON":
            raise ValueError("copy fixture evidence-kind mismatch")
        integrity = fixture.get("integrity")
        if not isinstance(integrity, Mapping) or integrity.get("probe_count") != 512:
            raise ValueError("original integrity/probe envelope missing")
        if "hard_flat_evaluation_parity_verified" in integrity:
            raise ValueError("original historical parity flag was missing")


def _key(row: Mapping[str, Any]) -> tuple[str, int]:
    family, case_id = row.get("family"), row.get("case_id")
    if family not in FAMILIES or type(case_id) is not int:
        raise ValueError("bad frozen family/case ID")
    return family, case_id


def analyze_archived_hops(
    copy_fixture: Mapping[str, Any],
    hop_fixture: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail-closed archive join; report positional behavior without causal claims."""
    _verify_envelope(copy_fixture, hop_fixture=False)
    _verify_envelope(hop_fixture, hop_fixture=True)
    copy_rows = copy_fixture.get("probe_rows")
    trace_rows = hop_fixture.get("trace_rows")
    if not isinstance(copy_rows, list) or len(copy_rows) != 512:
        raise ValueError("need exactly 512 archived copy rows")
    if not isinstance(trace_rows, list) or len(trace_rows) != 384:
        raise ValueError("need exactly 384 original address traces")

    copy_by_key: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in copy_rows:
        if not isinstance(row, Mapping):
            raise ValueError("invalid copy row")
        key = _key(row)
        if key in copy_by_key:
            raise ValueError("duplicate archived copy key")
        copy_by_key[key] = row
    for family in FAMILIES:
        if sum(k[0] == family for k in copy_by_key) != 128:
            raise ValueError("copy family must have exactly 128 probes")

    expected_keys: set[tuple[str, int]] = set()
    for key, row in copy_by_key.items():
        trace = row.get("daec_trace")
        if not isinstance(trace, Mapping):
            raise ValueError("missing archived copy trace")
        n = trace.get("memory_count")
        if key[0] in MEMORY_FAMILIES:
            if type(n) is not int or n != EXPECTED_MEMORY_TOKENS[key[0]]:
                raise ValueError("wrong family memory count")
            if row.get("daec_copy_matches_answer") is not False:
                raise ValueError("historical zero answer hits changed")
            token = trace.get("copied_token_id")
            if type(token) is not int or row.get("daec_copied_token_id") != token:
                raise ValueError("inconsistent archived copy token")
            expected_keys.add(key)
        elif n != 0 or trace.get("copied_token_id") is not None or row.get("daec_copied_token_id") is not None:
            raise ValueError("local negative was not memory-bearing")

    matched: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in trace_rows:
        if not isinstance(row, Mapping):
            raise ValueError("invalid hop row")
        key = _key(row)
        if key in matched:
            raise ValueError("duplicate archived hop key")
        if key not in expected_keys:
            raise ValueError("hop trace not part of archived memory probe set")
        old = copy_by_key[key]
        old_trace = old["daec_trace"]
        n = row.get("memory_count")
        if type(n) is not int or n != old_trace["memory_count"]:
            raise ValueError("hop trace memory count differs from archive")
        for field in ("first_hop_index", "second_hop_index"):
            value = row.get(field)
            if type(value) is not int or not (0 <= value < n):
                raise ValueError(f"{field} out of frozen memory bounds")
        copied = row.get("copied_token_id")
        if type(copied) is not int or copied != old["daec_copied_token_id"]:
            raise ValueError("hop/copy archive token mismatch")
        gate = row.get("gate")
        if type(gate) not in (float, int) or not math.isfinite(gate) or not 0 < gate < 1:
            raise ValueError("invalid frozen gate value")
        matched[key] = row

    if matched.keys() != expected_keys:
        raise ValueError("incomplete hop/copy archive join")

    def stats(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
        total = len(rows)
        same = sum(r["first_hop_index"] == r["second_hop_index"] for r in rows)
        tail = sum(r["second_hop_index"] >= r["memory_count"] - 32 for r in rows)
        head = sum(r["second_hop_index"] < 32 for r in rows)
        second_counts = Counter(r["second_hop_index"] for r in rows)
        copies = Counter(r["copied_token_id"] for r in rows)
        gates = [float(r["gate"]) for r in rows]
        return {
            "n": total,
            "same_first_second_hop": same,
            "different_first_second_hop": total - same,
            "same_hop_fraction": same / total,
            "second_hop_first_32": head,
            "second_hop_last_32": tail,
            "distinct_second_hop_indices": len(second_counts),
            "top_second_hop_indices": [
                {"index": i, "count": count}
                for i, count in sorted(second_counts.items(), key=lambda x: (-x[1], x[0]))[:10]
            ],
            "copied_token_counts": dict(sorted(copies.items())),
            "mean_copy_gate": math.fsum(gates) / total,
            "min_copy_gate": min(gates),
            "max_copy_gate": max(gates),
        }

    memory_rows = [matched[k] for k in sorted(matched)]
    return {
        "classification": "CHM_V3_1341_ARCHIVED_HOP_POSITION_FORENSICS_READ_ONLY",
        "original_scientific_classification": ORIGINAL_CLASSIFICATION,
        "original_seed_consumed": True,
        "historical_hard_flat_parity_proof": "missing",
        "scientific_result_reclassified": False,
        "gpu_or_checkpoint_inference_performed": False,
        "new_scientific_attempt_authorized": False,
        "copy_fixture_rows": len(copy_rows),
        "joined_memory_probe_rows": len(memory_rows),
        "overall": stats(memory_rows),
        "per_family": {
            family: stats([r for r in memory_rows if r["family"] == family])
            for family in MEMORY_FAMILIES
        },
        "local_negative_memory_rows": 0,
        "interpretation_ceiling": (
            "Observed address reuse and position concentration, not proof that "
            "the learned second-query update was inactive or proof of why the "
            "model selected boilerplate. Stage-C STOP and parity absence persist."
        ),
    }


__all__ = (
    "ORIGINAL_RUN_ID", "ORIGINAL_JOB_ID", "ORIGINAL_SOURCE_SHA",
    "CONSUMED_SCIENTIFIC_SEED", "ORIGINAL_CLASSIFICATION", "analyze_archived_hops",
)
