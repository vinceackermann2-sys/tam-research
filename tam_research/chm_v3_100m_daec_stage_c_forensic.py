from __future__ import annotations

"""Read-only CHM-v3 #1323 Stage-C forensic checks (#1332).

This helper NEVER modifies archived scientific outputs, certifies missing parity
proof, authorizes GPU, or changes the frozen scientific classifier.
"""

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

FROZEN_SOURCE_SHA = "eb4f0e36ee6c4095e550f26cb7a52c20c5116aa9"
FROZEN_SCIENTIFIC_SEED = 2_013_161
FROZEN_TERMINAL_CLASSIFICATION = "CHM_V3_100M_DAEC_STAGE_C_STOP"
FROZEN_RESULT_COMMENT_ID = 6_058_454_735
PROBE_FAMILIES = ("rare_fact", "overwrite", "two_hop", "local_negative")
PROBES_PER_FAMILY = 128
HARD_FLAT_PARITY_FLAG = "hard_flat_evaluation_parity_verified"

# Future evidence schema. It cannot be reconstructed by assigning an integrity
# flag to an old result or by proving a tiny synthetic scorer is correct.
REQUIRED_PARITY_PROOF_KEYS = (
    "source_sha",
    "trained_checkpoint_sha256",
    "scored_probe_rows_sha256",
    "independent_reference_impl_sha256",
    "probes_compared",
    "max_probability_absolute_error",
    "first_hop_index_mismatches",
    "second_hop_index_mismatches",
)


def parity_proof_status(
    integrity: Mapping[str, Any],
    evidence: Mapping[str, Any] | None = None,
) -> str:
    """Distinguish missing, asserted, and independently evidenced parity.

    An asserted bool without bound reference evidence is NOT a parity proof.
    The original #1323 result did not emit the flag at all.
    """
    if HARD_FLAT_PARITY_FLAG not in integrity:
        return "missing"
    if integrity[HARD_FLAT_PARITY_FLAG] is not True:
        return "not_verified"
    if evidence is None or any(key not in evidence for key in REQUIRED_PARITY_PROOF_KEYS):
        return "asserted_without_evidence"
    if evidence["source_sha"] != FROZEN_SOURCE_SHA:
        return "source_mismatch"
    if int(evidence["probes_compared"]) != 512:
        return "probe_count_mismatch"
    for key in ("trained_checkpoint_sha256", "scored_probe_rows_sha256",
                "independent_reference_impl_sha256"):
        digest = str(evidence[key]).lower()
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            return "invalid_evidence_hash"
    try:
        error = float(evidence["max_probability_absolute_error"])
        mismatch_a = int(evidence["first_hop_index_mismatches"])
        mismatch_b = int(evidence["second_hop_index_mismatches"])
    except (ValueError, TypeError, OverflowError):
        return "invalid_evidence_values"
    if not (0.0 <= error <= 1e-5 and mismatch_a == 0 and mismatch_b == 0):
        return "parity_mismatch"
    return "independent_evidence_present_not_checkpoint_authenticated"


def audit_archived_stage_c_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Produce only a descriptive, immutable-result audit; never a PASS rewrite."""
    if result.get("source_sha") != FROZEN_SOURCE_SHA:
        raise ValueError("wrong frozen Stage-C source")
    if result.get("scientific_seed") != FROZEN_SCIENTIFIC_SEED:
        raise ValueError("wrong consumed scientific seed")
    if result.get("classification") != FROZEN_TERMINAL_CLASSIFICATION:
        raise ValueError("frozen Stage-C STOP cannot be reclassified")
    if result.get("passed") is not False:
        raise ValueError("original scientific STOP must remain false")
    if result.get("scientific_seed_consumed") is not True:
        raise ValueError("consumed scientific seed must remain consumed")

    integrity = result.get("integrity")
    if not isinstance(integrity, Mapping):
        raise ValueError("missing frozen integrity mapping")
    rows = result.get("probe_rows")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError("missing frozen scored probe rows")
    if len(rows) != 512:
        raise ValueError("frozen aligned-v4 row count must be 512")

    counts = Counter(str(row["family"]) for row in rows)
    if counts != Counter({family: PROBES_PER_FAMILY for family in PROBE_FAMILIES}):
        raise ValueError("frozen aligned-v4 family counts are inconsistent")
    families: dict[str, dict[str, Any]] = {}
    for family in PROBE_FAMILIES:
        family_rows = [row for row in rows if row["family"] == family]
        hits = [
            row for row in family_rows if row.get("daec_copied_token_id") is not None
        ]
        families[family] = {
            "count": len(family_rows),
            "top1_correct": {
                k: sum(bool(row[f"{k}_correct"]) for row in family_rows)
                for k in ("local", "raw", "daec")
            },
            "copy_targets_hit": sum(
                bool(row.get("daec_copy_matches_answer")) for row in hits
            ),
            "memory_probe_count": len(hits),
        }

    parity = parity_proof_status(integrity)
    return {
        "classification": "CHM_V3_1323_FORENSIC_READ_ONLY_REPORT",
        "original_classification_unchanged": FROZEN_TERMINAL_CLASSIFICATION,
        "original_passed_unchanged": False,
        "scientific_seed": FROZEN_SCIENTIFIC_SEED,
        "scientific_seed_consumed": True,
        "parity_proof_status": parity,
        "can_certify_scientific_pass": False,
        "scientific_result_reinterpreted": False,
        "family_rows": families,
        "ordinary_language_nll": {
            k: float(result["ordinary_language"][name]["nll"])
            for k, name in (("local", "local"), ("raw", "raw_eiem"), ("daec", "daec_eiem"))
        },
        "long_range_nll_benefit_vs_local": float(
            result["metrics"]["aggregate_long_range"]["daec_vs_local"]["candidate_nll_benefit"]
        ),
        "long_range_nll_benefit_vs_raw": float(
            result["metrics"]["aggregate_long_range"]["daec_vs_raw"]["candidate_nll_benefit"]
        ),
        "no_training_or_gpu_authority": True,
    }


__all__ = [
    "FROZEN_RESULT_COMMENT_ID",
    "FROZEN_SCIENTIFIC_SEED",
    "FROZEN_SOURCE_SHA",
    "FROZEN_TERMINAL_CLASSIFICATION",
    "audit_archived_stage_c_result",
    "parity_proof_status",
]
