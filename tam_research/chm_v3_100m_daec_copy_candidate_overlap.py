from __future__ import annotations

"""#1337: CPU-only join of frozen DAEC Stage-C copies to GPT-2 candidate IDs.

No model loading, checkpoint inference, training, Modal, GPU, scientific
reruns, result rewriting, or authority to amend the original Stage-C STOP.
"""

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import math
from typing import Any

from .chm_v1_long_memory_eval import _single_token_values

ARCHIVED_RUN_ID = 37_760_374_218
ARCHIVED_JOB_ID = 113_255_052_797
ARCHIVED_SOURCE_SHA = "eb4f0e36ee6c4095e550f26cb7a52c20c5116aa9"
ARCHIVED_SCIENTIFIC_SEED = 2_013_161
ARCHIVED_STOP = "CHM_V3_100M_DAEC_STAGE_C_STOP"
RESULT_MARKER = "CHM_V3_100M_DAEC_STAGE_C_RESULT="
FROZEN_COPY_TOKEN_COUNTS = {317: 193, 20816: 107, 8125: 79, 383: 5}
# Sparse independent lookup from the public canonical GPT-2 encoder.json:
# https://github.com/SungjoonPark/OpenAI-GPT2/blob/master/gpt2/models/117M/encoder.json
# Vocabulary key is the GPT-2 byte-level leading-space marker (U+0120) + value.
# The frozen v3 generator takes the first eight such single-token _VALUES.
REFERENCE_GPT2_CANDIDATES = (
    ("amber", 36505), ("delta", 25979), ("granite", 41013),
    ("mosaic", 47076), ("north", 5093), ("pearl", 43836),
    ("quartz", 47969), ("river", 7850),
)
FAMILIES = ("rare_fact", "overwrite", "two_hop", "local_negative")


def _digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def extract_archived_result(log_text: str) -> dict[str, Any]:
    matches = [
        line.partition(RESULT_MARKER)[2].strip()
        for line in log_text.splitlines()
        if RESULT_MARKER in line and line.partition(RESULT_MARKER)[2].lstrip().startswith("{")
    ]
    if len(matches) != 1:
        raise ValueError(f"#1337 requires exactly one archived RESULT marker, got {len(matches)}")
    result = json.loads(matches[0])
    if not isinstance(result, dict):
        raise ValueError("archived RESULT must be a JSON object")
    return result


def frozen_candidate_ids(encode: Callable[[str], Sequence[int]]) -> tuple[tuple[str, int], ...]:
    """Reuse the frozen v3 generator's eight one-token candidate choices."""
    pairs = tuple(_single_token_values(encode)[:8])
    if len(pairs) != 8 or len({token for _, token in pairs}) != 8:
        raise ValueError("#1337 invalid eight-way generator candidate set")
    return pairs


def eight_way_candidate_nll(
    candidate_probabilities: Sequence[float],
    *,
    correct_index: int,
    gate: float,
    copied_candidate_index: int | None,
) -> float:
    """Mathematical check only, using synthetic candidate probabilities.

    If copied ID is outside candidates, (1-g) cancels exactly.
    """
    values = [float(x) for x in candidate_probabilities]
    if len(values) != 8 or any(not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError("eight positive candidate probabilities required")
    if not (0 <= correct_index < 8 and math.isfinite(gate) and 0 <= gate < 1):
        raise ValueError("invalid answer index or gate")
    if copied_candidate_index is not None and not 0 <= copied_candidate_index < 8:
        raise ValueError("invalid copied candidate index")
    mixed = [(1.0 - gate) * value for value in values]
    if copied_candidate_index is not None:
        mixed[copied_candidate_index] += gate
    return -math.log(mixed[correct_index] / sum(mixed))


def analyze_archive(
    result: Mapping[str, Any], encode: Callable[[str], Sequence[int]],
) -> dict[str, Any]:
    if (result.get("source_sha") != ARCHIVED_SOURCE_SHA or
        result.get("scientific_seed") != ARCHIVED_SCIENTIFIC_SEED or
        result.get("scientific_seed_consumed") is not True or
        result.get("classification") != ARCHIVED_STOP or
        result.get("passed") is not False or
        result.get("status") != "COMPLETE"):
        raise ValueError("wrong or untrustworthy frozen scientific RESULT")
    integrity = result.get("integrity")
    if not isinstance(integrity, Mapping):
        raise ValueError("missing integrity proof")
    if "hard_flat_evaluation_parity_verified" in integrity:
        raise ValueError("archived parity proof absence unexpectedly changed")
    if (integrity.get("probe_count") != 512 or
        integrity.get("training_tokens_per_model") != 33_554_432 or
        integrity.get("optimizer_steps_per_model") != 2048):
        raise ValueError("archived protocol envelope drift")

    rows = result.get("probe_rows")
    if not isinstance(rows, list) or len(rows) != 512:
        raise ValueError("expected 512 archived scored probe rows")
    by_family = Counter(row["family"] for row in rows)
    if by_family != Counter(dict.fromkeys(FAMILIES, 128)):
        raise ValueError("family counts do not match frozen aligned-v4")
    if len({(row["family"], row["case_id"]) for row in rows}) != 512:
        raise ValueError("duplicate archived probe IDs")
    copies = [row for row in rows if row["daec_trace"]["memory_count"] > 0]
    no_memory = [row for row in rows if row["daec_trace"]["memory_count"] == 0]
    if len(copies) != 384 or len(no_memory) != 128:
        raise ValueError("expected 384 memory probes and 128 local controls")
    if any(row["family"] == "local_negative" for row in copies):
        raise ValueError("a local-negative probe has historical memory")
    if any(row["daec_copied_token_id"] is not None for row in no_memory):
        raise ValueError("no-memory probes must not copy")
    counts = Counter(int(row["daec_copied_token_id"]) for row in copies)
    if dict(counts) != FROZEN_COPY_TOKEN_COUNTS:
        raise ValueError(f"archived hard-copy frequencies drifted: {dict(counts)}")
    if any(row["daec_copy_matches_answer"] for row in copies):
        raise ValueError("historical copy targets unexpectedly matched")
    for row in copies:
        if int(row["daec_trace"]["copied_token_id"]) != int(row["daec_copied_token_id"]):
            raise ValueError("trace/row copied ID mismatch")

    candidate_pairs = frozen_candidate_ids(encode)
    if candidate_pairs != REFERENCE_GPT2_CANDIDATES:
        raise ValueError("frozen GPT-2 candidate identity disagrees with public encoder.json")
    candidate_set = {token for _, token in candidate_pairs}
    family_reports = {}
    for family in FAMILIES:
        family_copies = [row for row in copies if row["family"] == family]
        overlaps = [row for row in family_copies if int(row["daec_copied_token_id"]) in candidate_set]
        family_reports[family] = {
            "memory_probe_count": len(family_copies),
            "copied_candidate_count": len(overlaps),
            "copied_outside_candidate_count": len(family_copies) - len(overlaps),
            "wrong_candidate_copies": len(overlaps),
        }
    inside = sum(x["copied_candidate_count"] for x in family_reports.values())
    report = {
        "classification": "CHM_V3_1337_CPU_ONLY_CANDIDATE_COPY_FORENSICS",
        "scientific_classification_unchanged": ARCHIVED_STOP,
        "scientific_seed_consumed": True,
        "scientific_result_reclassified": False,
        "new_scientific_attempt": False,
        "gpu_used": False,
        "archived_run_id": ARCHIVED_RUN_ID,
        "archived_job_id": ARCHIVED_JOB_ID,
        "archived_result_sha256": _digest(result),
        "archived_probe_rows_sha256": _digest(rows),
        "gpt2_candidates": [{"word":word, "token_id":token} for word,token in candidate_pairs],
        "archived_copied_token_counts": [
            {"token_id":token, "count":count} for token,count in sorted(counts.items())
        ],
        "memory_probe_count": len(copies),
        "copy_hits_on_authoritative_answers": 0,
        "copies_inside_eight_candidates": inside,
        "copies_outside_eight_candidates": len(copies) - inside,
        "per_family": family_reports,
        "interpretation": (
            "Outside-candidate copies rescale all eight candidates equally, hence cancel "
            "from the benchmark's eight-way normalization; effects on learned model "
            "weights, exact checkpoint parity, and all causal mechanisms remain unproven."
        ),
        "historical_parity_proof_status": "missing",
        "science_pass_or_retraining_authorized": False,
    }
    report["report_sha256"] = _digest(report)
    return report


__all__ = (
    "ARCHIVED_RUN_ID", "ARCHIVED_JOB_ID", "ARCHIVED_SCIENTIFIC_SEED",
    "extract_archived_result", "frozen_candidate_ids", "analyze_archive",
    "eight_way_candidate_nll",
    "REFERENCE_GPT2_CANDIDATES",
)
