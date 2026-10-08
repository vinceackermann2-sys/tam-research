from __future__ import annotations

"""#1340 strictly CPU-only frozen prompt-position and answer-oracle forensics.

Regenerates the preregistered GPT-2 aligned-v4 prompts, aligns immutable
scored copy traces, and audits which memory token positions were selected.
It does NOT load a trained model/checkpoint or run any inference/training.
"""

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
from typing import Any

from .chm_v1_100m_stage_c_eval import (
    CASES_PER_FAMILY, GENERATOR_VERSION, PROBE_SEED,
    generate_aligned_probe_suite,
)
from .chm_v3_100m_daec_copy_candidate_overlap import (
    ARCHIVED_SCIENTIFIC_SEED, ARCHIVED_SOURCE_SHA,
    FROZEN_COPY_TOKEN_COUNTS, REFERENCE_GPT2_CANDIDATES,
)

FAMILY_ORDER = ("rare_fact", "overwrite", "two_hop", "local_negative")
ARCHIVED_RUN_ID = 37_760_374_218
ARCHIVED_JOB_ID = 113_255_052_797
FROZEN_CLASSIFICATION = "CHM_V3_100M_DAEC_STAGE_C_STOP"
HISTORY_CLASSIFICATION = "CHM_V3_1340_CPU_ONLY_POSITION_ORACLE_FORENSICS"
LOCAL_WINDOW = 512
ENDPOINT_NEIGHBORHOOD = 16


def sha256_json(obj: Any) -> str:
    raw = json.dumps(obj, separators=(",", ":"), sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(raw.encode("ascii")).hexdigest()


def _top_counts(values: Sequence[int], limit: int = 8) -> list[dict[str, int]]:
    counts = Counter(values)
    return [
        {"position": int(pos), "count": count}
        for pos, count in sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:limit]
    ]


def analyze_position_oracle(
    archive: Mapping[str, Any],
    encode: Callable[[str], Sequence[int]],
    *,
    decode: Callable[[Sequence[int]], str] | None = None,
) -> dict[str, Any]:
    """Cross-check the old copy trace against freshly regenerated frozen prompts.

    Any mismatch is a forensic BLOCKER. Never interpret this as historical
    trained-checkpoint hard-flat evaluator parity proof.
    """
    if archive.get("source_kind") != "READ_ONLY_EXTRACT_OF_ARCHIVED_RESULT_FROM_COMPLETED_RUN":
        raise ValueError("archive kind drift")
    if archive.get("archived_actions_run_id") != ARCHIVED_RUN_ID or archive.get(
        "archived_actions_job_id"
    ) != ARCHIVED_JOB_ID:
        raise ValueError("wrong frozen run/job")
    if archive.get("source_sha") != ARCHIVED_SOURCE_SHA:
        raise ValueError("original scientific source mismatch")
    if archive.get("scientific_seed") != ARCHIVED_SCIENTIFIC_SEED or archive.get(
        "scientific_seed_consumed"
    ) is not True:
        raise ValueError("original scientific seed not consumed or identity mismatch")
    if archive.get("scientific_classification") != FROZEN_CLASSIFICATION or archive.get(
        "scientific_passed"
    ) is not False or archive.get("original_status") != "COMPLETE":
        raise ValueError("archived scientific STOP cannot be reclassified")
    if archive.get("generator_version") != GENERATOR_VERSION or archive.get(
        "probe_seed"
    ) != PROBE_SEED:
        raise ValueError("frozen aligned-v4 generator seed/version drift")
    counts = archive.get("counts")
    if not isinstance(counts, Mapping) or counts.get("probes") != 512 or counts.get(
        "optimizer_steps_per_model"
    ) != 2048 or counts.get("training_tokens_per_model") != 33_554_432:
        raise ValueError("training/probe integrity geometry mismatch")
    rows = archive.get("rows")
    if not isinstance(rows, list) or len(rows) != 4 * CASES_PER_FAMILY:
        raise ValueError("requires 512 original archived probe traces")

    probes = generate_aligned_probe_suite(encode, seed=PROBE_SEED, cases_per_family=CASES_PER_FAMILY)
    if len(probes) != len(rows):
        raise ValueError("probe count mismatch on deterministic reconstruction")
    candidates = tuple(int(x) for x in probes[0].candidate_token_ids)
    if candidates != tuple(token for _, token in REFERENCE_GPT2_CANDIDATES):
        raise ValueError("wrong GPT-2 tokenizer identity or candidate set")
    if any(tuple(p.candidate_token_ids) != candidates for p in probes):
        raise ValueError("candidate list changed within frozen suite")

    families: dict[str, dict[str, Any]] = {}
    selected_first: list[int] = []
    selected_second: list[int] = []
    selected_tokens: list[int] = []
    answers_in_memory = 0
    same_index = 0
    total_memory = 0
    case_digest: list[dict[str, Any]] = []

    for family in FAMILY_ORDER:
        families[family] = {
            "probes": 0, "memory_probes": 0, "oracle_answer_present": 0,
            "oracle_answer_absent": 0, "first_hop_on_answer_token": 0,
            "second_hop_on_answer_token": 0,
            "first_hop_near_authoritative_evidence_end_16": 0,
            "second_hop_near_authoritative_evidence_end_16": 0,
            "second_hop_near_first_relation_end_16": 0,
            "same_first_second_index": 0,
            "second_hop_mean_min_token_distance_to_target": None,
            "most_selected_second_hop_positions": [],
        }
    distance_lists: dict[str, list[int]] = {family: [] for family in FAMILY_ORDER}
    second_positions: dict[str, list[int]] = {family: [] for family in FAMILY_ORDER}

    for n, (probe, row) in enumerate(zip(probes, rows)):
        family = FAMILY_ORDER[n // CASES_PER_FAMILY]
        case_id = (n // CASES_PER_FAMILY) * 100_000 + (n % CASES_PER_FAMILY)
        if probe.family != family or int(probe.case_id) != case_id:
            raise ValueError(f"regenerated suite order/case drift at {n}")
        if row.get("family") != family or row.get("case_id") != case_id:
            raise ValueError(f"archived trace order/case drift at {n}")
        if int(row["evidence_distance"]) != int(probe.evidence_distance):
            raise ValueError(f"archived evidence distance mismatch at case {case_id}")
        if probe.query_token != len(probe.prompt_ids) - 1:
            raise ValueError("query must be the final prompt token")
        memory_len = (probe.query_token // LOCAL_WINDOW) * LOCAL_WINDOW
        if int(row["memory_count"]) != memory_len:
            raise ValueError(f"memory-window geometry drift at case {case_id}")
        if memory_len not in (0, 512, 1024):
            raise ValueError("unexpected memory slice size")

        summary = families[family]
        summary["probes"] += 1
        if memory_len == 0:
            if any(row.get(x) is not None for x in (
                "first_hop_index", "second_hop_index", "copied_token_id"
            )) or row["copied_matches_answer"]:
                raise ValueError("local-negative control fabricated memory")
            continue

        summary["memory_probes"] += 1
        total_memory += 1
        ids = probe.prompt_ids[:memory_len]
        answer_id = int(probe.answer_token_id)
        answer_positions = [i for i, token in enumerate(ids) if int(token) == answer_id]
        if answer_positions:
            summary["oracle_answer_present"] += 1
            answers_in_memory += 1
        else:
            summary["oracle_answer_absent"] += 1

        first, second = row["first_hop_index"], row["second_hop_index"]
        if (not isinstance(first, int) or isinstance(first, bool) or
            not isinstance(second, int) or isinstance(second, bool) or
            not (0 <= first < memory_len and 0 <= second < memory_len)):
            raise ValueError(f"archived hard-index outside completed memory: {case_id}")
        copied = int(row["copied_token_id"])
        if int(ids[second]) != copied:
            raise ValueError(
                f"copied token inconsistent with regenerated memory at {case_id}: "
                f"selected={ids[second]} archived={copied}"
            )
        if bool(row["copied_matches_answer"]) != (copied == answer_id):
            raise ValueError("copied-answer match flag inconsistent with frozen prompt")
        selected_first.append(first)
        selected_second.append(second)
        selected_tokens.append(copied)
        second_positions[family].append(second)
        if first == second:
            summary["same_first_second_index"] += 1
            same_index += 1
        summary["first_hop_on_answer_token"] += int(first in answer_positions)
        summary["second_hop_on_answer_token"] += int(second in answer_positions)
        summary["first_hop_near_authoritative_evidence_end_16"] += int(
            abs(first - int(probe.evidence_end_token)) <= ENDPOINT_NEIGHBORHOOD
        )
        summary["second_hop_near_authoritative_evidence_end_16"] += int(
            abs(second - int(probe.evidence_end_token)) <= ENDPOINT_NEIGHBORHOOD
        )
        if probe.first_evidence_end_token is not None:
            summary["second_hop_near_first_relation_end_16"] += int(
                abs(second - int(probe.first_evidence_end_token)) <= ENDPOINT_NEIGHBORHOOD
            )
        if answer_positions:
            distance_lists[family].append(min(abs(second - p) for p in answer_positions))

        case_digest.append({
            "family": family, "case_id": case_id, "memory_count": memory_len,
            "first_hop": first, "second_hop": second, "copied_token_id": copied,
            "target_token_id": answer_id, "oracle_target_positions": answer_positions,
            "authoritative_evidence_end": int(probe.evidence_end_token),
            "first_relation_end": (None if probe.first_evidence_end_token is None else
                                   int(probe.first_evidence_end_token)),
        })

    copied_counts = dict(Counter(selected_tokens))
    if copied_counts != FROZEN_COPY_TOKEN_COUNTS:
        raise ValueError("archived copy token counts changed")
    if total_memory != 384 or len(selected_tokens) != 384:
        raise ValueError("unexpected count of memory-bearing probes")
    if any(token in candidates for token in selected_tokens):
        raise ValueError("archived copies unexpectedly overlap eight candidates")
    if any(item["copied_token_id"] == item["target_token_id"] for item in case_digest):
        raise ValueError("archived zero-hit outcome changed")

    for family in FAMILY_ORDER:
        ds = distance_lists[family]
        families[family]["second_hop_mean_min_token_distance_to_target"] = (
            sum(ds) / len(ds) if ds else None
        )
        families[family]["most_selected_second_hop_positions"] = _top_counts(second_positions[family])

    report = {
        "classification": HISTORY_CLASSIFICATION,
        "original_scientific_classification": FROZEN_CLASSIFICATION,
        "scientific_seed_consumed": True,
        "scientific_result_reclassified": False,
        "historical_trained_checkpoint_parity_proved": False,
        "gpu_used": False,
        "new_scientific_attempt": False,
        "archived_run_id": ARCHIVED_RUN_ID,
        "archived_job_id": ARCHIVED_JOB_ID,
        "source_sha": ARCHIVED_SOURCE_SHA,
        "original_archived_trace_sha256": sha256_json(archive),
        "regenerated_case_alignment_sha256": sha256_json(case_digest),
        "probe_count": len(rows),
        "memory_probe_count": total_memory,
        "oracle_answer_present_count": answers_in_memory,
        "oracle_answer_absent_count": total_memory - answers_in_memory,
        "first_second_index_equal_count": same_index,
        "copy_hit_count": 0,
        "copy_candidate_overlap_count": 0,
        "copy_token_counts": [
            {"token_id": t, "count": count, "decoded_text": decode([t]) if decode else None}
            for t, count in sorted(copied_counts.items())
        ],
        "family_stats": families,
        "interpretation_ceiling": (
            "Read-only regenerated-prompt/index consistency, not parity proof for "
            "the original trained checkpoint and not evidence of scientific PASS."
        ),
        "new_training_or_gpu_authorized": False,
    }
    report["report_sha256"] = sha256_json(report)
    return report


__all__ = ["analyze_position_oracle", "sha256_json", "ARCHIVED_RUN_ID", "ARCHIVED_JOB_ID"]
