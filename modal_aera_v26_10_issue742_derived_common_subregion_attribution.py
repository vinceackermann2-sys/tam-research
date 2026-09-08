from __future__ import annotations

"""Issue #742 CPU-only derived common-subregion attribution from #736/#741."""

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable

import modal

APP_NAME = "aera-v26-10-issue742-derived-common-subregion-attribution"
VOLUME_NAME = "tam-research-data"
PARENT_RESULT_PATH = "/vol/aera-v26/issue736-stage-control-idle-attribution/result.json"
RESULT_PATH = "/vol/aera-v26/issue742-derived-common-subregion-attribution/result.json"
SOURCE_MAIN = "24ae68c393e098ec7535276e58776ea8ef11e238"
SOURCE_TREE = "94b7f64fc4c98f459d6fd45ec2b31f9ac3b9218d"
RESEARCH_ISSUE = 742
PARENT_ISSUE = 736
PARENT_TRIGGER = 741
PARENT_EVIDENCE_COMMENT = 5581226071
PARENT_RUN = 34200593238
PARENT_JOB = 101978242814
PARENT_ATTEMPT = 1
PARENT_LAUNCHER_BLOB = "2a004bb09eed8f644a61353f762f3ce1bd7d14f2"
TRIGGER_PREFIX = "[aera-v26-10-issue742-derived-subregion-cpu]"
MIN_SUBREGION_MS = 1.0
MIN_SUBREGION_SHARE = 0.10
RESULT_MARKER = "AERA_V26_10_ISSUE742_DERIVED_RESULT_JSON="
SUMMARY_MARKER = "AERA_V26_10_ISSUE742_DERIVED_SUMMARY_JSON="

EXPECTED_PARENT = {
    "8": {
        "median_ms": 50.930686950683594,
        "families": {
            "ficem_read_write_control": 8.766532585351221,
            "chunk_or_stage_glue_control": 8.675996231202186,
            "latent_reasoner_control": 8.0989977855759,
        },
    },
    "64": {
        "median_ms": 94.81072235107422,
        "families": {
            "latent_reasoner_control": 12.809936465918812,
            "ficem_read_write_control": 11.524780411744375,
            "chunk_or_stage_glue_control": 10.381195653185925,
        },
    },
}

CANDIDATES = (
    "ficem_read_control",
    "ficem_update_control",
    "ficem_parent_exclusive_control",
    "latent_reasoner_cell_control",
    "latent_reasoner_schedule_control",
    "stage_scope_exclusive_glue_control",
    "route_scope_exclusive_glue_control",
    "whole_call_outer_glue_control",
)

PARENT_FAMILY = {
    "ficem_read_control": "ficem_read_write_control",
    "ficem_update_control": "ficem_read_write_control",
    "ficem_parent_exclusive_control": "ficem_read_write_control",
    "latent_reasoner_cell_control": "latent_reasoner_control",
    "latent_reasoner_schedule_control": "latent_reasoner_control",
    "stage_scope_exclusive_glue_control": "chunk_or_stage_glue_control",
    "route_scope_exclusive_glue_control": "chunk_or_stage_glue_control",
    "whole_call_outer_glue_control": "chunk_or_stage_glue_control",
}

image = modal.Image.debian_slim(python_version="3.11")
app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite_nonnegative(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) >= 0.0


def _matches(candidate: str, name: str) -> bool:
    if candidate == "ficem_read_control":
        return name.startswith("issue736.ficem_scope.") and name.endswith(".read")
    if candidate == "ficem_update_control":
        return name.startswith("issue736.ficem_scope.") and (
            name.endswith(".update") or name.endswith(".update_from_projected")
        )
    if candidate == "ficem_parent_exclusive_control":
        return name == "issue736.ficem_parent_exclusive"
    if candidate == "latent_reasoner_cell_control":
        return name.startswith("issue736.reasoner_cell.")
    if candidate == "latent_reasoner_schedule_control":
        return name.startswith("issue736.reasoner_scope.")
    if candidate == "stage_scope_exclusive_glue_control":
        return name.startswith("issue736.stage_scope.")
    if candidate == "route_scope_exclusive_glue_control":
        return name.startswith("issue736.route_scope.")
    if candidate == "whole_call_outer_glue_control":
        return name == "issue736.whole_call"
    raise KeyError(candidate)


def _validate_parent(parent: dict[str, Any]) -> None:
    if parent.get("scope") != "aera_v26_10_issue736_stage_control_idle_attribution":
        raise RuntimeError("issue742 parent scope drift")
    if parent.get("research_issue") != 736 or parent.get("next_target_region") is not None:
        raise RuntimeError("issue742 parent identity/target drift")
    if parent.get("device") != "NVIDIA L4":
        raise RuntimeError("issue742 parent device drift")
    for key in (
        "optimization_authorized",
        "full_e2e_systems_gate_authorized",
        "systems_pass_earned",
        "architecture_freeze_authorized",
        "s2_authorized",
        "fresh_scientific_seed_authorized",
        "independent_replication_credit",
        "100m_authorized",
        "breakthrough_proven",
    ):
        if parent.get(key) is not False:
            raise RuntimeError(f"issue742 parent authority drift: {key}")
    rows = parent.get("rows")
    if not isinstance(rows, dict) or set(rows) != {"8", "64"}:
        raise RuntimeError("issue742 parent row drift")
    for batch, expected in EXPECTED_PARENT.items():
        row = rows[batch]
        unprofiled = row.get("unprofiled_whole_call")
        trace = row.get("trace")
        if not isinstance(unprofiled, dict) or not isinstance(trace, dict):
            raise RuntimeError(f"issue742 parent structure missing batch {batch}")
        median = float(unprofiled.get("median_ms"))
        if not math.isclose(median, expected["median_ms"], rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"issue742 parent median drift batch {batch}: {median}")
        if trace.get("exclusive_partition_no_double_count") is not True:
            raise RuntimeError(f"issue742 parent exclusive partition drift batch {batch}")
        scale = trace.get("normalization_scale")
        if not isinstance(scale, (int, float)) or not math.isfinite(float(scale)) or float(scale) <= 0.0:
            raise RuntimeError(f"issue742 parent normalization scale invalid batch {batch}")
        raw = trace.get("raw_invocations")
        if not isinstance(raw, list) or not raw:
            raise RuntimeError(f"issue742 parent raw_invocations missing batch {batch}")
        table = trace.get("exclusive_family_table")
        if not isinstance(table, dict):
            raise RuntimeError(f"issue742 parent family table missing batch {batch}")
        for family, expected_ms in expected["families"].items():
            entry = table.get(family)
            if not isinstance(entry, dict):
                raise RuntimeError(f"issue742 parent family missing {batch}/{family}")
            got = float(entry.get("distortion_normalized_host_control_idle_ms"))
            if not math.isclose(got, expected_ms, rel_tol=0.0, abs_tol=1e-12):
                raise RuntimeError(f"issue742 parent family drift {batch}/{family}: {got}")


def _aggregate_candidate(
    candidate: str,
    raw: list[dict[str, Any]],
    scale: float,
    median_ms: float,
    parent_cap_ms: float,
) -> dict[str, Any]:
    matched = [row for row in raw if isinstance(row, dict) and _matches(candidate, str(row.get("name", "")))]
    if not matched:
        return {
            "available": False,
            "reason": "no matching persisted child-subtracted #736 raw invocation rows",
            "matched_names": [],
        }
    for row in matched:
        if row.get("children_subtracted") is not True:
            return {
                "available": False,
                "reason": "matching raw invocation was not persisted as children_subtracted=true",
                "matched_names": sorted({str(r.get("name", "")) for r in matched}),
            }
        for key in (
            "exclusive_cpu_us",
            "exclusive_device_active_union_us",
            "exclusive_residual_us",
            "exclusive_inter_device_idle_total_us",
        ):
            if not _finite_nonnegative(row.get(key)):
                raise RuntimeError(f"issue742 invalid persisted {key} for {row.get('name')}")
        counts = row.get("cuda_runtime_counts_exclusive")
        if not isinstance(counts, dict):
            raise RuntimeError(f"issue742 missing persisted runtime counts for {row.get('name')}")
        for key in ("launch", "synchronize_or_wait", "memcpy_or_memset", "total"):
            value = counts.get(key)
            if not isinstance(value, int) or value < 0:
                raise RuntimeError(f"issue742 invalid runtime count {key} for {row.get('name')}")
    cpu_us = sum(float(row["exclusive_cpu_us"]) for row in matched)
    device_us = sum(float(row["exclusive_device_active_union_us"]) for row in matched)
    residual_us = sum(float(row["exclusive_residual_us"]) for row in matched)
    idle_us = sum(float(row["exclusive_inter_device_idle_total_us"]) for row in matched)
    counts = {
        key: sum(int(row["cuda_runtime_counts_exclusive"][key]) for row in matched)
        for key in ("launch", "synchronize_or_wait", "memcpy_or_memset", "total")
    }
    normalized_ms = residual_us / 1000.0 * scale
    epsilon = 1e-9
    if normalized_ms > parent_cap_ms + epsilon:
        raise RuntimeError(
            f"issue742 derived candidate exceeds frozen parent family cap: {candidate} {normalized_ms} > {parent_cap_ms}"
        )
    if normalized_ms > median_ms + epsilon:
        raise RuntimeError(f"issue742 derived candidate exceeds whole-call median: {candidate}")
    return {
        "available": True,
        "directly_derived_from_persisted_child_subtracted_rows": True,
        "invocation_count": len(matched),
        "matched_names": sorted({str(row["name"]) for row in matched}),
        "exclusive_cpu_us": cpu_us,
        "exclusive_device_active_union_us": device_us,
        "exclusive_residual_us": residual_us,
        "exclusive_inter_device_idle_total_us": idle_us,
        "cuda_runtime_counts_exclusive": counts,
        "normalization_scale": scale,
        "distortion_normalized_residual_ms": normalized_ms,
        "share_of_unprofiled_median": normalized_ms / median_ms if median_ms > 0.0 else 0.0,
        "parent_family": PARENT_FAMILY[candidate],
        "parent_family_cap_ms": parent_cap_ms,
    }


def _derive(parent: dict[str, Any]) -> tuple[dict[str, Any], str | None, dict[str, Any]]:
    rows_out: dict[str, Any] = {}
    qualifying: dict[str, set[str]] = {}
    rankings: dict[str, list[dict[str, Any]]] = {}
    for batch in ("8", "64"):
        row = parent["rows"][batch]
        trace = row["trace"]
        median_ms = float(row["unprofiled_whole_call"]["median_ms"])
        scale = float(trace["normalization_scale"])
        raw = trace["raw_invocations"]
        table = trace["exclusive_family_table"]
        candidates: dict[str, Any] = {}
        for candidate in CANDIDATES:
            parent_family = PARENT_FAMILY[candidate]
            parent_cap = float(table[parent_family]["distortion_normalized_host_control_idle_ms"])
            candidates[candidate] = _aggregate_candidate(
                candidate,
                raw,
                scale,
                median_ms,
                parent_cap,
            )
        ranking = sorted(
            (
                {
                    "candidate": candidate,
                    "ms": float(entry["distortion_normalized_residual_ms"]),
                    "share": float(entry["share_of_unprofiled_median"]),
                }
                for candidate, entry in candidates.items()
                if entry.get("available") is True
            ),
            key=lambda item: (item["ms"], item["candidate"]),
            reverse=True,
        )
        q = {
            item["candidate"]
            for item in ranking
            if item["ms"] >= MIN_SUBREGION_MS and item["share"] >= MIN_SUBREGION_SHARE
        }
        rows_out[batch] = {
            "unprofiled_median_ms": median_ms,
            "normalization_scale": scale,
            "candidates": candidates,
            "ranking": ranking,
            "qualifying": sorted(q),
        }
        rankings[batch] = ranking
        qualifying[batch] = q
    common = qualifying["8"] & qualifying["64"]
    target: str | None = None
    top_common: dict[str, str | None] = {}
    for batch in ("8", "64"):
        top_common[batch] = next(
            (item["candidate"] for item in rankings[batch] if item["candidate"] in common),
            None,
        )
    if common and top_common["8"] == top_common["64"]:
        target = top_common["8"]
    evidence = {
        "qualifying": {batch: sorted(values) for batch, values in qualifying.items()},
        "common_qualifying": sorted(common),
        "largest_common_by_batch": top_common,
        "same_largest_common_required": True,
        "min_subregion_ms": MIN_SUBREGION_MS,
        "min_subregion_share": MIN_SUBREGION_SHARE,
        "unavailable_or_inferred_cannot_authorize": True,
    }
    return rows_out, target, evidence


@app.function(
    image=image,
    cpu=1,
    memory=1024,
    timeout=120,
    volumes={"/vol": volume},
)
def run_derived() -> dict[str, Any]:
    volume.reload()
    parent_path = Path(PARENT_RESULT_PATH)
    result_path = Path(RESULT_PATH)
    if not parent_path.exists():
        raise RuntimeError("issue742 parent #736 durable result missing")
    if result_path.exists():
        raise RuntimeError("issue742 derived result already exists")
    parent_hash_before = _sha256(parent_path)
    parent = json.loads(parent_path.read_text())
    _validate_parent(parent)
    rows, target, evidence = _derive(parent)
    parent_hash_after = _sha256(parent_path)
    if parent_hash_after != parent_hash_before:
        raise RuntimeError("issue742 parent result changed during derived read")
    result = {
        "scope": "aera_v26_10_issue742_cpu_derived_common_subregion_attribution",
        "research_issue": RESEARCH_ISSUE,
        "source_main": SOURCE_MAIN,
        "source_tree": SOURCE_TREE,
        "parent_issue": PARENT_ISSUE,
        "parent_trigger": PARENT_TRIGGER,
        "parent_evidence_comment": PARENT_EVIDENCE_COMMENT,
        "parent_run": PARENT_RUN,
        "parent_job": PARENT_JOB,
        "parent_attempt": PARENT_ATTEMPT,
        "parent_launcher_blob": PARENT_LAUNCHER_BLOB,
        "parent_result_path": PARENT_RESULT_PATH,
        "parent_result_sha256_before": parent_hash_before,
        "parent_result_sha256_after": parent_hash_after,
        "parent_result_unchanged": True,
        "result_path": RESULT_PATH,
        "trigger_prefix": TRIGGER_PREFIX,
        "min_subregion_ms": MIN_SUBREGION_MS,
        "min_subregion_share": MIN_SUBREGION_SHARE,
        "candidate_subregions": list(CANDIDATES),
        "rows": rows,
        "next_target_subregion": target,
        "decision_evidence": evidence,
        "derived_only": True,
        "cpu_only": True,
        "gpu_used": False,
        "model_constructed": False,
        "checkpoint_loaded": False,
        "new_measurement_performed": False,
        "profiler_run": False,
        "timing_loop_run": False,
        "scientific_seed_consumed": False,
        "optimization_authorized": False,
        "full_e2e_systems_gate_authorized": False,
        "systems_pass_earned": False,
        "architecture_freeze_authorized": False,
        "s2_authorized": False,
        "fresh_scientific_seed_authorized": False,
        "independent_replication_credit": False,
        "100m_authorized": False,
        "breakthrough_proven": False,
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    volume.commit()
    summary = {
        "research_issue": RESEARCH_ISSUE,
        "parent_result_sha256": parent_hash_before,
        "next_target_subregion": target,
        "rows": {
            batch: {
                "ranking": row["ranking"],
                "qualifying": row["qualifying"],
                "unavailable": {
                    name: entry.get("reason")
                    for name, entry in row["candidates"].items()
                    if entry.get("available") is not True
                },
            }
            for batch, row in rows.items()
        },
        "decision_evidence": evidence,
        "derived_only": True,
        "cpu_only": True,
        "gpu_used": False,
        "model_constructed": False,
        "checkpoint_loaded": False,
        "new_measurement_performed": False,
        "profiler_run": False,
        "timing_loop_run": False,
        "scientific_seed_consumed": False,
        "optimization_authorized": False,
        "full_e2e_systems_gate_authorized": False,
        "systems_pass_earned": False,
        "architecture_freeze_authorized": False,
        "s2_authorized": False,
        "fresh_scientific_seed_authorized": False,
        "independent_replication_credit": False,
        "100m_authorized": False,
        "breakthrough_proven": False,
    }
    print(RESULT_MARKER + json.dumps(summary, sort_keys=True))
    return summary


@app.local_entrypoint()
def main() -> None:
    summary = run_derived.remote()
    print(SUMMARY_MARKER + json.dumps(summary, sort_keys=True))
