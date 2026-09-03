from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import modal

RESEARCH_ISSUE = 597
SOURCE_MAIN = "a6515564aef1738b6dc95cc50102a432751be24e"
SOURCE_TREE = "396d00f04132c14d987e9953015bb5bf76eecf0d"
APP_NAME = "aera-v26-8-issue597-readonly-issue594-result-inspector"
VOLUME_NAME = "tam-research-data"
SOURCE_RESULT_PATH = "/vol/aera-v26/issue594-stage0-post-read-amplification-localizer/result.json"
SOURCE_RESULT_SHA256 = "c950d8fa50e70a48ec64a87f860d70d854cf1a2b58e1acbdfbcb0052495e809e"
SOURCE_TRIGGER = 596
SOURCE_RUN = 33772104621
SOURCE_JOB = 100704667286
SOURCE_ATTEMPT = 1
SOURCE_RESEARCH_ISSUE = 594
SOURCE_DECISION = "COMPLETE_STAGE0_POST_READ_AMPLIFICATION_LOCALIZATION"
RESULT_MARKER = "AERA_V26_8_ISSUE597_READONLY_ISSUE594_RESULT_INSPECTOR_JSON="

EXPECTED_FIRST = {
    "8": {
        "bitwise_name": "chunk1.stage0.tokenwise_context.context",
        "bitwise_max_abs": 0.001953125,
        "integrated_name": "chunk1.stage0.end_summary",
        "integrated_max_abs": 0.03125,
        "failure_count": 2,
    },
    "64": {
        "bitwise_name": "chunk1.stage0.tokenwise_context.context",
        "bitwise_max_abs": 0.001953125,
        "integrated_name": "chunk1.stage0.end_summary",
        "integrated_max_abs": 0.031494140625,
        "failure_count": 2,
    },
}

FOCUS_BOUNDARY_NAMES = (
    "chunk1.stage0.norm.output",
    "chunk1.stage0.tokenwise_context.h_input",
    "chunk1.stage0.tokenwise_context.context",
    "chunk1.stage0.tokenwise_context.memory_read",
    "chunk1.stage0.post_context.attn_input",
    "chunk1.stage0.attn.output",
    "chunk1.stage0.post_attention.experts_input",
    "chunk1.stage0.experts.expert_logits",
    "chunk1.stage0.experts.count_logits",
    "chunk1.stage0.experts.chosen_count",
    "chunk1.stage0.experts.run_selected_call_count",
    "chunk1.stage0.experts.output",
    "chunk1.stage0.end_summary",
    "chunk1.stage0.end_controller.stream",
    "chunk1.stage0.end_controller.proj_input",
    "chunk1.stage0.end_controller.raw",
)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.11")


def _require_dict(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"issue597 expected {label} object")
    return value


def _require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise RuntimeError(f"issue597 expected {label} list")
    return value


def _boundary_name(row: dict[str, Any], label: str) -> str:
    name = row.get("name")
    if not isinstance(name, str) or not name:
        raise RuntimeError(f"issue597 expected {label}.name")
    _require_dict(row.get("comparison"), f"{label}.comparison")
    return name


def _boundary_index(boundaries: list[dict[str, Any]], target: dict[str, Any], label: str) -> int:
    target_name = _boundary_name(target, label)
    matches = [index for index, row in enumerate(boundaries) if row.get("name") == target_name]
    if len(matches) != 1:
        raise RuntimeError(
            f"issue597 expected one {label} boundary named {target_name!r}, got {matches}"
        )
    return matches[0]


def _validate_first_summary(
    batch: str,
    comparison: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], None]:
    frozen = EXPECTED_FIRST[batch]
    bitwise = _require_dict(
        comparison.get("first_bitwise_difference"),
        f"batch{batch}.first_bitwise_difference",
    )
    integrated = _require_dict(
        comparison.get("first_integrated_tolerance_or_metadata_failure"),
        f"batch{batch}.first_integrated_failure",
    )
    discrete = comparison.get("first_discrete_decision_difference")
    if discrete is not None:
        raise RuntimeError(f"issue597 batch{batch} unexpected first discrete difference")
    if bitwise.get("name") != frozen["bitwise_name"]:
        raise RuntimeError(f"issue597 batch{batch} first-bitwise name drift")
    if integrated.get("name") != frozen["integrated_name"]:
        raise RuntimeError(f"issue597 batch{batch} first-integrated name drift")
    bitwise_cmp = _require_dict(
        bitwise.get("comparison"), f"batch{batch}.first_bitwise.comparison"
    )
    integrated_cmp = _require_dict(
        integrated.get("comparison"), f"batch{batch}.first_integrated.comparison"
    )
    if bitwise_cmp.get("max_abs") != frozen["bitwise_max_abs"]:
        raise RuntimeError(f"issue597 batch{batch} first-bitwise max_abs drift")
    if integrated_cmp.get("max_abs") != frozen["integrated_max_abs"]:
        raise RuntimeError(f"issue597 batch{batch} first-integrated max_abs drift")
    if bitwise_cmp.get("allclose") is not True or bitwise_cmp.get("failure") is True:
        raise RuntimeError(f"issue597 batch{batch} first bitwise drift classification changed")
    if integrated_cmp.get("allclose") is not False or integrated_cmp.get("failure") is not True:
        raise RuntimeError(f"issue597 batch{batch} integrated failure classification changed")
    return bitwise, integrated, None


def _inspect_batch(batch: str, row: dict[str, Any]) -> dict[str, Any]:
    comparison = _require_dict(row.get("comparison"), f"batch{batch}.comparison")
    boundaries_raw = _require_list(comparison.get("boundaries"), f"batch{batch}.boundaries")
    boundaries = [
        _require_dict(value, f"batch{batch}.boundaries[{index}]")
        for index, value in enumerate(boundaries_raw)
    ]
    names: list[str] = []
    for index, boundary in enumerate(boundaries):
        names.append(_boundary_name(boundary, f"batch{batch}.boundaries[{index}]"))
    if len(set(names)) != len(names):
        raise RuntimeError(f"issue597 batch{batch} boundary names are not unique")

    bitwise, integrated, discrete = _validate_first_summary(batch, comparison)
    bitwise_index = _boundary_index(boundaries, bitwise, f"batch{batch}.first_bitwise")
    integrated_index = _boundary_index(
        boundaries, integrated, f"batch{batch}.first_integrated"
    )
    if bitwise_index > integrated_index:
        raise RuntimeError(
            f"issue597 batch{batch} execution ordering drift: "
            f"bitwise={bitwise_index}, integrated={integrated_index}"
        )

    failures = [
        _require_dict(value, f"batch{batch}.failures[{index}]")
        for index, value in enumerate(
            _require_list(comparison.get("failures"), f"batch{batch}.failures")
        )
    ]
    stored_failure_names = [
        _boundary_name(value, f"batch{batch}.failure[{index}]")
        for index, value in enumerate(failures)
    ]
    recomputed_failure_names = [
        boundary["name"]
        for boundary in boundaries
        if _require_dict(
            boundary.get("comparison"),
            f"batch{batch}.{boundary['name']}.comparison",
        ).get("failure")
        is True
    ]
    if recomputed_failure_names != stored_failure_names:
        raise RuntimeError(f"issue597 batch{batch} stored failure ordering drift")
    if len(failures) != EXPECTED_FIRST[batch]["failure_count"]:
        raise RuntimeError(f"issue597 batch{batch} failure count drift")

    boundary_index = {name: index for index, name in enumerate(names)}
    missing_focus = [name for name in FOCUS_BOUNDARY_NAMES if name not in boundary_index]
    if missing_focus:
        raise RuntimeError(f"issue597 batch{batch} missing focus boundaries: {missing_focus}")

    focus = {
        name: {
            "index": boundary_index[name],
            "row": boundaries[boundary_index[name]],
        }
        for name in FOCUS_BOUNDARY_NAMES
    }
    run_selected = [
        {"index": index, "row": boundary}
        for index, boundary in enumerate(boundaries)
        if boundary["name"].startswith("chunk1.stage0.experts.run_selected")
    ]
    end_controller = [
        {"index": index, "row": boundary}
        for index, boundary in enumerate(boundaries)
        if boundary["name"].startswith("chunk1.stage0.end_controller.")
        or boundary["name"].startswith("chunk1.stage0.route_returned_end.")
    ]

    unavailable = comparison.get("unavailable_fields")
    if unavailable is None:
        unavailable = []
    unavailable = _require_list(unavailable, f"batch{batch}.unavailable_fields")

    return {
        "batch": int(batch),
        "boundary_count": len(boundaries),
        "boundary_indices": {
            "first_bitwise": bitwise_index,
            "first_integrated_failure": integrated_index,
            "first_discrete": None,
        },
        "first_bitwise_difference": bitwise,
        "first_integrated_tolerance_or_metadata_failure": integrated,
        "first_discrete_decision_difference": discrete,
        "boundaries_execution_order": [
            {"index": index, "row": boundary}
            for index, boundary in enumerate(boundaries)
        ],
        "focus_boundaries": focus,
        "run_selected_boundaries": run_selected,
        "end_controller_and_returned_end_boundaries": end_controller,
        "failures_execution_order": failures,
        "failure_names_execution_order": stored_failure_names,
        "failure_count": len(failures),
        "unavailable_fields": unavailable,
    }


def _inspect_payload(result: dict[str, Any]) -> dict[str, Any]:
    if result.get("research_issue") != SOURCE_RESEARCH_ISSUE:
        raise RuntimeError(f"issue597 source research_issue drift: {result.get('research_issue')!r}")
    if result.get("decision") != SOURCE_DECISION:
        raise RuntimeError(f"issue597 source decision drift: {result.get('decision')!r}")
    if result.get("localization_complete") is not True:
        raise RuntimeError("issue597 source localization is not complete")
    if result.get("parameter_versions_unchanged") is not True:
        raise RuntimeError("issue597 source parameter versions are not immutable")
    if result.get("checkpoint_hashes_unchanged") is not True:
        raise RuntimeError("issue597 source checkpoint hashes are not immutable")
    if result.get("repair_authorized") is not False:
        raise RuntimeError("issue597 source unexpectedly authorizes repair")

    rows = _require_dict(result.get("rows"), "rows")
    if set(rows) != {"8", "64"}:
        raise RuntimeError(f"issue597 source batch inventory drift: {sorted(rows)}")
    batches = {
        batch: _inspect_batch(batch, _require_dict(rows[batch], f"batch{batch}"))
        for batch in ("8", "64")
    }
    return {
        "research_issue": RESEARCH_ISSUE,
        "inspection_kind": "read_only_existing_immutable_issue594_full_boundary_result",
        "source_main": SOURCE_MAIN,
        "source_tree": SOURCE_TREE,
        "source_trigger": SOURCE_TRIGGER,
        "source_run": SOURCE_RUN,
        "source_job": SOURCE_JOB,
        "source_attempt": SOURCE_ATTEMPT,
        "source_research_issue": SOURCE_RESEARCH_ISSUE,
        "source_result_path": SOURCE_RESULT_PATH,
        "source_result_sha256": SOURCE_RESULT_SHA256,
        "source_decision": result.get("decision"),
        "source_localization_complete": result.get("localization_complete"),
        "source_device": result.get("device"),
        "candidate_backend_names": result.get("candidate_backend_names"),
        "parameter_versions_unchanged": result.get("parameter_versions_unchanged"),
        "checkpoint_hashes_unchanged": result.get("checkpoint_hashes_unchanged"),
        "batches": batches,
        "volume_mutated": False,
        "accelerator_used": False,
        "model_or_checkpoint_execution_performed": False,
        "experiment_rerun": False,
        "production_repair_authorized": False,
        "evaluator_repair_authorized": False,
        "performance_tuning_authorized": False,
        "architecture_freeze_authorized": False,
        "s2_authorized": False,
        "fresh_scientific_seed_authorized": False,
        "independent_replication_credit": False,
        "100m_authorized": False,
        "breakthrough_proven": False,
    }


@app.function(image=image, cpu=1, memory=1024, timeout=120, volumes={"/vol": volume})
def inspect_existing_issue594_result() -> dict[str, Any]:
    volume.reload()
    path = Path(SOURCE_RESULT_PATH)
    if not path.is_file():
        raise RuntimeError(f"frozen issue594 durable result is missing: {SOURCE_RESULT_PATH}")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SOURCE_RESULT_SHA256:
        raise RuntimeError(
            f"issue597 source SHA256 drift: got={digest} expected={SOURCE_RESULT_SHA256}"
        )
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise RuntimeError("issue597 expected source JSON object")
    return _inspect_payload(payload)


@app.local_entrypoint()
def main() -> None:
    result = inspect_existing_issue594_result.remote()
    print(
        RESULT_MARKER + json.dumps(result, sort_keys=True, separators=(",", ":")),
        flush=True,
    )
