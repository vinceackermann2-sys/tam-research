from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import modal

RESEARCH_ISSUE = 591
SOURCE_MAIN = "e08e49e2e5d01010f42dc7119d7bcacc12dd1f83"
SOURCE_TREE = "24417dd9297e76d8899796ccccff8f5ff2462222"
APP_NAME = "aera-v26-8-issue591-readonly-issue588-result-inspector"
VOLUME_NAME = "tam-research-data"
SOURCE_RESULT_PATH = "/vol/aera-v26/issue588-first-divergence-guard-repair1/result.json"
SOURCE_RESULT_SHA256 = "495c6f49210074580553aa4b55bf0970624a8abaee910f6d2bf7315e26d2a540"
SOURCE_TRIGGER = 590
SOURCE_RUN = 33753926605
SOURCE_JOB = 100643674944
SOURCE_ATTEMPT = 1
SOURCE_RESEARCH_ISSUE = 588
SOURCE_DECISION = "COMPLETE_FIRST_DIVERGENCE_LOCALIZATION"
RESULT_MARKER = "AERA_V26_8_ISSUE591_READONLY_ISSUE588_RESULT_INSPECTOR_JSON="
DUPLICATE_THRESHOLD = 0.95

EXPECTED_FIRST = {
    "8": {
        "bitwise_name": "chunk1.stage0.read.recalled",
        "bitwise_max_abs": 0.00048828125,
        "integrated_name": "chunk1.stage0.end_controller.event",
        "integrated_max_abs": 0.03125,
        "discrete_name": "chunk1.stage3.adjudication_replay.shadowed_incoming",
    },
    "64": {
        "bitwise_name": "chunk1.stage0.read.recalled",
        "bitwise_max_abs": 0.00048828125,
        "integrated_name": "chunk1.stage0.end_controller.event",
        "integrated_max_abs": 0.031494140625,
        "discrete_name": "chunk1.stage3.route.gate",
    },
}

REQUESTED_BOUNDARY_TOKENS = (
    "read.",
    "applied_read",
    "controller",
    "end_summary",
    "reasoner",
    "stream",
    "write",
    "selected_write",
    "new_valid",
    "post_state",
    "adjudication_replay",
    "route.",
)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.11")


def _require_dict(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"issue591 expected {label} object")
    return value


def _require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise RuntimeError(f"issue591 expected {label} list")
    return value


def _boundary_name(value: Any, label: str) -> str:
    row = _require_dict(value, label)
    name = row.get("name")
    if not isinstance(name, str) or not name:
        raise RuntimeError(f"issue591 expected {label}.name")
    _require_dict(row.get("comparison"), f"{label}.comparison")
    return name


def _boundary_index(boundaries: list[Any], target: dict[str, Any], label: str) -> int:
    target_name = _boundary_name(target, label)
    matches = [index for index, row in enumerate(boundaries) if isinstance(row, dict) and row.get("name") == target_name]
    if len(matches) != 1:
        raise RuntimeError(f"issue591 expected one {label} boundary named {target_name!r}, got {matches}")
    return matches[0]


def _validate_first_summary(batch: str, comparison: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    frozen = EXPECTED_FIRST[batch]
    bitwise = _require_dict(comparison.get("first_bitwise_difference"), f"batch{batch}.first_bitwise")
    integrated = _require_dict(
        comparison.get("first_integrated_tolerance_or_metadata_failure"),
        f"batch{batch}.first_integrated_failure",
    )
    discrete = _require_dict(comparison.get("first_discrete_decision_difference"), f"batch{batch}.first_discrete")
    if bitwise.get("name") != frozen["bitwise_name"]:
        raise RuntimeError(f"issue591 batch{batch} first-bitwise name drift")
    if integrated.get("name") != frozen["integrated_name"]:
        raise RuntimeError(f"issue591 batch{batch} first-integrated name drift")
    if discrete.get("name") != frozen["discrete_name"]:
        raise RuntimeError(f"issue591 batch{batch} first-discrete name drift")
    bitwise_cmp = _require_dict(bitwise.get("comparison"), f"batch{batch}.first_bitwise.comparison")
    integrated_cmp = _require_dict(integrated.get("comparison"), f"batch{batch}.first_integrated.comparison")
    if bitwise_cmp.get("max_abs") != frozen["bitwise_max_abs"]:
        raise RuntimeError(f"issue591 batch{batch} first-bitwise max_abs drift")
    if integrated_cmp.get("max_abs") != frozen["integrated_max_abs"]:
        raise RuntimeError(f"issue591 batch{batch} first-integrated max_abs drift")
    if bitwise_cmp.get("allclose") is not True or bitwise_cmp.get("failure") is True:
        raise RuntimeError(f"issue591 batch{batch} first bitwise drift is no longer tolerated")
    if integrated_cmp.get("allclose") is not False or integrated_cmp.get("failure") is not True:
        raise RuntimeError(f"issue591 batch{batch} first integrated failure classification drift")
    return bitwise, integrated, discrete


def _presence(boundaries: list[dict[str, Any]], token: str) -> list[str]:
    return [row["name"] for row in boundaries if token in row["name"]]


def _inspect_batch(batch: str, row: dict[str, Any]) -> dict[str, Any]:
    comparison = _require_dict(row.get("comparison"), f"batch{batch}.comparison")
    boundaries_raw = _require_list(comparison.get("boundaries"), f"batch{batch}.boundaries")
    boundaries = [_require_dict(value, f"batch{batch}.boundaries[{index}]") for index, value in enumerate(boundaries_raw)]
    for index, boundary in enumerate(boundaries):
        _boundary_name(boundary, f"batch{batch}.boundaries[{index}]")

    bitwise, integrated, discrete = _validate_first_summary(batch, comparison)
    bitwise_index = _boundary_index(boundaries, bitwise, f"batch{batch}.first_bitwise")
    integrated_index = _boundary_index(boundaries, integrated, f"batch{batch}.first_integrated")
    discrete_index = _boundary_index(boundaries, discrete, f"batch{batch}.first_discrete")
    if not (bitwise_index <= integrated_index <= discrete_index):
        raise RuntimeError(
            f"issue591 batch{batch} execution ordering drift: "
            f"bitwise={bitwise_index}, integrated={integrated_index}, discrete={discrete_index}"
        )

    failures = [_require_dict(value, f"batch{batch}.failures[{index}]") for index, value in enumerate(
        _require_list(comparison.get("failures"), f"batch{batch}.failures")
    )]
    stored_failure_count = len(failures)
    recomputed_failure_names = [
        boundary["name"]
        for boundary in boundaries
        if _require_dict(boundary.get("comparison"), f"batch{batch}.{boundary['name']}.comparison").get("failure") is True
    ]
    stored_failure_names = [_boundary_name(value, f"batch{batch}.failure") for value in failures]
    if recomputed_failure_names != stored_failure_names:
        raise RuntimeError(f"issue591 batch{batch} stored failure ordering drift")

    through_integrated = boundaries[bitwise_index : integrated_index + 1]
    after_integrated_through_discrete = boundaries[integrated_index + 1 : discrete_index + 1]
    bitwise_through_discrete = boundaries[bitwise_index : discrete_index + 1]
    relevant = [
        boundary
        for boundary in bitwise_through_discrete
        if any(token in boundary["name"] for token in REQUESTED_BOUNDARY_TOKENS)
    ]

    requested_presence = {token: _presence(bitwise_through_discrete, token) for token in REQUESTED_BOUNDARY_TOKENS}
    unavailable = [token for token, names in requested_presence.items() if not names]

    threshold_margins = _require_list(
        comparison.get("threshold_margin_diagnostics"), f"batch{batch}.threshold_margin_diagnostics"
    )
    if comparison.get("candidate_backend_internal_adjudication_decisions_available") is not False:
        raise RuntimeError(f"issue591 batch{batch} candidate adjudication availability drift")
    if comparison.get("adjudication_replay_is_diagnostic_only") is not True:
        raise RuntimeError(f"issue591 batch{batch} adjudication replay labeling drift")

    route_context = []
    if batch == "64":
        lo = max(0, discrete_index - 4)
        hi = min(len(boundaries), discrete_index + 5)
        route_context = [boundary for boundary in boundaries[lo:hi] if ".route." in boundary["name"]]

    return {
        "batch": int(batch),
        "boundary_count": len(boundaries),
        "failure_count": stored_failure_count,
        "first_bitwise_difference": bitwise,
        "first_integrated_tolerance_or_metadata_failure": integrated,
        "first_discrete_decision_difference": discrete,
        "boundary_indices": {
            "first_bitwise": bitwise_index,
            "first_integrated_failure": integrated_index,
            "first_discrete": discrete_index,
        },
        "boundary_chain_first_bitwise_through_first_integrated_failure": through_integrated,
        "boundary_chain_after_first_integrated_failure_through_first_discrete": after_integrated_through_discrete,
        "relevant_observed_boundaries_first_bitwise_through_first_discrete": relevant,
        "failures_execution_order": failures,
        "failure_names_execution_order": stored_failure_names,
        "threshold_margin_diagnostics": threshold_margins,
        "duplicate_threshold": DUPLICATE_THRESHOLD,
        "candidate_backend_internal_adjudication_decisions_available": False,
        "adjudication_replay_is_diagnostic_only": True,
        "batch64_route_context_around_first_discrete": route_context,
        "requested_boundary_presence": requested_presence,
        "requested_but_unavailable_tokens": unavailable,
    }


def _inspect_payload(result: dict[str, Any]) -> dict[str, Any]:
    if result.get("research_issue") != SOURCE_RESEARCH_ISSUE:
        raise RuntimeError(f"issue591 source research_issue drift: {result.get('research_issue')!r}")
    if result.get("decision") != SOURCE_DECISION:
        raise RuntimeError(f"issue591 source decision drift: {result.get('decision')!r}")
    if result.get("localization_complete") is not True:
        raise RuntimeError("issue591 source localization is not complete")
    if result.get("parameter_versions_unchanged") is not True:
        raise RuntimeError("issue591 source parameter versions are not immutable")
    if result.get("checkpoint_hashes_unchanged") is not True:
        raise RuntimeError("issue591 source checkpoint hashes are not immutable")
    rows = _require_dict(result.get("rows"), "rows")
    if set(rows) != {"8", "64"}:
        raise RuntimeError(f"issue591 source batch inventory drift: {sorted(rows)}")

    batches = {
        batch: _inspect_batch(batch, _require_dict(rows[batch], f"batch{batch}"))
        for batch in ("8", "64")
    }
    return {
        "research_issue": RESEARCH_ISSUE,
        "inspection_kind": "read_only_existing_immutable_issue588_first_divergence_result",
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
        "gpu_used": False,
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
def inspect_existing_issue588_result() -> dict[str, Any]:
    volume.reload()
    path = Path(SOURCE_RESULT_PATH)
    if not path.is_file():
        raise RuntimeError(f"frozen issue588 durable result is missing: {SOURCE_RESULT_PATH}")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SOURCE_RESULT_SHA256:
        raise RuntimeError(f"issue591 source SHA256 drift: got={digest} expected={SOURCE_RESULT_SHA256}")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise RuntimeError("issue591 expected source JSON object")
    return _inspect_payload(payload)


@app.local_entrypoint()
def main() -> None:
    result = inspect_existing_issue588_result.remote()
    print(RESULT_MARKER + json.dumps(result, sort_keys=True, separators=(",", ":")), flush=True)
