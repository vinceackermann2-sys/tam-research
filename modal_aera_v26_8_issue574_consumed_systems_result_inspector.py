from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import modal

RESEARCH_ISSUE = 574
SOURCE_MAIN = "c913d817ee1c1a1fca2d7c7622f4c8ca5353772f"
SOURCE_TREE = "646ffe7f96ed2f1322408fcd2dc3eee2ff886161"
APP_NAME = "aera-v26-8-issue574-issue571-result-inspector"
VOLUME_NAME = "tam-research-data"
SOURCE_RESULT_PATH = "/vol/aera-v26/issue571-memory-safe-end-to-end-systems/result.json"
SOURCE_RESULT_SHA256 = "afeeb62351cc4fb97d272c5b55c9621839e26f83753ae1fb237733d58a5ee472"
SOURCE_TRIGGER = 573
SOURCE_RUN = 33741700781
SOURCE_JOB = 100604889696
SOURCE_ATTEMPT = 1
SOURCE_DECISION = "FAIL_FROZEN_E2E_SYSTEMS_GATE"
SOURCE_OVERALL_PASS = False
SOURCE_CANDIDATE_BACKEND = "triton-ficem-read-v26.8-mixed-strength-precision-write-v26.6-materialize-cast"
FROZEN_501_BACKEND_NAME = "triton-ficem-read-repair5-write-v26.4"
RESULT_MARKER = "AERA_V26_8_ISSUE574_ISSUE571_RESULT_CLASSIFICATION_JSON="

FROZEN_BATCH_PARTITION = {
    "8": {
        "routing_exact": True,
        "logit_equivalence_pass": True,
        "logit_max_abs": 0.0625,
        "state_equivalence_pass": False,
        "physical_sparse_pass": False,
        "write_geometry_pass": True,
        "finite": True,
        "persistent_state_bytes_pass": True,
        "candidate_full_vs_transformer_speed_ratio": 0.2665054349604248,
        "required_full_speed_ratio": 0.25,
        "throughput_pass": True,
        "reference_full_latency_ms": 43.729408264160156,
        "candidate_full_latency_ms": 35.385873794555664,
        "no_reference_full_latency_regression": True,
    },
    "64": {
        "routing_exact": False,
        "logit_equivalence_pass": False,
        "logit_max_abs": 2.34765625,
        "state_equivalence_pass": False,
        "physical_sparse_pass": False,
        "write_geometry_pass": True,
        "finite": True,
        "persistent_state_bytes_pass": True,
        "candidate_full_vs_transformer_speed_ratio": 1.164232659436335,
        "required_full_speed_ratio": 1.25,
        "throughput_pass": False,
        "reference_full_latency_ms": 90.45303726196289,
        "candidate_full_latency_ms": 78.88516616821289,
        "no_reference_full_latency_regression": True,
    },
}

PASS_GATES = (
    "routing_exact",
    "logit_equivalence",
    "state_equivalence",
    "physical_sparse",
    "write_geometry",
    "finite",
    "persistent_state_bytes_pass",
    "throughput_pass",
    "no_reference_full_latency_regression",
)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.11")


def _require_dict(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"issue574 expected {label} object")
    return value


def _json_diff(reference: Any, candidate: Any, path: str = "$") -> list[dict[str, Any]]:
    differences: list[dict[str, Any]] = []
    if isinstance(reference, dict) and isinstance(candidate, dict):
        keys = sorted(set(reference) | set(candidate))
        for key in keys:
            child = f"{path}.{key}"
            if key not in reference:
                differences.append({"path": child, "reference": "<missing>", "candidate": candidate[key]})
            elif key not in candidate:
                differences.append({"path": child, "reference": reference[key], "candidate": "<missing>"})
            else:
                differences.extend(_json_diff(reference[key], candidate[key], child))
        return differences
    if isinstance(reference, list) and isinstance(candidate, list):
        length = max(len(reference), len(candidate))
        for index in range(length):
            child = f"{path}[{index}]"
            if index >= len(reference):
                differences.append({"path": child, "reference": "<missing>", "candidate": candidate[index]})
            elif index >= len(candidate):
                differences.append({"path": child, "reference": reference[index], "candidate": "<missing>"})
            else:
                differences.extend(_json_diff(reference[index], candidate[index], child))
        return differences
    if type(reference) is not type(candidate) or reference != candidate:
        differences.append({"path": path, "reference": reference, "candidate": candidate})
    return differences


def _gate_value(row: dict[str, Any], gate: str) -> bool:
    if gate in ("logit_equivalence", "state_equivalence", "physical_sparse", "write_geometry"):
        nested = row.get(gate)
        return bool(isinstance(nested, dict) and nested.get("pass") is True)
    return bool(row.get(gate) is True)


def _frozen_partition_from_row(row: dict[str, Any]) -> dict[str, Any]:
    logit = _require_dict(row.get("logit_equivalence"), "logit_equivalence")
    state = _require_dict(row.get("state_equivalence"), "state_equivalence")
    sparse = _require_dict(row.get("physical_sparse"), "physical_sparse")
    write = _require_dict(row.get("write_geometry"), "write_geometry")
    return {
        "routing_exact": row.get("routing_exact"),
        "logit_equivalence_pass": logit.get("pass"),
        "logit_max_abs": logit.get("max_abs"),
        "state_equivalence_pass": state.get("pass"),
        "physical_sparse_pass": sparse.get("pass"),
        "write_geometry_pass": write.get("pass"),
        "finite": row.get("finite"),
        "persistent_state_bytes_pass": row.get("persistent_state_bytes_pass"),
        "candidate_full_vs_transformer_speed_ratio": row.get("candidate_full_vs_transformer_speed_ratio"),
        "required_full_speed_ratio": row.get("required_full_speed_ratio"),
        "throughput_pass": row.get("throughput_pass"),
        "reference_full_latency_ms": row.get("reference_full_latency_ms"),
        "candidate_full_latency_ms": row.get("candidate_full_latency_ms"),
        "no_reference_full_latency_regression": row.get("no_reference_full_latency_regression"),
    }


def _classify_batch(batch: str, row: dict[str, Any]) -> dict[str, Any]:
    routing_reference = _require_dict(row.get("routing_reference"), f"batch{batch}.routing_reference")
    routing_candidate = _require_dict(row.get("routing_candidate"), f"batch{batch}.routing_candidate")
    logit = _require_dict(row.get("logit_equivalence"), f"batch{batch}.logit_equivalence")
    state = _require_dict(row.get("state_equivalence"), f"batch{batch}.state_equivalence")
    sparse = _require_dict(row.get("physical_sparse"), f"batch{batch}.physical_sparse")
    write = _require_dict(row.get("write_geometry"), f"batch{batch}.write_geometry")
    timings = _require_dict(row.get("timings"), f"batch{batch}.timings")
    peak_vram = _require_dict(row.get("peak_vram"), f"batch{batch}.peak_vram")
    profiler = _require_dict(row.get("profiler_candidate_full"), f"batch{batch}.profiler_candidate_full")

    failed_gates = [gate for gate in PASS_GATES if not _gate_value(row, gate)]
    backend_names_raw = sparse.get("backend_names")
    backend_names = backend_names_raw if isinstance(backend_names_raw, list) else []
    frozen_backend_identity_incompatible = bool(
        backend_names
        and all(name == SOURCE_CANDIDATE_BACKEND for name in backend_names)
        and SOURCE_CANDIDATE_BACKEND != FROZEN_501_BACKEND_NAME
        and sparse.get("backend_names_exact") is False
    )

    state_reason = state.get("reason")
    continuous_state_numerical_mismatch = bool(
        state.get("pass") is False
        and state_reason is None
        and state.get("continuous_allclose") is False
    )
    validity_or_state_schema_mismatch = bool(
        state.get("pass") is False
        and (
            state_reason is not None
            or state.get("validity_exact") is False
            or state.get("dtype_device_shape_exact") is False
        )
    )
    output_logit_numerical_mismatch = bool(
        logit.get("pass") is False
        and logit.get("dtype_device_shape_exact") is True
        and logit.get("allclose") is False
    )
    output_logit_metadata_mismatch = bool(
        logit.get("pass") is False and logit.get("dtype_device_shape_exact") is False
    )

    return {
        "batch": int(batch),
        "authoritative_row_pass": all(_gate_value(row, gate) for gate in PASS_GATES),
        "failed_gates": failed_gates,
        "routing": {
            "routing_exact_authoritative": row.get("routing_exact"),
            "raw_gate_signatures_persisted": False,
            "raw_gate_signature_note": (
                "The frozen evaluator compared raw thresholded gate tensors in memory but persisted only "
                "routing accounting plus the combined routing_exact boolean; raw signatures cannot be "
                "reconstructed from the immutable result."
            ),
            "reference_accounting": routing_reference,
            "candidate_accounting": routing_candidate,
            "accounting_diff": _json_diff(routing_reference, routing_candidate),
        },
        "logit_equivalence": logit,
        "state_equivalence": state,
        "physical_sparse": sparse,
        "write_geometry": write,
        "finite": row.get("finite"),
        "persistent_state_bytes_per_session_actual": row.get("persistent_state_bytes_per_session_actual"),
        "persistent_state_bytes_pass": row.get("persistent_state_bytes_pass"),
        "performance": {
            "candidate_full_vs_transformer_speed_ratio": row.get("candidate_full_vs_transformer_speed_ratio"),
            "required_full_speed_ratio": row.get("required_full_speed_ratio"),
            "throughput_pass": row.get("throughput_pass"),
            "reference_full_latency_ms": row.get("reference_full_latency_ms"),
            "candidate_full_latency_ms": row.get("candidate_full_latency_ms"),
            "candidate_vs_reference_latency_ratio": row.get("candidate_vs_reference_latency_ratio"),
            "no_reference_full_latency_regression": row.get("no_reference_full_latency_regression"),
        },
        "timings": timings,
        "peak_vram_diagnostic_only": peak_vram,
        "profiler_candidate_full_diagnostic_only": profiler,
        "classification": {
            "evaluator_contract_incompatibility": frozen_backend_identity_incompatible,
            "routing_mismatch": row.get("routing_exact") is False,
            "continuous_state_numerical_mismatch": continuous_state_numerical_mismatch,
            "validity_or_state_schema_mismatch": validity_or_state_schema_mismatch,
            "output_logit_numerical_mismatch": output_logit_numerical_mismatch,
            "output_logit_metadata_mismatch": output_logit_metadata_mismatch,
            "performance_threshold_miss": row.get("throughput_pass") is False,
        },
        "backend_identity_diagnostic": {
            "frozen_501_expected_backend_name": FROZEN_501_BACKEND_NAME,
            "issue571_intended_backend_name": SOURCE_CANDIDATE_BACKEND,
            "stored_backend_names": backend_names,
            "stored_backend_names_exact": sparse.get("backend_names_exact"),
            "identity_incompatibility_proven": frozen_backend_identity_incompatible,
            "retroactive_pass_granted": False,
        },
    }


def _inspect_payload(result: dict[str, Any]) -> dict[str, Any]:
    if result.get("decision") != SOURCE_DECISION or result.get("overall_pass") is not SOURCE_OVERALL_PASS:
        raise RuntimeError("issue574 source decision drift")
    if result.get("device") != "NVIDIA L4":
        raise RuntimeError(f"issue574 source device drift: {result.get('device')!r}")
    candidate_names = result.get("candidate_backend_names")
    if not isinstance(candidate_names, list) or not candidate_names:
        raise RuntimeError("issue574 source result missing candidate backend names")
    if any(name != SOURCE_CANDIDATE_BACKEND for name in candidate_names):
        raise RuntimeError("issue574 source candidate backend drift")

    rows = _require_dict(result.get("rows"), "rows")
    if set(rows) != {"8", "64"}:
        raise RuntimeError(f"issue574 source batch inventory drift: {sorted(rows)}")
    for batch, frozen in FROZEN_BATCH_PARTITION.items():
        row = _require_dict(rows.get(batch), f"batch{batch}")
        observed = _frozen_partition_from_row(row)
        if observed != frozen:
            raise RuntimeError(
                f"issue574 frozen batch{batch} partition drift: observed={observed!r} expected={frozen!r}"
            )

    classified = {batch: _classify_batch(batch, _require_dict(rows[batch], f"batch{batch}")) for batch in ("8", "64")}
    metadata = _require_dict(result.get("issue571_gate_metadata"), "issue571_gate_metadata")
    claims = _require_dict(result.get("claims"), "claims")

    return {
        "research_issue": RESEARCH_ISSUE,
        "inspection_kind": "read_only_existing_immutable_issue571_durable_result",
        "source_main": SOURCE_MAIN,
        "source_tree": SOURCE_TREE,
        "source_trigger": SOURCE_TRIGGER,
        "source_run": SOURCE_RUN,
        "source_job": SOURCE_JOB,
        "source_attempt": SOURCE_ATTEMPT,
        "source_result_path": SOURCE_RESULT_PATH,
        "source_result_sha256": SOURCE_RESULT_SHA256,
        "source_decision": result.get("decision"),
        "source_overall_pass": result.get("overall_pass"),
        "source_device": result.get("device"),
        "candidate_backend_names": candidate_names,
        "per_batch_pass_stored": result.get("per_batch_pass"),
        "batches": classified,
        "parameter_versions_before": result.get("parameter_versions_before"),
        "parameter_versions_after": result.get("parameter_versions_after"),
        "parameter_versions_unchanged": result.get("parameter_versions_unchanged"),
        "checkpoint_hashes_before": result.get("checkpoint_hashes_before"),
        "checkpoint_hashes_after": result.get("checkpoint_hashes_after"),
        "checkpoint_hashes_unchanged": result.get("checkpoint_hashes_unchanged"),
        "training_performed": result.get("training_performed"),
        "optimizer_created": result.get("optimizer_created"),
        "backward_performed": result.get("backward_performed"),
        "corpus_accessed": result.get("corpus_accessed"),
        "checkpoint_written": result.get("checkpoint_written"),
        "scientific_seed_consumed": result.get("scientific_seed_consumed"),
        "claims_stored": claims,
        "issue571_gate_metadata": metadata,
        "routing_signatures_persisted": False,
        "volume_mutated": False,
        "gpu_used": False,
        "experiment_rerun": False,
        "repair_authorized": False,
        "end_to_end_systems_authorized": False,
        "architecture_freeze_authorized": False,
        "s2_authorized": False,
        "fresh_scientific_seed_authorized": False,
        "independent_replication_credit": False,
        "100m_authorized": False,
        "breakthrough_proven": False,
    }


@app.function(image=image, cpu=1, memory=512, timeout=120, volumes={"/vol": volume})
def inspect_existing_issue571_result() -> dict[str, Any]:
    volume.reload()
    path = Path(SOURCE_RESULT_PATH)
    if not path.is_file():
        raise RuntimeError(f"frozen issue571 durable result is missing: {SOURCE_RESULT_PATH}")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SOURCE_RESULT_SHA256:
        raise RuntimeError(
            f"issue571 durable SHA mismatch: got={digest} expected={SOURCE_RESULT_SHA256}"
        )
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise RuntimeError("issue571 durable result root must be an object")
    inspection = _inspect_payload(parsed)
    inspection["verified_source_bytes"] = len(raw)
    return inspection


@app.local_entrypoint()
def main() -> None:
    result = inspect_existing_issue571_result.remote()
    print(RESULT_MARKER + json.dumps(result, sort_keys=True, separators=(",", ":")), flush=True)
