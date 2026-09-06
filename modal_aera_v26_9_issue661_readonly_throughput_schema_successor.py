from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import modal

RESEARCH_ISSUE = 661
SOURCE_RESEARCH_ISSUE = 654
SOURCE_MAIN = "af48dc731f2dabfc8015a9d0b7fd189e9c905aad"
SOURCE_TREE = "c87f3f24bdf35b118f8f8aac8f3ccb426196df14"
SOURCE_RESULT_PATH = "/vol/aera-v26/issue650-e2e-cli-guard-continuation/result.json"
SOURCE_RESULT_SHA256 = "914615db5267565563dcc9e82bfc31f444a656a68bd560f50447a8fd03588431"
SOURCE_TRIGGER = 653
SOURCE_RUN = 34022331841
SOURCE_JOB = 101457058965
SOURCE_ATTEMPT = 1
SOURCE_DECISION = "FAIL_FROZEN_E2E_SYSTEMS_GATE"
SOURCE_OVERALL_PASS = False
SOURCE_DEVICE = "NVIDIA L4"
SOURCE_CANDIDATE_BACKEND = "triton-ficem-read-v26.9-identity-weight-visibility-write-v26.6-materialize-cast"
PREDECESSOR_TRIGGER = 660
PREDECESSOR_RUN = 34036347677
PREDECESSOR_JOB = 101495015912
APP_NAME = "aera-v26-9-issue661-readonly-throughput-schema-successor"
VOLUME_NAME = "tam-research-data"
RESULT_MARKER = "AERA_V26_9_ISSUE661_READONLY_THROUGHPUT_EVIDENCE_JSON="

EXPECTED_TIMING_CONDITIONS = (
    "transformer",
    "v26_torch_reference_full_ficem",
    "v26_6_triton_full_ficem",
)

FROZEN_ROWS = {
    "8": {
        "routing_exact": True,
        "logit_equivalence_pass": True,
        "logit_max_abs": 0.03125,
        "state_equivalence_pass": True,
        "physical_sparse_pass": True,
        "write_geometry_pass": True,
        "finite": True,
        "persistent_state_bytes_pass": True,
        "candidate_full_vs_transformer_speed_ratio": 0.20816584116044815,
        "required_full_speed_ratio": 0.25,
        "throughput_pass": False,
        "reference_full_latency_ms": 67.08428955078125,
        "candidate_full_latency_ms": 54.38617515563965,
        "candidate_vs_reference_latency_ratio": 0.8107140363239679,
        "no_reference_full_latency_regression": True,
    },
    "64": {
        "routing_exact": True,
        "logit_equivalence_pass": True,
        "logit_max_abs": 0.0625,
        "state_equivalence_pass": True,
        "physical_sparse_pass": True,
        "write_geometry_pass": True,
        "finite": True,
        "persistent_state_bytes_pass": True,
        "candidate_full_vs_transformer_speed_ratio": 0.9196618814832817,
        "required_full_speed_ratio": 1.25,
        "throughput_pass": False,
        "reference_full_latency_ms": 115.4748306274414,
        "candidate_full_latency_ms": 99.48311996459961,
        "candidate_vs_reference_latency_ratio": 0.8615134521007772,
        "no_reference_full_latency_regression": True,
    },
}

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.11")


def _require_dict(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"issue661 expected {label} object")
    return value


def _require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise RuntimeError(f"issue661 expected {label} list")
    return value


def _frozen_observed(row: dict[str, Any]) -> dict[str, Any]:
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
        "candidate_vs_reference_latency_ratio": row.get("candidate_vs_reference_latency_ratio"),
        "no_reference_full_latency_regression": row.get("no_reference_full_latency_regression"),
    }


def _extract_batch(batch: str, row: dict[str, Any]) -> dict[str, Any]:
    timings = _require_dict(row.get("timings"), f"batch{batch}.timings")
    if set(timings) != set(EXPECTED_TIMING_CONDITIONS):
        raise RuntimeError(
            f"issue661 batch{batch} timing-key set drifted: {sorted(timings)!r}"
        )
    canonical_timings = {
        condition: _require_dict(timings.get(condition), f"batch{batch}.timings.{condition}")
        for condition in EXPECTED_TIMING_CONDITIONS
    }

    profiler = _require_dict(
        row.get("profiler_candidate_full"), f"batch{batch}.profiler_candidate_full"
    )
    top_ops = _require_list(
        profiler.get("top_cuda_operators"),
        f"batch{batch}.profiler_candidate_full.top_cuda_operators",
    )
    fragmentation = _require_dict(
        profiler.get("fragmentation_operator_calls"),
        f"batch{batch}.profiler_candidate_full.fragmentation_operator_calls",
    )
    peak_vram = _require_dict(row.get("peak_vram"), f"batch{batch}.peak_vram")
    physical_sparse = _require_dict(row.get("physical_sparse"), f"batch{batch}.physical_sparse")
    routing_reference = _require_dict(
        row.get("routing_reference"), f"batch{batch}.routing_reference"
    )
    routing_candidate = _require_dict(
        row.get("routing_candidate"), f"batch{batch}.routing_candidate"
    )

    return {
        "batch": int(batch),
        "gate": {
            "candidate_full_vs_transformer_speed_ratio": row.get("candidate_full_vs_transformer_speed_ratio"),
            "required_full_speed_ratio": row.get("required_full_speed_ratio"),
            "throughput_pass": row.get("throughput_pass"),
            "candidate_full_latency_ms": row.get("candidate_full_latency_ms"),
            "reference_full_latency_ms": row.get("reference_full_latency_ms"),
            "candidate_vs_reference_latency_ratio": row.get("candidate_vs_reference_latency_ratio"),
        },
        "timings": canonical_timings,
        "candidate_profiler": {
            "top_cuda_operators": top_ops,
            "fragmentation_operator_calls": fragmentation,
        },
        "peak_vram": peak_vram,
        "physical_sparse_counters": physical_sparse,
        "routing_accounting": {
            "reference": routing_reference,
            "candidate": routing_candidate,
        },
    }


def _inspect_payload(result: dict[str, Any]) -> dict[str, Any]:
    if result.get("decision") != SOURCE_DECISION:
        raise RuntimeError(f"issue661 source decision drift: {result.get('decision')!r}")
    if result.get("overall_pass") is not SOURCE_OVERALL_PASS:
        raise RuntimeError(f"issue661 source overall_pass drift: {result.get('overall_pass')!r}")
    if result.get("device") != SOURCE_DEVICE:
        raise RuntimeError(f"issue661 source device drift: {result.get('device')!r}")

    candidate_names = _require_list(result.get("candidate_backend_names"), "candidate_backend_names")
    if len(candidate_names) != 4 or any(name != SOURCE_CANDIDATE_BACKEND for name in candidate_names):
        raise RuntimeError("issue661 source candidate backend identity drift")

    rows = _require_dict(result.get("rows"), "rows")
    if set(rows) != {"8", "64"}:
        raise RuntimeError(f"issue661 source batch inventory drift: {sorted(rows)!r}")

    extracted: dict[str, Any] = {}
    for batch in ("8", "64"):
        row = _require_dict(rows.get(batch), f"batch{batch}")
        observed = _frozen_observed(row)
        if observed != FROZEN_ROWS[batch]:
            raise RuntimeError(
                f"issue661 frozen batch{batch} evidence drift: observed={observed!r} expected={FROZEN_ROWS[batch]!r}"
            )
        extracted[batch] = _extract_batch(batch, row)

    return {
        "research_issue": RESEARCH_ISSUE,
        "source_research_issue": SOURCE_RESEARCH_ISSUE,
        "inspection_kind": "read_only_existing_immutable_issue650_throughput_evidence_schema_successor",
        "source_main": SOURCE_MAIN,
        "source_tree": SOURCE_TREE,
        "source_trigger": SOURCE_TRIGGER,
        "source_run": SOURCE_RUN,
        "source_job": SOURCE_JOB,
        "source_attempt": SOURCE_ATTEMPT,
        "predecessor_trigger": PREDECESSOR_TRIGGER,
        "predecessor_run": PREDECESSOR_RUN,
        "predecessor_job": PREDECESSOR_JOB,
        "source_result_path": SOURCE_RESULT_PATH,
        "source_result_sha256": SOURCE_RESULT_SHA256,
        "source_decision": result.get("decision"),
        "source_overall_pass": result.get("overall_pass"),
        "source_device": result.get("device"),
        "candidate_backend_names": candidate_names,
        "timing_key_order_is_scientific_invariant": False,
        "timing_key_set_verified": True,
        "canonical_timing_display_order": list(EXPECTED_TIMING_CONDITIONS),
        "batches": extracted,
        "parameter_versions_unchanged": result.get("parameter_versions_unchanged"),
        "checkpoint_hashes_unchanged": result.get("checkpoint_hashes_unchanged"),
        "training_performed": result.get("training_performed"),
        "optimizer_created": result.get("optimizer_created"),
        "backward_performed": result.get("backward_performed"),
        "corpus_accessed": result.get("corpus_accessed"),
        "checkpoint_written": result.get("checkpoint_written"),
        "scientific_seed_consumed": result.get("scientific_seed_consumed"),
        "claims_stored": result.get("claims"),
        "issue643_adapter_metadata": result.get("issue643_adapter_metadata"),
        "issue650_continuation_metadata": result.get("issue650_continuation_metadata"),
        "volume_mutated": False,
        "gpu_used": False,
        "model_constructed": False,
        "checkpoint_read": False,
        "experiment_rerun": False,
        "new_benchmark_performed": False,
        "threshold_changed": False,
        "optimization_authorized": False,
        "systems_pass_earned": False,
        "architecture_freeze_authorized": False,
        "s2_authorized": False,
        "fresh_scientific_seed_authorized": False,
        "independent_replication_credit": False,
        "100m_authorized": False,
        "breakthrough_proven": False,
    }


@app.function(image=image, cpu=1, memory=512, timeout=120, volumes={"/vol": volume})
def inspect_existing_issue650_throughput_evidence_schema_successor() -> dict[str, Any]:
    volume.reload()
    path = Path(SOURCE_RESULT_PATH)
    if not path.is_file():
        raise RuntimeError(f"frozen issue650 durable result is missing: {SOURCE_RESULT_PATH}")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SOURCE_RESULT_SHA256:
        raise RuntimeError(
            f"issue661 source SHA drift: observed={digest} expected={SOURCE_RESULT_SHA256}"
        )
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise RuntimeError("issue661 source result is not a JSON object")
    inspection = _inspect_payload(parsed)
    inspection["verified_source_bytes"] = len(raw)
    return inspection


@app.local_entrypoint()
def main() -> None:
    result = inspect_existing_issue650_throughput_evidence_schema_successor.remote()
    print(RESULT_MARKER + json.dumps(result, sort_keys=True, separators=(",", ":")), flush=True)
