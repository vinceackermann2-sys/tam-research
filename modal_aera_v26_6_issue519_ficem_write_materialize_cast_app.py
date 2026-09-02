from __future__ import annotations

import hashlib
import json
from pathlib import Path

import modal

RESEARCH_ISSUE = 519
SOURCE_MAIN = "cc9f401d7d3b5ed5c75dc8905ffc8f12df32616b"
APP_NAME = "tam-research-aera-v26-6-issue519-ficem-write-materialize-cast"
VOLUME_NAME = "tam-research-data"
RESULT_PATH = "/vol/aera-v26/issue519-ficem-write-materialize-cast/result.json"
MAX_GPU_SECONDS = 300

CANDIDATE_BLOB = "d45c262314a0b4691f26812a279937a225043ad9"
PREDECESSOR_BLOB = "e54570292489bd17570038dca7518419ac00418c"
FAILED_V26_5_BLOB = "dab24c733eff7aa08e5f818614f7504eaac48dc3"
READ_BACKEND_BLOB = "263f68eb1186a8ac14a08fc4b4df1fc5b292c711"
V26_INTERFACE_BLOB = "268644ac4edee15a4cc4e29d3fed7f61eeb3caa7"
STABLE_REFERENCE_BLOB = "4e336b6e1a6238dac782fa320751d68281493ee1"
HISTORICAL_PROBE_BLOB = "7d8c2c4990beb4c7b4a719d02d009ffefe94671f"
FROZEN_ISSUE514_PROBE_BLOB = "e48dc308bef3b4ef9b6590ab0324db4a50c0f7af"
FROZEN_ISSUE514_LAUNCHER_BLOB = "1ab50f7b184feda61a6f6e1c7553296bed8863a6"
FROZEN_ISSUE514_WORKFLOW_BLOB = "5871b0a12e6168f16b59a1e7f1895feea6e8426c"
PROBE_BLOB = "ec22807434192f58e292bffc3de9828be2b44272"
FROZEN_ISSUE514_RESULT_SHA256 = "c1a8936458c57e975787a27288d3caf494e360ec8ae8acb8d0f5742aef6bf505"
FROZEN_ISSUE514_RUN = 33664645415
FROZEN_ISSUE514_JOB = 100363263710
REPAIR_ISSUE = 517
REPAIR_PR = 518
REPAIR_CPU_HEAD = "c2edcfeb28efebe3818a92c5e00d53ea72689c75"
REPAIR_CPU_RUN = 33668780903
REPAIR_CPU_JOB = 100376942271
REPAIR_MERGE_MAIN = SOURCE_MAIN

PREFLIGHT_MARKER = "AERA_V26_6_ISSUE519_FICEM_WRITE_MATERIALIZE_CAST_PREFLIGHT_JSON="
L4_START_MARKER = "AERA_V26_6_ISSUE519_FICEM_WRITE_MATERIALIZE_CAST_L4_START_JSON="
RESULT_MARKER = "AERA_V26_6_ISSUE519_FICEM_WRITE_MATERIALIZE_CAST_RESULT_JSON="
SUMMARY_MARKER = "AERA_V26_6_ISSUE519_FICEM_WRITE_MATERIALIZE_CAST_SUMMARY_JSON="

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.10,<2.11", "numpy>=2.0,<3")
    .add_local_python_source("tam_research")
)


def _git_blob_sha(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def _summary(result: dict) -> dict:
    direct_failed = [
        key for key, row in result["direct_matrix"].items() if not row.get("pass", False)
    ]
    edge_failed = [
        key for key, row in result["edge_cases"].items() if not row.get("pass", False)
    ]
    public_failed = [
        key for key, row in result["public_rows"].items() if not row.get("pass", False)
    ]
    topology_failed = [
        key for key, row in result["topology_rows"].items() if not row.get("pass", False)
    ]
    return {
        "decision": result["decision"],
        "overall_pass": result["overall_pass"],
        "device": result["device"],
        "direct_matrix_case_count": result["direct_matrix_case_count"],
        "direct_matrix_pass": result["direct_matrix_pass"],
        "direct_failed": direct_failed,
        "edge_case_count": result["edge_case_count"],
        "edge_cases_pass": result["edge_cases_pass"],
        "edge_failed": edge_failed,
        "public_row_count": result["public_row_count"],
        "public_rows_pass": result["public_rows_pass"],
        "public_failed": public_failed,
        "topology_row_count": result["topology_row_count"],
        "topology_pass": result["topology_pass"],
        "topology_failed": topology_failed,
        "claims": result["claims"],
    }


@app.function(image=image, cpu=4, memory=8192, timeout=300, volumes={"/vol": volume})
def preflight() -> dict:
    import tam_research.aera_hardware_core_v25_1_compact as stable
    import tam_research.aera_hardware_core_v26 as v26
    import tam_research.aera_hardware_core_v26_3_ficem_read_triton as read_backend
    import tam_research.aera_hardware_core_v26_4_ficem_write_triton as predecessor
    import tam_research.aera_hardware_core_v26_5_ficem_write_mixed_dtype as failed_v26_5
    import tam_research.aera_hardware_core_v26_6_ficem_write_materialize_cast as candidate
    import tam_research.aera_v26_4_ficem_write_probe as historical_probe
    import tam_research.aera_v26_5_issue514_ficem_write_mixed_dtype_probe as frozen_issue514_probe
    import tam_research.aera_v26_6_issue519_ficem_write_materialize_cast_probe as probe

    volume.reload()
    result_path = Path(RESULT_PATH)
    if result_path.exists():
        raise RuntimeError(
            f"refusing duplicate issue519 materialize-cast WRITE run because result exists: {RESULT_PATH}"
        )

    blobs = {
        "candidate": _git_blob_sha(Path(candidate.__file__)),
        "predecessor": _git_blob_sha(Path(predecessor.__file__)),
        "failed_v26_5": _git_blob_sha(Path(failed_v26_5.__file__)),
        "read_backend": _git_blob_sha(Path(read_backend.__file__)),
        "v26_interface": _git_blob_sha(Path(v26.__file__)),
        "stable_reference": _git_blob_sha(Path(stable.__file__)),
        "historical_probe": _git_blob_sha(Path(historical_probe.__file__)),
        "frozen_issue514_probe": _git_blob_sha(Path(frozen_issue514_probe.__file__)),
        "probe": _git_blob_sha(Path(probe.__file__)),
    }
    expected = {
        "candidate": CANDIDATE_BLOB,
        "predecessor": PREDECESSOR_BLOB,
        "failed_v26_5": FAILED_V26_5_BLOB,
        "read_backend": READ_BACKEND_BLOB,
        "v26_interface": V26_INTERFACE_BLOB,
        "stable_reference": STABLE_REFERENCE_BLOB,
        "historical_probe": HISTORICAL_PROBE_BLOB,
        "frozen_issue514_probe": FROZEN_ISSUE514_PROBE_BLOB,
        "probe": PROBE_BLOB,
    }
    if blobs != expected:
        raise RuntimeError(f"issue519 frozen blob drift: got={blobs} expected={expected}")

    candidate_protocol = candidate.materialize_cast_ficem_read_write_v26_6_protocol()
    required_candidate = {
        "research_issue": 517,
        "predecessor_write_backend_blob": PREDECESSOR_BLOB,
        "failed_v26_5_backend_blob": FAILED_V26_5_BLOB,
        "historical_v26_4_backend_mutated": False,
        "failed_v26_5_backend_mutated": False,
        "write_global_cross_field_dtype_equality_required": False,
        "write_supported_float_dtypes": ["float32", "bfloat16"],
        "write_fieldwise_mixed_dtype_supported": True,
        "write_materialization_output_follows_durable_state_field_dtype": True,
        "write_duplicate_decisions_before_materialization": True,
        "write_materialize_both_branches_cast_to_output_element_type": True,
        "write_materialize_cast_numeric_not_bitcast": True,
        "write_explicit_pre_tail_cast_kernels": 0,
        "write_new_triton_kernels": 1,
        "write_adjudicate_kernel_changed_by_v26_6": False,
        "write_materialize_kernel_versioned_by_v26_6": True,
        "write_tail_triton_launches_target": 2,
        "mixed_dtype_gpu_gate_authorized": False,
        "end_to_end_systems_authorized": False,
        "architecture_freeze_authorized": False,
        "fresh_scientific_seed_authorized": False,
        "independent_replication_credit": False,
        "100m_authorized": False,
        "breakthrough_proven": False,
    }
    for key, expected_value in required_candidate.items():
        if candidate_protocol.get(key) != expected_value:
            raise RuntimeError(
                f"issue519 candidate protocol drift: {key}={candidate_protocol.get(key)!r} expected={expected_value!r}"
            )

    contract = probe.cpu_contract_preflight()
    gate_protocol = contract["protocol"]
    required_gate = {
        "gate_research_issue": 519,
        "source_main": SOURCE_MAIN,
        "candidate_blob": CANDIDATE_BLOB,
        "frozen_issue514_probe_blob": FROZEN_ISSUE514_PROBE_BLOB,
        "frozen_issue514_result_sha256": FROZEN_ISSUE514_RESULT_SHA256,
        "design_seed": 408514,
        "design_seed_is_scientific_seed": False,
        "decision_surface_reused_from_issue514": True,
        "matrix_case_count": 256,
        "edge_case_count": 32,
        "public_row_count": 6,
        "topology_row_count": 4,
        "expected_adjudicate_kernel": "_write_adjudicate_map_kernel",
        "expected_materialize_kernel": "_write_materialize_cast_kernel",
        "performance_threshold_added": False,
        "synthetic_only": True,
        "scientific_seed_consumed": False,
        "end_to_end_systems_authorized": False,
        "architecture_freeze_authorized": False,
        "fresh_scientific_seed_authorized": False,
        "independent_replication_credit": False,
        "100m_authorized": False,
        "breakthrough_proven": False,
    }
    for key, expected_value in required_gate.items():
        if gate_protocol.get(key) != expected_value:
            raise RuntimeError(
                f"issue519 probe protocol drift: {key}={gate_protocol.get(key)!r} expected={expected_value!r}"
            )

    return {
        "research_issue": RESEARCH_ISSUE,
        "source_main": SOURCE_MAIN,
        "blobs": blobs,
        "contract": contract,
        "frozen_issue514_result_sha256": FROZEN_ISSUE514_RESULT_SHA256,
        "frozen_issue514_run": FROZEN_ISSUE514_RUN,
        "frozen_issue514_job": FROZEN_ISSUE514_JOB,
        "repair_issue": REPAIR_ISSUE,
        "repair_pr": REPAIR_PR,
        "repair_cpu_head": REPAIR_CPU_HEAD,
        "repair_cpu_run": REPAIR_CPU_RUN,
        "repair_cpu_job": REPAIR_CPU_JOB,
        "repair_merge_main": REPAIR_MERGE_MAIN,
        "result_path_absent": True,
        "synthetic_only": True,
        "scientific_seed_consumed": False,
        "end_to_end_systems_authorized": False,
        "architecture_freeze_authorized": False,
        "100m_authorized": False,
        "breakthrough_proven": False,
    }


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=16384,
    timeout=MAX_GPU_SECONDS,
    volumes={"/vol": volume},
)
def run_probe() -> dict:
    import torch
    from tam_research.aera_v26_6_issue519_ficem_write_materialize_cast_probe import (
        run_materialize_cast_write_probe,
    )

    volume.reload()
    result_path = Path(RESULT_PATH)
    if result_path.exists():
        raise RuntimeError(
            f"refusing duplicate issue519 materialize-cast WRITE run because result exists: {RESULT_PATH}"
        )

    print(
        L4_START_MARKER
        + json.dumps(
            {
                "device": torch.cuda.get_device_name(0),
                "research_issue": RESEARCH_ISSUE,
                "source_main": SOURCE_MAIN,
                "candidate_blob": CANDIDATE_BLOB,
                "probe_blob": PROBE_BLOB,
                "frozen_issue514_result_sha256": FROZEN_ISSUE514_RESULT_SHA256,
                "design_seed": 408514,
                "synthetic_only": True,
                "scientific_seed_consumed": False,
                "end_to_end_systems_authorized": False,
                "architecture_freeze_authorized": False,
                "100m_authorized": False,
                "breakthrough_proven": False,
            },
            separators=(",", ":"),
        ),
        flush=True,
    )

    result = run_materialize_cast_write_probe()
    result["issue519_gate_metadata"] = {
        "source_main": SOURCE_MAIN,
        "candidate_blob": CANDIDATE_BLOB,
        "predecessor_blob": PREDECESSOR_BLOB,
        "failed_v26_5_blob": FAILED_V26_5_BLOB,
        "read_backend_blob": READ_BACKEND_BLOB,
        "v26_interface_blob": V26_INTERFACE_BLOB,
        "stable_reference_blob": STABLE_REFERENCE_BLOB,
        "historical_probe_blob": HISTORICAL_PROBE_BLOB,
        "frozen_issue514_probe_blob": FROZEN_ISSUE514_PROBE_BLOB,
        "probe_blob": PROBE_BLOB,
        "frozen_issue514_result_sha256": FROZEN_ISSUE514_RESULT_SHA256,
        "frozen_issue514_run": FROZEN_ISSUE514_RUN,
        "repair_cpu_run": REPAIR_CPU_RUN,
        "repair_merge_main": REPAIR_MERGE_MAIN,
        "synthetic_only": True,
        "scientific_seed_consumed": False,
    }
    durable_json = json.dumps(result, sort_keys=True, separators=(",", ":"))
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(durable_json)
    volume.commit()
    digest = hashlib.sha256(durable_json.encode()).hexdigest()

    marker = {
        "research_issue": RESEARCH_ISSUE,
        "result_path": RESULT_PATH,
        "result_sha256": digest,
        "decision": result["decision"],
        "overall_pass": result["overall_pass"],
        "device": result["device"],
        "source_main": SOURCE_MAIN,
        "candidate_blob": CANDIDATE_BLOB,
        "probe_blob": PROBE_BLOB,
    }
    print(RESULT_MARKER + json.dumps(marker, separators=(",", ":")), flush=True)
    print(SUMMARY_MARKER + json.dumps(_summary(result), separators=(",", ":")), flush=True)
    return result


@app.local_entrypoint()
def main() -> None:
    check = preflight.remote()
    print(PREFLIGHT_MARKER + json.dumps(check, separators=(",", ":")), flush=True)
    result = run_probe.remote()
    print(json.dumps(_summary(result), indent=2), flush=True)
