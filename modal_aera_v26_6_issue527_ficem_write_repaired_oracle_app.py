from __future__ import annotations

import hashlib
import json
from pathlib import Path

import modal

RESEARCH_ISSUE = 527
SOURCE_MAIN = "e18aa12f1ddd96ba30f1b3f5e2be67d5f0922116"
APP_NAME = "tam-research-aera-v26-6-issue527-ficem-write-repaired-oracle"
VOLUME_NAME = "tam-research-data"
RESULT_PATH = "/vol/aera-v26/issue527-ficem-write-repaired-oracle/result.json"
FROZEN_ISSUE519_RESULT_PATH = "/vol/aera-v26/issue519-ficem-write-materialize-cast/result.json"
MAX_GPU_SECONDS = 300

CANDIDATE_BLOB = "d45c262314a0b4691f26812a279937a225043ad9"
FROZEN_ISSUE519_PROBE_BLOB = "ec22807434192f58e292bffc3de9828be2b44272"
FROZEN_ISSUE519_RESULT_SHA256 = (
    "b9fba0fca96644ef8db9bc46faf2c73d0c0cc1f1aaac6a321abe2411d3703cd5"
)
ORACLE_BLOB = "8f472451af4024bb3faacb56d814f7d6bdb25cc9"
ORACLE_CPU_TEST_BLOB = "de3ae08b9db04803359d216f601d5c68dac3a542"
PROBE_BLOB = "bcfeb6a93ed062b7d00359603dc9fbc7aca5767f"
FROZEN_ISSUE519_RUN = 33672232063
FROZEN_ISSUE519_JOB = 100388368044
FROZEN_ISSUE522_RUN = 33675476637
FROZEN_ISSUE522_JOB = 100398984660
ORACLE_CPU_HEAD = "275fed17e0e0c855a8f9a5fd39bfa484e1b78ed2"
ORACLE_CPU_RUN = 33676365160
ORACLE_CPU_JOB = 100401938039
ORACLE_MERGE_MAIN = SOURCE_MAIN

PREFLIGHT_MARKER = "AERA_V26_6_ISSUE527_FICEM_WRITE_REPAIRED_ORACLE_PREFLIGHT_JSON="
L4_START_MARKER = "AERA_V26_6_ISSUE527_FICEM_WRITE_REPAIRED_ORACLE_L4_START_JSON="
RESULT_MARKER = "AERA_V26_6_ISSUE527_FICEM_WRITE_REPAIRED_ORACLE_RESULT_JSON="
SUMMARY_MARKER = "AERA_V26_6_ISSUE527_FICEM_WRITE_REPAIRED_ORACLE_SUMMARY_JSON="

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
    import tam_research.aera_hardware_core_v26_6_ficem_write_materialize_cast as candidate
    import tam_research.aera_v26_6_issue519_ficem_write_materialize_cast_probe as issue519_probe
    import tam_research.aera_v26_6_issue525_mixed_dtype_write_oracle as oracle
    import tam_research.aera_v26_6_issue527_ficem_write_repaired_oracle_probe as probe

    volume.reload()
    result_path = Path(RESULT_PATH)
    if result_path.exists():
        raise RuntimeError(
            f"refusing duplicate issue527 repaired-oracle WRITE run because result exists: {RESULT_PATH}"
        )

    frozen_path = Path(FROZEN_ISSUE519_RESULT_PATH)
    if not frozen_path.is_file():
        raise RuntimeError(f"frozen issue519 durable result missing: {FROZEN_ISSUE519_RESULT_PATH}")
    frozen_bytes = frozen_path.read_bytes()
    frozen_sha = hashlib.sha256(frozen_bytes).hexdigest()
    if frozen_sha != FROZEN_ISSUE519_RESULT_SHA256:
        raise RuntimeError(
            f"issue519 durable SHA drift: got={frozen_sha} expected={FROZEN_ISSUE519_RESULT_SHA256}"
        )

    blobs = {
        "candidate": _git_blob_sha(Path(candidate.__file__)),
        "issue519_probe": _git_blob_sha(Path(issue519_probe.__file__)),
        "oracle": _git_blob_sha(Path(oracle.__file__)),
        "probe": _git_blob_sha(Path(probe.__file__)),
    }
    expected = {
        "candidate": CANDIDATE_BLOB,
        "issue519_probe": FROZEN_ISSUE519_PROBE_BLOB,
        "oracle": ORACLE_BLOB,
        "probe": PROBE_BLOB,
    }
    if blobs != expected:
        raise RuntimeError(f"issue527 frozen blob drift: got={blobs} expected={expected}")

    candidate_protocol = candidate.materialize_cast_ficem_read_write_v26_6_protocol()
    if candidate_protocol.get("write_materialization_output_follows_durable_state_field_dtype") is not True:
        raise RuntimeError("issue527 candidate protocol lost durable field output semantics")
    if candidate_protocol.get("write_tail_triton_launches_target") != 2:
        raise RuntimeError("issue527 candidate protocol lost two-launch target")

    oracle_protocol = oracle.issue525_oracle_protocol()
    required_oracle = {
        "research_issue": 525,
        "source_main": "6d5cfddd7b5b9359fb6e7e31c2da3f14c65203f3",
        "source_issue519_result_sha256": FROZEN_ISSUE519_RESULT_SHA256,
        "production_backend_changed": False,
        "gpu_authorized": False,
        "scientific_seed_authorized": False,
        "end_to_end_systems_authorized": False,
        "100m_authorized": False,
        "breakthrough_proven": False,
    }
    for key, expected_value in required_oracle.items():
        if oracle_protocol.get(key) != expected_value:
            raise RuntimeError(
                f"issue527 oracle protocol drift: {key}={oracle_protocol.get(key)!r} expected={expected_value!r}"
            )

    contract = probe.cpu_contract_preflight()
    gate_protocol = contract["protocol"]
    required_gate = {
        "gate_research_issue": 527,
        "source_main": SOURCE_MAIN,
        "candidate_blob": CANDIDATE_BLOB,
        "frozen_issue519_probe_blob": FROZEN_ISSUE519_PROBE_BLOB,
        "frozen_issue519_result_sha256": FROZEN_ISSUE519_RESULT_SHA256,
        "oracle_blob": ORACLE_BLOB,
        "oracle_cpu_test_blob": ORACLE_CPU_TEST_BLOB,
        "oracle_cpu_head": ORACLE_CPU_HEAD,
        "oracle_cpu_run": ORACLE_CPU_RUN,
        "oracle_cpu_job": ORACLE_CPU_JOB,
        "oracle_merge_main": ORACLE_MERGE_MAIN,
        "direct_edge_oracle": "issue525_durable_mixed_dtype_reference_tail",
        "public_reference_reused_from_issue519": True,
        "topology_contract_reused_from_issue519": True,
        "decision_surface_reused_from_issue519": True,
        "design_seed": 408514,
        "design_seed_is_scientific_seed": False,
        "matrix_case_count": 256,
        "edge_case_count": 32,
        "public_row_count": 6,
        "topology_row_count": 4,
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
                f"issue527 probe protocol drift: {key}={gate_protocol.get(key)!r} expected={expected_value!r}"
            )

    return {
        "research_issue": RESEARCH_ISSUE,
        "source_main": SOURCE_MAIN,
        "blobs": blobs,
        "contract": contract,
        "frozen_issue519_result_path": FROZEN_ISSUE519_RESULT_PATH,
        "frozen_issue519_result_sha256": frozen_sha,
        "frozen_issue519_run": FROZEN_ISSUE519_RUN,
        "frozen_issue519_job": FROZEN_ISSUE519_JOB,
        "frozen_issue522_run": FROZEN_ISSUE522_RUN,
        "frozen_issue522_job": FROZEN_ISSUE522_JOB,
        "oracle_cpu_run": ORACLE_CPU_RUN,
        "oracle_cpu_job": ORACLE_CPU_JOB,
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
    from tam_research.aera_v26_6_issue527_ficem_write_repaired_oracle_probe import (
        run_repaired_oracle_write_probe,
    )

    volume.reload()
    result_path = Path(RESULT_PATH)
    if result_path.exists():
        raise RuntimeError(
            f"refusing duplicate issue527 repaired-oracle WRITE run because result exists: {RESULT_PATH}"
        )

    print(
        L4_START_MARKER
        + json.dumps(
            {
                "device": torch.cuda.get_device_name(0),
                "research_issue": RESEARCH_ISSUE,
                "source_main": SOURCE_MAIN,
                "candidate_blob": CANDIDATE_BLOB,
                "oracle_blob": ORACLE_BLOB,
                "probe_blob": PROBE_BLOB,
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

    result = run_repaired_oracle_write_probe()
    result["issue527_gate_metadata"] = {
        "source_main": SOURCE_MAIN,
        "candidate_blob": CANDIDATE_BLOB,
        "issue519_probe_blob": FROZEN_ISSUE519_PROBE_BLOB,
        "issue519_result_sha256": FROZEN_ISSUE519_RESULT_SHA256,
        "oracle_blob": ORACLE_BLOB,
        "oracle_cpu_test_blob": ORACLE_CPU_TEST_BLOB,
        "probe_blob": PROBE_BLOB,
        "oracle_cpu_head": ORACLE_CPU_HEAD,
        "oracle_cpu_run": ORACLE_CPU_RUN,
        "oracle_cpu_job": ORACLE_CPU_JOB,
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
        "oracle_blob": ORACLE_BLOB,
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
