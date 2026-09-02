from __future__ import annotations

import hashlib
import json
from pathlib import Path

import modal

RESEARCH_ISSUE = 514
SOURCE_MAIN = "d9bda2bd3143308407c0d11e640d984385eb095a"
APP_NAME = "tam-research-aera-v26-5-issue514-ficem-write-mixed-dtype"
VOLUME_NAME = "tam-research-data"
RESULT_PATH = "/vol/aera-v26/issue514-ficem-write-mixed-dtype/result.json"
MAX_GPU_SECONDS = 300

CANDIDATE_BLOB = "dab24c733eff7aa08e5f818614f7504eaac48dc3"
PREDECESSOR_BLOB = "e54570292489bd17570038dca7518419ac00418c"
READ_BACKEND_BLOB = "263f68eb1186a8ac14a08fc4b4df1fc5b292c711"
V26_INTERFACE_BLOB = "268644ac4edee15a4cc4e29d3fed7f61eeb3caa7"
STABLE_REFERENCE_BLOB = "4e336b6e1a6238dac782fa320751d68281493ee1"
HISTORICAL_PROBE_BLOB = "7d8c2c4990beb4c7b4a719d02d009ffefe94671f"
PROBE_BLOB = "e48dc308bef3b4ef9b6590ab0324db4a50c0f7af"
EXHAUSTED_508_LAUNCHER_BLOB = "5597dbbd79c782420d48ed538ef2669aebfe5fae"
EXHAUSTED_508_WORKFLOW_BLOB = "556ea59ebc2d95272caa774a9fef62efbf66a302"
EXHAUSTED_508_RUN = 33661498305
EXHAUSTED_508_JOB = 100352870198
REPAIR_ISSUE = 511
REPAIR_PR = 513
REPAIR_CPU_HEAD = "2268dd022a3bfcb0eda8ab65a9a6b78874231650"
REPAIR_CPU_RUN = 33662720255
REPAIR_CPU_JOB = 100356904904
REPAIR_MERGE_MAIN = "d9bda2bd3143308407c0d11e640d984385eb095a"

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
    import tam_research.aera_hardware_core_v26_5_ficem_write_mixed_dtype as candidate
    import tam_research.aera_v26_4_ficem_write_probe as historical_probe
    import tam_research.aera_v26_5_issue514_ficem_write_mixed_dtype_probe as probe

    volume.reload()
    result_path = Path(RESULT_PATH)
    if result_path.exists():
        raise RuntimeError(
            f"refusing duplicate issue514 mixed-dtype WRITE run because result exists: {RESULT_PATH}"
        )

    blobs = {
        "candidate": _git_blob_sha(Path(candidate.__file__)),
        "predecessor": _git_blob_sha(Path(predecessor.__file__)),
        "read_backend": _git_blob_sha(Path(read_backend.__file__)),
        "v26_interface": _git_blob_sha(Path(v26.__file__)),
        "stable_reference": _git_blob_sha(Path(stable.__file__)),
        "historical_probe": _git_blob_sha(Path(historical_probe.__file__)),
        "probe": _git_blob_sha(Path(probe.__file__)),
    }
    expected = {
        "candidate": CANDIDATE_BLOB,
        "predecessor": PREDECESSOR_BLOB,
        "read_backend": READ_BACKEND_BLOB,
        "v26_interface": V26_INTERFACE_BLOB,
        "stable_reference": STABLE_REFERENCE_BLOB,
        "historical_probe": HISTORICAL_PROBE_BLOB,
        "probe": PROBE_BLOB,
    }
    if blobs != expected:
        raise RuntimeError(f"issue514 frozen blob drift: got={blobs} expected={expected}")

    protocol = candidate.mixed_dtype_ficem_read_write_v26_5_protocol()
    required_candidate = {
        "research_issue": 511,
        "predecessor_write_backend_blob": PREDECESSOR_BLOB,
        "historical_v26_4_backend_mutated": False,
        "write_global_cross_field_dtype_equality_required": False,
        "write_supported_float_dtypes": ["float32", "bfloat16"],
        "write_fieldwise_mixed_dtype_supported": True,
        "write_materialization_output_follows_durable_state_field_dtype": True,
        "write_duplicate_decisions_before_materialization": True,
        "write_explicit_pre_tail_cast_kernels": 0,
        "write_triton_kernel_bodies_changed": False,
        "write_tail_triton_launches_target": 2,
        "mixed_dtype_gpu_gate_authorized": False,
        "end_to_end_systems_authorized": False,
        "architecture_freeze_authorized": False,
        "fresh_scientific_seed_authorized": False,
        "100m_authorized": False,
        "breakthrough_proven": False,
    }
    for key, expected_value in required_candidate.items():
        if protocol.get(key) != expected_value:
            raise RuntimeError(
                f"issue514 candidate protocol drift: {key}={protocol.get(key)!r} expected={expected_value!r}"
            )

    contract = probe.cpu_contract_preflight()
    gate_protocol = contract["protocol"]
    required_gate = {
        "gate_research_issue": 514,
        "design_seed": 408514,
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
        "100m_authorized": False,
        "breakthrough_proven": False,
    }
    for key, expected_value in required_gate.items():
        if gate_protocol.get(key) != expected_value:
            raise RuntimeError(
                f"issue514 probe protocol drift: {key}={gate_protocol.get(key)!r} expected={expected_value!r}"
            )

    return {
        "research_issue": RESEARCH_ISSUE,
        "source_main": SOURCE_MAIN,
        "blobs": blobs,
        "contract": contract,
        "repair_issue": REPAIR_ISSUE,
        "repair_pr": REPAIR_PR,
        "repair_cpu_head": REPAIR_CPU_HEAD,
        "repair_cpu_run": REPAIR_CPU_RUN,
        "repair_cpu_job": REPAIR_CPU_JOB,
        "repair_merge_main": REPAIR_MERGE_MAIN,
        "exhausted_508_run": EXHAUSTED_508_RUN,
        "exhausted_508_job": EXHAUSTED_508_JOB,
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
    from tam_research.aera_v26_5_issue514_ficem_write_mixed_dtype_probe import (
        run_mixed_dtype_write_probe,
    )

    volume.reload()
    result_path = Path(RESULT_PATH)
    if result_path.exists():
        raise RuntimeError(
            f"refusing duplicate issue514 mixed-dtype WRITE run because result exists: {RESULT_PATH}"
        )

    print(
        "AERA_V26_5_ISSUE514_FICEM_WRITE_MIXED_DTYPE_L4_START_JSON="
        + json.dumps(
            {
                "device": torch.cuda.get_device_name(0),
                "research_issue": RESEARCH_ISSUE,
                "source_main": SOURCE_MAIN,
                "candidate_blob": CANDIDATE_BLOB,
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

    result = run_mixed_dtype_write_probe()
    result["issue514_gate_metadata"] = {
        "source_main": SOURCE_MAIN,
        "candidate_blob": CANDIDATE_BLOB,
        "predecessor_blob": PREDECESSOR_BLOB,
        "read_backend_blob": READ_BACKEND_BLOB,
        "v26_interface_blob": V26_INTERFACE_BLOB,
        "stable_reference_blob": STABLE_REFERENCE_BLOB,
        "historical_probe_blob": HISTORICAL_PROBE_BLOB,
        "probe_blob": PROBE_BLOB,
        "repair_cpu_run": REPAIR_CPU_RUN,
        "exhausted_508_run": EXHAUSTED_508_RUN,
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
    print(
        "AERA_V26_5_ISSUE514_FICEM_WRITE_MIXED_DTYPE_RESULT_JSON="
        + json.dumps(marker, separators=(",", ":")),
        flush=True,
    )
    print(
        "AERA_V26_5_ISSUE514_FICEM_WRITE_MIXED_DTYPE_SUMMARY_JSON="
        + json.dumps(_summary(result), separators=(",", ":")),
        flush=True,
    )
    return result


@app.local_entrypoint()
def main() -> None:
    check = preflight.remote()
    print(
        "AERA_V26_5_ISSUE514_FICEM_WRITE_MIXED_DTYPE_PREFLIGHT_JSON="
        + json.dumps(check, separators=(",", ":")),
        flush=True,
    )
    result = run_probe.remote()
    print(json.dumps(_summary(result), indent=2), flush=True)
