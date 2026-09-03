from __future__ import annotations

import hashlib
import json
from pathlib import Path

import modal

APP_NAME = "tam-research-aera-v26-7-issue553-ficem-read-mixed-dtype-l4"
VOLUME_NAME = "tam-research-data"
RESULT_PATH = "/vol/aera-v26/issue553-ficem-read-mixed-dtype/result.json"
MAX_GPU_SECONDS = 300
RESEARCH_ISSUE = 553
SOURCE_MAIN = "89ef42e447fd797146a45cf4ea869e3012542761"
SOURCE_TREE = "c9890b3985976265e68d785ed1ba9854b89eb8a1"
PROBE_BLOB = "ff9a47f510be07e8adeff018f327338147163cdb"
V26_7_BACKEND_BLOB = "d8133c6b204b1ee5f23955255fb2fb09d09bd723"
REPAIR5_BACKEND_BLOB = "263f68eb1186a8ac14a08fc4b4df1fc5b292c711"
V26_6_WRITE_BACKEND_BLOB = "d45c262314a0b4691f26812a279937a225043ad9"
HISTORICAL_PROBE_BLOB = "16aa99b9f6f0a1d11bd7bf5f36b2f0b1fb97047b"
REPAIR5_PROBE_BLOB = "6fd6518e10ed1ef4115863f98ac591ffd77ce903"
ISSUE530_SYSTEMS_BLOB = "9d5a3c31f4a3862f96b957540baa2e0ec6a84c6b"
V26_INTERFACE_BLOB = "268644ac4edee15a4cc4e29d3fed7f61eeb3caa7"
STABLE_REFERENCE_BLOB = "4e336b6e1a6238dac782fa320751d68281493ee1"

ISSUE552_HEAD = "c7b74cac061f5d0233df38261c60d12d18126eed"
ISSUE552_CPU_RUN = 33722918693
ISSUE552_CPU_JOB = 100545652710
ISSUE479_TRIGGER = 484
ISSUE479_RUN = 33618950619
ISSUE479_JOB = 100211244996
ISSUE545_TRIGGER = 550
ISSUE545_RUN = 33686037672
ISSUE545_JOB = 100433658768
ISSUE545_FAILURE = "FICEM read-tail floating dtypes must match"

PRECHECK_MARKER = "AERA_V26_7_ISSUE553_FICEM_READ_MIXED_DTYPE_PRECHECK_JSON="
PREFLIGHT_MARKER = "AERA_V26_7_ISSUE553_FICEM_READ_MIXED_DTYPE_PREFLIGHT_JSON="
L4_START_MARKER = "AERA_V26_7_ISSUE553_FICEM_READ_MIXED_DTYPE_L4_START_JSON="
RESULT_MARKER = "AERA_V26_7_ISSUE553_FICEM_READ_MIXED_DTYPE_RESULT_JSON="
SUMMARY_MARKER = "AERA_V26_7_ISSUE553_FICEM_READ_MIXED_DTYPE_SUMMARY_JSON="

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


def precheck() -> dict:
    return {
        "research_issue": RESEARCH_ISSUE,
        "source_main": SOURCE_MAIN,
        "source_tree": SOURCE_TREE,
        "probe_blob": PROBE_BLOB,
        "v26_7_backend_blob": V26_7_BACKEND_BLOB,
        "result_path": RESULT_PATH,
        "max_gpu_seconds": MAX_GPU_SECONDS,
        "gpu": "L4",
        "gpu_remote_functions": 1,
        "scientific_seed_consumed": False,
        "end_to_end_systems_authorized": False,
        "architecture_freeze_authorized": False,
        "fresh_scientific_seed_authorized": False,
        "independent_replication_credit": False,
        "100m_authorized": False,
        "breakthrough_proven": False,
    }


def _summary(result: dict) -> dict:
    historical = result["historical"]
    mixed = result["mixed"]
    mixed_rows = {
        key: {
            "pass": row["pass"],
            "compute_dtype": row["compute_dtype"],
            "durable_dtype": row["durable_dtype"],
            "batch_size": row["batch_size"],
            "validity_kind": row["validity_kind"],
            "topology_pass": row["topology_pass"],
            "full_backend_required": row["full_backend"]["required"],
            "full_backend_exercised": row["full_backend"]["exercised"],
            "full_backend_pass": row["full_backend"]["pass"],
            "diagnostic_latency_ratio_candidate_over_reference": row[
                "diagnostic_latency_ratio_candidate_over_reference"
            ],
            "latency_decision_bearing": row["latency_decision_bearing"],
        }
        for key, row in mixed["rows"].items()
    }
    return {
        "research_issue": RESEARCH_ISSUE,
        "decision": result["decision"],
        "overall_pass": result["overall_pass"],
        "device": result["device"],
        "historical": {
            "decision": historical["decision"],
            "overall_pass": historical["overall_pass"],
            "correctness_pass": historical["correctness_pass"],
            "known_empty_pass": historical["known_empty_pass"],
            "near_tie_pass": historical["near_tie_pass"],
            "row_latency_pass": historical["row_latency_pass"],
            "full_event_ratio_pass": historical["full_event_ratio_pass"],
            "single_tail_kernel_pass": historical["single_tail_kernel_pass"],
            "candidate_no_reference_tail_ops_pass": historical[
                "candidate_no_reference_tail_ops_pass"
            ],
            "geomean_latency_ratio_by_dtype": historical[
                "geomean_latency_ratio_by_dtype"
            ],
            "geomean_latency_pass_by_dtype": historical[
                "geomean_latency_pass_by_dtype"
            ],
        },
        "mixed": {
            "overall_pass": mixed["overall_pass"],
            "rows_pass": mixed["rows_pass"],
            "near_tie_pass": mixed["near_tie_pass"],
            "known_empty_pass": mixed["known_empty_pass"],
            "timing_decision_bearing": mixed["timing_decision_bearing"],
            "rows": mixed_rows,
        },
        "scientific_seed_consumed": result["scientific_seed_consumed"],
        "end_to_end_systems_authorized": result["end_to_end_systems_authorized"],
        "architecture_freeze_authorized": result["architecture_freeze_authorized"],
        "fresh_scientific_seed_authorized": result[
            "fresh_scientific_seed_authorized"
        ],
        "independent_replication_credit": result["independent_replication_credit"],
        "100m_authorized": result["100m_authorized"],
        "breakthrough_proven": result["breakthrough_proven"],
    }


@app.function(image=image, cpu=4, memory=8192, timeout=300, volumes={"/vol": volume})
def preflight() -> dict:
    import tam_research.aera_hardware_core_v25_1_compact as stable
    import tam_research.aera_hardware_core_v26 as v26
    import tam_research.aera_hardware_core_v26_3_ficem_read_triton as repair5_backend
    import tam_research.aera_hardware_core_v26_6_ficem_write_materialize_cast as write_backend
    import tam_research.aera_hardware_core_v26_7_ficem_read_mixed_dtype as candidate
    import tam_research.aera_v26_3_ficem_read_probe as historical_probe
    import tam_research.aera_v26_3_ficem_read_probe_repair5 as repair5_probe
    import tam_research.aera_v26_6_issue530_end_to_end_systems as systems
    import tam_research.aera_v26_7_issue553_ficem_read_mixed_dtype_probe as probe

    volume.reload()
    result_path = Path(RESULT_PATH)
    if result_path.exists():
        raise RuntimeError(
            f"refusing duplicate issue553 run because durable result exists: {RESULT_PATH}"
        )

    blobs = {
        "probe": _git_blob_sha(Path(probe.__file__)),
        "v26_7_backend": _git_blob_sha(Path(candidate.__file__)),
        "repair5_backend": _git_blob_sha(Path(repair5_backend.__file__)),
        "v26_6_write_backend": _git_blob_sha(Path(write_backend.__file__)),
        "historical_probe": _git_blob_sha(Path(historical_probe.__file__)),
        "repair5_probe": _git_blob_sha(Path(repair5_probe.__file__)),
        "issue530_systems": _git_blob_sha(Path(systems.__file__)),
        "v26_interface": _git_blob_sha(Path(v26.__file__)),
        "stable_reference": _git_blob_sha(Path(stable.__file__)),
    }
    expected = {
        "probe": PROBE_BLOB,
        "v26_7_backend": V26_7_BACKEND_BLOB,
        "repair5_backend": REPAIR5_BACKEND_BLOB,
        "v26_6_write_backend": V26_6_WRITE_BACKEND_BLOB,
        "historical_probe": HISTORICAL_PROBE_BLOB,
        "repair5_probe": REPAIR5_PROBE_BLOB,
        "issue530_systems": ISSUE530_SYSTEMS_BLOB,
        "v26_interface": V26_INTERFACE_BLOB,
        "stable_reference": STABLE_REFERENCE_BLOB,
    }
    if blobs != expected:
        raise RuntimeError(f"issue553 frozen blob drift: got={blobs} expected={expected}")

    contract = probe.cpu_contract_preflight_issue553()
    if contract["gpu_authorized_by_cpu_preflight"] is not False:
        raise RuntimeError("issue553 CPU preflight unexpectedly authorizes GPU")
    protocol = probe.issue553_protocol()
    required = {
        "research_issue": RESEARCH_ISSUE,
        "source_main_issue553": SOURCE_MAIN,
        "v26_7_backend_blob": V26_7_BACKEND_BLOB,
        "historical_surface_preserved": True,
        "historical_surface_candidate_is_v26_7": True,
        "mixed_layouts": [["bfloat16", "float32"], ["float32", "bfloat16"]],
        "mixed_regular_generator_continues_historical_stream": True,
        "mixed_timing_decision_bearing": False,
        "historical_timing_decision_bearing": True,
        "integration_bf16_compute_fp32_durable_full_backend_required": True,
        "complementary_fp32_compute_bf16_durable_full_backend_required": False,
        "end_to_end_systems_authorized": False,
        "architecture_freeze_authorized": False,
        "fresh_scientific_seed_authorized": False,
        "independent_replication_credit": False,
        "100m_authorized": False,
        "breakthrough_proven": False,
    }
    for key, expected_value in required.items():
        if protocol.get(key) != expected_value:
            raise RuntimeError(
                f"issue553 probe protocol drift: {key}={protocol.get(key)!r} "
                f"expected={expected_value!r}"
            )

    candidate_protocol = candidate.mixed_dtype_ficem_read_write_v26_7_protocol()
    candidate_required = {
        "read_kernel_reused_from_repair5": True,
        "read_new_triton_kernels": 0,
        "read_tail_triton_launches_target": 1,
        "read_global_cross_field_dtype_equality_required": False,
        "read_strengths_values_dtype_equality_required": True,
        "read_supported_new_mixed_dtypes": ["float32", "bfloat16"],
        "read_host_pre_tail_cast_kernels": 0,
        "mixed_dtype_read_gpu_gate_authorized": False,
        "end_to_end_systems_authorized": False,
        "architecture_freeze_authorized": False,
        "fresh_scientific_seed_authorized": False,
        "100m_authorized": False,
        "breakthrough_proven": False,
    }
    for key, expected_value in candidate_required.items():
        if candidate_protocol.get(key) != expected_value:
            raise RuntimeError(
                f"issue553 candidate protocol drift: {key}={candidate_protocol.get(key)!r} "
                f"expected={expected_value!r}"
            )

    return {
        "research_issue": RESEARCH_ISSUE,
        "source_main": SOURCE_MAIN,
        "source_tree": SOURCE_TREE,
        "blobs": blobs,
        "issue552_head": ISSUE552_HEAD,
        "issue552_cpu_run": ISSUE552_CPU_RUN,
        "issue552_cpu_job": ISSUE552_CPU_JOB,
        "issue479_trigger": ISSUE479_TRIGGER,
        "issue479_run": ISSUE479_RUN,
        "issue479_job": ISSUE479_JOB,
        "issue545_trigger": ISSUE545_TRIGGER,
        "issue545_run": ISSUE545_RUN,
        "issue545_job": ISSUE545_JOB,
        "issue545_failure": ISSUE545_FAILURE,
        "result_path_absent": True,
        "scientific_seed_consumed": False,
        "end_to_end_systems_authorized": False,
        "architecture_freeze_authorized": False,
        "fresh_scientific_seed_authorized": False,
        "independent_replication_credit": False,
        "100m_authorized": False,
        "breakthrough_proven": False,
    }


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=32768,
    timeout=MAX_GPU_SECONDS,
    volumes={"/vol": volume},
)
def run_gate() -> dict:
    import torch
    from tam_research.aera_v26_7_issue553_ficem_read_mixed_dtype_probe import (
        run_ficem_read_mixed_dtype_probe_v26_7,
    )

    volume.reload()
    result_path = Path(RESULT_PATH)
    if result_path.exists():
        raise RuntimeError(
            f"refusing duplicate issue553 run because durable result exists: {RESULT_PATH}"
        )

    print(
        L4_START_MARKER
        + json.dumps(
            {
                "research_issue": RESEARCH_ISSUE,
                "device": torch.cuda.get_device_name(0),
                "source_main": SOURCE_MAIN,
                "probe_blob": PROBE_BLOB,
                "v26_7_backend_blob": V26_7_BACKEND_BLOB,
                "scientific_seed_consumed": False,
                "end_to_end_systems_authorized": False,
                "architecture_freeze_authorized": False,
                "fresh_scientific_seed_authorized": False,
                "independent_replication_credit": False,
                "100m_authorized": False,
                "breakthrough_proven": False,
            },
            separators=(",", ":"),
        ),
        flush=True,
    )

    result = run_ficem_read_mixed_dtype_probe_v26_7()
    result["issue553_gate_metadata"] = {
        "source_main": SOURCE_MAIN,
        "source_tree": SOURCE_TREE,
        "probe_blob": PROBE_BLOB,
        "v26_7_backend_blob": V26_7_BACKEND_BLOB,
        "repair5_backend_blob": REPAIR5_BACKEND_BLOB,
        "v26_6_write_backend_blob": V26_6_WRITE_BACKEND_BLOB,
        "historical_probe_blob": HISTORICAL_PROBE_BLOB,
        "repair5_probe_blob": REPAIR5_PROBE_BLOB,
        "issue530_systems_blob": ISSUE530_SYSTEMS_BLOB,
        "v26_interface_blob": V26_INTERFACE_BLOB,
        "stable_reference_blob": STABLE_REFERENCE_BLOB,
        "issue552_head": ISSUE552_HEAD,
        "issue552_cpu_run": ISSUE552_CPU_RUN,
        "issue552_cpu_job": ISSUE552_CPU_JOB,
        "issue479_trigger": ISSUE479_TRIGGER,
        "issue479_run": ISSUE479_RUN,
        "issue479_job": ISSUE479_JOB,
        "issue545_trigger": ISSUE545_TRIGGER,
        "issue545_run": ISSUE545_RUN,
        "issue545_job": ISSUE545_JOB,
        "issue545_failure": ISSUE545_FAILURE,
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
        "probe_blob": PROBE_BLOB,
        "v26_7_backend_blob": V26_7_BACKEND_BLOB,
    }
    print(RESULT_MARKER + json.dumps(marker, separators=(",", ":")), flush=True)
    print(
        SUMMARY_MARKER + json.dumps(_summary(result), separators=(",", ":")),
        flush=True,
    )
    return result


@app.local_entrypoint()
def main() -> None:
    print(PRECHECK_MARKER + json.dumps(precheck(), separators=(",", ":")), flush=True)
    check = preflight.remote()
    print(PREFLIGHT_MARKER + json.dumps(check, separators=(",", ":")), flush=True)
    result = run_gate.remote()
    print(json.dumps(_summary(result), indent=2), flush=True)
