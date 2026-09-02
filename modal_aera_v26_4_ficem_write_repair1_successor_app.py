from __future__ import annotations

import hashlib
import json
from pathlib import Path

import modal

APP_NAME = "tam-research-aera-v26-4-issue498-ficem-write-repair1-successor"
VOLUME_NAME = "tam-research-data"
RESULT_PATH = "/vol/aera-v26/issue498-ficem-write-repair1-successor/result.json"
MAX_GPU_SECONDS = 300
SOURCE_MAIN = "852edbe5261f75f56f45e52b33d6acf3bea02f2b"
WRITE_BACKEND_BLOB = "e54570292489bd17570038dca7518419ac00418c"
V26_INTERFACE_BLOB = "268644ac4edee15a4cc4e29d3fed7f61eeb3caa7"
STABLE_REFERENCE_BLOB = "4e336b6e1a6238dac782fa320751d68281493ee1"
READ_BACKEND_BLOB = "263f68eb1186a8ac14a08fc4b4df1fc5b292c711"
WRITE_PROBE_BLOB = "7d8c2c4990beb4c7b4a719d02d009ffefe94671f"
WRITE_REPAIR_ISSUE = 495
WRITE_REPAIR_PR = 497
WRITE_REPAIR_CPU_RUN = 33647126135
WRITE_REPAIR_CPU_JOB = 100304584393
WRITE_REPAIR_CPU_HEAD = "515d3fad8463c8e61da664b2667b7a4616ab3864"
EXHAUSTED_WRITE_GATE_ISSUE = 488
EXHAUSTED_WRITE_TRIGGER_ISSUE = 490
EXHAUSTED_WRITE_RUN = 33638047466
EXHAUSTED_WRITE_JOB = 100273784137
EXHAUSTED_WRITE_RESULT_SHA256 = "0149516113076534db1cbdcacb122447c3604f70394817eddf456bba316a195e"
READ_PASS_RUN = 33618950619
READ_PASS_JOB = 100211244996

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
    edge = result["edge_fixtures"]
    return {
        "decision": result["decision"],
        "overall_pass": result["overall_pass"],
        "device": result["device"],
        "correctness_pass": result["correctness_pass"],
        "row_latency_pass": result["row_latency_pass"],
        "tail_event_ratio_pass": result["tail_event_ratio_pass"],
        "two_kernel_topology_pass": result["two_kernel_topology_pass"],
        "geomean_latency_ratio_by_dtype": result["geomean_latency_ratio_by_dtype"],
        "geomean_latency_pass": result["geomean_latency_pass"],
        "previously_failed_bfloat16_threshold_fixture": edge[
            "bfloat16_threshold_inclusive_and_below_control"
        ]["pass"],
        "rows": {
            key: {
                "pass": row["pass"],
                "latency_ratio": row["latency_ratio_candidate_over_reference"],
                "tail_event_ratio": row[
                    "tail_cuda_event_ratio_candidate_over_reference"
                ],
                "projected_correctness": row["projected_correctness"]["pass"],
                "tail_correctness": row["tail_correctness"]["pass"],
                "topology": row["topology"]["pass"],
            }
            for key, row in result["rows"].items()
        },
        "edge_pass": {key: row["pass"] for key, row in edge.items()},
    }


def _required_repair_protocol() -> dict[str, object]:
    return {
        "duplicate_similarity": 0.95,
        "write_tail_triton_launches_target": 2,
        "write_threshold_input_dtype_visibility_repair1": True,
        "write_numeric_duplicate_threshold_changed_by_repair1": False,
        "float32_write_threshold_semantics_changed_by_repair1": False,
        "write_tail_kernel_count_changed_by_repair1": False,
        "write_threshold_repair1_issue": 495,
        "write_pre_repair_backend_blob": "5d703bbba296328ca2f49407e56192d10541349d",
        "write_gpu_gate_authorized": False,
        "end_to_end_systems_authorized": False,
        "architecture_freeze_authorized": False,
        "s2_authorized": False,
        "fresh_scientific_seed_authorized": False,
        "100m_authorized": False,
        "breakthrough_proven": False,
    }


@app.function(image=image, cpu=4, memory=8192, timeout=300, volumes={"/vol": volume})
def preflight() -> dict:
    import tam_research.aera_hardware_core_v26 as v26
    import tam_research.aera_hardware_core_v25_1_compact as stable
    import tam_research.aera_hardware_core_v26_3_ficem_read_triton as read_backend
    import tam_research.aera_hardware_core_v26_4_ficem_write_triton as write_backend
    import tam_research.aera_v26_4_ficem_write_probe as probe

    volume.reload()
    result_path = Path(RESULT_PATH)
    if result_path.exists():
        raise RuntimeError(
            f"refusing duplicate issue498 repaired WRITE successor run because result exists: {RESULT_PATH}"
        )

    blobs = {
        "write_backend": _git_blob_sha(Path(write_backend.__file__)),
        "v26_interface": _git_blob_sha(Path(v26.__file__)),
        "stable_reference": _git_blob_sha(Path(stable.__file__)),
        "read_backend": _git_blob_sha(Path(read_backend.__file__)),
        "write_probe": _git_blob_sha(Path(probe.__file__)),
    }
    expected = {
        "write_backend": WRITE_BACKEND_BLOB,
        "v26_interface": V26_INTERFACE_BLOB,
        "stable_reference": STABLE_REFERENCE_BLOB,
        "read_backend": READ_BACKEND_BLOB,
        "write_probe": WRITE_PROBE_BLOB,
    }
    if blobs != expected:
        raise RuntimeError(f"issue498 frozen blob drift: got={blobs} expected={expected}")

    protocol = write_backend.fused_ficem_read_write_v26_4_protocol()
    for key, expected_value in _required_repair_protocol().items():
        if protocol.get(key) != expected_value:
            raise RuntimeError(
                f"issue498 repaired WRITE protocol drift: {key}={protocol.get(key)!r} expected={expected_value!r}"
            )

    contract = probe.cpu_contract_preflight()
    return {
        "research_issue": 498,
        "source_main": SOURCE_MAIN,
        "blobs": blobs,
        "contract": contract,
        "write_repair_issue": WRITE_REPAIR_ISSUE,
        "write_repair_pr": WRITE_REPAIR_PR,
        "write_repair_cpu_run": WRITE_REPAIR_CPU_RUN,
        "write_repair_cpu_job": WRITE_REPAIR_CPU_JOB,
        "write_repair_cpu_head": WRITE_REPAIR_CPU_HEAD,
        "exhausted_write_run": EXHAUSTED_WRITE_RUN,
        "exhausted_write_job": EXHAUSTED_WRITE_JOB,
        "exhausted_write_result_sha256": EXHAUSTED_WRITE_RESULT_SHA256,
        "read_pass_run": READ_PASS_RUN,
        "read_pass_job": READ_PASS_JOB,
        "result_path_absent": True,
        "synthetic_only": True,
        "scientific_seed_consumed": False,
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
    from tam_research.aera_v26_4_ficem_write_probe import run_ficem_write_probe

    volume.reload()
    result_path = Path(RESULT_PATH)
    if result_path.exists():
        raise RuntimeError(
            f"refusing duplicate issue498 repaired WRITE successor run because result exists: {RESULT_PATH}"
        )

    print(
        "AERA_V26_ISSUE498_FICEM_WRITE_REPAIR1_SUCCESSOR_L4_START_JSON="
        + json.dumps(
            {
                "device": torch.cuda.get_device_name(0),
                "research_issue": 498,
                "source_main": SOURCE_MAIN,
                "write_backend_blob": WRITE_BACKEND_BLOB,
                "write_probe_blob": WRITE_PROBE_BLOB,
                "design_seed": 408487,
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

    result = run_ficem_write_probe()
    result["issue498_gate_metadata"] = {
        "source_main": SOURCE_MAIN,
        "write_backend_blob": WRITE_BACKEND_BLOB,
        "v26_interface_blob": V26_INTERFACE_BLOB,
        "stable_reference_blob": STABLE_REFERENCE_BLOB,
        "read_backend_blob": READ_BACKEND_BLOB,
        "write_probe_blob": WRITE_PROBE_BLOB,
        "write_repair_cpu_run": WRITE_REPAIR_CPU_RUN,
        "exhausted_write_run": EXHAUSTED_WRITE_RUN,
        "synthetic_only": True,
        "scientific_seed_consumed": False,
    }
    durable_json = json.dumps(result, sort_keys=True, separators=(",", ":"))
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(durable_json)
    volume.commit()
    digest = hashlib.sha256(durable_json.encode()).hexdigest()
    marker = {
        "research_issue": 498,
        "result_path": RESULT_PATH,
        "result_sha256": digest,
        "decision": result["decision"],
        "overall_pass": result["overall_pass"],
        "device": result["device"],
        "source_main": SOURCE_MAIN,
        "write_backend_blob": WRITE_BACKEND_BLOB,
        "write_probe_blob": WRITE_PROBE_BLOB,
    }
    print(
        "AERA_V26_ISSUE498_FICEM_WRITE_REPAIR1_SUCCESSOR_RESULT_JSON="
        + json.dumps(marker, separators=(",", ":")),
        flush=True,
    )
    print(
        "AERA_V26_ISSUE498_FICEM_WRITE_REPAIR1_SUCCESSOR_SUMMARY_JSON="
        + json.dumps(_summary(result), separators=(",", ":")),
        flush=True,
    )
    return result


@app.local_entrypoint()
def main() -> None:
    check = preflight.remote()
    print(
        "AERA_V26_ISSUE498_FICEM_WRITE_REPAIR1_SUCCESSOR_PREFLIGHT_JSON="
        + json.dumps(check, separators=(",", ":")),
        flush=True,
    )
    result = run_probe.remote()
    print(json.dumps(_summary(result), indent=2), flush=True)
