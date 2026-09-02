from __future__ import annotations

import hashlib
import json
from pathlib import Path

import modal

APP_NAME = "tam-research-aera-v26-4-issue487-ficem-write"
VOLUME_NAME = "tam-research-data"
RESULT_PATH = "/vol/aera-v26/issue487-ficem-write/result.json"
MAX_GPU_SECONDS = 300
SOURCE_MAIN = "c0ee36ba66e11d24bb9990787e125e986171a46e"
WRITE_BACKEND_GIT_BLOB = "5d703bbba296328ca2f49407e56192d10541349d"
READ_BACKEND_GIT_BLOB = "263f68eb1186a8ac14a08fc4b4df1fc5b292c711"
V26_INTERFACE_GIT_BLOB = "268644ac4edee15a4cc4e29d3fed7f61eeb3caa7"
STABLE_REFERENCE_GIT_BLOB = "4e336b6e1a6238dac782fa320751d68281493ee1"
PROBE_GIT_BLOB = "8497bb6e5f077c5cfd190aa7b95d1632b2e4fa1a"
SOURCE_WRITE_ISSUE = 485
SOURCE_WRITE_PR = 486
SOURCE_WRITE_CPU_RUN = 33620850681
SOURCE_WRITE_CPU_JOB = 100217278171
SOURCE_WRITE_CPU_HEAD = "c545f3c40b090183030c0ea68d411493db2b444c"
SOURCE_READ_TRIGGER = 484
SOURCE_READ_RUN = 33618950619
SOURCE_READ_JOB = 100211244996

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
    return {
        "decision": result["decision"],
        "overall_pass": result["overall_pass"],
        "device": result["device"],
        "geomean_latency_ratio_by_dtype": result["geomean_latency_ratio_by_dtype"],
        "correctness_pass": result["correctness_pass"],
        "row_latency_pass": result["row_latency_pass"],
        "geomean_latency_pass": result["geomean_latency_pass"],
        "isolated_tail_exact_pass": result["isolated_tail_exact_pass"],
        "tail_kernel_pass": result["tail_kernel_pass"],
        "tail_ops_pass": result["tail_ops_pass"],
        "tail_event_ratio_pass": result["tail_event_ratio_pass"],
        "stress_pass": all(row["pass"] for row in result["stress"].values()),
        "ordinary_update_pass": all(
            row["pass"] for row in result["ordinary_update"].values()
        ),
        "row_ratios": {
            key: {
                "latency": row["latency_ratio_candidate_over_reference"],
                "tail_events": row[
                    "isolated_tail_event_ratio_candidate_over_reference"
                ],
                "correctness": row["correctness"]["pass"],
                "tail_exact": row["isolated_tail_state_bit_exact"],
                "two_kernel_tail": row["candidate_tail_kernel_exact"],
            }
            for key, row in result["rows"].items()
        },
    }


@app.function(image=image, cpu=4, memory=8192, timeout=300, volumes={"/vol": volume})
def preflight() -> dict:
    import tam_research.aera_hardware_core_v25_1_compact as stable_module
    import tam_research.aera_hardware_core_v26 as v26_module
    import tam_research.aera_hardware_core_v26_3_ficem_read_triton as read_module
    import tam_research.aera_hardware_core_v26_4_ficem_write_triton as write_module
    import tam_research.aera_v26_4_ficem_write_probe as probe_module
    from tam_research.aera_hardware_core_v26_4_ficem_write_triton import (
        fused_ficem_read_write_v26_4_protocol,
    )
    from tam_research.aera_v26_4_ficem_write_probe import cpu_contract_preflight

    volume.reload()
    result_path = Path(RESULT_PATH)
    if result_path.exists():
        raise RuntimeError(
            f"refusing duplicate issue487 FICEM write run because result exists: {RESULT_PATH}"
        )

    blobs = {
        "write_backend": _git_blob_sha(Path(write_module.__file__)),
        "read_backend": _git_blob_sha(Path(read_module.__file__)),
        "v26_interface": _git_blob_sha(Path(v26_module.__file__)),
        "stable_reference": _git_blob_sha(Path(stable_module.__file__)),
        "probe": _git_blob_sha(Path(probe_module.__file__)),
    }
    expected = {
        "write_backend": WRITE_BACKEND_GIT_BLOB,
        "read_backend": READ_BACKEND_GIT_BLOB,
        "v26_interface": V26_INTERFACE_GIT_BLOB,
        "stable_reference": STABLE_REFERENCE_GIT_BLOB,
        "probe": PROBE_GIT_BLOB,
    }
    if blobs != expected:
        raise RuntimeError(f"issue487 frozen blob drift: observed={blobs} expected={expected}")

    protocol = fused_ficem_read_write_v26_4_protocol()
    required = {
        "write_count": 16,
        "capacity": 48,
        "memory_dim": 50,
        "duplicate_similarity": 0.95,
        "write_tail_triton_launches_target": 2,
        "read_backend_changed_by_v26_4": False,
        "write_similarity_einsums_changed": False,
        "write_value_projection_changed": False,
        "write_duplicate_semantics_changed": False,
        "write_stable_compaction_semantics_changed": False,
        "write_invalid_storage_semantics_changed": False,
        "write_training_backend_changed": False,
        "write_persistent_state_changed": False,
        "write_gpu_gate_authorized": False,
        "end_to_end_systems_authorized": False,
        "architecture_freeze_authorized": False,
        "fresh_scientific_seed_authorized": False,
        "100m_authorized": False,
        "breakthrough_proven": False,
    }
    for key, value in required.items():
        if protocol.get(key) != value:
            raise RuntimeError(
                f"issue487 production protocol drifted: {key}={protocol.get(key)!r}"
            )

    check = {
        "contract": cpu_contract_preflight(),
        "research_issue": 487,
        "source_main": SOURCE_MAIN,
        "source_write_issue": SOURCE_WRITE_ISSUE,
        "source_write_pr": SOURCE_WRITE_PR,
        "source_write_cpu_run": SOURCE_WRITE_CPU_RUN,
        "source_write_cpu_job": SOURCE_WRITE_CPU_JOB,
        "source_write_cpu_head": SOURCE_WRITE_CPU_HEAD,
        "source_read_trigger": SOURCE_READ_TRIGGER,
        "source_read_run": SOURCE_READ_RUN,
        "source_read_job": SOURCE_READ_JOB,
        "blobs": blobs,
        "result_path_absent": True,
        "synthetic_only": True,
        "model_loaded": False,
        "checkpoint_loaded": False,
        "corpus_accessed": False,
        "training_performed": False,
        "scientific_seed_consumed": False,
    }
    print(
        "AERA_V26_ISSUE487_FICEM_WRITE_PREFLIGHT_JSON="
        + json.dumps(check, separators=(",", ":")),
        flush=True,
    )
    return check


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
            f"refusing duplicate issue487 FICEM write run because result exists: {RESULT_PATH}"
        )

    print(
        "AERA_V26_ISSUE487_FICEM_WRITE_L4_START_JSON="
        + json.dumps(
            {
                "device": torch.cuda.get_device_name(0),
                "research_issue": 487,
                "source_main": SOURCE_MAIN,
                "candidate_backend_git_blob": WRITE_BACKEND_GIT_BLOB,
                "probe_git_blob": PROBE_GIT_BLOB,
                "hard_gpu_timeout_seconds": MAX_GPU_SECONDS,
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

    # Exactly one frozen issue487 probe invocation. Any pre-marker exception consumes
    # the sole attempt and is never retried under this preregistration.
    result = run_ficem_write_probe()
    result["issue487_gate_metadata"] = {
        "research_issue": 487,
        "source_main": SOURCE_MAIN,
        "source_write_cpu_run": SOURCE_WRITE_CPU_RUN,
        "source_write_cpu_job": SOURCE_WRITE_CPU_JOB,
        "source_write_cpu_head": SOURCE_WRITE_CPU_HEAD,
        "source_read_run": SOURCE_READ_RUN,
        "source_read_job": SOURCE_READ_JOB,
        "candidate_backend_git_blob": WRITE_BACKEND_GIT_BLOB,
        "probe_git_blob": PROBE_GIT_BLOB,
        "synthetic_only": True,
        "scientific_seed_consumed": False,
    }

    result_path.parent.mkdir(parents=True, exist_ok=True)
    durable_json = json.dumps(result, sort_keys=True, separators=(",", ":"))
    result_path.write_text(durable_json)
    volume.commit()

    result_sha256 = hashlib.sha256(durable_json.encode()).hexdigest()
    marker = {
        "research_issue": 487,
        "result_path": RESULT_PATH,
        "result_sha256": result_sha256,
        "decision": result["decision"],
        "overall_pass": result["overall_pass"],
        "device": result["device"],
        "source_main": SOURCE_MAIN,
        "candidate_backend_git_blob": WRITE_BACKEND_GIT_BLOB,
        "probe_git_blob": PROBE_GIT_BLOB,
    }
    print(
        "AERA_V26_ISSUE487_FICEM_WRITE_RESULT_JSON="
        + json.dumps(marker, separators=(",", ":")),
        flush=True,
    )
    print(
        "AERA_V26_ISSUE487_FICEM_WRITE_SUMMARY_JSON="
        + json.dumps(_summary(result), separators=(",", ":")),
        flush=True,
    )
    return result


@app.local_entrypoint()
def main() -> None:
    check = preflight.remote()
    print(
        "AERA_V26_ISSUE487_FICEM_WRITE_PREFLIGHT_LOCAL="
        + json.dumps(check, separators=(",", ":")),
        flush=True,
    )
    result = run_probe.remote()
    print(json.dumps(_summary(result), indent=2), flush=True)
