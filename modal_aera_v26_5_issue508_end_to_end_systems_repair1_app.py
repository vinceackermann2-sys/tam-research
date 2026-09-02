from __future__ import annotations

import hashlib
import json
from pathlib import Path

import modal

APP_NAME = "tam-research-aera-v26-5-issue508-end-to-end-systems-repair1"
VOLUME_NAME = "tam-research-data"
RESULT_PATH = "/vol/aera-v26/issue508-end-to-end-systems-repair1/result.json"
MAX_GPU_SECONDS = 600
RESEARCH_ISSUE = 508
SOURCE_MAIN = "371c97380c1488689a6a1ddacfb89f47a64aabfc"
HISTORICAL_SYSTEMS_BLOB = "c9731cae7e386f09b2a190b045532591c4fa00be"
REPAIRED_SYSTEMS_BLOB = "b3f7082b188644007b873db3733492f424d4941a"
V26_INTERFACE_BLOB = "268644ac4edee15a4cc4e29d3fed7f61eeb3caa7"
READ_BACKEND_BLOB = "263f68eb1186a8ac14a08fc4b4df1fc5b292c711"
WRITE_BACKEND_BLOB = "e54570292489bd17570038dca7518419ac00418c"
STABLE_REFERENCE_BLOB = "4e336b6e1a6238dac782fa320751d68281493ee1"
REPAIR_ISSUE = 503
REPAIR_PR = 504
REPAIR_CPU_HEAD = "58d04a12edaef50caf5daa24eb0386e8c624c6ca"
REPAIR_CPU_RUN = 33657968851
REPAIR_CPU_JOB = 100341171002
READ_PASS_RUN = 33618950619
READ_PASS_JOB = 100211244996
WRITE_PASS_RUN = 33651216734
WRITE_PASS_JOB = 100318422299
EXHAUSTED_ISSUE = 505
EXHAUSTED_TRIGGER = 507
EXHAUSTED_RUN = 33660377370
EXHAUSTED_JOB = 100349177580
EXHAUSTED_LAUNCHER_BLOB = "491fe31b0f701c1a67e91a9bd877069eeba42e55"
EXHAUSTED_WORKFLOW_BLOB = "06240f93cf779050a8615957fe431bc7e5fd2ccd"
CHECKPOINT_HASH_KEYS = frozenset({"aera", "transformer"})

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


def _required_protocol() -> dict[str, object]:
    return {
        "repair_issue": 503,
        "repair_source_main": "bd7fe8aab50af30006b7cb8a5f790699736379e0",
        "predecessor_module_blob": HISTORICAL_SYSTEMS_BLOB,
        "top_level_inference_decorated": False,
        "model_construction_outside_inference_mode": True,
        "parameter_version_snapshots_outside_inference_mode": True,
        "measurements_inside_explicit_inference_mode": True,
        "historical_issue501_module_mutated": False,
        "gpu_authorized_by_issue503": False,
        "batch_sizes": [8, 64],
        "random_token_seed_rule": "138471 + 10000 + batch_size",
        "timing_order": "rotated interleaved conditions per issue381",
        "timing_clock": "CUDA events with synchronize before/after",
        "hard": True,
        "route_mode": "hard_sparse",
        "physically_real_sparse_required": True,
        "dense_masked_sparse_credit": False,
        "architecture_freeze_authorized": False,
        "s2_authorized": False,
        "fresh_scientific_seed_authorized": False,
        "independent_replication_credit": False,
        "100m_authorized": False,
        "breakthrough_proven": False,
    }


def _summary(result: dict) -> dict:
    rows: dict[str, dict] = {}
    for batch, row in result["rows"].items():
        rows[batch] = {
            "pass": result["per_batch_pass"][batch],
            "routing_exact": row["routing_exact"],
            "logit_equivalence_pass": row["logit_equivalence"]["pass"],
            "state_equivalence_pass": row["state_equivalence"]["pass"],
            "physical_sparse_pass": row["physical_sparse"]["pass"],
            "write_geometry_pass": row["write_geometry"]["pass"],
            "finite": row["finite"],
            "persistent_state_bytes_pass": row["persistent_state_bytes_pass"],
            "candidate_full_vs_transformer_speed_ratio": row[
                "candidate_full_vs_transformer_speed_ratio"
            ],
            "required_full_speed_ratio": row["required_full_speed_ratio"],
            "throughput_pass": row["throughput_pass"],
            "candidate_vs_reference_latency_ratio": row[
                "candidate_vs_reference_latency_ratio"
            ],
            "no_reference_full_latency_regression": row[
                "no_reference_full_latency_regression"
            ],
            "profiler_available": bool(row.get("profiler_candidate_full")),
            "peak_vram_available": bool(row.get("peak_vram")),
        }
    return {
        "research_issue": RESEARCH_ISSUE,
        "decision": result["decision"],
        "overall_pass": result["overall_pass"],
        "device": result["device"],
        "parameter_versions_unchanged": result["parameter_versions_unchanged"],
        "checkpoint_hashes_unchanged": result["checkpoint_hashes_unchanged"],
        "training_performed": result["training_performed"],
        "optimizer_created": result["optimizer_created"],
        "backward_performed": result["backward_performed"],
        "corpus_accessed": result["corpus_accessed"],
        "checkpoint_written": result["checkpoint_written"],
        "scientific_seed_consumed": result["scientific_seed_consumed"],
        "rows": rows,
        "claims": result["claims"],
    }


@app.function(image=image, cpu=4, memory=8192, timeout=300, volumes={"/vol": volume})
def preflight() -> dict:
    import tam_research.aera_hardware_core_v25_1_compact as stable
    import tam_research.aera_hardware_core_v26 as v26
    import tam_research.aera_hardware_core_v26_3_ficem_read_triton as read_backend
    import tam_research.aera_hardware_core_v26_4_ficem_write_triton as write_backend
    import tam_research.aera_v26_5_end_to_end_systems as historical
    import tam_research.aera_v26_5_end_to_end_systems_repair1 as repaired

    volume.reload()
    result_path = Path(RESULT_PATH)
    if result_path.exists():
        raise RuntimeError(
            f"refusing duplicate issue508 end-to-end systems repair1 run because result exists: {RESULT_PATH}"
        )

    blobs = {
        "historical_systems": _git_blob_sha(Path(historical.__file__)),
        "repaired_systems": _git_blob_sha(Path(repaired.__file__)),
        "v26_interface": _git_blob_sha(Path(v26.__file__)),
        "read_backend": _git_blob_sha(Path(read_backend.__file__)),
        "write_backend": _git_blob_sha(Path(write_backend.__file__)),
        "stable_reference": _git_blob_sha(Path(stable.__file__)),
    }
    expected = {
        "historical_systems": HISTORICAL_SYSTEMS_BLOB,
        "repaired_systems": REPAIRED_SYSTEMS_BLOB,
        "v26_interface": V26_INTERFACE_BLOB,
        "read_backend": READ_BACKEND_BLOB,
        "write_backend": WRITE_BACKEND_BLOB,
        "stable_reference": STABLE_REFERENCE_BLOB,
    }
    if blobs != expected:
        raise RuntimeError(f"issue508 frozen blob drift: got={blobs} expected={expected}")

    protocol = repaired.repair1_protocol()
    for key, expected_value in _required_protocol().items():
        if protocol.get(key) != expected_value:
            raise RuntimeError(
                f"issue508 repaired systems protocol drift: {key}={protocol.get(key)!r} expected={expected_value!r}"
            )

    if historical.CHECKPOINT_RELATIVE_DIR != "/vol/aera-real-language/v25-dev-seed8471":
        raise RuntimeError("issue508 checkpoint path drift")
    hashes = historical.checkpoint_hashes(historical.CHECKPOINT_RELATIVE_DIR)
    if set(hashes) != CHECKPOINT_HASH_KEYS:
        raise RuntimeError(
            f"issue508 checkpoint hash inventory drift: got={sorted(hashes)} expected={sorted(CHECKPOINT_HASH_KEYS)}"
        )

    contract = repaired.cpu_contract_preflight_repair1()
    if contract["gpu_authorized_by_issue503"] is not False:
        raise RuntimeError("issue508 inherited repair contract unexpectedly authorizes GPU")

    return {
        "research_issue": RESEARCH_ISSUE,
        "source_main": SOURCE_MAIN,
        "blobs": blobs,
        "checkpoint_hashes": hashes,
        "checkpoint_hash_keys": sorted(hashes),
        "checkpoint_hash_key_repair": True,
        "repair_issue": REPAIR_ISSUE,
        "repair_pr": REPAIR_PR,
        "repair_cpu_head": REPAIR_CPU_HEAD,
        "repair_cpu_run": REPAIR_CPU_RUN,
        "repair_cpu_job": REPAIR_CPU_JOB,
        "read_pass_run": READ_PASS_RUN,
        "read_pass_job": READ_PASS_JOB,
        "write_pass_run": WRITE_PASS_RUN,
        "write_pass_job": WRITE_PASS_JOB,
        "exhausted_issue": EXHAUSTED_ISSUE,
        "exhausted_trigger": EXHAUSTED_TRIGGER,
        "exhausted_run": EXHAUSTED_RUN,
        "exhausted_job": EXHAUSTED_JOB,
        "exhausted_launcher_blob": EXHAUSTED_LAUNCHER_BLOB,
        "exhausted_workflow_blob": EXHAUSTED_WORKFLOW_BLOB,
        "result_path_absent": True,
        "systems_gate_authorized_by_issue508": True,
        "scientific_seed_consumed": False,
        "architecture_freeze_authorized": False,
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
    from tam_research.aera_v26_5_end_to_end_systems_repair1 import (
        run_end_to_end_systems_repair1,
    )

    volume.reload()
    result_path = Path(RESULT_PATH)
    if result_path.exists():
        raise RuntimeError(
            f"refusing duplicate issue508 end-to-end systems repair1 run because result exists: {RESULT_PATH}"
        )

    print(
        "AERA_V26_5_ISSUE508_END_TO_END_SYSTEMS_REPAIR1_L4_START_JSON="
        + json.dumps(
            {
                "device": torch.cuda.get_device_name(0),
                "research_issue": RESEARCH_ISSUE,
                "source_main": SOURCE_MAIN,
                "repaired_systems_blob": REPAIRED_SYSTEMS_BLOB,
                "checkpoint_seed": 8471,
                "checkpoint_hash_key_repair": True,
                "exhausted_issue505_run": EXHAUSTED_RUN,
                "scientific_seed_consumed": False,
                "architecture_freeze_authorized": False,
                "100m_authorized": False,
                "breakthrough_proven": False,
            },
            separators=(",", ":"),
        ),
        flush=True,
    )

    result = run_end_to_end_systems_repair1()
    result["issue508_repair1_gate_metadata"] = {
        "source_main": SOURCE_MAIN,
        "historical_systems_blob": HISTORICAL_SYSTEMS_BLOB,
        "repaired_systems_blob": REPAIRED_SYSTEMS_BLOB,
        "v26_interface_blob": V26_INTERFACE_BLOB,
        "read_backend_blob": READ_BACKEND_BLOB,
        "write_backend_blob": WRITE_BACKEND_BLOB,
        "stable_reference_blob": STABLE_REFERENCE_BLOB,
        "repair_cpu_head": REPAIR_CPU_HEAD,
        "repair_cpu_run": REPAIR_CPU_RUN,
        "repair_cpu_job": REPAIR_CPU_JOB,
        "read_pass_run": READ_PASS_RUN,
        "write_pass_run": WRITE_PASS_RUN,
        "exhausted_issue": EXHAUSTED_ISSUE,
        "exhausted_trigger": EXHAUSTED_TRIGGER,
        "exhausted_run": EXHAUSTED_RUN,
        "exhausted_job": EXHAUSTED_JOB,
        "checkpoint_hash_key_repair": True,
        "checkpoint_hash_keys": sorted(CHECKPOINT_HASH_KEYS),
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
        "repaired_systems_blob": REPAIRED_SYSTEMS_BLOB,
        "checkpoint_hash_key_repair": True,
    }
    print(
        "AERA_V26_5_ISSUE508_END_TO_END_SYSTEMS_REPAIR1_RESULT_JSON="
        + json.dumps(marker, separators=(",", ":")),
        flush=True,
    )
    print(
        "AERA_V26_5_ISSUE508_END_TO_END_SYSTEMS_REPAIR1_SUMMARY_JSON="
        + json.dumps(_summary(result), separators=(",", ":")),
        flush=True,
    )
    return result


@app.local_entrypoint()
def main() -> None:
    check = preflight.remote()
    print(
        "AERA_V26_5_ISSUE508_END_TO_END_SYSTEMS_REPAIR1_PREFLIGHT_JSON="
        + json.dumps(check, separators=(",", ":")),
        flush=True,
    )
    result = run_gate.remote()
    print(json.dumps(_summary(result), indent=2), flush=True)
