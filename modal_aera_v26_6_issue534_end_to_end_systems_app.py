from __future__ import annotations

import hashlib
import json
from pathlib import Path

import modal

APP_NAME = "tam-research-aera-v26-6-issue534-end-to-end-systems"
VOLUME_NAME = "tam-research-data"
RESULT_PATH = "/vol/aera-v26/issue534-end-to-end-systems/result.json"
PRIMITIVE_RESULT_PATH = "/vol/aera-v26/issue527-ficem-write-repaired-oracle/result.json"
MAX_GPU_SECONDS = 600
RESEARCH_ISSUE = 534
SOURCE_MAIN = "67b9559cafaf72d08261ff5c988233f2bc20932b"

SYSTEMS_ADAPTER_BLOB = "9d5a3c31f4a3862f96b957540baa2e0ec6a84c6b"
SYSTEMS_ADAPTER_PR = 533
SYSTEMS_ADAPTER_PR_HEAD = "af3ae4d721ccf218f2d5ebcd41458fd7ff5b8ad3"
SYSTEMS_ADAPTER_CPU_RUN = 33682266234
SYSTEMS_ADAPTER_CPU_JOB = 100421371756
SYSTEMS_ADAPTER_MERGE = SOURCE_MAIN

BASE_SYSTEMS_BLOB = "c9731cae7e386f09b2a190b045532591c4fa00be"
REPAIR1_SYSTEMS_BLOB = "b3f7082b188644007b873db3733492f424d4941a"
V26_INTERFACE_BLOB = "268644ac4edee15a4cc4e29d3fed7f61eeb3caa7"
READ_BACKEND_BLOB = "263f68eb1186a8ac14a08fc4b4df1fc5b292c711"
HISTORICAL_V26_4_WRITE_BACKEND_BLOB = "e54570292489bd17570038dca7518419ac00418c"
STABLE_REFERENCE_BLOB = "4e336b6e1a6238dac782fa320751d68281493ee1"
V26_6_CANDIDATE_BLOB = "d45c262314a0b4691f26812a279937a225043ad9"

ISSUE527_TRIGGER = 529
ISSUE527_RUN = 33680028132
ISSUE527_JOB = 100414089065
ISSUE527_ATTEMPT = 1
ISSUE527_BOUND_MAIN = "2c0c28005bff8d9b4f36a96de86144dd74107e39"
ISSUE527_RESULT_SHA256 = "a07790c2d55d7a696baf0e903a05dbdf925e3ef2a08b84c784c77d1fbdd31874"
ISSUE527_ORACLE_BLOB = "8f472451af4024bb3faacb56d814f7d6bdb25cc9"
ISSUE527_PROBE_BLOB = "bcfeb6a93ed062b7d00359603dc9fbc7aca5767f"

ISSUE508_TRIGGER = 510
ISSUE508_RUN = 33661498305
ISSUE508_JOB = 100352870198
ISSUE508_ATTEMPT = 1
ISSUE508_BOUND_MAIN = "1d475a199cfd2b14d5e94e5cffa29e05ac868ab1"
ISSUE508_FAILURE = "FICEM write state/value floating dtypes must match"
CHECKPOINT_HASH_KEYS = frozenset({"aera", "transformer"})

PREFLIGHT_MARKER = "AERA_V26_6_ISSUE534_END_TO_END_SYSTEMS_PREFLIGHT_JSON="
L4_START_MARKER = "AERA_V26_6_ISSUE534_END_TO_END_SYSTEMS_L4_START_JSON="
RESULT_MARKER = "AERA_V26_6_ISSUE534_END_TO_END_SYSTEMS_RESULT_JSON="
SUMMARY_MARKER = "AERA_V26_6_ISSUE534_END_TO_END_SYSTEMS_SUMMARY_JSON="

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


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _required_protocol() -> dict[str, object]:
    return {
        "version": "aera-v26.6-issue530-end-to-end-systems-adapter",
        "research_issue": 530,
        "source_main": ISSUE527_BOUND_MAIN,
        "base_systems_blob": BASE_SYSTEMS_BLOB,
        "repair1_systems_blob": REPAIR1_SYSTEMS_BLOB,
        "v26_6_candidate_blob": V26_6_CANDIDATE_BLOB,
        "historical_v26_4_candidate_backend_decision_bearing": False,
        "frozen_loader_v26_4_backend_replaced_before_parameter_snapshot": True,
        "frozen_loader_v26_4_backend_replaced_before_any_model_call": True,
        "only_candidate_semantic_change": "execution_backend_v26_4_to_v26_6_materialize_cast",
        "issue527_trigger": ISSUE527_TRIGGER,
        "issue527_run": ISSUE527_RUN,
        "issue527_job": ISSUE527_JOB,
        "issue527_attempt": ISSUE527_ATTEMPT,
        "issue527_bound_main": ISSUE527_BOUND_MAIN,
        "issue527_result_sha256": ISSUE527_RESULT_SHA256,
        "issue527_decision": "PASS",
        "issue527_direct_pass": [256, 256],
        "issue527_edge_pass": [32, 32],
        "issue527_public_pass": [6, 6],
        "issue527_topology_pass": [4, 4],
        "issue527_oracle_blob": ISSUE527_ORACLE_BLOB,
        "issue527_probe_blob": ISSUE527_PROBE_BLOB,
        "issue508_trigger": ISSUE508_TRIGGER,
        "issue508_run": ISSUE508_RUN,
        "issue508_job": ISSUE508_JOB,
        "issue508_attempt": ISSUE508_ATTEMPT,
        "issue508_bound_main": ISSUE508_BOUND_MAIN,
        "issue508_authoritative_result_emitted": False,
        "issue508_integrated_failure": ISSUE508_FAILURE,
        "batch_sizes": [8, 64],
        "warmup_calls": 3,
        "timed_calls_per_round": 20,
        "rounds": 5,
        "batch8_min_full_speed_ratio": 0.25,
        "batch64_min_full_speed_ratio": 1.25,
        "integrated_atol": 1e-2,
        "integrated_rtol": 1e-2,
        "persistent_state_bytes_per_session": 77_760,
        "production_write_geometry": [16, 255, 1],
        "source_checkpoint_seed": 8471,
        "checkpoint_relative_dir": "/vol/aera-real-language/v25-dev-seed8471",
        "random_token_seed_rule": "138471 + 10000 + batch_size",
        "timing_order": "rotated interleaved conditions per issue381",
        "timing_clock": "CUDA events with synchronize before/after",
        "hard": True,
        "route_mode": "hard_sparse",
        "physically_real_sparse_required": True,
        "dense_masked_sparse_credit": False,
        "systems_gpu_authorized_by_issue530": False,
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
            "pass": bool(result["per_batch_pass"][batch]),
            "routing_exact": bool(row["routing_exact"]),
            "logit_equivalence_pass": bool(row["logit_equivalence"]["pass"]),
            "state_equivalence_pass": bool(row["state_equivalence"]["pass"]),
            "physical_sparse_pass": bool(row["physical_sparse"]["pass"]),
            "write_geometry_pass": bool(row["write_geometry"]["pass"]),
            "finite": bool(row["finite"]),
            "persistent_state_bytes_actual": row["persistent_state_bytes_per_session_actual"],
            "persistent_state_bytes_pass": bool(row["persistent_state_bytes_pass"]),
            "candidate_full_vs_transformer_speed_ratio": row[
                "candidate_full_vs_transformer_speed_ratio"
            ],
            "required_full_speed_ratio": row["required_full_speed_ratio"],
            "throughput_pass": bool(row["throughput_pass"]),
            "reference_full_latency_ms": row["reference_full_latency_ms"],
            "candidate_full_latency_ms": row["candidate_full_latency_ms"],
            "candidate_vs_reference_latency_ratio": row[
                "candidate_vs_reference_latency_ratio"
            ],
            "no_reference_full_latency_regression": bool(
                row["no_reference_full_latency_regression"]
            ),
            "profiler_available": bool(row.get("profiler_candidate_full")),
            "peak_vram_available": bool(row.get("peak_vram")),
        }
    return {
        "research_issue": RESEARCH_ISSUE,
        "decision": result["decision"],
        "overall_pass": bool(result["overall_pass"]),
        "device": result["device"],
        "candidate_backend_names": result.get("candidate_backend_names", []),
        "parameter_versions_unchanged": bool(result["parameter_versions_unchanged"]),
        "checkpoint_hashes_unchanged": bool(result["checkpoint_hashes_unchanged"]),
        "training_performed": bool(result["training_performed"]),
        "optimizer_created": bool(result["optimizer_created"]),
        "backward_performed": bool(result["backward_performed"]),
        "corpus_accessed": bool(result["corpus_accessed"]),
        "checkpoint_written": bool(result["checkpoint_written"]),
        "scientific_seed_consumed": bool(result["scientific_seed_consumed"]),
        "rows": rows,
        "claims": result["claims"],
    }


@app.function(image=image, cpu=4, memory=8192, timeout=300, volumes={"/vol": volume})
def preflight() -> dict:
    import tam_research.aera_hardware_core_v25_1_compact as stable
    import tam_research.aera_hardware_core_v26 as v26
    import tam_research.aera_hardware_core_v26_3_ficem_read_triton as read_backend
    import tam_research.aera_hardware_core_v26_4_ficem_write_triton as historical_write
    import tam_research.aera_hardware_core_v26_6_ficem_write_materialize_cast as candidate_backend
    import tam_research.aera_v26_5_end_to_end_systems as base
    import tam_research.aera_v26_5_end_to_end_systems_repair1 as repair1
    import tam_research.aera_v26_6_issue530_end_to_end_systems as systems

    volume.reload()
    result_path = Path(RESULT_PATH)
    if result_path.exists():
        raise RuntimeError(
            f"refusing duplicate issue534 end-to-end systems run because result exists: {RESULT_PATH}"
        )

    primitive_path = Path(PRIMITIVE_RESULT_PATH)
    if not primitive_path.exists():
        raise RuntimeError(f"issue534 missing frozen #527 primitive result: {PRIMITIVE_RESULT_PATH}")
    primitive_sha = _sha256_bytes(primitive_path.read_bytes())
    if primitive_sha != ISSUE527_RESULT_SHA256:
        raise RuntimeError(
            f"issue534 #527 durable SHA drift: got={primitive_sha} expected={ISSUE527_RESULT_SHA256}"
        )

    root = Path(systems.__file__).parent
    blobs = {
        "systems_adapter": _git_blob_sha(Path(systems.__file__)),
        "base_systems": _git_blob_sha(Path(base.__file__)),
        "repair1_systems": _git_blob_sha(Path(repair1.__file__)),
        "v26_interface": _git_blob_sha(Path(v26.__file__)),
        "read_backend": _git_blob_sha(Path(read_backend.__file__)),
        "historical_v26_4_write": _git_blob_sha(Path(historical_write.__file__)),
        "stable_reference": _git_blob_sha(Path(stable.__file__)),
        "v26_6_candidate": _git_blob_sha(Path(candidate_backend.__file__)),
        "issue527_oracle": _git_blob_sha(root / "aera_v26_6_issue525_mixed_dtype_write_oracle.py"),
        "issue527_probe": _git_blob_sha(root / "aera_v26_6_issue527_ficem_write_repaired_oracle_probe.py"),
    }
    expected = {
        "systems_adapter": SYSTEMS_ADAPTER_BLOB,
        "base_systems": BASE_SYSTEMS_BLOB,
        "repair1_systems": REPAIR1_SYSTEMS_BLOB,
        "v26_interface": V26_INTERFACE_BLOB,
        "read_backend": READ_BACKEND_BLOB,
        "historical_v26_4_write": HISTORICAL_V26_4_WRITE_BACKEND_BLOB,
        "stable_reference": STABLE_REFERENCE_BLOB,
        "v26_6_candidate": V26_6_CANDIDATE_BLOB,
        "issue527_oracle": ISSUE527_ORACLE_BLOB,
        "issue527_probe": ISSUE527_PROBE_BLOB,
    }
    if blobs != expected:
        raise RuntimeError(f"issue534 frozen blob drift: got={blobs} expected={expected}")

    protocol = systems.issue530_systems_protocol()
    for key, expected_value in _required_protocol().items():
        if protocol.get(key) != expected_value:
            raise RuntimeError(
                f"issue534 systems protocol drift: {key}={protocol.get(key)!r} expected={expected_value!r}"
            )

    contract = systems.cpu_contract_preflight_issue530()
    if contract["gpu_authorized_by_cpu_preflight"] is not False:
        raise RuntimeError("issue534 inherited CPU contract unexpectedly authorizes GPU")
    if contract["scientific_seed_consumed"] is not False:
        raise RuntimeError("issue534 inherited CPU contract unexpectedly consumes scientific seed")

    if base.CHECKPOINT_RELATIVE_DIR != "/vol/aera-real-language/v25-dev-seed8471":
        raise RuntimeError("issue534 checkpoint path drift")
    hashes = base.checkpoint_hashes(base.CHECKPOINT_RELATIVE_DIR)
    if set(hashes) != CHECKPOINT_HASH_KEYS:
        raise RuntimeError(
            f"issue534 checkpoint hash inventory drift: got={sorted(hashes)} expected={sorted(CHECKPOINT_HASH_KEYS)}"
        )

    return {
        "research_issue": RESEARCH_ISSUE,
        "source_main": SOURCE_MAIN,
        "systems_adapter_pr": SYSTEMS_ADAPTER_PR,
        "systems_adapter_pr_head": SYSTEMS_ADAPTER_PR_HEAD,
        "systems_adapter_cpu_run": SYSTEMS_ADAPTER_CPU_RUN,
        "systems_adapter_cpu_job": SYSTEMS_ADAPTER_CPU_JOB,
        "systems_adapter_merge": SYSTEMS_ADAPTER_MERGE,
        "blobs": blobs,
        "primitive_trigger": ISSUE527_TRIGGER,
        "primitive_run": ISSUE527_RUN,
        "primitive_job": ISSUE527_JOB,
        "primitive_result_sha256": primitive_sha,
        "consumed_issue508_trigger": ISSUE508_TRIGGER,
        "consumed_issue508_run": ISSUE508_RUN,
        "consumed_issue508_job": ISSUE508_JOB,
        "consumed_issue508_failure": ISSUE508_FAILURE,
        "checkpoint_hashes": hashes,
        "checkpoint_hash_keys": sorted(hashes),
        "result_path_absent": True,
        "gpu_authorized_by_preflight": False,
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
    from tam_research.aera_v26_6_issue530_end_to_end_systems import (
        run_end_to_end_systems_v26_6,
    )

    volume.reload()
    result_path = Path(RESULT_PATH)
    if result_path.exists():
        raise RuntimeError(
            f"refusing duplicate issue534 end-to-end systems run because result exists: {RESULT_PATH}"
        )

    print(
        L4_START_MARKER
        + json.dumps(
            {
                "research_issue": RESEARCH_ISSUE,
                "device": torch.cuda.get_device_name(0),
                "source_main": SOURCE_MAIN,
                "systems_adapter_blob": SYSTEMS_ADAPTER_BLOB,
                "v26_6_candidate_blob": V26_6_CANDIDATE_BLOB,
                "checkpoint_seed": 8471,
                "primitive_result_sha256": ISSUE527_RESULT_SHA256,
                "scientific_seed_consumed": False,
                "architecture_freeze_authorized": False,
                "100m_authorized": False,
                "breakthrough_proven": False,
            },
            separators=(",", ":"),
        ),
        flush=True,
    )

    result = run_end_to_end_systems_v26_6()
    result["issue534_gate_metadata"] = {
        "source_main": SOURCE_MAIN,
        "systems_adapter_blob": SYSTEMS_ADAPTER_BLOB,
        "systems_adapter_pr": SYSTEMS_ADAPTER_PR,
        "systems_adapter_pr_head": SYSTEMS_ADAPTER_PR_HEAD,
        "systems_adapter_cpu_run": SYSTEMS_ADAPTER_CPU_RUN,
        "systems_adapter_cpu_job": SYSTEMS_ADAPTER_CPU_JOB,
        "systems_adapter_merge": SYSTEMS_ADAPTER_MERGE,
        "v26_6_candidate_blob": V26_6_CANDIDATE_BLOB,
        "issue527_trigger": ISSUE527_TRIGGER,
        "issue527_run": ISSUE527_RUN,
        "issue527_job": ISSUE527_JOB,
        "issue527_result_sha256": ISSUE527_RESULT_SHA256,
        "issue508_trigger": ISSUE508_TRIGGER,
        "issue508_run": ISSUE508_RUN,
        "issue508_job": ISSUE508_JOB,
        "issue508_failure": ISSUE508_FAILURE,
        "actions_attempt_required": 1,
        "gpu": "L4",
        "max_gpu_seconds": MAX_GPU_SECONDS,
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
        "systems_adapter_blob": SYSTEMS_ADAPTER_BLOB,
        "v26_6_candidate_blob": V26_6_CANDIDATE_BLOB,
        "primitive_result_sha256": ISSUE527_RESULT_SHA256,
    }
    print(RESULT_MARKER + json.dumps(marker, separators=(",", ":")), flush=True)
    print(SUMMARY_MARKER + json.dumps(_summary(result), separators=(",", ":")), flush=True)
    return result


@app.local_entrypoint()
def main() -> None:
    check = preflight.remote()
    print(PREFLIGHT_MARKER + json.dumps(check, separators=(",", ":")), flush=True)
    result = run_gate.remote()
    print(json.dumps(_summary(result), indent=2), flush=True)
