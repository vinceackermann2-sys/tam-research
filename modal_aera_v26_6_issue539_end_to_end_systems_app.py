from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import modal

APP_NAME = "tam-research-aera-v26-6-issue539-end-to-end-systems"
VOLUME_NAME = "tam-research-data"
RESULT_PATH = "/vol/aera-v26/issue539-end-to-end-systems/result.json"
MAX_GPU_SECONDS = 600
RESEARCH_ISSUE = 539
SOURCE_MAIN = "c4400a56ab293588f44d678965f577698fd345cf"
SOURCE_TREE = "d0ac3b9a82268d6707642f1d5e5b3a148bb309e3"
ISSUE530_EVALUATOR_BLOB = "9d5a3c31f4a3862f96b957540baa2e0ec6a84c6b"
V26_6_CANDIDATE_BLOB = "d45c262314a0b4691f26812a279937a225043ad9"
BASE_SYSTEMS_BLOB = "c9731cae7e386f09b2a190b045532591c4fa00be"
REPAIR1_SYSTEMS_BLOB = "b3f7082b188644007b873db3733492f424d4941a"
V26_INTERFACE_BLOB = "268644ac4edee15a4cc4e29d3fed7f61eeb3caa7"
READ_BACKEND_BLOB = "263f68eb1186a8ac14a08fc4b4df1fc5b292c711"
HISTORICAL_V26_4_WRITE_BACKEND_BLOB = "e54570292489bd17570038dca7518419ac00418c"
STABLE_REFERENCE_BLOB = "4e336b6e1a6238dac782fa320751d68281493ee1"
ISSUE533_HEAD = "af3ae4d721ccf218f2d5ebcd41458fd7ff5b8ad3"
ISSUE533_CPU_RUN = 33682266234
ISSUE533_CPU_JOB = 100421371756
ISSUE527_TRIGGER = 529
ISSUE527_RUN = 33680028132
ISSUE527_JOB = 100414089065
ISSUE527_RESULT_SHA256 = "a07790c2d55d7a696baf0e903a05dbdf925e3ef2a08b84c784c77d1fbdd31874"
ISSUE508_TRIGGER = 510
ISSUE508_RUN = 33661498305
ISSUE508_JOB = 100352870198
ISSUE508_FAILURE = "FICEM write state/value floating dtypes must match"
CHECKPOINT_HASH_KEYS = frozenset({"aera", "transformer"})
PREFLIGHT_MARKER = "AERA_V26_6_ISSUE539_END_TO_END_SYSTEMS_PREFLIGHT_JSON="
L4_START_MARKER = "AERA_V26_6_ISSUE539_END_TO_END_SYSTEMS_L4_START_JSON="
RESULT_MARKER = "AERA_V26_6_ISSUE539_END_TO_END_SYSTEMS_RESULT_JSON="
SUMMARY_MARKER = "AERA_V26_6_ISSUE539_END_TO_END_SYSTEMS_SUMMARY_JSON="

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


def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


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
        "candidate_backend_names": result.get("candidate_backend_names", []),
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
    import tam_research.aera_hardware_core_v26_4_ficem_write_triton as historical_write
    import tam_research.aera_hardware_core_v26_6_ficem_write_materialize_cast as candidate_backend
    import tam_research.aera_v26_5_end_to_end_systems as base
    import tam_research.aera_v26_5_end_to_end_systems_repair1 as repair1
    import tam_research.aera_v26_6_issue530_end_to_end_systems as evaluator

    volume.reload()
    result_path = Path(RESULT_PATH)
    if result_path.exists():
        raise RuntimeError(
            f"refusing duplicate issue539 end-to-end systems run because result exists: {RESULT_PATH}"
        )

    blobs = {
        "issue530_evaluator": _git_blob_sha(Path(evaluator.__file__)),
        "v26_6_candidate": _git_blob_sha(Path(candidate_backend.__file__)),
        "base_systems": _git_blob_sha(Path(base.__file__)),
        "repair1_systems": _git_blob_sha(Path(repair1.__file__)),
        "v26_interface": _git_blob_sha(Path(v26.__file__)),
        "read_backend": _git_blob_sha(Path(read_backend.__file__)),
        "historical_v26_4_write": _git_blob_sha(Path(historical_write.__file__)),
        "stable_reference": _git_blob_sha(Path(stable.__file__)),
    }
    expected = {
        "issue530_evaluator": ISSUE530_EVALUATOR_BLOB,
        "v26_6_candidate": V26_6_CANDIDATE_BLOB,
        "base_systems": BASE_SYSTEMS_BLOB,
        "repair1_systems": REPAIR1_SYSTEMS_BLOB,
        "v26_interface": V26_INTERFACE_BLOB,
        "read_backend": READ_BACKEND_BLOB,
        "historical_v26_4_write": HISTORICAL_V26_4_WRITE_BACKEND_BLOB,
        "stable_reference": STABLE_REFERENCE_BLOB,
    }
    if blobs != expected:
        raise RuntimeError(f"issue539 frozen blob drift: got={blobs} expected={expected}")

    protocol = evaluator.issue530_systems_protocol()
    required = {
        "research_issue": 530,
        "v26_6_candidate_blob": V26_6_CANDIDATE_BLOB,
        "historical_v26_4_candidate_backend_decision_bearing": False,
        "frozen_loader_v26_4_backend_replaced_before_parameter_snapshot": True,
        "frozen_loader_v26_4_backend_replaced_before_any_model_call": True,
        "only_candidate_semantic_change": "execution_backend_v26_4_to_v26_6_materialize_cast",
        "issue527_trigger": ISSUE527_TRIGGER,
        "issue527_run": ISSUE527_RUN,
        "issue527_job": ISSUE527_JOB,
        "issue527_result_sha256": ISSUE527_RESULT_SHA256,
        "issue527_decision": "PASS",
        "issue527_direct_pass": [256, 256],
        "issue527_edge_pass": [32, 32],
        "issue527_public_pass": [6, 6],
        "issue527_topology_pass": [4, 4],
        "issue508_trigger": ISSUE508_TRIGGER,
        "issue508_run": ISSUE508_RUN,
        "issue508_job": ISSUE508_JOB,
        "issue508_integrated_failure": ISSUE508_FAILURE,
        "candidate_fieldwise_mixed_dtype_supported": True,
        "candidate_global_cross_field_dtype_equality_required": False,
        "candidate_write_supported_float_dtypes": ["float32", "bfloat16"],
        "candidate_write_tail_triton_launches_target": 2,
        "candidate_read_backend_changed_by_v26_6": False,
        "architecture_freeze_authorized": False,
        "fresh_scientific_seed_authorized": False,
        "independent_replication_credit": False,
        "100m_authorized": False,
        "breakthrough_proven": False,
    }
    for key, expected_value in required.items():
        if protocol.get(key) != expected_value:
            raise RuntimeError(
                f"issue539 evaluator protocol drift: {key}={protocol.get(key)!r} expected={expected_value!r}"
            )

    if base.CHECKPOINT_RELATIVE_DIR != "/vol/aera-real-language/v25-dev-seed8471":
        raise RuntimeError("issue539 checkpoint path drift")
    hashes = base.checkpoint_hashes(base.CHECKPOINT_RELATIVE_DIR)
    if set(hashes) != CHECKPOINT_HASH_KEYS:
        raise RuntimeError(
            f"issue539 checkpoint hash inventory drift: got={sorted(hashes)} expected={sorted(CHECKPOINT_HASH_KEYS)}"
        )
    if not all(_valid_sha256(value) for value in hashes.values()):
        raise RuntimeError("issue539 checkpoint hash value drift")

    contract = evaluator.cpu_contract_preflight_issue530()
    if contract["gpu_authorized_by_cpu_preflight"] is not False:
        raise RuntimeError("issue539 inherited #530 contract unexpectedly authorizes GPU")

    return {
        "research_issue": RESEARCH_ISSUE,
        "source_main": SOURCE_MAIN,
        "source_tree": SOURCE_TREE,
        "blobs": blobs,
        "checkpoint_hashes": hashes,
        "checkpoint_hash_keys": sorted(hashes),
        "issue533_head": ISSUE533_HEAD,
        "issue533_cpu_run": ISSUE533_CPU_RUN,
        "issue533_cpu_job": ISSUE533_CPU_JOB,
        "issue527_trigger": ISSUE527_TRIGGER,
        "issue527_run": ISSUE527_RUN,
        "issue527_job": ISSUE527_JOB,
        "issue527_result_sha256": ISSUE527_RESULT_SHA256,
        "issue508_trigger": ISSUE508_TRIGGER,
        "issue508_run": ISSUE508_RUN,
        "issue508_job": ISSUE508_JOB,
        "result_path_absent": True,
        "systems_gate_authorized_by_issue539_after_harness_merge_only": True,
        "scientific_seed_consumed": False,
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
    from tam_research.aera_v26_6_issue530_end_to_end_systems import (
        run_end_to_end_systems_v26_6,
    )

    volume.reload()
    result_path = Path(RESULT_PATH)
    if result_path.exists():
        raise RuntimeError(
            f"refusing duplicate issue539 end-to-end systems run because result exists: {RESULT_PATH}"
        )

    print(
        L4_START_MARKER
        + json.dumps(
            {
                "device": torch.cuda.get_device_name(0),
                "research_issue": RESEARCH_ISSUE,
                "source_main": SOURCE_MAIN,
                "issue530_evaluator_blob": ISSUE530_EVALUATOR_BLOB,
                "v26_6_candidate_blob": V26_6_CANDIDATE_BLOB,
                "checkpoint_seed": 8471,
                "scientific_seed_consumed": False,
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

    result = run_end_to_end_systems_v26_6()
    result["issue539_gate_metadata"] = {
        "source_main": SOURCE_MAIN,
        "source_tree": SOURCE_TREE,
        "issue530_evaluator_blob": ISSUE530_EVALUATOR_BLOB,
        "v26_6_candidate_blob": V26_6_CANDIDATE_BLOB,
        "base_systems_blob": BASE_SYSTEMS_BLOB,
        "repair1_systems_blob": REPAIR1_SYSTEMS_BLOB,
        "v26_interface_blob": V26_INTERFACE_BLOB,
        "read_backend_blob": READ_BACKEND_BLOB,
        "historical_v26_4_write_backend_blob": HISTORICAL_V26_4_WRITE_BACKEND_BLOB,
        "stable_reference_blob": STABLE_REFERENCE_BLOB,
        "issue533_head": ISSUE533_HEAD,
        "issue533_cpu_run": ISSUE533_CPU_RUN,
        "issue533_cpu_job": ISSUE533_CPU_JOB,
        "issue527_trigger": ISSUE527_TRIGGER,
        "issue527_run": ISSUE527_RUN,
        "issue527_job": ISSUE527_JOB,
        "issue527_result_sha256": ISSUE527_RESULT_SHA256,
        "issue508_trigger": ISSUE508_TRIGGER,
        "issue508_run": ISSUE508_RUN,
        "issue508_job": ISSUE508_JOB,
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
        "issue530_evaluator_blob": ISSUE530_EVALUATOR_BLOB,
        "v26_6_candidate_blob": V26_6_CANDIDATE_BLOB,
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
