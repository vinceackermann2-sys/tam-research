from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import modal

APP_NAME = "aera-v26-9-issue643-bounded-memory-e2e-systems-l4"
VOLUME_NAME = "tam-research-data"
RESULT_PATH = "/vol/aera-v26/issue643-bounded-memory-end-to-end-systems/result.json"
MAX_GPU_SECONDS = 600
RESEARCH_ISSUE = 643
SOURCE_MAIN = "ef93e787e6d01585307f05f046d7fd3806374511"
SOURCE_TREE = "a44bcdb61b3124494e58902cad3d233cf7926cff"

ISSUE643_ADAPTER_BLOB = "512572340cc09e2e7ad6729712258c12cb377ef2"
BASE_SYSTEMS_BLOB = "c9731cae7e386f09b2a190b045532591c4fa00be"
REPAIR1_SYSTEMS_BLOB = "b3f7082b188644007b873db3733492f424d4941a"
ISSUE530_SYSTEMS_BLOB = "9d5a3c31f4a3862f96b957540baa2e0ec6a84c6b"
ISSUE562_SYSTEMS_BLOB = "3534103eea21f7c4d9d31798ad34601fd47090d6"
ISSUE569_SYSTEMS_BLOB = "1eeaa80adb2ae960e2a8dad06031c4ed5ca99203"
V26_9_BACKEND_BLOB = "b81cc209f5d95abbe1fb8bd620c78e87c067bc19"
ISSUE625_REPAIR_BLOB = "92d06a4954bca1b302355e81f5bf09b06fcee222"
ISSUE641_ADAPTER_BLOB = "5ea1919d15904add0f9e0fb714757f32b11442cb"
ISSUE641_CPU_TEST_BLOB = "e620a9874958bda78d586269f597095f5cf70670"

ISSUE642_HEAD = "504b5b3fdd254645eeb31bcb059831a8a6ee3164"
ISSUE642_CPU_RUN = 33989770634
ISSUE642_CPU_JOB = 101369805445
ISSUE642_MERGE = SOURCE_MAIN

ISSUE571_TRIGGER = 573
ISSUE571_RUN = 33741700781
ISSUE571_JOB = 100604889696
ISSUE571_RESULT_PATH = "/vol/aera-v26/issue571-memory-safe-end-to-end-systems/result.json"
ISSUE571_RESULT_SHA256 = "afeeb62351cc4fb97d272c5b55c9621839e26f83753ae1fb237733d58a5ee472"
ISSUE571_DECISION = "FAIL_FROZEN_E2E_SYSTEMS_GATE"

ISSUE630_TRIGGER = 640
ISSUE630_RUN = 33985543569
ISSUE630_JOB = 101358253857
ISSUE630_RESULT_PATH = "/vol/aera-v26/issue630-runner-allocation-continuation/result.json"
ISSUE630_RESULT_SHA256 = "ef92c85b55484b3ce191cd4016be86bf52da997a153f737194976164b29554b4"
ISSUE630_DECISION = "PASS"
ISSUE630_DESIGN_SEED = 891475817

CHECKPOINT_HASHES = {
    "aera": "f8aa92421801e8f190247e420632be5f0c20bc5ea8bf6bdeefe06686b3a31b30",
    "transformer": "cdd5cab4439a709468d6607d45d82081b33e876b2e40d91d4a38ba139b219dd7",
}
CHECKPOINT_HASH_KEYS = frozenset(CHECKPOINT_HASHES)

PRECHECK_MARKER = "AERA_V26_9_ISSUE643_BOUNDED_MEMORY_END_TO_END_SYSTEMS_PRECHECK_JSON="
PREFLIGHT_MARKER = "AERA_V26_9_ISSUE643_BOUNDED_MEMORY_END_TO_END_SYSTEMS_PREFLIGHT_JSON="
L4_START_MARKER = "AERA_V26_9_ISSUE643_BOUNDED_MEMORY_END_TO_END_SYSTEMS_L4_START_JSON="
RESULT_MARKER = "AERA_V26_9_ISSUE643_BOUNDED_MEMORY_END_TO_END_SYSTEMS_RESULT_JSON="
SUMMARY_MARKER = "AERA_V26_9_ISSUE643_BOUNDED_MEMORY_END_TO_END_SYSTEMS_SUMMARY_JSON="

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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _summary(result: dict) -> dict:
    rows: dict[str, dict] = {}
    for batch, row in result["rows"].items():
        physical = row["physical_sparse"]
        rows[str(batch)] = {
            "pass": result["per_batch_pass"][batch],
            "routing_exact": row["routing_exact"],
            "logit_equivalence_pass": row["logit_equivalence"]["pass"],
            "logit_max_abs": row["logit_equivalence"]["max_abs"],
            "state_equivalence_pass": row["state_equivalence"]["pass"],
            "physical_sparse_pass": physical["pass"],
            "physical_sparse_backend_names_exact": physical["backend_names_exact"],
            "physical_sparse_historical_backend_names_exact": physical.get(
                "historical_backend_names_exact"
            ),
            "write_geometry_pass": row["write_geometry"]["pass"],
            "finite": row["finite"],
            "persistent_state_bytes_pass": row["persistent_state_bytes_pass"],
            "candidate_full_vs_transformer_speed_ratio": row[
                "candidate_full_vs_transformer_speed_ratio"
            ],
            "required_full_speed_ratio": row["required_full_speed_ratio"],
            "throughput_pass": row["throughput_pass"],
            "reference_full_latency_ms": row["reference_full_latency_ms"],
            "candidate_full_latency_ms": row["candidate_full_latency_ms"],
            "candidate_vs_reference_latency_ratio": row[
                "candidate_vs_reference_latency_ratio"
            ],
            "no_reference_full_latency_regression": row[
                "no_reference_full_latency_regression"
            ],
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
        "issue643_adapter_metadata": result.get("issue643_adapter_metadata", {}),
        "issue643_gate_metadata": result.get("issue643_gate_metadata", {}),
    }


@app.function(image=image, cpu=4, memory=8192, timeout=300, volumes={"/vol": volume})
def preflight() -> dict:
    import tam_research.aera_hardware_core_v26_9_ficem_read_identity_weight_visibility as v26_9
    import tam_research.aera_v26_5_end_to_end_systems as base
    import tam_research.aera_v26_5_end_to_end_systems_repair1 as repair1
    import tam_research.aera_v26_6_issue530_end_to_end_systems as issue530
    import tam_research.aera_v26_8_issue562_end_to_end_systems as issue562
    import tam_research.aera_v26_8_issue569_end_to_end_systems_memory_safe as memory_safe
    import tam_research.aera_v26_9_issue625_schema_guard_repair1 as issue625
    import tam_research.aera_v26_9_issue641_physical_sparse_backend_identity_compat as issue641
    import tam_research.aera_v26_9_issue643_bounded_memory_end_to_end_systems as systems

    volume.reload()
    result_path = Path(RESULT_PATH)
    if result_path.exists():
        raise RuntimeError(
            f"refusing duplicate issue643 bounded-memory systems run because result exists: {RESULT_PATH}"
        )

    blobs = {
        "issue643_adapter": _git_blob_sha(Path(systems.__file__)),
        "base_systems": _git_blob_sha(Path(base.__file__)),
        "repair1_systems": _git_blob_sha(Path(repair1.__file__)),
        "issue530_systems": _git_blob_sha(Path(issue530.__file__)),
        "issue562_systems": _git_blob_sha(Path(issue562.__file__)),
        "issue569_systems": _git_blob_sha(Path(memory_safe.__file__)),
        "v26_9_candidate": _git_blob_sha(Path(v26_9.__file__)),
        "issue625_repair": _git_blob_sha(Path(issue625.__file__)),
        "issue641_adapter": _git_blob_sha(Path(issue641.__file__)),
    }
    expected = {
        "issue643_adapter": ISSUE643_ADAPTER_BLOB,
        "base_systems": BASE_SYSTEMS_BLOB,
        "repair1_systems": REPAIR1_SYSTEMS_BLOB,
        "issue530_systems": ISSUE530_SYSTEMS_BLOB,
        "issue562_systems": ISSUE562_SYSTEMS_BLOB,
        "issue569_systems": ISSUE569_SYSTEMS_BLOB,
        "v26_9_candidate": V26_9_BACKEND_BLOB,
        "issue625_repair": ISSUE625_REPAIR_BLOB,
        "issue641_adapter": ISSUE641_ADAPTER_BLOB,
    }
    if blobs != expected:
        raise RuntimeError(f"issue643 frozen blob drift: got={blobs} expected={expected}")

    contract = systems.cpu_contract_preflight_issue643()
    if contract["gpu_authorized_by_cpu_preflight"] is not False:
        raise RuntimeError("issue643 CPU contract unexpectedly authorizes GPU")
    if contract["model_construction_performed"] is not False:
        raise RuntimeError("issue643 CPU contract unexpectedly constructed a model")
    if contract["checkpoint_loaded"] is not False:
        raise RuntimeError("issue643 CPU contract unexpectedly loaded a checkpoint")
    if contract["systems_measurement_performed"] is not False:
        raise RuntimeError("issue643 CPU contract unexpectedly performed systems measurement")

    historical571 = Path(ISSUE571_RESULT_PATH)
    if not historical571.exists():
        raise RuntimeError("issue643 immutable #571 result is missing")
    if _sha256_file(historical571) != ISSUE571_RESULT_SHA256:
        raise RuntimeError("issue643 immutable #571 result SHA drifted")
    payload571 = json.loads(historical571.read_text())
    if payload571.get("decision") != ISSUE571_DECISION or payload571.get("overall_pass") is not False:
        raise RuntimeError("issue643 immutable #571 decision drifted")

    primitive630 = Path(ISSUE630_RESULT_PATH)
    if not primitive630.exists():
        raise RuntimeError("issue643 immutable #630 primitive result is missing")
    if _sha256_file(primitive630) != ISSUE630_RESULT_SHA256:
        raise RuntimeError("issue643 immutable #630 primitive result SHA drifted")
    payload630 = json.loads(primitive630.read_text())
    if payload630.get("decision") != ISSUE630_DECISION or payload630.get("all_pass") is not True:
        raise RuntimeError("issue643 immutable #630 primitive decision drifted")

    if base.CHECKPOINT_RELATIVE_DIR != "/vol/aera-real-language/v25-dev-seed8471":
        raise RuntimeError("issue643 checkpoint path drift")
    hashes = base.checkpoint_hashes(base.CHECKPOINT_RELATIVE_DIR)
    if set(hashes) != CHECKPOINT_HASH_KEYS:
        raise RuntimeError(
            f"issue643 checkpoint hash inventory drift: got={sorted(hashes)} "
            f"expected={sorted(CHECKPOINT_HASH_KEYS)}"
        )
    if hashes != CHECKPOINT_HASHES:
        raise RuntimeError(f"issue643 checkpoint hash drift: got={hashes} expected={CHECKPOINT_HASHES}")
    if not all(_valid_sha256(value) for value in hashes.values()):
        raise RuntimeError("issue643 checkpoint hash value drift")

    return {
        "research_issue": RESEARCH_ISSUE,
        "source_main": SOURCE_MAIN,
        "source_tree": SOURCE_TREE,
        "blobs": blobs,
        "checkpoint_hashes": hashes,
        "checkpoint_hash_keys": sorted(hashes),
        "issue642_head": ISSUE642_HEAD,
        "issue642_cpu_run": ISSUE642_CPU_RUN,
        "issue642_cpu_job": ISSUE642_CPU_JOB,
        "issue642_merge": ISSUE642_MERGE,
        "issue571_trigger": ISSUE571_TRIGGER,
        "issue571_run": ISSUE571_RUN,
        "issue571_job": ISSUE571_JOB,
        "issue571_result_sha256": ISSUE571_RESULT_SHA256,
        "issue571_decision": ISSUE571_DECISION,
        "issue630_trigger": ISSUE630_TRIGGER,
        "issue630_run": ISSUE630_RUN,
        "issue630_job": ISSUE630_JOB,
        "issue630_result_sha256": ISSUE630_RESULT_SHA256,
        "issue630_decision": ISSUE630_DECISION,
        "issue630_design_seed_consumed": ISSUE630_DESIGN_SEED,
        "result_path_absent": True,
        "chunk_batch_rows": 1,
        "scientific_seed_consumed": False,
        "architecture_freeze_authorized": False,
        "s2_authorized": False,
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
    from tam_research.aera_v26_9_issue643_bounded_memory_end_to_end_systems import (
        run_end_to_end_systems_v26_9_bounded_memory,
    )

    volume.reload()
    result_path = Path(RESULT_PATH)
    if result_path.exists():
        raise RuntimeError(
            f"refusing duplicate issue643 bounded-memory systems run because result exists: {RESULT_PATH}"
        )

    device = torch.cuda.get_device_name(0)
    if device != "NVIDIA L4":
        raise RuntimeError(f"issue643 requires exact NVIDIA L4, got {device!r}")

    print(
        L4_START_MARKER
        + json.dumps(
            {
                "research_issue": RESEARCH_ISSUE,
                "device": device,
                "source_main": SOURCE_MAIN,
                "issue643_adapter_blob": ISSUE643_ADAPTER_BLOB,
                "v26_9_candidate_blob": V26_9_BACKEND_BLOB,
                "issue641_adapter_blob": ISSUE641_ADAPTER_BLOB,
                "checkpoint_seed": 8471,
                "chunk_batch_rows": 1,
                "issue571_historical_fail_preserved": True,
                "issue630_primitive_pass_preserved": True,
                "issue630_design_seed_consumed": ISSUE630_DESIGN_SEED,
                "scientific_seed_consumed": False,
                "architecture_freeze_authorized": False,
                "s2_authorized": False,
                "fresh_scientific_seed_authorized": False,
                "independent_replication_credit": False,
                "100m_authorized": False,
                "breakthrough_proven": False,
            },
            separators=(",", ":"),
        ),
        flush=True,
    )

    result = run_end_to_end_systems_v26_9_bounded_memory()
    result["issue643_gate_metadata"] = {
        "source_main": SOURCE_MAIN,
        "source_tree": SOURCE_TREE,
        "issue643_adapter_blob": ISSUE643_ADAPTER_BLOB,
        "base_systems_blob": BASE_SYSTEMS_BLOB,
        "repair1_systems_blob": REPAIR1_SYSTEMS_BLOB,
        "issue530_systems_blob": ISSUE530_SYSTEMS_BLOB,
        "issue562_systems_blob": ISSUE562_SYSTEMS_BLOB,
        "issue569_systems_blob": ISSUE569_SYSTEMS_BLOB,
        "v26_9_candidate_blob": V26_9_BACKEND_BLOB,
        "issue625_repair_blob": ISSUE625_REPAIR_BLOB,
        "issue641_adapter_blob": ISSUE641_ADAPTER_BLOB,
        "issue641_cpu_test_blob": ISSUE641_CPU_TEST_BLOB,
        "issue642_head": ISSUE642_HEAD,
        "issue642_cpu_run": ISSUE642_CPU_RUN,
        "issue642_cpu_job": ISSUE642_CPU_JOB,
        "issue642_merge": ISSUE642_MERGE,
        "issue571_trigger": ISSUE571_TRIGGER,
        "issue571_run": ISSUE571_RUN,
        "issue571_job": ISSUE571_JOB,
        "issue571_result_path": ISSUE571_RESULT_PATH,
        "issue571_result_sha256": ISSUE571_RESULT_SHA256,
        "issue571_decision": ISSUE571_DECISION,
        "issue630_trigger": ISSUE630_TRIGGER,
        "issue630_run": ISSUE630_RUN,
        "issue630_job": ISSUE630_JOB,
        "issue630_result_path": ISSUE630_RESULT_PATH,
        "issue630_result_sha256": ISSUE630_RESULT_SHA256,
        "issue630_decision": ISSUE630_DECISION,
        "issue630_design_seed_consumed": ISSUE630_DESIGN_SEED,
        "checkpoint_hashes": dict(CHECKPOINT_HASHES),
        "chunk_batch_rows": 1,
        "scientific_seed_consumed": False,
        "architecture_freeze_authorized": False,
        "s2_authorized": False,
        "fresh_scientific_seed_authorized": False,
        "independent_replication_credit": False,
        "100m_authorized": False,
        "breakthrough_proven": False,
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
        "issue643_adapter_blob": ISSUE643_ADAPTER_BLOB,
        "v26_9_candidate_blob": V26_9_BACKEND_BLOB,
        "issue641_adapter_blob": ISSUE641_ADAPTER_BLOB,
        "chunk_batch_rows": 1,
    }
    print(RESULT_MARKER + json.dumps(marker, separators=(",", ":")), flush=True)
    print(SUMMARY_MARKER + json.dumps(_summary(result), separators=(",", ":")), flush=True)
    return result


@app.local_entrypoint()
def main() -> None:
    print(
        PRECHECK_MARKER
        + json.dumps(
            {
                "research_issue": RESEARCH_ISSUE,
                "source_main": SOURCE_MAIN,
                "source_tree": SOURCE_TREE,
                "issue643_adapter_blob": ISSUE643_ADAPTER_BLOB,
                "result_path": RESULT_PATH,
                "max_gpu_seconds": MAX_GPU_SECONDS,
                "gpu": "L4",
                "gpu_remote_functions": 1,
                "chunk_batch_rows": 1,
                "issue571_historical_fail_preserved": True,
                "issue630_primitive_pass_preserved": True,
                "issue630_design_seed_consumed": ISSUE630_DESIGN_SEED,
                "scientific_seed_consumed": False,
                "architecture_freeze_authorized": False,
                "s2_authorized": False,
                "fresh_scientific_seed_authorized": False,
                "independent_replication_credit": False,
                "100m_authorized": False,
                "breakthrough_proven": False,
            },
            separators=(",", ":"),
        ),
        flush=True,
    )
    check = preflight.remote()
    print(PREFLIGHT_MARKER + json.dumps(check, separators=(",", ":")), flush=True)
    result = run_gate.remote()
    print(json.dumps(_summary(result), indent=2), flush=True)
