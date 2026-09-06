from __future__ import annotations

import hashlib
import json
from pathlib import Path

import modal

APP_NAME = "aera-v26-9-issue646-issue643-e2e-guard-continuation-l4"
VOLUME_NAME = "tam-research-data"
RESULT_PATH = "/vol/aera-v26/issue646-issue643-e2e-guard-continuation/result.json"
MAX_GPU_SECONDS = 600

RESEARCH_ISSUE = 643
CONTINUATION_ISSUE = 646
BOUND_SOURCE_MAIN = "25fd672e923ea66bab5a529de0c3e8a8680bf41e"
BOUND_SOURCE_TREE = "118f66652b7767a979fb7126aa71603e41c29723"

ISSUE643_ADAPTER_BLOB = "512572340cc09e2e7ad6729712258c12cb377ef2"
BASE_SYSTEMS_BLOB = "c9731cae7e386f09b2a190b045532591c4fa00be"
REPAIR1_SYSTEMS_BLOB = "b3f7082b188644007b873db3733492f424d4941a"
ISSUE530_SYSTEMS_BLOB = "9d5a3c31f4a3862f96b957540baa2e0ec6a84c6b"
ISSUE562_SYSTEMS_BLOB = "3534103eea21f7c4d9d31798ad34601fd47090d6"
ISSUE569_SYSTEMS_BLOB = "1eeaa80adb2ae960e2a8dad06031c4ed5ca99203"
V26_9_BACKEND_BLOB = "b81cc209f5d95abbe1fb8bd620c78e87c067bc19"
ISSUE625_REPAIR_BLOB = "92d06a4954bca1b302355e81f5bf09b06fcee222"
ISSUE641_ADAPTER_BLOB = "5ea1919d15904add0f9e0fb714757f32b11442cb"

ISSUE645_TRIGGER = 645
ISSUE645_RUN = 33991449361
ISSUE645_JOB = 101374299440

ISSUE571_RESULT_PATH = "/vol/aera-v26/issue571-memory-safe-end-to-end-systems/result.json"
ISSUE571_RESULT_SHA256 = "afeeb62351cc4fb97d272c5b55c9621839e26f83753ae1fb237733d58a5ee472"
ISSUE571_DECISION = "FAIL_FROZEN_E2E_SYSTEMS_GATE"

ISSUE630_RESULT_PATH = "/vol/aera-v26/issue630-runner-allocation-continuation/result.json"
ISSUE630_RESULT_SHA256 = "ef92c85b55484b3ce191cd4016be86bf52da997a153f737194976164b29554b4"
ISSUE630_DECISION = "PASS"

CHECKPOINT_HASHES = {
    "aera": "f8aa92421801e8f190247e420632be5f0c20bc5ea8bf6bdeefe06686b3a31b30",
    "transformer": "cdd5cab4439a709468d6607d45d82081b33e876b2e40d91d4a38ba139b219dd7",
}

PRECHECK_MARKER = "AERA_V26_9_ISSUE646_ISSUE643_E2E_GUARD_CONTINUATION_PRECHECK_JSON="
PREFLIGHT_MARKER = "AERA_V26_9_ISSUE646_ISSUE643_E2E_GUARD_CONTINUATION_PREFLIGHT_JSON="
L4_START_MARKER = "AERA_V26_9_ISSUE646_ISSUE643_E2E_GUARD_CONTINUATION_L4_START_JSON="
RESULT_MARKER = "AERA_V26_9_ISSUE646_ISSUE643_E2E_GUARD_CONTINUATION_RESULT_JSON="
SUMMARY_MARKER = "AERA_V26_9_ISSUE646_ISSUE643_E2E_GUARD_CONTINUATION_SUMMARY_JSON="

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
        "continuation_issue": CONTINUATION_ISSUE,
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
        "issue646_continuation_metadata": result.get(
            "issue646_continuation_metadata", {}
        ),
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
            f"refusing duplicate issue646 continuation because result exists: {RESULT_PATH}"
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
        raise RuntimeError(
            f"issue646 frozen scientific blob drift: got={blobs} expected={expected}"
        )

    contract = systems.cpu_contract_preflight_issue643()
    for key in (
        "gpu_authorized_by_cpu_preflight",
        "model_construction_performed",
        "checkpoint_loaded",
        "systems_measurement_performed",
        "scientific_seed_consumed",
        "architecture_freeze_authorized",
        "s2_authorized",
        "fresh_scientific_seed_authorized",
        "independent_replication_credit",
        "100m_authorized",
        "breakthrough_proven",
    ):
        if contract[key] is not False:
            raise RuntimeError(f"issue646 inherited #643 CPU contract drifted: {key}")

    historical571 = Path(ISSUE571_RESULT_PATH)
    if not historical571.exists():
        raise RuntimeError("issue646 immutable #571 result is missing")
    if _sha256_file(historical571) != ISSUE571_RESULT_SHA256:
        raise RuntimeError("issue646 immutable #571 result SHA drifted")
    payload571 = json.loads(historical571.read_text())
    if (
        payload571.get("decision") != ISSUE571_DECISION
        or payload571.get("overall_pass") is not False
    ):
        raise RuntimeError("issue646 immutable #571 decision drifted")

    primitive630 = Path(ISSUE630_RESULT_PATH)
    if not primitive630.exists():
        raise RuntimeError("issue646 immutable #630 result is missing")
    if _sha256_file(primitive630) != ISSUE630_RESULT_SHA256:
        raise RuntimeError("issue646 immutable #630 result SHA drifted")
    payload630 = json.loads(primitive630.read_text())
    if (
        payload630.get("decision") != ISSUE630_DECISION
        or payload630.get("overall_pass") is not True
    ):
        raise RuntimeError("issue646 immutable #630 decision drifted")

    if base.CHECKPOINT_RELATIVE_DIR != "/vol/aera-real-language/v25-dev-seed8471":
        raise RuntimeError("issue646 checkpoint path drift")
    hashes = base.checkpoint_hashes(base.CHECKPOINT_RELATIVE_DIR)
    if hashes != CHECKPOINT_HASHES:
        raise RuntimeError(
            f"issue646 checkpoint hash drift: got={hashes} expected={CHECKPOINT_HASHES}"
        )

    return {
        "research_issue": RESEARCH_ISSUE,
        "continuation_issue": CONTINUATION_ISSUE,
        "bound_source_main": BOUND_SOURCE_MAIN,
        "bound_source_tree": BOUND_SOURCE_TREE,
        "blobs": blobs,
        "checkpoint_hashes": hashes,
        "issue645_trigger": ISSUE645_TRIGGER,
        "issue645_run": ISSUE645_RUN,
        "issue645_job": ISSUE645_JOB,
        "result_path_absent": True,
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
            f"refusing duplicate issue646 continuation because result exists: {RESULT_PATH}"
        )

    device = torch.cuda.get_device_name(0)
    if device != "NVIDIA L4":
        raise RuntimeError(f"issue646 requires exact NVIDIA L4, got {device!r}")

    print(
        L4_START_MARKER
        + json.dumps(
            {
                "research_issue": RESEARCH_ISSUE,
                "continuation_issue": CONTINUATION_ISSUE,
                "device": device,
                "bound_source_main": BOUND_SOURCE_MAIN,
                "issue643_adapter_blob": ISSUE643_ADAPTER_BLOB,
                "issue645_consumed_pre_modal": True,
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
    result["issue646_continuation_metadata"] = {
        "continuation_issue": CONTINUATION_ISSUE,
        "bound_source_main": BOUND_SOURCE_MAIN,
        "bound_source_tree": BOUND_SOURCE_TREE,
        "fresh_result_path": RESULT_PATH,
        "issue645_trigger": ISSUE645_TRIGGER,
        "issue645_run": ISSUE645_RUN,
        "issue645_job": ISSUE645_JOB,
        "issue645_consumed_pre_modal": True,
        "issue643_adapter_blob": ISSUE643_ADAPTER_BLOB,
        "base_systems_blob": BASE_SYSTEMS_BLOB,
        "repair1_systems_blob": REPAIR1_SYSTEMS_BLOB,
        "issue530_systems_blob": ISSUE530_SYSTEMS_BLOB,
        "issue562_systems_blob": ISSUE562_SYSTEMS_BLOB,
        "issue569_systems_blob": ISSUE569_SYSTEMS_BLOB,
        "v26_9_candidate_blob": V26_9_BACKEND_BLOB,
        "issue625_repair_blob": ISSUE625_REPAIR_BLOB,
        "issue641_adapter_blob": ISSUE641_ADAPTER_BLOB,
        "checkpoint_hashes": dict(CHECKPOINT_HASHES),
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
        "continuation_issue": CONTINUATION_ISSUE,
        "result_path": RESULT_PATH,
        "result_sha256": digest,
        "decision": result["decision"],
        "overall_pass": result["overall_pass"],
        "device": result["device"],
        "bound_source_main": BOUND_SOURCE_MAIN,
        "issue643_adapter_blob": ISSUE643_ADAPTER_BLOB,
        "issue645_consumed_pre_modal": True,
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
                "continuation_issue": CONTINUATION_ISSUE,
                "bound_source_main": BOUND_SOURCE_MAIN,
                "bound_source_tree": BOUND_SOURCE_TREE,
                "issue643_adapter_blob": ISSUE643_ADAPTER_BLOB,
                "result_path": RESULT_PATH,
                "max_gpu_seconds": MAX_GPU_SECONDS,
                "gpu": "L4",
                "gpu_remote_functions": 1,
                "issue645_consumed_pre_modal": True,
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
