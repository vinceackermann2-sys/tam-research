from __future__ import annotations

import hashlib
import json
from pathlib import Path

import modal

APP_NAME = "aera-v26-8-issue588-first-divergence-guard-repair1"
VOLUME_NAME = "tam-research-data"
RESULT_PATH = "/vol/aera-v26/issue588-first-divergence-guard-repair1/result.json"
MAX_GPU_SECONDS = 300
RESEARCH_ISSUE = 588
SOURCE_MAIN = "ad3b56106ea80956dcd1e8b457a5ef3169507351"
SOURCE_TREE = "3c3940f4e910c5bc7d6155829183c03889f305bd"

ISSUE584_LAUNCHER_BLOB = "25842ba4c02546e4773764354919de683d5fd6f0"
ISSUE584_WORKFLOW_BLOB = "25d420e4f30fbbe8ac2fbb93419039db7b54bfa6"
ISSUE584_CPU_TEST_BLOB = "0e5995adb6bf6aa99e875115a9490783fb2287ff"
ISSUE584_TESTED_HEAD = "58bfecc6a00214f827862a9e38befd5b03d8db89"
ISSUE584_CI_RUN = 33751191196
ISSUE584_CI_JOB = 100634879254
ISSUE584_MERGE = SOURCE_MAIN

ISSUE587_TRIGGER = 587
ISSUE587_RUN = 33752190666
ISSUE587_JOB = 100638055124
ISSUE587_ATTEMPT = 1
ISSUE587_BOUND_MAIN = SOURCE_MAIN
ISSUE587_FAILURE = "Resource not accessible by integration (HTTP 403)"
ISSUE587_MODAL_AUTH_SKIPPED = True
ISSUE587_L4_SKIPPED = True

ISSUE581_WRAPPER_BLOB = "8800bb399e21b691e0d7703cc3eeaf486d3223b6"
ISSUE581_CPU_TEST_BLOB = "6b5218bcc744dd8d9cbca65c9b5c7c0c1578f5e9"
ISSUE578_LOCALIZER_BLOB = "5933d1113b950348bb63e5f3eb8713ad36dbf580"
ISSUE562_ADAPTER_BLOB = "3534103eea21f7c4d9d31798ad34601fd47090d6"
V26_8_CANDIDATE_BLOB = "3575c58d1cd730be77649f087908c51dbf3e6088"
V26_6_WRITE_BLOB = "d45c262314a0b4691f26812a279937a225043ad9"
V26_INTERFACE_BLOB = "268644ac4edee15a4cc4e29d3fed7f61eeb3caa7"
STABLE_REFERENCE_BLOB = "4e336b6e1a6238dac782fa320751d68281493ee1"
ISSUE571_RESULT_SHA256 = "afeeb62351cc4fb97d272c5b55c9621839e26f83753ae1fb237733d58a5ee472"

CHECKPOINT_HASHES = {
    "aera": "f8aa92421801e8f190247e420632be5f0c20bc5ea8bf6bdeefe06686b3a31b30",
    "transformer": "cdd5cab4439a709468d6607d45d82081b33e876b2e40d91d4a38ba139b219dd7",
}

PRECHECK_MARKER = "AERA_V26_8_ISSUE588_FIRST_DIVERGENCE_GUARD_REPAIR1_PRECHECK_JSON="
PREFLIGHT_MARKER = "AERA_V26_8_ISSUE588_FIRST_DIVERGENCE_GUARD_REPAIR1_PREFLIGHT_JSON="
L4_START_MARKER = "AERA_V26_8_ISSUE588_FIRST_DIVERGENCE_GUARD_REPAIR1_L4_START_JSON="
RESULT_MARKER = "AERA_V26_8_ISSUE588_FIRST_DIVERGENCE_GUARD_REPAIR1_RESULT_JSON="
SUMMARY_MARKER = "AERA_V26_8_ISSUE588_FIRST_DIVERGENCE_GUARD_REPAIR1_SUMMARY_JSON="

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.10,<2.11", "numpy>=2.0,<3", "triton>=3.6,<3.7")
    .add_local_python_source("tam_research")
)


def _git_blob_sha(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _summary(result: dict) -> dict:
    rows = {}
    for batch, row in result["rows"].items():
        comparison = row["comparison"]
        rows[batch] = {
            "first_bitwise_difference": comparison["first_bitwise_difference"],
            "first_integrated_tolerance_or_metadata_failure": comparison[
                "first_integrated_tolerance_or_metadata_failure"
            ],
            "first_discrete_decision_difference": comparison[
                "first_discrete_decision_difference"
            ],
            "failure_count": len(comparison["failures"]),
            "candidate_backend_internal_adjudication_decisions_available": comparison[
                "candidate_backend_internal_adjudication_decisions_available"
            ],
        }
    return {
        "research_issue": RESEARCH_ISSUE,
        "decision": result["decision"],
        "localization_complete": result["localization_complete"],
        "device": result["device"],
        "candidate_backend_names": result["candidate_backend_names"],
        "parameter_versions_unchanged": result["parameter_versions_unchanged"],
        "checkpoint_hashes_unchanged": result["checkpoint_hashes_unchanged"],
        "rows": rows,
        "repair_authorized": False,
        "architecture_freeze_authorized": False,
        "s2_authorized": False,
        "fresh_scientific_seed_authorized": False,
        "independent_replication_credit": False,
        "100m_authorized": False,
        "breakthrough_proven": False,
    }


@app.function(image=image, cpu=4, memory=8192, timeout=240, volumes={"/vol": volume})
def preflight() -> dict:
    import tam_research.aera_hardware_core_v25_1_compact as stable
    import tam_research.aera_hardware_core_v26 as v26
    import tam_research.aera_hardware_core_v26_6_ficem_write_materialize_cast as write_v26_6
    import tam_research.aera_hardware_core_v26_8_ficem_read_mixed_strength_precision as v26_8
    import tam_research.aera_v26_8_issue562_end_to_end_systems as issue562
    import tam_research.aera_v26_8_issue578_first_divergence_localizer as issue578
    import tam_research.aera_v26_8_issue581_first_divergence_no_grad as issue581
    import tam_research.aera_v26_5_end_to_end_systems as base

    volume.reload()
    result_path = Path(RESULT_PATH)
    if result_path.exists():
        raise RuntimeError(
            f"refusing duplicate issue588 localization because result exists: {RESULT_PATH}"
        )

    blobs = {
        "issue581_wrapper": _git_blob_sha(Path(issue581.__file__)),
        "issue578_localizer": _git_blob_sha(Path(issue578.__file__)),
        "issue562_adapter": _git_blob_sha(Path(issue562.__file__)),
        "v26_8_candidate": _git_blob_sha(Path(v26_8.__file__)),
        "v26_6_write": _git_blob_sha(Path(write_v26_6.__file__)),
        "v26_interface": _git_blob_sha(Path(v26.__file__)),
        "stable_reference": _git_blob_sha(Path(stable.__file__)),
    }
    expected = {
        "issue581_wrapper": ISSUE581_WRAPPER_BLOB,
        "issue578_localizer": ISSUE578_LOCALIZER_BLOB,
        "issue562_adapter": ISSUE562_ADAPTER_BLOB,
        "v26_8_candidate": V26_8_CANDIDATE_BLOB,
        "v26_6_write": V26_6_WRITE_BLOB,
        "v26_interface": V26_INTERFACE_BLOB,
        "stable_reference": STABLE_REFERENCE_BLOB,
    }
    if blobs != expected:
        raise RuntimeError(f"issue588 frozen blob drift: got={blobs} expected={expected}")

    contract = issue581.cpu_contract_preflight_issue581()
    if contract["gpu_authorized_by_cpu_preflight"] is not False:
        raise RuntimeError("issue581 CPU preflight unexpectedly authorizes GPU")
    if contract["localization_measurement_performed"] is not False:
        raise RuntimeError("issue581 CPU preflight unexpectedly performed localization")
    if contract["checkpoint_loaded"] is not False:
        raise RuntimeError("issue581 CPU preflight unexpectedly loaded checkpoint")
    if contract["protocol"]["issue580_authoritative_result_emitted"] is not False:
        raise RuntimeError("issue581 contract unexpectedly reclassifies consumed #580")

    hashes = base.checkpoint_hashes(base.CHECKPOINT_RELATIVE_DIR)
    if hashes != CHECKPOINT_HASHES:
        raise RuntimeError(f"issue588 checkpoint hash drift: got={hashes} expected={CHECKPOINT_HASHES}")

    return {
        "research_issue": RESEARCH_ISSUE,
        "source_main": SOURCE_MAIN,
        "source_tree": SOURCE_TREE,
        "blobs": blobs,
        "checkpoint_hashes": hashes,
        "result_path": RESULT_PATH,
        "result_path_absent": True,
        "issue584_tested_head": ISSUE584_TESTED_HEAD,
        "issue584_ci_run": ISSUE584_CI_RUN,
        "issue584_ci_job": ISSUE584_CI_JOB,
        "issue584_merge": ISSUE584_MERGE,
        "issue587_trigger": ISSUE587_TRIGGER,
        "issue587_run": ISSUE587_RUN,
        "issue587_job": ISSUE587_JOB,
        "issue587_attempt": ISSUE587_ATTEMPT,
        "issue587_bound_main": ISSUE587_BOUND_MAIN,
        "issue587_failure": ISSUE587_FAILURE,
        "issue587_modal_auth_skipped": ISSUE587_MODAL_AUTH_SKIPPED,
        "issue587_l4_skipped": ISSUE587_L4_SKIPPED,
        "issue571_result_sha256": ISSUE571_RESULT_SHA256,
        "gpu_authorized_by_preflight": False,
        "localization_measurement_performed": False,
        "scientific_seed_consumed": False,
        "repair_authorized": False,
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
    memory=24576,
    timeout=MAX_GPU_SECONDS,
    volumes={"/vol": volume},
)
def run_localization() -> dict:
    import torch

    import tam_research.aera_v26_8_issue581_first_divergence_no_grad as issue581
    import tam_research.aera_v26_5_end_to_end_systems as base

    volume.reload()
    path = Path(RESULT_PATH)
    if path.exists():
        raise RuntimeError(f"issue588 result already exists: {RESULT_PATH}")

    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"issue588 requires exactly one CUDA device, got {torch.cuda.device_count()}")
    device_name = torch.cuda.get_device_name(0)
    if device_name != "NVIDIA L4":
        raise RuntimeError(f"issue588 requires NVIDIA L4, got {device_name}")

    hashes_before = base.checkpoint_hashes(base.CHECKPOINT_RELATIVE_DIR)
    if hashes_before != CHECKPOINT_HASHES:
        raise RuntimeError("issue588 checkpoint hashes changed before L4 localization")

    print(
        L4_START_MARKER
        + json.dumps(
            {
                "research_issue": RESEARCH_ISSUE,
                "device": device_name,
                "device_count": 1,
                "result_path": RESULT_PATH,
                "checkpoint_hashes": hashes_before,
                "training_performed": False,
                "scientific_seed_consumed": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    result = issue581.run_first_divergence_localization_issue581(
        run_dir=base.CHECKPOINT_RELATIVE_DIR
    )

    hashes_after = base.checkpoint_hashes(base.CHECKPOINT_RELATIVE_DIR)
    if hashes_after != hashes_before:
        raise RuntimeError("issue588 checkpoint hashes changed during localization")
    if result["parameter_versions_unchanged"] is not True:
        raise RuntimeError("issue588 parameter versions changed during localization")
    if result["checkpoint_hashes_unchanged"] is not True:
        raise RuntimeError("issue588 localizer reported checkpoint hash mutation")

    result["research_issue"] = RESEARCH_ISSUE
    result["successor_of_consumed_trigger"] = ISSUE587_TRIGGER
    result["localization_complete"] = True
    result["decision"] = "COMPLETE_FIRST_DIVERGENCE_LOCALIZATION"
    result["result_path"] = RESULT_PATH
    result["volume_mutated_only_by_new_result_write"] = True
    result["repair_authorized"] = False
    result["architecture_freeze_authorized"] = False
    result["s2_authorized"] = False
    result["fresh_scientific_seed_authorized"] = False
    result["independent_replication_credit"] = False
    result["100m_authorized"] = False
    result["breakthrough_proven"] = False

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    encoded = json.dumps(result, sort_keys=True, separators=(",", ":"))
    tmp.write_text(encoded)
    tmp.replace(path)
    volume.commit()
    digest = _sha256(path)

    authoritative = {
        "research_issue": RESEARCH_ISSUE,
        "decision": result["decision"],
        "localization_complete": True,
        "result_path": RESULT_PATH,
        "result_sha256": digest,
        "parameter_versions_unchanged": True,
        "checkpoint_hashes_unchanged": True,
        "repair_authorized": False,
        "architecture_freeze_authorized": False,
        "s2_authorized": False,
        "fresh_scientific_seed_authorized": False,
        "independent_replication_credit": False,
        "100m_authorized": False,
        "breakthrough_proven": False,
    }
    print(RESULT_MARKER + json.dumps(authoritative, sort_keys=True), flush=True)
    print(SUMMARY_MARKER + json.dumps(_summary(result), sort_keys=True), flush=True)
    return authoritative


@app.local_entrypoint()
def main() -> None:
    contract = preflight.remote()
    print(
        PRECHECK_MARKER
        + json.dumps(
            {
                "research_issue": RESEARCH_ISSUE,
                "source_main": SOURCE_MAIN,
                "source_tree": SOURCE_TREE,
                "result_path": RESULT_PATH,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    print(PREFLIGHT_MARKER + json.dumps(contract, sort_keys=True), flush=True)
    run_localization.remote()
