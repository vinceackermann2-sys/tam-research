from __future__ import annotations

import hashlib
import json
from pathlib import Path

import modal

APP_NAME = "tam-research-aera-v26-8-issue594-stage0-post-read-amplification-localizer"
VOLUME_NAME = "tam-research-data"
RESULT_PATH = "/vol/aera-v26/issue594-stage0-post-read-amplification-localizer/result.json"
MAX_GPU_SECONDS = 300
RESEARCH_ISSUE = 594
SOURCE_MAIN = "9547bb0b4c340d793acb1e12c655dc3d22513234"
SOURCE_TREE = "6cd794ae3d893ab8f9f29e3d342c71aa081da324"

ISSUE594_LOCALIZER_BLOB = "2b72454ea74929ac7254cfc399bb2ab201dfc2cb"
ISSUE578_LOCALIZER_BLOB = "5933d1113b950348bb63e5f3eb8713ad36dbf580"
ISSUE581_WRAPPER_BLOB = "8800bb399e21b691e0d7703cc3eeaf486d3223b6"
ISSUE562_ADAPTER_BLOB = "3534103eea21f7c4d9d31798ad34601fd47090d6"
V26_8_CANDIDATE_BLOB = "3575c58d1cd730be77649f087908c51dbf3e6088"
V26_INTERFACE_BLOB = "268644ac4edee15a4cc4e29d3fed7f61eeb3caa7"
STABLE_REFERENCE_BLOB = "4e336b6e1a6238dac782fa320751d68281493ee1"

ISSUE588_RESULT_PATH = "/vol/aera-v26/issue588-first-divergence-guard-repair1/result.json"
ISSUE588_RESULT_SHA256 = "495c6f49210074580553aa4b55bf0970624a8abaee910f6d2bf7315e26d2a540"
ISSUE590_TRIGGER = 590
ISSUE590_RUN = 33753926605
ISSUE590_JOB = 100643674944
ISSUE593_TRIGGER = 593
ISSUE593_RUN = 33764045085
ISSUE593_JOB = 100677235816
ISSUE592_TESTED_HEAD = "45c007b52811c0b62f5da254fd2ae862d6ed81aa"
ISSUE592_CI_RUN = 33763382230
ISSUE592_CI_JOB = 100674987954
ISSUE592_MERGE = SOURCE_MAIN

CHECKPOINT_HASHES = {
    "aera": "f8aa92421801e8f190247e420632be5f0c20bc5ea8bf6bdeefe06686b3a31b30",
    "transformer": "cdd5cab4439a709468d6607d45d82081b33e876b2e40d91d4a38ba139b219dd7",
}

PRECHECK_MARKER = "AERA_V26_8_ISSUE594_STAGE0_POST_READ_AMPLIFICATION_PRECHECK_JSON="
PREFLIGHT_MARKER = "AERA_V26_8_ISSUE594_STAGE0_POST_READ_AMPLIFICATION_PREFLIGHT_JSON="
L4_START_MARKER = "AERA_V26_8_ISSUE594_STAGE0_POST_READ_AMPLIFICATION_L4_START_JSON="
RESULT_MARKER = "AERA_V26_8_ISSUE594_STAGE0_POST_READ_AMPLIFICATION_RESULT_JSON="
SUMMARY_MARKER = "AERA_V26_8_ISSUE594_STAGE0_POST_READ_AMPLIFICATION_SUMMARY_JSON="

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
            "unavailable_fields": comparison["unavailable_fields"],
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
    import tam_research.aera_hardware_core_v26_8_ficem_read_mixed_strength_precision as v26_8
    import tam_research.aera_v26_8_issue562_end_to_end_systems as issue562
    import tam_research.aera_v26_8_issue578_first_divergence_localizer as issue578
    import tam_research.aera_v26_8_issue581_first_divergence_no_grad as issue581
    import tam_research.aera_v26_8_issue594_stage0_post_read_amplification_localizer as issue594
    import tam_research.aera_v26_5_end_to_end_systems as base

    volume.reload()
    result_path = Path(RESULT_PATH)
    if result_path.exists():
        raise RuntimeError(
            f"refusing duplicate issue594 localization because result exists: {RESULT_PATH}"
        )

    predecessor_path = Path(ISSUE588_RESULT_PATH)
    if not predecessor_path.is_file():
        raise RuntimeError(f"issue594 predecessor durable result missing: {ISSUE588_RESULT_PATH}")
    predecessor_sha = _sha256(predecessor_path)
    if predecessor_sha != ISSUE588_RESULT_SHA256:
        raise RuntimeError(
            f"issue594 predecessor result SHA drift: got={predecessor_sha} expected={ISSUE588_RESULT_SHA256}"
        )

    blobs = {
        "issue594_localizer": _git_blob_sha(Path(issue594.__file__)),
        "issue578_localizer": _git_blob_sha(Path(issue578.__file__)),
        "issue581_wrapper": _git_blob_sha(Path(issue581.__file__)),
        "issue562_adapter": _git_blob_sha(Path(issue562.__file__)),
        "v26_8_candidate": _git_blob_sha(Path(v26_8.__file__)),
        "v26_interface": _git_blob_sha(Path(v26.__file__)),
        "stable_reference": _git_blob_sha(Path(stable.__file__)),
    }
    expected = {
        "issue594_localizer": ISSUE594_LOCALIZER_BLOB,
        "issue578_localizer": ISSUE578_LOCALIZER_BLOB,
        "issue581_wrapper": ISSUE581_WRAPPER_BLOB,
        "issue562_adapter": ISSUE562_ADAPTER_BLOB,
        "v26_8_candidate": V26_8_CANDIDATE_BLOB,
        "v26_interface": V26_INTERFACE_BLOB,
        "stable_reference": STABLE_REFERENCE_BLOB,
    }
    if blobs != expected:
        raise RuntimeError(f"issue594 frozen blob drift: got={blobs} expected={expected}")

    contract = issue594.cpu_contract_preflight_issue594()
    if contract["gpu_authorized_by_cpu_preflight"] is not False:
        raise RuntimeError("issue594 CPU preflight unexpectedly authorizes GPU")
    if contract["localization_measurement_performed"] is not False:
        raise RuntimeError("issue594 CPU preflight unexpectedly performed localization")
    if contract["checkpoint_loaded"] is not False:
        raise RuntimeError("issue594 CPU preflight unexpectedly loaded checkpoint")

    hashes = base.checkpoint_hashes(base.CHECKPOINT_RELATIVE_DIR)
    if hashes != CHECKPOINT_HASHES:
        raise RuntimeError(f"issue594 checkpoint hash drift: got={hashes} expected={CHECKPOINT_HASHES}")

    return {
        "research_issue": RESEARCH_ISSUE,
        "source_main": SOURCE_MAIN,
        "source_tree": SOURCE_TREE,
        "blobs": blobs,
        "checkpoint_hashes": hashes,
        "predecessor_result_path": ISSUE588_RESULT_PATH,
        "predecessor_result_sha256": predecessor_sha,
        "result_path": RESULT_PATH,
        "result_path_absent": True,
        "issue590_trigger": ISSUE590_TRIGGER,
        "issue590_run": ISSUE590_RUN,
        "issue590_job": ISSUE590_JOB,
        "issue593_trigger": ISSUE593_TRIGGER,
        "issue593_run": ISSUE593_RUN,
        "issue593_job": ISSUE593_JOB,
        "issue592_tested_head": ISSUE592_TESTED_HEAD,
        "issue592_ci_run": ISSUE592_CI_RUN,
        "issue592_ci_job": ISSUE592_CI_JOB,
        "issue592_merge": ISSUE592_MERGE,
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

    import tam_research.aera_v26_8_issue594_stage0_post_read_amplification_localizer as issue594
    import tam_research.aera_v26_5_end_to_end_systems as base

    volume.reload()
    path = Path(RESULT_PATH)
    if path.exists():
        raise RuntimeError(f"issue594 result already exists: {RESULT_PATH}")

    predecessor_path = Path(ISSUE588_RESULT_PATH)
    if _sha256(predecessor_path) != ISSUE588_RESULT_SHA256:
        raise RuntimeError("issue594 predecessor durable result changed before L4")

    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"issue594 requires exactly one CUDA device, got {torch.cuda.device_count()}")
    device_name = torch.cuda.get_device_name(0)
    if device_name != "NVIDIA L4":
        raise RuntimeError(f"issue594 requires NVIDIA L4, got {device_name}")

    hashes_before = base.checkpoint_hashes(base.CHECKPOINT_RELATIVE_DIR)
    if hashes_before != CHECKPOINT_HASHES:
        raise RuntimeError("issue594 checkpoint hashes changed before L4 localization")

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

    result = issue594.run_stage0_post_read_amplification_localization(
        run_dir=base.CHECKPOINT_RELATIVE_DIR
    )

    hashes_after = base.checkpoint_hashes(base.CHECKPOINT_RELATIVE_DIR)
    if hashes_after != hashes_before:
        raise RuntimeError("issue594 checkpoint hashes changed during localization")
    if result["parameter_versions_unchanged"] is not True:
        raise RuntimeError("issue594 parameter versions changed during localization")
    if result["checkpoint_hashes_unchanged"] is not True:
        raise RuntimeError("issue594 localizer reported checkpoint hash mutation")

    result["research_issue"] = RESEARCH_ISSUE
    result["predecessor_authority_trigger"] = ISSUE590_TRIGGER
    result["predecessor_readonly_inspector_trigger"] = ISSUE593_TRIGGER
    result["localization_complete"] = True
    result["decision"] = "COMPLETE_STAGE0_POST_READ_AMPLIFICATION_LOCALIZATION"
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
                "predecessor_result_sha256": ISSUE588_RESULT_SHA256,
                "result_path": RESULT_PATH,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    print(PREFLIGHT_MARKER + json.dumps(contract, sort_keys=True), flush=True)
    run_localization.remote()
