from __future__ import annotations

import hashlib
import json
from pathlib import Path

import modal

APP_NAME = "aera-v26-9-issue622-corrected-autocast-dtype-gate-l4"
VOLUME_NAME = "tam-research-data"
MOUNT_PATH = "/vol"
RESULT_PATH = "/vol/aera-v26/issue622-corrected-autocast-dtype-gate/result.json"
ISSUE602_RESULT_PATH = "/vol/aera-v26/issue602-identity-weight-visibility/result.json"
ISSUE602_RESULT_SHA256 = "5ab64b2aa9750babebec6e681c7be587f079436436b5a3cda86ac809018256fb"
MAX_GPU_SECONDS = 300
RESEARCH_ISSUE = 622
SOURCE_MAIN = "caa7b019e9232d607d69b0e422e6d9550d675ff4"
SOURCE_TREE = "fd76f479a16036bcc81d3e48ba70956fc79c409e"
DESIGN_SEED = 891_475_817

PROBE_BLOB = "4e08ac9af18f666f09009e4d2c5822b11e91c2c1"
V26_9_BACKEND_BLOB = "b81cc209f5d95abbe1fb8bd620c78e87c067bc19"
ISSUE602_PROBE_BLOB = "456203f515d67d1c92b0a9c3e0e59ce4137ac10a"

PRECHECK_MARKER = "AERA_V26_9_ISSUE622_CORRECTED_AUTOCAST_DTYPE_GATE_PRECHECK_JSON="
PREFLIGHT_MARKER = "AERA_V26_9_ISSUE622_CORRECTED_AUTOCAST_DTYPE_GATE_PREFLIGHT_JSON="
L4_START_MARKER = "AERA_V26_9_ISSUE622_CORRECTED_AUTOCAST_DTYPE_GATE_L4_START_JSON="
RESULT_MARKER = "AERA_V26_9_ISSUE622_CORRECTED_AUTOCAST_DTYPE_GATE_RESULT_JSON="
SUMMARY_MARKER = "AERA_V26_9_ISSUE622_CORRECTED_AUTOCAST_DTYPE_GATE_SUMMARY_JSON="

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


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def precheck() -> dict:
    return {
        "research_issue": RESEARCH_ISSUE,
        "source_main": SOURCE_MAIN,
        "source_tree": SOURCE_TREE,
        "design_seed": DESIGN_SEED,
        "probe_blob": PROBE_BLOB,
        "v26_9_backend_blob": V26_9_BACKEND_BLOB,
        "issue602_result_sha256": ISSUE602_RESULT_SHA256,
        "result_path": RESULT_PATH,
        "max_gpu_seconds": MAX_GPU_SECONDS,
        "gpu": "L4",
        "gpu_remote_functions": 1,
        "scientific_seed_consumed": False,
        "end_to_end_systems_authorized": False,
        "architecture_freeze_authorized": False,
        "s2_authorized": False,
        "fresh_scientific_seed_authorized": False,
        "independent_replication_credit": False,
        "100m_authorized": False,
        "breakthrough_proven": False,
    }


def _summary(result: dict) -> dict:
    rows = {
        key: {
            "pass": row["pass"],
            "batch_size": row["batch_size"],
            "validity_kind": row["validity_kind"],
            "issue602_dtype_split_exact": row["issue602_dtype_split_exact"],
            "dtype_split_exact": row["dtype_split_exact"],
            "non_dtype_pass": row["non_dtype_pass"],
            "identity_dtype": row["identity_dtype"],
            "context_dtype": row["context_dtype"],
            "projected_query_dtype": row["projected_query_dtype"],
            "similarity_dtype": row["similarity_dtype"],
            "keys_dtype": row["keys_dtype"],
            "values_dtype": row["values_dtype"],
            "strengths_dtype": row["strengths_dtype"],
            "valid_dtype": row["valid_dtype"],
            "normalized_keys_dtype": row["normalized_keys_dtype"],
            "selection_semantically_equivalent": row["selection_semantically_equivalent"],
            "pre_out_recalled_close": row["pre_out_recalled_close"],
            "final_out_close": row["final_out_close"],
            "query_and_normalized_keys_bit_exact": row["query_and_normalized_keys_bit_exact"],
            "source_unchanged": row["source_unchanged"],
            "finite": row["finite"],
            "dtype_device_shape_exact": row["dtype_device_shape_exact"],
            "direct_tail_topology_pass": row["direct_tail_topology_pass"],
            "full_backend_no_reference_tail_ops": row["full_backend_no_reference_tail_ops"],
            "pre_out_max_abs_diff": row["pre_out_max_abs_diff"],
            "final_out_max_abs_diff": row["final_out_max_abs_diff"],
            "atol": row["atol"],
            "rtol": row["rtol"],
            "timing_decision_bearing": row["timing_decision_bearing"],
        }
        for key, row in result["fresh_integrated"]["rows"].items()
    }
    return {
        "research_issue": RESEARCH_ISSUE,
        "decision": result["decision"],
        "overall_pass": result["overall_pass"],
        "device": result["device"],
        "design_seed": DESIGN_SEED,
        "issue602_authority": result["issue602_authority"],
        "fresh_integrated": {
            "rows_pass": result["fresh_integrated"]["rows_pass"],
            "overall_pass": result["fresh_integrated"]["overall_pass"],
            "timing_decision_bearing": False,
            "rows": rows,
        },
        "scientific_seed_consumed": False,
        "end_to_end_systems_authorized": False,
        "architecture_freeze_authorized": False,
        "s2_authorized": False,
        "fresh_scientific_seed_authorized": False,
        "independent_replication_credit": False,
        "100m_authorized": False,
        "breakthrough_proven": False,
    }


@app.function(image=image, cpu=4, memory=8192, timeout=300, volumes={MOUNT_PATH: volume})
def preflight() -> dict:
    import tam_research.aera_hardware_core_v26_9_ficem_read_identity_weight_visibility as candidate
    import tam_research.aera_v26_9_issue602_identity_weight_visibility_probe as issue602
    import tam_research.aera_v26_9_issue622_corrected_autocast_dtype_gate as probe

    volume.reload()
    result_path = Path(RESULT_PATH)
    if result_path.exists():
        raise RuntimeError(
            f"refusing duplicate issue622 run because durable result exists: {RESULT_PATH}"
        )
    source_result = Path(ISSUE602_RESULT_PATH)
    if not source_result.is_file():
        raise RuntimeError("issue622 immutable issue602 result missing")
    source_digest = _sha256_path(source_result)
    if source_digest != ISSUE602_RESULT_SHA256:
        raise RuntimeError(
            f"issue622 immutable issue602 SHA drift: {source_digest}"
        )

    blobs = {
        "probe": _git_blob_sha(Path(probe.__file__)),
        "v26_9_backend": _git_blob_sha(Path(candidate.__file__)),
        "issue602_probe": _git_blob_sha(Path(issue602.__file__)),
    }
    expected = {
        "probe": PROBE_BLOB,
        "v26_9_backend": V26_9_BACKEND_BLOB,
        "issue602_probe": ISSUE602_PROBE_BLOB,
    }
    if blobs != expected:
        raise RuntimeError(f"issue622 frozen blob drift: got={blobs} expected={expected}")

    contract = probe.cpu_contract_preflight_issue622()
    if contract["gpu_authorized_by_cpu_preflight"] is not False:
        raise RuntimeError("issue622 CPU preflight unexpectedly authorizes GPU")
    authority = probe.load_issue602_preserved_authority(source_result)
    return {
        "research_issue": RESEARCH_ISSUE,
        "source_main": SOURCE_MAIN,
        "source_tree": SOURCE_TREE,
        "design_seed": DESIGN_SEED,
        "blobs": blobs,
        "issue602_result_sha256": source_digest,
        "issue602_decision": authority["issue602_decision"],
        "preserved_issue558_overall_pass": authority["preserved_issue558"]["overall_pass"],
        "result_path_absent": True,
        "gpu_executed": False,
        "model_loaded": False,
        "checkpoint_loaded": False,
        "corpus_accessed": False,
        "training_performed": False,
        "optimizer_created": False,
        "backward_performed": False,
        "scientific_seed_consumed": False,
        "end_to_end_systems_authorized": False,
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
    volumes={MOUNT_PATH: volume},
)
def run_gate() -> dict:
    import torch
    from tam_research.aera_v26_9_issue622_corrected_autocast_dtype_gate import (
        run_corrected_autocast_dtype_gate_v26_9_issue622,
    )

    volume.reload()
    result_path = Path(RESULT_PATH)
    if result_path.exists():
        raise RuntimeError(
            f"refusing duplicate issue622 run because durable result exists: {RESULT_PATH}"
        )
    source_result = Path(ISSUE602_RESULT_PATH)
    if _sha256_path(source_result) != ISSUE602_RESULT_SHA256:
        raise RuntimeError("issue622 immutable issue602 result changed before L4")

    print(
        L4_START_MARKER
        + json.dumps(
            {
                "research_issue": RESEARCH_ISSUE,
                "device": torch.cuda.get_device_name(0),
                "source_main": SOURCE_MAIN,
                "probe_blob": PROBE_BLOB,
                "v26_9_backend_blob": V26_9_BACKEND_BLOB,
                "design_seed": DESIGN_SEED,
                "scientific_seed_consumed": False,
                "end_to_end_systems_authorized": False,
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

    result = run_corrected_autocast_dtype_gate_v26_9_issue622(
        issue602_result_path=ISSUE602_RESULT_PATH
    )
    result["issue622_gate_metadata"] = {
        "source_main": SOURCE_MAIN,
        "source_tree": SOURCE_TREE,
        "probe_blob": PROBE_BLOB,
        "v26_9_backend_blob": V26_9_BACKEND_BLOB,
        "issue602_probe_blob": ISSUE602_PROBE_BLOB,
        "issue602_result_sha256": ISSUE602_RESULT_SHA256,
        "design_seed": DESIGN_SEED,
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
        "v26_9_backend_blob": V26_9_BACKEND_BLOB,
        "design_seed": DESIGN_SEED,
    }
    print(RESULT_MARKER + json.dumps(marker, separators=(",", ":")), flush=True)
    print(SUMMARY_MARKER + json.dumps(_summary(result), separators=(",", ":")), flush=True)
    return result


@app.local_entrypoint()
def main() -> None:
    print(PRECHECK_MARKER + json.dumps(precheck(), separators=(",", ":")), flush=True)
    check = preflight.remote()
    print(PREFLIGHT_MARKER + json.dumps(check, separators=(",", ":")), flush=True)
    result = run_gate.remote()
    print(json.dumps(_summary(result), indent=2), flush=True)
