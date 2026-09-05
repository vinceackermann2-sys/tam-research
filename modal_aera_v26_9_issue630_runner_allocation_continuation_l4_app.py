from __future__ import annotations

import hashlib
import json
from pathlib import Path

import modal

APP_NAME = "aera-v26-9-issue630-runner-allocation-continuation-l4"
VOLUME_NAME = "tam-research-data"
SOURCE_RESULT_PATH = "/vol/aera-v26/issue602-identity-weight-visibility/result.json"
SOURCE_RESULT_SHA256 = "5ab64b2aa9750babebec6e681c7be587f079436436b5a3cda86ac809018256fb"
RESULT_PATH = "/vol/aera-v26/issue630-runner-allocation-continuation/result.json"
MAX_GPU_SECONDS = 300
RESEARCH_ISSUE = 630
SOURCE_MAIN = "96b42b0b3dfae4e06209bac6fb55ce02e2960034"
SOURCE_TREE = "6145cc9cba48ac878b3149f34a4666fe310e4852"
ISSUE625_REPAIR_BLOB = "92d06a4954bca1b302355e81f5bf09b06fcee222"
V26_9_BACKEND_BLOB = "b81cc209f5d95abbe1fb8bd620c78e87c067bc19"
ISSUE602_PROBE_BLOB = "456203f515d67d1c92b0a9c3e0e59ce4137ac10a"
DESIGN_SEED = 891_475_817

PRECHECK_MARKER = "AERA_V26_9_ISSUE630_RUNNER_ALLOCATION_CONTINUATION_PRECHECK_JSON="
PREFLIGHT_MARKER = "AERA_V26_9_ISSUE630_RUNNER_ALLOCATION_CONTINUATION_PREFLIGHT_JSON="
L4_START_MARKER = "AERA_V26_9_ISSUE630_RUNNER_ALLOCATION_CONTINUATION_L4_START_JSON="
RESULT_MARKER = "AERA_V26_9_ISSUE630_RUNNER_ALLOCATION_CONTINUATION_RESULT_JSON="
SUMMARY_MARKER = "AERA_V26_9_ISSUE630_RUNNER_ALLOCATION_CONTINUATION_SUMMARY_JSON="

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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def precheck() -> dict:
    return {
        "research_issue": RESEARCH_ISSUE,
        "source_main": SOURCE_MAIN,
        "source_tree": SOURCE_TREE,
        "source_result_path": SOURCE_RESULT_PATH,
        "source_result_sha256": SOURCE_RESULT_SHA256,
        "result_path": RESULT_PATH,
        "design_seed": DESIGN_SEED,
        "issue625_repair_blob": ISSUE625_REPAIR_BLOB,
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
            "dtype_split_exact": row["dtype_split_exact"],
            "issue602_dtype_split_exact": row["issue602_dtype_split_exact"],
            "non_dtype_pass": row["non_dtype_pass"],
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
            "timing_decision_bearing": row["timing_decision_bearing"],
        }
        for key, row in result["fresh_integrated"]["rows"].items()
    }
    return {
        "research_issue": RESEARCH_ISSUE,
        "decision": result["decision"],
        "overall_pass": result["overall_pass"],
        "device": result["device"],
        "issue602_authority": result["issue602_authority"],
        "fresh_integrated": {
            "rows_pass": result["fresh_integrated"]["rows_pass"],
            "overall_pass": result["fresh_integrated"]["overall_pass"],
            "timing_decision_bearing": result["fresh_integrated"]["timing_decision_bearing"],
            "rows": rows,
        },
        "scientific_seed_consumed": result["scientific_seed_consumed"],
        "end_to_end_systems_authorized": result["end_to_end_systems_authorized"],
        "architecture_freeze_authorized": result["architecture_freeze_authorized"],
        "s2_authorized": result["s2_authorized"],
        "fresh_scientific_seed_authorized": result["fresh_scientific_seed_authorized"],
        "independent_replication_credit": result["independent_replication_credit"],
        "100m_authorized": result["100m_authorized"],
        "breakthrough_proven": result["breakthrough_proven"],
    }


@app.function(image=image, cpu=4, memory=8192, timeout=300, volumes={"/vol": volume})
def preflight() -> dict:
    import tam_research.aera_hardware_core_v26_9_ficem_read_identity_weight_visibility as backend
    import tam_research.aera_v26_9_issue602_identity_weight_visibility_probe as issue602
    import tam_research.aera_v26_9_issue625_schema_guard_repair1 as repair

    volume.reload()
    source = Path(SOURCE_RESULT_PATH)
    result_path = Path(RESULT_PATH)
    if result_path.exists():
        raise RuntimeError(
            f"refusing duplicate issue630 run because durable result exists: {RESULT_PATH}"
        )
    if not source.is_file() or _sha256(source) != SOURCE_RESULT_SHA256:
        raise RuntimeError("issue630 immutable issue602 source missing or SHA drifted")

    blobs = {
        "issue625_repair": _git_blob_sha(Path(repair.__file__)),
        "v26_9_backend": _git_blob_sha(Path(backend.__file__)),
        "issue602_probe": _git_blob_sha(Path(issue602.__file__)),
    }
    expected = {
        "issue625_repair": ISSUE625_REPAIR_BLOB,
        "v26_9_backend": V26_9_BACKEND_BLOB,
        "issue602_probe": ISSUE602_PROBE_BLOB,
    }
    if blobs != expected:
        raise RuntimeError(f"issue630 frozen source blob drift: got={blobs} expected={expected}")

    contract = repair.cpu_contract_preflight_issue625()
    if contract["gpu_authorized_by_cpu_preflight"] is not False:
        raise RuntimeError("issue630 CPU preflight unexpectedly authorizes GPU")
    authority = repair.load_issue602_preserved_authority_issue625(source)
    if (
        authority["issue602_decision"] != "FAIL"
        or authority["issue602_overall_pass"] is not False
        or authority["preserved_issue558"]["overall_pass"] is not True
    ):
        raise RuntimeError("issue630 immutable authority validation drifted")
    return {
        "research_issue": RESEARCH_ISSUE,
        "source_main": SOURCE_MAIN,
        "source_tree": SOURCE_TREE,
        "source_result_sha256": SOURCE_RESULT_SHA256,
        "result_path_absent": True,
        "blobs": blobs,
        "authority": authority,
        "design_seed": DESIGN_SEED,
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
    volumes={"/vol": volume},
)
def run_gate() -> dict:
    import torch
    from tam_research.aera_v26_9_issue625_schema_guard_repair1 import (
        run_schema_guard_repair1_gate_v26_9_issue625,
    )

    volume.reload()
    source = Path(SOURCE_RESULT_PATH)
    result_path = Path(RESULT_PATH)
    if result_path.exists():
        raise RuntimeError(
            f"refusing duplicate issue630 run because durable result exists: {RESULT_PATH}"
        )
    if not source.is_file() or _sha256(source) != SOURCE_RESULT_SHA256:
        raise RuntimeError(
            "issue630 immutable issue602 source missing or SHA drifted immediately before GPU row execution"
        )

    print(
        L4_START_MARKER
        + json.dumps(
            {
                "research_issue": RESEARCH_ISSUE,
                "device": torch.cuda.get_device_name(0),
                "source_main": SOURCE_MAIN,
                "source_result_sha256": SOURCE_RESULT_SHA256,
                "issue625_repair_blob": ISSUE625_REPAIR_BLOB,
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

    result = run_schema_guard_repair1_gate_v26_9_issue625(
        issue602_result_path=source
    )
    result["issue630_orchestration_metadata"] = {
        "source_main": SOURCE_MAIN,
        "source_tree": SOURCE_TREE,
        "issue625_repair_blob": ISSUE625_REPAIR_BLOB,
        "source_result_sha256": SOURCE_RESULT_SHA256,
        "design_seed": DESIGN_SEED,
        "issue629_consumed_pre_runner": True,
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
        "source_result_sha256": SOURCE_RESULT_SHA256,
        "issue625_repair_blob": ISSUE625_REPAIR_BLOB,
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
