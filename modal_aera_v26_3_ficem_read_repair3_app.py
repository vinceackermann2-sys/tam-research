from __future__ import annotations

import hashlib
import json
from pathlib import Path

import modal

APP_NAME = "tam-research-aera-v26-3-issue433-ficem-read-repair3"
VOLUME_NAME = "tam-research-data"
RESULT_PATH = "/vol/aera-v26/issue433-ficem-read-repair3/result.json"
MAX_GPU_SECONDS = 300
FROZEN_PROBE_GIT_BLOB = "16aa99b9f6f0a1d11bd7bf5f36b2f0b1fb97047b"
MERGED_REPAIR3_BACKEND_GIT_BLOB = "b6b37f0379b280eea4e5c2b16f349951dadc4df9"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.10,<2.11", "numpy>=2.0,<3")
    .add_local_python_source("tam_research")
)


def _git_blob_sha(path: Path) -> str:
    data = path.read_bytes()
    header = f"blob {len(data)}\0".encode()
    return hashlib.sha1(header + data).hexdigest()


def _summary(result: dict) -> dict:
    localized = result["rows"]["bfloat16_batch8_mixed"]["correctness"]
    return {
        "decision": result["decision"],
        "overall_pass": result["overall_pass"],
        "device": result["device"],
        "geomean_latency_ratio_by_dtype": result["geomean_latency_ratio_by_dtype"],
        "correctness_pass": result["correctness_pass"],
        "known_empty_pass": result["known_empty_pass"],
        "near_tie_pass": result["near_tie_pass"],
        "row_latency_pass": result["row_latency_pass"],
        "full_event_ratio_pass": result["full_event_ratio_pass"],
        "single_tail_kernel_pass": result["single_tail_kernel_pass"],
        "candidate_no_reference_tail_ops_pass": result[
            "candidate_no_reference_tail_ops_pass"
        ],
        "localized_bfloat16_batch8_mixed": {
            key: localized[key]
            for key in (
                "pass",
                "selection_semantically_equivalent",
                "distinct_selected_set_exact",
                "pre_out_recalled_close",
                "final_out_close",
                "query_and_normalized_keys_bit_exact",
                "source_unchanged",
                "finite",
                "dtype_device_shape_exact",
                "pre_out_max_abs_diff",
                "final_out_max_abs_diff",
                "tie_query_count",
                "tie_query_fraction",
            )
        },
        "row_ratios": {
            key: {
                "latency": row["latency_ratio_candidate_over_reference"],
                "events": row["full_cuda_event_ratio_candidate_over_reference"],
                "correctness": row["correctness"]["pass"],
                "tie_query_count": row["correctness"].get("tie_query_count", 0),
                "tie_query_fraction": row["correctness"].get("tie_query_fraction", 0.0),
                "selection_semantically_equivalent": row["correctness"].get(
                    "selection_semantically_equivalent", False
                ),
            }
            for key, row in result["rows"].items()
        },
    }


@app.function(image=image, cpu=4, memory=8192, timeout=300, volumes={"/vol": volume})
def preflight() -> dict:
    import tam_research.aera_hardware_core_v26_3_ficem_read_triton as backend_module
    import tam_research.aera_v26_3_ficem_read_probe as probe_module
    from tam_research.aera_hardware_core_v26_3_ficem_read_triton import (
        fused_ficem_read_v26_3_protocol,
    )
    from tam_research.aera_v26_3_ficem_read_probe import cpu_contract_preflight

    volume.reload()
    result_path = Path(RESULT_PATH)
    if result_path.exists():
        raise RuntimeError(
            f"refusing duplicate issue433 FICEM read repair3 run because result exists: {RESULT_PATH}"
        )

    probe_blob = _git_blob_sha(Path(probe_module.__file__))
    backend_blob = _git_blob_sha(Path(backend_module.__file__))
    if probe_blob != FROZEN_PROBE_GIT_BLOB:
        raise RuntimeError(f"issue433 frozen probe blob drifted: {probe_blob}")
    if backend_blob != MERGED_REPAIR3_BACKEND_GIT_BLOB:
        raise RuntimeError(f"issue433 merged repair3 backend blob drifted: {backend_blob}")

    protocol = fused_ficem_read_v26_3_protocol()
    if protocol.get("bf16_reference_rounding_repair3") is not True:
        raise RuntimeError("issue433 repair3 protocol marker is missing")
    if protocol.get("float32_path_changed_by_repair3") is not False:
        raise RuntimeError("issue433 float32 path boundary drifted")

    check = {
        "contract": cpu_contract_preflight(),
        "research_issue": 433,
        "source_repair_issue": 426,
        "source_localization_issue": 423,
        "source_cpu_pr": 427,
        "source_cpu_actions_run": 33503499118,
        "source_cpu_job": 99842027190,
        "source_cpu_head": "0ad68f50c5ca937a0de9e4bd1c5464e1c0aeab24",
        "source_merged_main": "7e1346709d3c1eb158c9ec7d621cafdf498da315",
        "probe_git_blob": probe_blob,
        "backend_git_blob": backend_blob,
        "result_path_absent": True,
        "gpu_authorized_by_preflight": False,
        "synthetic_only": True,
        "original_case_generation_preserved": True,
        "tie_aware_correctness_preserved": True,
        "repair3_backend_frozen": True,
        "model_loaded": False,
        "checkpoint_loaded": False,
        "corpus_accessed": False,
        "training_performed": False,
        "optimizer_created": False,
        "backward_performed": False,
        "scientific_seed_consumed": False,
    }
    print(
        "AERA_V26_ISSUE433_FICEM_READ_REPAIR3_PREFLIGHT_JSON="
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

    from tam_research.aera_v26_3_ficem_read_probe import run_ficem_read_probe

    volume.reload()
    result_path = Path(RESULT_PATH)
    if result_path.exists():
        raise RuntimeError(
            f"refusing duplicate issue433 FICEM read repair3 run because result exists: {RESULT_PATH}"
        )
    print(
        "AERA_V26_ISSUE433_FICEM_READ_REPAIR3_L4_START_JSON="
        + json.dumps(
            {
                "device": torch.cuda.get_device_name(0),
                "research_issue": 433,
                "source_repair_issue": 426,
                "source_localization_issue": 423,
                "source_cpu_pr": 427,
                "hard_gpu_timeout_seconds": MAX_GPU_SECONDS,
                "synthetic_only": True,
                "original_case_generation_preserved": True,
                "tie_aware_correctness_preserved": True,
                "repair3_backend_frozen": True,
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
                "100m_authorized": False,
                "breakthrough_proven": False,
            },
            separators=(",", ":"),
        ),
        flush=True,
    )

    result = run_ficem_read_probe()
    localized = result["rows"]["bfloat16_batch8_mixed"]["correctness"]
    required_localized = (
        "pass",
        "selection_semantically_equivalent",
        "distinct_selected_set_exact",
        "pre_out_recalled_close",
        "final_out_close",
        "query_and_normalized_keys_bit_exact",
        "source_unchanged",
        "finite",
        "dtype_device_shape_exact",
    )
    if not all(bool(localized[key]) for key in required_localized):
        raise RuntimeError(
            "issue433 localized bfloat16_batch8_mixed row did not resolve all #423 correctness subgates"
        )

    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2))
    volume.commit()
    print(
        "AERA_V26_ISSUE433_FICEM_READ_REPAIR3_RESULT_JSON="
        + json.dumps(result, separators=(",", ":")),
        flush=True,
    )
    print(
        "AERA_V26_ISSUE433_FICEM_READ_REPAIR3_SUMMARY_JSON="
        + json.dumps(_summary(result), separators=(",", ":")),
        flush=True,
    )
    return result


@app.local_entrypoint()
def main() -> None:
    check = preflight.remote()
    print(
        "AERA_V26_ISSUE433_FICEM_READ_REPAIR3_PREFLIGHT_LOCAL="
        + json.dumps(check, separators=(",", ":")),
        flush=True,
    )
    result = run_probe.remote()
    print(json.dumps(_summary(result), indent=2), flush=True)
