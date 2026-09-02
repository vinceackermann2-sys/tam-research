from __future__ import annotations

import hashlib
import json
from pathlib import Path

import modal

APP_NAME = "tam-research-aera-v26-3-issue474-ficem-read-repair5"
VOLUME_NAME = "tam-research-data"
RESULT_PATH = "/vol/aera-v26/issue474-ficem-read-repair5/result.json"
MAX_GPU_SECONDS = 300
FROZEN_PROBE_GIT_BLOB = "16aa99b9f6f0a1d11bd7bf5f36b2f0b1fb97047b"
MERGED_REPAIR5_BACKEND_GIT_BLOB = "263f68eb1186a8ac14a08fc4b4df1fc5b292c711"
SOURCE_REPAIR_ISSUE = 470
SOURCE_CPU_PR = 473
SOURCE_CPU_RUN = 33606059884
SOURCE_CPU_JOB = 100170186089
SOURCE_CPU_HEAD = "55a8262d144a4644a392ea2ab81eda99124518ca"
SOURCE_MERGED_MAIN = "34be2d4f1311fb00acdc5acf14b4914fb80c6bd5"

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
            }
            for key, row in result["rows"].items()
        },
    }


def _required_repair5_protocol() -> dict[str, bool]:
    return {
        "bf16_reference_rounding_repair3": True,
        "float32_path_changed_by_repair3": False,
        "bf16_product_rounding_repair4": True,
        "float32_path_changed_by_repair4": False,
        "bf16_actual_autocast_tail_repair5": True,
        "bf16_strength_bias_fp32_repair5": True,
        "bf16_logits_fp32_repair5": True,
        "bf16_final_weights_fp32_repair5": True,
        "bf16_recalled_fp32_repair5": True,
        "bf16_product_rounding_active_after_repair5": False,
        "float32_path_changed_by_repair5": False,
        "gpu_authorized_by_module": False,
        "scientific_training_authorized": False,
        "end_to_end_systems_authorized": False,
        "architecture_freeze_authorized": False,
        "s2_authorized": False,
        "fresh_scientific_seed_authorized": False,
        "100m_authorized": False,
        "breakthrough_proven": False,
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
            f"refusing duplicate issue474 FICEM read repair5 run because result exists: {RESULT_PATH}"
        )

    probe_blob = _git_blob_sha(Path(probe_module.__file__))
    backend_blob = _git_blob_sha(Path(backend_module.__file__))
    if probe_blob != FROZEN_PROBE_GIT_BLOB:
        raise RuntimeError(f"issue474 frozen probe blob drifted: {probe_blob}")
    if backend_blob != MERGED_REPAIR5_BACKEND_GIT_BLOB:
        raise RuntimeError(f"issue474 merged repair5 backend blob drifted: {backend_blob}")

    protocol = fused_ficem_read_v26_3_protocol()
    for key, value in _required_repair5_protocol().items():
        if protocol.get(key) is not value:
            raise RuntimeError(
                f"issue474 protocol boundary drifted: {key}={protocol.get(key)!r}"
            )

    check = {
        "contract": cpu_contract_preflight(),
        "research_issue": 474,
        "source_repair_issue": SOURCE_REPAIR_ISSUE,
        "source_cpu_pr": SOURCE_CPU_PR,
        "source_cpu_actions_run": SOURCE_CPU_RUN,
        "source_cpu_job": SOURCE_CPU_JOB,
        "source_cpu_head": SOURCE_CPU_HEAD,
        "source_merged_main": SOURCE_MERGED_MAIN,
        "probe_git_blob": probe_blob,
        "backend_git_blob": backend_blob,
        "result_path_absent": True,
        "synthetic_only": True,
        "original_case_generation_preserved": True,
        "tie_aware_correctness_preserved": True,
        "model_loaded": False,
        "checkpoint_loaded": False,
        "corpus_accessed": False,
        "training_performed": False,
        "scientific_seed_consumed": False,
    }
    print(
        "AERA_V26_ISSUE474_FICEM_READ_REPAIR5_PREFLIGHT_JSON="
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
            f"refusing duplicate issue474 FICEM read repair5 run because result exists: {RESULT_PATH}"
        )

    print(
        "AERA_V26_ISSUE474_FICEM_READ_REPAIR5_L4_START_JSON="
        + json.dumps(
            {
                "device": torch.cuda.get_device_name(0),
                "research_issue": 474,
                "source_repair_issue": SOURCE_REPAIR_ISSUE,
                "source_merged_main": SOURCE_MERGED_MAIN,
                "candidate_backend_git_blob": MERGED_REPAIR5_BACKEND_GIT_BLOB,
                "frozen_probe_git_blob": FROZEN_PROBE_GIT_BLOB,
                "hard_gpu_timeout_seconds": MAX_GPU_SECONDS,
                "synthetic_only": True,
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

    # Exact frozen #411/#418 probe invocation. It intentionally raises immediately
    # on a correctness failure; #474 does not replace or weaken that behavior.
    result = run_ficem_read_probe()
    result["issue474_gate_metadata"] = {
        "research_issue": 474,
        "source_repair_issue": SOURCE_REPAIR_ISSUE,
        "source_cpu_pr": SOURCE_CPU_PR,
        "source_cpu_run": SOURCE_CPU_RUN,
        "source_cpu_job": SOURCE_CPU_JOB,
        "source_cpu_head": SOURCE_CPU_HEAD,
        "source_merged_main": SOURCE_MERGED_MAIN,
        "candidate_backend_git_blob": MERGED_REPAIR5_BACKEND_GIT_BLOB,
        "frozen_probe_git_blob": FROZEN_PROBE_GIT_BLOB,
        "synthetic_only": True,
        "scientific_seed_consumed": False,
    }

    result_path.parent.mkdir(parents=True, exist_ok=True)
    durable_json = json.dumps(result, sort_keys=True, separators=(",", ":"))
    result_path.write_text(durable_json)
    volume.commit()

    result_sha256 = hashlib.sha256(durable_json.encode()).hexdigest()
    marker = {
        "research_issue": 474,
        "result_path": RESULT_PATH,
        "result_sha256": result_sha256,
        "decision": result["decision"],
        "overall_pass": result["overall_pass"],
        "device": result["device"],
        "source_merged_main": SOURCE_MERGED_MAIN,
        "candidate_backend_git_blob": MERGED_REPAIR5_BACKEND_GIT_BLOB,
        "frozen_probe_git_blob": FROZEN_PROBE_GIT_BLOB,
    }
    print(
        "AERA_V26_ISSUE474_FICEM_READ_REPAIR5_RESULT_JSON="
        + json.dumps(marker, separators=(",", ":")),
        flush=True,
    )
    print(
        "AERA_V26_ISSUE474_FICEM_READ_REPAIR5_SUMMARY_JSON="
        + json.dumps(_summary(result), separators=(",", ":")),
        flush=True,
    )
    return result


@app.local_entrypoint()
def main() -> None:
    check = preflight.remote()
    print(
        "AERA_V26_ISSUE474_FICEM_READ_REPAIR5_PREFLIGHT_LOCAL="
        + json.dumps(check, separators=(",", ":")),
        flush=True,
    )
    result = run_probe.remote()
    print(json.dumps(_summary(result), indent=2), flush=True)
