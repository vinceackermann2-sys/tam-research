from __future__ import annotations

import hashlib
import json
from pathlib import Path

import modal

APP_NAME = "tam-research-aera-v26-3-issue479-ficem-read-repair5-successor"
VOLUME_NAME = "tam-research-data"
RESULT_PATH = "/vol/aera-v26/issue479-ficem-read-repair5-successor/result.json"
MAX_GPU_SECONDS = 300
SOURCE_MAIN = "fb1cc86f51f1b012cf2f74bfaf30d6d9b389ee34"
REPAIR5_BACKEND_GIT_BLOB = "263f68eb1186a8ac14a08fc4b4df1fc5b292c711"
HISTORICAL_PROBE_GIT_BLOB = "16aa99b9f6f0a1d11bd7bf5f36b2f0b1fb97047b"
SUCCESSOR_PROBE_GIT_BLOB = "6fd6518e10ed1ef4115863f98ac591ffd77ce903"
SOURCE_SUCCESSOR_ISSUE = 477
SOURCE_SUCCESSOR_PR = 478
SOURCE_SUCCESSOR_CPU_RUN = 33611259063
SOURCE_SUCCESSOR_CPU_JOB = 100186694663
SOURCE_SUCCESSOR_CPU_HEAD = "d092464d3b62a0703c53b238164ee16e975142ce"
EXHAUSTED_GATE_ISSUE = 474
EXHAUSTED_TRIGGER_ISSUE = 476
EXHAUSTED_GATE_RUN = 33608906596
EXHAUSTED_GATE_JOB = 100179200965

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
                "tied_selection_semantically_valid",
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


def _required_successor_protocol() -> dict[str, bool]:
    return {
        "diagnostic_reference_precision_context_corrected": True,
        "historical_probe_modified": False,
        "candidate_path_changed_by_probe_successor": False,
        "fixtures_changed_by_probe_successor": False,
        "thresholds_changed_by_probe_successor": False,
        "timing_changed_by_probe_successor": False,
        "gpu_authorized_by_issue477": False,
        "scientific_seed_consumed": False,
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
    import tam_research.aera_v26_3_ficem_read_probe as historical_probe_module
    import tam_research.aera_v26_3_ficem_read_probe_repair5 as successor_probe_module
    from tam_research.aera_hardware_core_v26_3_ficem_read_triton import (
        fused_ficem_read_v26_3_protocol,
    )
    from tam_research.aera_v26_3_ficem_read_probe_repair5 import (
        cpu_contract_preflight,
        issue477_protocol,
    )

    volume.reload()
    result_path = Path(RESULT_PATH)
    if result_path.exists():
        raise RuntimeError(
            f"refusing duplicate issue479 FICEM read repair5 successor run because result exists: {RESULT_PATH}"
        )

    backend_blob = _git_blob_sha(Path(backend_module.__file__))
    historical_blob = _git_blob_sha(Path(historical_probe_module.__file__))
    successor_blob = _git_blob_sha(Path(successor_probe_module.__file__))
    if backend_blob != REPAIR5_BACKEND_GIT_BLOB:
        raise RuntimeError(f"issue479 repair5 backend blob drifted: {backend_blob}")
    if historical_blob != HISTORICAL_PROBE_GIT_BLOB:
        raise RuntimeError(f"issue479 historical probe blob drifted: {historical_blob}")
    if successor_blob != SUCCESSOR_PROBE_GIT_BLOB:
        raise RuntimeError(f"issue479 successor probe blob drifted: {successor_blob}")

    backend_protocol = fused_ficem_read_v26_3_protocol()
    for key, value in _required_repair5_protocol().items():
        if backend_protocol.get(key) is not value:
            raise RuntimeError(
                f"issue479 repair5 protocol boundary drifted: {key}={backend_protocol.get(key)!r}"
            )

    successor_protocol = issue477_protocol()
    for key, value in _required_successor_protocol().items():
        if successor_protocol.get(key) is not value:
            raise RuntimeError(
                f"issue479 successor protocol boundary drifted: {key}={successor_protocol.get(key)!r}"
            )

    check = {
        "contract": cpu_contract_preflight(),
        "research_issue": 479,
        "source_main": SOURCE_MAIN,
        "source_successor_issue": SOURCE_SUCCESSOR_ISSUE,
        "source_successor_pr": SOURCE_SUCCESSOR_PR,
        "source_successor_cpu_run": SOURCE_SUCCESSOR_CPU_RUN,
        "source_successor_cpu_job": SOURCE_SUCCESSOR_CPU_JOB,
        "source_successor_cpu_head": SOURCE_SUCCESSOR_CPU_HEAD,
        "exhausted_gate_issue": EXHAUSTED_GATE_ISSUE,
        "exhausted_trigger_issue": EXHAUSTED_TRIGGER_ISSUE,
        "exhausted_gate_run": EXHAUSTED_GATE_RUN,
        "exhausted_gate_job": EXHAUSTED_GATE_JOB,
        "backend_git_blob": backend_blob,
        "historical_probe_git_blob": historical_blob,
        "successor_probe_git_blob": successor_blob,
        "result_path_absent": True,
        "synthetic_only": True,
        "original_case_generation_preserved": True,
        "tie_aware_correctness_preserved": True,
        "diagnostic_reference_precision_context_corrected": True,
        "model_loaded": False,
        "checkpoint_loaded": False,
        "corpus_accessed": False,
        "training_performed": False,
        "scientific_seed_consumed": False,
    }
    print(
        "AERA_V26_ISSUE479_FICEM_READ_REPAIR5_SUCCESSOR_PREFLIGHT_JSON="
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
    from tam_research.aera_v26_3_ficem_read_probe_repair5 import (
        run_ficem_read_probe_repair5,
    )

    volume.reload()
    result_path = Path(RESULT_PATH)
    if result_path.exists():
        raise RuntimeError(
            f"refusing duplicate issue479 FICEM read repair5 successor run because result exists: {RESULT_PATH}"
        )

    print(
        "AERA_V26_ISSUE479_FICEM_READ_REPAIR5_SUCCESSOR_L4_START_JSON="
        + json.dumps(
            {
                "device": torch.cuda.get_device_name(0),
                "research_issue": 479,
                "source_main": SOURCE_MAIN,
                "candidate_backend_git_blob": REPAIR5_BACKEND_GIT_BLOB,
                "historical_probe_git_blob": HISTORICAL_PROBE_GIT_BLOB,
                "successor_probe_git_blob": SUCCESSOR_PROBE_GIT_BLOB,
                "source_successor_cpu_run": SOURCE_SUCCESSOR_CPU_RUN,
                "exhausted_gate_run": EXHAUSTED_GATE_RUN,
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

    # Exactly one frozen successor-probe invocation. If correctness raises before a
    # returned result, the sole attempt fails before a marker and is not retried.
    result = run_ficem_read_probe_repair5()
    result["issue479_gate_metadata"] = {
        "research_issue": 479,
        "source_main": SOURCE_MAIN,
        "source_successor_issue": SOURCE_SUCCESSOR_ISSUE,
        "source_successor_pr": SOURCE_SUCCESSOR_PR,
        "source_successor_cpu_run": SOURCE_SUCCESSOR_CPU_RUN,
        "source_successor_cpu_job": SOURCE_SUCCESSOR_CPU_JOB,
        "source_successor_cpu_head": SOURCE_SUCCESSOR_CPU_HEAD,
        "exhausted_gate_issue": EXHAUSTED_GATE_ISSUE,
        "exhausted_trigger_issue": EXHAUSTED_TRIGGER_ISSUE,
        "exhausted_gate_run": EXHAUSTED_GATE_RUN,
        "exhausted_gate_job": EXHAUSTED_GATE_JOB,
        "candidate_backend_git_blob": REPAIR5_BACKEND_GIT_BLOB,
        "historical_probe_git_blob": HISTORICAL_PROBE_GIT_BLOB,
        "successor_probe_git_blob": SUCCESSOR_PROBE_GIT_BLOB,
        "synthetic_only": True,
        "scientific_seed_consumed": False,
    }

    result_path.parent.mkdir(parents=True, exist_ok=True)
    durable_json = json.dumps(result, sort_keys=True, separators=(",", ":"))
    result_path.write_text(durable_json)
    volume.commit()

    result_sha256 = hashlib.sha256(durable_json.encode()).hexdigest()
    marker = {
        "research_issue": 479,
        "result_path": RESULT_PATH,
        "result_sha256": result_sha256,
        "decision": result["decision"],
        "overall_pass": result["overall_pass"],
        "device": result["device"],
        "source_main": SOURCE_MAIN,
        "candidate_backend_git_blob": REPAIR5_BACKEND_GIT_BLOB,
        "historical_probe_git_blob": HISTORICAL_PROBE_GIT_BLOB,
        "successor_probe_git_blob": SUCCESSOR_PROBE_GIT_BLOB,
    }
    print(
        "AERA_V26_ISSUE479_FICEM_READ_REPAIR5_SUCCESSOR_RESULT_JSON="
        + json.dumps(marker, separators=(",", ":")),
        flush=True,
    )
    print(
        "AERA_V26_ISSUE479_FICEM_READ_REPAIR5_SUCCESSOR_SUMMARY_JSON="
        + json.dumps(_summary(result), separators=(",", ":")),
        flush=True,
    )
    return result


@app.local_entrypoint()
def main() -> None:
    check = preflight.remote()
    print(
        "AERA_V26_ISSUE479_FICEM_READ_REPAIR5_SUCCESSOR_PREFLIGHT_LOCAL="
        + json.dumps(check, separators=(",", ":")),
        flush=True,
    )
    result = run_probe.remote()
    print(json.dumps(_summary(result), indent=2), flush=True)
