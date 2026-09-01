from __future__ import annotations

import json
from pathlib import Path

import modal

APP_NAME = "tam-research-aera-v26-3-issue418-ficem-read-repair2"
VOLUME_NAME = "tam-research-data"
RESULT_PATH = "/vol/aera-v26/issue418-ficem-read-repair2/result.json"
MAX_GPU_SECONDS = 300

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.10,<2.11", "numpy>=2.0,<3")
    .add_local_python_source("tam_research")
)


def _summary(result: dict) -> dict:
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
    from tam_research.aera_v26_3_ficem_read_probe import cpu_contract_preflight

    volume.reload()
    result_path = Path(RESULT_PATH)
    if result_path.exists():
        raise RuntimeError(
            f"refusing duplicate issue418 FICEM read repair2 run because result exists: {RESULT_PATH}"
        )
    check = {
        "contract": cpu_contract_preflight(),
        "repair_issue": 418,
        "source_repair_issue": 414,
        "source_failed_trigger": 416,
        "source_failed_actions_run": 33497787407,
        "source_failure": "reference_only_fourth_fifth_tie_before_timing",
        "result_path_absent": True,
        "gpu_authorized_by_preflight": False,
        "synthetic_only": True,
        "original_case_generation_preserved": True,
        "tie_aware_correctness_only": True,
        "candidate_backend_changed_by_repair2": False,
        "model_loaded": False,
        "checkpoint_loaded": False,
        "corpus_accessed": False,
        "training_performed": False,
        "optimizer_created": False,
        "backward_performed": False,
        "scientific_seed_consumed": False,
    }
    print(
        "AERA_V26_ISSUE418_FICEM_READ_REPAIR2_PREFLIGHT_JSON="
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
            f"refusing duplicate issue418 FICEM read repair2 run because result exists: {RESULT_PATH}"
        )
    print(
        "AERA_V26_ISSUE418_FICEM_READ_REPAIR2_L4_START_JSON="
        + json.dumps(
            {
                "device": torch.cuda.get_device_name(0),
                "repair_issue": 418,
                "source_repair_issue": 414,
                "source_failed_trigger": 416,
                "hard_gpu_timeout_seconds": MAX_GPU_SECONDS,
                "synthetic_only": True,
                "original_case_generation_preserved": True,
                "tie_aware_correctness_only": True,
                "candidate_backend_changed_by_repair2": False,
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
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2))
    volume.commit()
    print(
        "AERA_V26_ISSUE418_FICEM_READ_REPAIR2_RESULT_JSON="
        + json.dumps(result, separators=(",", ":")),
        flush=True,
    )
    print(
        "AERA_V26_ISSUE418_FICEM_READ_REPAIR2_SUMMARY_JSON="
        + json.dumps(_summary(result), separators=(",", ":")),
        flush=True,
    )
    return result


@app.local_entrypoint()
def main() -> None:
    check = preflight.remote()
    print(
        "AERA_V26_ISSUE418_FICEM_READ_REPAIR2_PREFLIGHT_LOCAL="
        + json.dumps(check, separators=(",", ":")),
        flush=True,
    )
    result = run_probe.remote()
    print(json.dumps(_summary(result), indent=2), flush=True)
