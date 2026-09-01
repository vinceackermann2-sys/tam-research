from __future__ import annotations

import json
from pathlib import Path

import modal

APP_NAME = "tam-research-aera-v26-3-issue423-bf16-localize"
VOLUME_NAME = "tam-research-data"
RESULT_PATH = "/vol/aera-v26/issue423-bf16-correctness-localize/result.json"
MAX_GPU_SECONDS = 180

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.10,<2.11", "numpy>=2.0,<3")
    .add_local_python_source("tam_research")
)


@app.function(image=image, cpu=2, memory=4096, timeout=180, volumes={"/vol": volume})
def preflight() -> dict:
    from tam_research.aera_v26_3_bf16_correctness_localize import cpu_contract_preflight

    volume.reload()
    result_path = Path(RESULT_PATH)
    if result_path.exists():
        raise RuntimeError(
            f"refusing duplicate #423 localization because result exists: {RESULT_PATH}"
        )
    result = cpu_contract_preflight()
    result["result_path_absent"] = True
    result["gpu_authorized_by_preflight"] = False
    print(
        "AERA_V26_ISSUE423_BF16_CORRECTNESS_LOCALIZE_PREFLIGHT_JSON="
        + json.dumps(result, separators=(",", ":")),
        flush=True,
    )
    return result


@app.function(
    image=image,
    gpu="L4",
    cpu=2,
    memory=8192,
    timeout=MAX_GPU_SECONDS,
    volumes={"/vol": volume},
)
def run_localization_once() -> dict:
    import torch

    from tam_research.aera_v26_3_bf16_correctness_localize import run_localization

    volume.reload()
    result_path = Path(RESULT_PATH)
    if result_path.exists():
        raise RuntimeError(
            f"refusing duplicate #423 localization because result exists: {RESULT_PATH}"
        )
    print(
        "AERA_V26_ISSUE423_BF16_CORRECTNESS_LOCALIZE_L4_START_JSON="
        + json.dumps(
            {
                "device": torch.cuda.get_device_name(0),
                "research_issue": 423,
                "source_failed_issue": 418,
                "source_failed_trigger": 422,
                "source_failed_actions_run": 33499743719,
                "hard_gpu_timeout_seconds": MAX_GPU_SECONDS,
                "target_row": "bfloat16_batch8_mixed",
                "localization_only": True,
                "timing_authorized": False,
                "profiling_authorized": False,
                "performance_decision_authorized": False,
                "model_loaded": False,
                "checkpoint_loaded": False,
                "corpus_accessed": False,
                "training_performed": False,
                "optimizer_created": False,
                "backward_performed": False,
                "scientific_seed_consumed": False,
            },
            separators=(",", ":"),
        ),
        flush=True,
    )
    result = run_localization()
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2))
    volume.commit()
    print(
        "AERA_V26_ISSUE423_BF16_CORRECTNESS_LOCALIZE_RESULT_JSON="
        + json.dumps(result, separators=(",", ":")),
        flush=True,
    )
    return result


@app.local_entrypoint()
def main() -> None:
    check = preflight.remote()
    print(
        "AERA_V26_ISSUE423_BF16_CORRECTNESS_LOCALIZE_PREFLIGHT_LOCAL="
        + json.dumps(check, separators=(",", ":")),
        flush=True,
    )
    result = run_localization_once.remote()
    print(
        "AERA_V26_ISSUE423_BF16_CORRECTNESS_LOCALIZE_SUMMARY_JSON="
        + json.dumps(
            {
                "device": result["device"],
                "target_row": result["target_row"],
                "aggregate_correctness_pass": result["aggregate_correctness_pass"],
                "failed_subgates": result["failed_subgates"],
                "subgates": result["subgates"],
                "selection": result["selection"],
                "distinct_query_mismatch_count": result[
                    "distinct_query_mismatch_count"
                ],
                "tied_query_count": result["tied_query_count"],
                "pre_out_recalled_error": result["pre_out_recalled_error"],
                "final_out_error": result["final_out_error"],
                "reference_boundary_diagnostics": result[
                    "reference_boundary_diagnostics"
                ],
                "localization_only": True,
            },
            separators=(",", ":"),
        ),
        flush=True,
    )
