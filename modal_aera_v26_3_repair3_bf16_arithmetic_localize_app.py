from __future__ import annotations

import hashlib
import json
from pathlib import Path

import modal

APP_NAME = "tam-research-aera-v26-3-issue439-repair3-bf16-arithmetic-localize"
VOLUME_NAME = "tam-research-data"
RESULT_PATH = "/vol/aera-v26/issue439-repair3-bf16-arithmetic-localize/result.json"
MAX_GPU_SECONDS = 180
FROZEN_PROBE_GIT_BLOB = "16aa99b9f6f0a1d11bd7bf5f36b2f0b1fb97047b"
FROZEN_REPAIR3_BACKEND_GIT_BLOB = "b6b37f0379b280eea4e5c2b16f349951dadc4df9"

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
    checkpoints = result["checkpoint_reports"]
    reduction = result["microdiagnostics"]["reference_reduction_semantics"]
    softmax = checkpoints["softmax_fp32_aligned_on_equal_sets"]
    return {
        "device": result["device"],
        "target_row": result["target_row"],
        "diagnostic_mirror_valid": result["diagnostic_mirror_valid"],
        "first_reference_vs_repair3_divergence": result[
            "first_reference_vs_repair3_divergence"
        ],
        "selection_semantically_equivalent": result["selection"][
            "selection_semantically_equivalent"
        ],
        "distinct_query_mismatch_count": result["distinct_query_mismatch_count"],
        "tied_query_count": result["tied_query_count"],
        "strength_bias_bit_equal": checkpoints["strength_bias"]["bit_equal"],
        "unmasked_logits_bit_equal": checkpoints["unmasked_logits"]["bit_equal"],
        "softmax_equal_set_report": softmax,
        "pre_out_recalled": checkpoints["pre_out_recalled_bf16"],
        "repair3_fp32_product_accum_vs_reference": reduction[
            "repair3_fp32_product_accum_vs_reference"
        ],
        "bf16_rounded_product_fp32_accum_vs_reference": reduction[
            "bf16_rounded_product_fp32_accum_vs_reference"
        ],
        "localization_only": True,
    }


@app.function(image=image, cpu=2, memory=4096, timeout=180, volumes={"/vol": volume})
def preflight() -> dict:
    import tam_research.aera_hardware_core_v26_3_ficem_read_triton as backend_module
    import tam_research.aera_v26_3_ficem_read_probe as probe_module
    from tam_research.aera_v26_3_repair3_bf16_arithmetic_localize import (
        cpu_contract_preflight,
    )

    volume.reload()
    result_path = Path(RESULT_PATH)
    if result_path.exists():
        raise RuntimeError(
            f"refusing duplicate #439 localization because result exists: {RESULT_PATH}"
        )

    probe_blob = _git_blob_sha(Path(probe_module.__file__))
    backend_blob = _git_blob_sha(Path(backend_module.__file__))
    if probe_blob != FROZEN_PROBE_GIT_BLOB:
        raise RuntimeError(f"#439 frozen probe blob drifted: {probe_blob}")
    if backend_blob != FROZEN_REPAIR3_BACKEND_GIT_BLOB:
        raise RuntimeError(f"#439 frozen repair3 backend blob drifted: {backend_blob}")

    result = cpu_contract_preflight()
    result.update(
        {
            "probe_git_blob": probe_blob,
            "backend_git_blob": backend_blob,
            "result_path_absent": True,
            "gpu_authorized_by_preflight": False,
        }
    )
    print(
        "AERA_V26_ISSUE439_REPAIR3_BF16_ARITHMETIC_LOCALIZE_PREFLIGHT_JSON="
        + json.dumps(result, separators=(",", ":")),
        flush=True,
    )
    return result


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=16384,
    timeout=MAX_GPU_SECONDS,
    volumes={"/vol": volume},
)
def run_localization_once() -> dict:
    import torch

    from tam_research.aera_v26_3_repair3_bf16_arithmetic_localize import (
        run_localization,
    )

    volume.reload()
    result_path = Path(RESULT_PATH)
    if result_path.exists():
        raise RuntimeError(
            f"refusing duplicate #439 localization because result exists: {RESULT_PATH}"
        )

    print(
        "AERA_V26_ISSUE439_REPAIR3_BF16_ARITHMETIC_LOCALIZE_L4_START_JSON="
        + json.dumps(
            {
                "device": torch.cuda.get_device_name(0),
                "research_issue": 439,
                "source_failed_trigger": 438,
                "source_failed_actions_run": 33510242472,
                "hard_gpu_timeout_seconds": MAX_GPU_SECONDS,
                "target_row": "bfloat16_batch8_mixed",
                "localization_only": True,
                "timing_authorized": False,
                "performance_decision_authorized": False,
                "production_backend_frozen": True,
                "production_probe_frozen": True,
                "model_loaded": False,
                "checkpoint_loaded": False,
                "corpus_accessed": False,
                "training_performed": False,
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
        "AERA_V26_ISSUE439_REPAIR3_BF16_ARITHMETIC_LOCALIZE_RESULT_JSON="
        + json.dumps(result, separators=(",", ":")),
        flush=True,
    )
    print(
        "AERA_V26_ISSUE439_REPAIR3_BF16_ARITHMETIC_LOCALIZE_SUMMARY_JSON="
        + json.dumps(_summary(result), separators=(",", ":")),
        flush=True,
    )
    return result


@app.local_entrypoint()
def main() -> None:
    check = preflight.remote()
    print(
        "AERA_V26_ISSUE439_REPAIR3_BF16_ARITHMETIC_LOCALIZE_PREFLIGHT_LOCAL="
        + json.dumps(check, separators=(",", ":")),
        flush=True,
    )
    result = run_localization_once.remote()
    print(json.dumps(_summary(result), indent=2), flush=True)
