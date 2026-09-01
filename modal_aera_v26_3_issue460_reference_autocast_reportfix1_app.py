from __future__ import annotations

import hashlib
import json
from pathlib import Path

import modal

APP_NAME = "tam-research-aera-v26-3-issue460-reference-autocast-reportfix1"
VOLUME_NAME = "tam-research-data"
RESULT_PATH = "/vol/aera-v26/issue460-reference-autocast-localize-reportfix1/result.json"
EXHAUSTED_ISSUE456_RESULT_PATH = "/vol/aera-v26/issue456-repair4-reference-autocast-localize/result.json"
MAX_GPU_SECONDS = 180
FROZEN_LOCALIZATION_GIT_BLOB = "8ed7de14a0f29f3ac66d6228a71892fbf97e150f"
FROZEN_PROBE_GIT_BLOB = "16aa99b9f6f0a1d11bd7bf5f36b2f0b1fb97047b"
FROZEN_REPAIR4_BACKEND_GIT_BLOB = "a3a603c8a2d4b20ebcccd7663970978f4288a760"

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
    return {
        "device": result["device"],
        "target_row": result["target_row"],
        "primary_pass": result["primary_pass"],
        "primary_false_subgates": result["primary_false_subgates"],
        "first_differing_checkpoint": result[
            "first_differing_checkpoint_actual_reference_vs_outside"
        ],
        "actual_reference_vs_outside_pre_out": result[
            "actual_reference_vs_outside_pre_out"
        ],
        "actual_reference_vs_production_pre_out": result[
            "actual_reference_vs_production_pre_out"
        ],
        "outside_vs_production_pre_out": result["outside_vs_production_pre_out"],
        "actual_reference_vs_production_final": result[
            "actual_reference_vs_production_final"
        ],
        "same_recalled_out_inside_vs_outside": result[
            "outside_recalled_memory_out_inside_vs_outside_autocast"
        ],
        "reference_capture_matches_backend": result[
            "actual_reference_capture_vs_backend_final"
        ],
        "production_capture_matches_backend": result[
            "production_capture_vs_backend_final"
        ],
        "reportfix1_applied": result["reportfix1_applied"],
        "reporting_only_dtype_promotion": result["reporting_only_dtype_promotion"],
        "native_execution_tensors_changed": result["native_execution_tensors_changed"],
        "localization_only": True,
    }


@app.function(image=image, cpu=2, memory=4096, timeout=180, volumes={"/vol": volume})
def preflight() -> dict:
    import tam_research.aera_hardware_core_v26_3_ficem_read_triton as backend_module
    import tam_research.aera_v26_3_ficem_read_probe as probe_module
    import tam_research.aera_v26_3_repair4_reference_autocast_localize as localization_module
    from tam_research.aera_v26_3_issue460_reference_autocast_reportfix1 import (
        cpu_contract_preflight,
    )

    volume.reload()
    result_path = Path(RESULT_PATH)
    exhausted_path = Path(EXHAUSTED_ISSUE456_RESULT_PATH)
    if result_path.exists():
        raise RuntimeError(f"refusing duplicate #460 reportfix1 localization: {RESULT_PATH}")
    if exhausted_path.exists():
        raise RuntimeError(
            "unexpected durable #456 result exists despite #459 failing before its marker"
        )

    localization_blob = _git_blob_sha(Path(localization_module.__file__))
    probe_blob = _git_blob_sha(Path(probe_module.__file__))
    backend_blob = _git_blob_sha(Path(backend_module.__file__))
    if localization_blob != FROZEN_LOCALIZATION_GIT_BLOB:
        raise RuntimeError(f"#460 frozen #456 localization blob drifted: {localization_blob}")
    if probe_blob != FROZEN_PROBE_GIT_BLOB:
        raise RuntimeError(f"#460 frozen probe blob drifted: {probe_blob}")
    if backend_blob != FROZEN_REPAIR4_BACKEND_GIT_BLOB:
        raise RuntimeError(f"#460 frozen repair4 backend blob drifted: {backend_blob}")

    result = cpu_contract_preflight()
    result.update(
        {
            "localization_git_blob": localization_blob,
            "probe_git_blob": probe_blob,
            "backend_git_blob": backend_blob,
            "successor_result_path_absent": True,
            "exhausted_issue456_result_path_absent": True,
            "gpu_authorized_by_preflight": False,
        }
    )
    print(
        "AERA_V26_ISSUE460_REFERENCE_AUTOCAST_LOCALIZE_REPORTFIX1_PREFLIGHT_JSON="
        + json.dumps(result, separators=(",", ":")),
        flush=True,
    )
    return result


@app.function(
    image=image,
    gpu="L4",
    cpu=2,
    memory=12288,
    timeout=MAX_GPU_SECONDS,
    volumes={"/vol": volume},
)
def run_localization_once() -> dict:
    import torch
    from tam_research.aera_v26_3_issue460_reference_autocast_reportfix1 import (
        run_localization_reportfix1,
    )

    volume.reload()
    result_path = Path(RESULT_PATH)
    if result_path.exists():
        raise RuntimeError(f"refusing duplicate #460 reportfix1 localization: {RESULT_PATH}")
    if Path(EXHAUSTED_ISSUE456_RESULT_PATH).exists():
        raise RuntimeError(
            "unexpected durable #456 result exists despite #459 failing before its marker"
        )

    print(
        "AERA_V26_ISSUE460_REFERENCE_AUTOCAST_LOCALIZE_REPORTFIX1_L4_START_JSON="
        + json.dumps(
            {
                "device": torch.cuda.get_device_name(0),
                "research_issue": 460,
                "source_failed_trigger": 459,
                "source_failed_run": 33546452532,
                "source_failed_job": 99985042556,
                "target_row": "bfloat16_batch8_mixed",
                "hard_gpu_timeout_seconds": MAX_GPU_SECONDS,
                "dtype_safe_reporting_only": True,
                "localization_only": True,
                "timing_authorized": False,
                "performance_decision_authorized": False,
                "production_backend_frozen": True,
                "production_probe_frozen": True,
                "source_localization_frozen": True,
                "scientific_seed_consumed": False,
            },
            separators=(",", ":"),
        ),
        flush=True,
    )

    result = run_localization_reportfix1()
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2))
    volume.commit()
    print(
        "AERA_V26_ISSUE460_REFERENCE_AUTOCAST_LOCALIZE_REPORTFIX1_RESULT_JSON="
        + json.dumps(result, separators=(",", ":")),
        flush=True,
    )
    print(
        "AERA_V26_ISSUE460_REFERENCE_AUTOCAST_LOCALIZE_REPORTFIX1_SUMMARY_JSON="
        + json.dumps(_summary(result), separators=(",", ":")),
        flush=True,
    )
    return result


@app.local_entrypoint()
def main() -> None:
    check = preflight.remote()
    print(json.dumps(check, indent=2), flush=True)
    result = run_localization_once.remote()
    print(json.dumps(_summary(result), indent=2), flush=True)
