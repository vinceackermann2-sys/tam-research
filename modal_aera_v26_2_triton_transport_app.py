from __future__ import annotations

import json
from pathlib import Path

import modal

APP_NAME = "tam-research-aera-v26-2-issue408-triton-transport"
VOLUME_NAME = "tam-research-data"
RESULT_PATH = "/vol/aera-v26/issue408-triton-transport/result.json"
MAX_GPU_SECONDS = 300

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.10,<2.11", "numpy>=2.0,<3")
    .add_local_python_source("tam_research")
)


@app.function(image=image, cpu=4, memory=8192, timeout=300, volumes={"/vol": volume})
def preflight() -> dict:
    from tam_research.aera_v26_2_triton_transport_probe import cpu_contract_preflight

    volume.reload()
    result_path = Path(RESULT_PATH)
    if result_path.exists():
        raise RuntimeError(
            f"refusing duplicate issue408 Triton transport run because result exists: {RESULT_PATH}"
        )
    check = {
        "contract": cpu_contract_preflight(),
        "result_path_absent": True,
        "gpu_authorized_by_preflight": False,
        "synthetic_only": True,
        "model_loaded": False,
        "checkpoint_loaded": False,
        "corpus_accessed": False,
        "training_performed": False,
        "optimizer_created": False,
        "backward_performed": False,
        "scientific_seed_consumed": False,
    }
    print(
        "AERA_V26_ISSUE408_TRITON_TRANSPORT_PREFLIGHT_JSON="
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

    from tam_research.aera_v26_2_triton_transport_probe import run_triton_transport_probe

    volume.reload()
    result_path = Path(RESULT_PATH)
    if result_path.exists():
        raise RuntimeError(
            f"refusing duplicate issue408 Triton transport run because result exists: {RESULT_PATH}"
        )
    print(
        "AERA_V26_ISSUE408_TRITON_TRANSPORT_L4_START_JSON="
        + json.dumps(
            {
                "device": torch.cuda.get_device_name(0),
                "research_issue": 408,
                "hard_gpu_timeout_seconds": MAX_GPU_SECONDS,
                "synthetic_only": True,
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
    result = run_triton_transport_probe()
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2))
    volume.commit()
    print(
        "AERA_V26_ISSUE408_TRITON_TRANSPORT_RESULT_JSON="
        + json.dumps(result, separators=(",", ":")),
        flush=True,
    )
    return result


@app.local_entrypoint()
def main() -> None:
    check = preflight.remote()
    print(
        "AERA_V26_ISSUE408_TRITON_TRANSPORT_PREFLIGHT_LOCAL="
        + json.dumps(check, separators=(",", ":")),
        flush=True,
    )
    result = run_probe.remote()
    print(json.dumps(result, indent=2), flush=True)
