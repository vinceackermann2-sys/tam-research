from __future__ import annotations

import json
from pathlib import Path

import modal

APP_NAME = "tam-research-aera-v23-seed8461-posthoc-341"
VOLUME_NAME = "tam-research-data"
DATA_DIR = "/vol/data/tam100m-2b-curated-v1"
CHECKPOINT_PATH = "/vol/aera-real-language/v23-dev-seed8461/aera.pt"
RESULT_PATH = "/vol/aera-real-language/v23-posthoc-issue341/result.json"
MAX_GPU_SECONDS = 600

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.10,<2.11", "numpy>=2.0,<3")
    .add_local_python_source("tam_research")
)


@app.function(image=image, cpu=4, memory=8192, timeout=300, volumes={"/vol": volume})
def preflight() -> dict:
    import torch

    from tam_research.aera_v23_posthoc_diagnosis import frozen_protocol

    volume.reload()
    checkpoint = Path(CHECKPOINT_PATH)
    result_path = Path(RESULT_PATH)
    if not checkpoint.exists():
        raise RuntimeError(f"seed8461 checkpoint missing: {checkpoint}")
    if result_path.exists():
        raise RuntimeError(f"refusing duplicate diagnostic result: {result_path}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("seed") != 8461:
        raise RuntimeError("post-hoc checkpoint seed mismatch")
    protocol = frozen_protocol()
    check = {
        "protocol": protocol,
        "checkpoint_exists": True,
        "checkpoint_seed": int(payload["seed"]),
        "result_path_absent": True,
        "gpu_authorized": False,
        "training_authorized": False,
    }
    print(
        "AERA_V23_POSTHOC_PREFLIGHT_JSON=" + json.dumps(check, separators=(",", ":")),
        flush=True,
    )
    return check


@app.function(
    image=image,
    gpu="L4",
    cpu=8,
    memory=32768,
    timeout=MAX_GPU_SECONDS,
    volumes={"/vol": volume},
)
def run_diagnostic() -> dict:
    import torch

    from tam_research.aera_v23_posthoc_diagnosis import run_posthoc_diagnosis

    volume.reload()
    result_path = Path(RESULT_PATH)
    if result_path.exists():
        raise RuntimeError(f"refusing duplicate post-hoc diagnostic: {result_path}")
    print(
        "AERA_V23_POSTHOC_L4_START_JSON="
        + json.dumps(
            {
                "device": torch.cuda.get_device_name(0),
                "checkpoint_seed": 8461,
                "research_issue": 341,
                "hard_gpu_timeout_seconds": MAX_GPU_SECONDS,
                "training_performed": False,
                "optimizer_created": False,
                "checkpoint_write_authorized": False,
                "100m_authorized": False,
            },
            separators=(",", ":"),
        ),
        flush=True,
    )
    result = run_posthoc_diagnosis(
        data_dir=DATA_DIR,
        checkpoint_path=CHECKPOINT_PATH,
    )
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2))
    volume.commit()
    print(
        "AERA_V23_POSTHOC_RESULT_JSON=" + json.dumps(result, separators=(",", ":")),
        flush=True,
    )
    return result


@app.local_entrypoint()
def main() -> None:
    check = preflight.remote()
    print(
        "AERA_V23_POSTHOC_PREFLIGHT_LOCAL=" + json.dumps(check, separators=(",", ":")),
        flush=True,
    )
    result = run_diagnostic.remote()
    print(json.dumps(result, indent=2), flush=True)
