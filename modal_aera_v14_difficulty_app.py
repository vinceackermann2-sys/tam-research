from __future__ import annotations

import json
from pathlib import Path

import modal

APP_NAME = "tam-research-aera-v14-difficulty-seed8271"
VOLUME_NAME = "tam-research-data"
DATA_DIR = "/vol/data/tam100m-2b-curated-v1"
RUN_DIR = "/vol/aera-real-language/v14-dev-seed8271"
RESULT_PATH = f"{RUN_DIR}/difficulty_diagnostic.json"
MAX_GPU_SECONDS = 300

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.10,<2.11", "numpy>=2.0,<3")
    .add_local_python_source("tam_research")
)


@app.function(image=image, cpu=2, memory=4096, timeout=120, volumes={"/vol": volume})
def preflight() -> dict:
    from tam_research.aera_v14_difficulty_diagnostic import EXPECTED_CHUNKS, SEED

    volume.reload()
    required = [
        Path(RUN_DIR) / "aera.pt",
        Path(RUN_DIR) / "transformer.pt",
        Path(DATA_DIR) / "val.bin",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing frozen diagnostic inputs: {missing}")
    if Path(RESULT_PATH).exists():
        raise RuntimeError(f"refusing duplicate diagnostic: {RESULT_PATH} already exists")
    result = {
        "seed": SEED,
        "expected_chunks": EXPECTED_CHUNKS,
        "training_tokens_added": 0,
        "weights_updated": False,
        "gpu_scope": "one inference-only L4 diagnostic <=300s",
    }
    print("AERA_V14_DIFFICULTY_PREFLIGHT_JSON=" + json.dumps(result, separators=(",", ":")), flush=True)
    return result


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=16384,
    timeout=MAX_GPU_SECONDS,
    volumes={"/vol": volume},
)
def run_diagnostic() -> dict:
    import torch
    from tam_research.aera_v14_difficulty_diagnostic import run_diagnostic as evaluate

    volume.reload()
    if Path(RESULT_PATH).exists():
        raise RuntimeError(f"refusing duplicate diagnostic: {RESULT_PATH} already exists")
    print(
        "AERA_V14_DIFFICULTY_L4_START_JSON="
        + json.dumps(
            {
                "device": torch.cuda.get_device_name(0),
                "hard_gpu_timeout_seconds": MAX_GPU_SECONDS,
                "training_tokens_added": 0,
                "weights_updated": False,
            },
            separators=(",", ":"),
        ),
        flush=True,
    )
    result = evaluate(data_dir=DATA_DIR, run_dir=RUN_DIR, result_path=RESULT_PATH)
    volume.commit()
    print("AERA_V14_DIFFICULTY_RESULT_JSON=" + json.dumps(result, separators=(",", ":")), flush=True)
    return result


@app.local_entrypoint()
def main() -> None:
    check = preflight.remote()
    print("AERA_V14_DIFFICULTY_PREFLIGHT_LOCAL=" + json.dumps(check, separators=(",", ":")), flush=True)
    result = run_diagnostic.remote()
    print(json.dumps(result, indent=2), flush=True)
