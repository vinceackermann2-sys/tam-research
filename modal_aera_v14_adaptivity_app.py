from __future__ import annotations

import json
from pathlib import Path

import modal

APP_NAME = "tam-research-aera-v14-adaptivity-seed8271"
VOLUME_NAME = "tam-research-data"
DATA_DIR = "/vol/data/tam100m-2b-curated-v1"
SOURCE_RUN_DIR = "/vol/aera-real-language/v14-dev-seed8271"
RESULT_DIR = "/vol/aera-real-language/v14-adaptivity-seed8271"
RESULT_PATH = f"{RESULT_DIR}/result.json"
MAX_GPU_SECONDS = 600

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.10,<2.11", "numpy>=2.0,<3")
    .add_local_python_source("tam_research")
)


@app.function(image=image, cpu=2, memory=4096, timeout=180, volumes={"/vol": volume})
def preflight() -> dict:
    volume.reload()
    required = [
        Path(DATA_DIR) / "val.bin",
        Path(SOURCE_RUN_DIR) / "aera.pt",
        Path(SOURCE_RUN_DIR) / "transformer.pt",
        Path(SOURCE_RUN_DIR) / "result.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"required frozen v14 artifacts missing: {missing}")
    if Path(RESULT_PATH).exists():
        raise RuntimeError(f"refusing duplicate adaptivity evaluation: {RESULT_PATH} already exists")
    result = {
        "source_run_dir": SOURCE_RUN_DIR,
        "result_path": RESULT_PATH,
        "training_authorized": False,
        "checkpoint_mutation_authorized": False,
        "gpu_authorization_scope": "one checkpoint-only AERA-v14 seed8271 adaptivity evaluation",
        "hard_gpu_timeout_seconds": MAX_GPU_SECONDS,
    }
    print("AERA_V14_ADAPTIVITY_PREFLIGHT_JSON=" + json.dumps(result, separators=(",", ":")), flush=True)
    return result


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=16384,
    timeout=MAX_GPU_SECONDS,
    volumes={"/vol": volume},
)
def run_gate() -> dict:
    import torch
    from tam_research.aera_v14_adaptivity import evaluate_checkpoint, write_result

    volume.reload()
    if Path(RESULT_PATH).exists():
        raise RuntimeError(f"refusing duplicate adaptivity evaluation: {RESULT_PATH} already exists")
    print(
        "AERA_V14_ADAPTIVITY_L4_START_JSON="
        + json.dumps(
            {
                "device": torch.cuda.get_device_name(0),
                "source_seed": 8271,
                "training_performed": False,
                "checkpoint_mutation": False,
                "hard_gpu_timeout_seconds": MAX_GPU_SECONDS,
            },
            separators=(",", ":"),
        ),
        flush=True,
    )
    result = evaluate_checkpoint(
        data_dir=DATA_DIR,
        run_dir=SOURCE_RUN_DIR,
        device=torch.device("cuda"),
    )
    write_result(result, RESULT_PATH)
    volume.commit()
    print("AERA_V14_ADAPTIVITY_RESULT_JSON=" + json.dumps(result, separators=(",", ":")), flush=True)
    return result


@app.local_entrypoint()
def main() -> None:
    check = preflight.remote()
    print("AERA_V14_ADAPTIVITY_PREFLIGHT_LOCAL=" + json.dumps(check, separators=(",", ":")), flush=True)
    result = run_gate.remote()
    print(json.dumps(result, indent=2), flush=True)
