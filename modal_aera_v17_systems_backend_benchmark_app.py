from __future__ import annotations

import json
from pathlib import Path

import modal

APP_NAME = "tam-research-aera-v17-systems-backend-benchmark"
VOLUME_NAME = "tam-research-data"
DATA_DIR = "/vol/data/tam100m-2b-curated-v1"
CHECKPOINT_DIR = "/vol/aera-real-language/v17-dev-seed8331"
RESULT_DIR = "/vol/aera-systems/v17-systems-backend-benchmark-v1"
MAX_GPU_SECONDS = 900

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.10,<2.11", "numpy>=2.0,<3")
    .add_local_python_source("tam_research")
)


@app.function(image=image, cpu=4, memory=8192, timeout=300, volumes={"/vol": volume})
def preflight() -> dict:
    from tam_research.aera_v17_systems_backend_benchmark import validate_protocol
    volume.reload()
    result = validate_protocol(DATA_DIR, CHECKPOINT_DIR)
    print("AERA_V17_SYSTEMS_BACKEND_PREFLIGHT_JSON=" + json.dumps(result, separators=(",", ":")), flush=True)
    return result


@app.function(image=image, gpu="L4", cpu=8, memory=32768, timeout=MAX_GPU_SECONDS, volumes={"/vol": volume})
def run_gate() -> dict:
    import torch
    from tam_research.aera_v17_systems_backend_benchmark import BATCH_SIZES, CHECKPOINT_SEED, run_benchmark

    volume.reload()
    result_path = Path(RESULT_DIR) / "result.json"
    if result_path.exists():
        raise RuntimeError(f"refusing duplicate systems-backend benchmark: {result_path}")
    print("AERA_V17_SYSTEMS_BACKEND_L4_START_JSON=" + json.dumps({
        "device": torch.cuda.get_device_name(0),
        "checkpoint_seed": CHECKPOINT_SEED,
        "batch_sizes": list(BATCH_SIZES),
        "hard_gpu_timeout_seconds": MAX_GPU_SECONDS,
        "measurement_only": True,
    }, separators=(",", ":")), flush=True)
    result = run_benchmark(data_dir=DATA_DIR, checkpoint_dir=CHECKPOINT_DIR)
    Path(RESULT_DIR).mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2))
    volume.commit()
    print("AERA_V17_SYSTEMS_BACKEND_RESULT_JSON=" + json.dumps(result, separators=(",", ":")), flush=True)
    return result


@app.local_entrypoint()
def main() -> None:
    check = preflight.remote()
    print("AERA_V17_SYSTEMS_BACKEND_PREFLIGHT_LOCAL=" + json.dumps(check, separators=(",", ":")), flush=True)
    result = run_gate.remote()
    print(json.dumps(result, indent=2), flush=True)
