from __future__ import annotations

import json
from pathlib import Path

import modal

from tam_research.aera_real_language import SEED, train_matched_pair, validate_protocol

APP_NAME = "tam-research-aera-v10-real-language"
VOLUME_NAME = "tam-research-data"
DATA_DIR = "/vol/data/tam100m-2b-curated-v1"
RUN_DIR = f"/vol/aera-real-language/v10-seed{SEED}"
MAX_GPU_SECONDS = 2700

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.10,<2.11", "numpy>=2.0,<3")
    .add_local_python_source("tam_research")
)


@app.function(
    image=image,
    cpu=4,
    memory=8192,
    timeout=300,
    volumes={"/vol": volume},
)
def preflight() -> dict:
    volume.reload()
    result = validate_protocol(DATA_DIR)
    print("AERA_V10_LANGUAGE_PREFLIGHT_JSON=" + json.dumps(result, separators=(",", ":")), flush=True)
    return result


@app.function(
    image=image,
    gpu="L4",
    cpu=8,
    memory=32768,
    timeout=MAX_GPU_SECONDS,
    volumes={"/vol": volume},
)
def run_gate() -> dict:
    volume.reload()
    result_path = Path(RUN_DIR) / "result.json"
    if result_path.exists():
        raise RuntimeError(
            f"refusing duplicate GPU run: durable result already exists at {result_path}"
        )
    result = train_matched_pair(data_dir=DATA_DIR, run_dir=RUN_DIR, seed=SEED)
    volume.commit()
    print("AERA_V10_LANGUAGE_RESULT_JSON=" + json.dumps(result, separators=(",", ":")), flush=True)
    return result


@app.local_entrypoint()
def main() -> None:
    check = preflight.remote()
    print("AERA_V10_LANGUAGE_PREFLIGHT_LOCAL=" + json.dumps(check, separators=(",", ":")), flush=True)
    result = run_gate.remote()
    print(json.dumps(result, indent=2), flush=True)
