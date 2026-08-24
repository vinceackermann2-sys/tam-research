from __future__ import annotations

import json
from pathlib import Path

import modal

APP_NAME = "tam-research-aera-v22-real-language-dev-seed8391"
VOLUME_NAME = "tam-research-data"
SEED = 8391
DATA_DIR = "/vol/data/tam100m-2b-curated-v1"
RUN_DIR = f"/vol/aera-real-language/v22-dev-seed{SEED}"
MAX_GPU_SECONDS = 1800

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.10,<2.11", "numpy>=2.0,<3")
    .add_local_python_source("tam_research")
)


@app.function(image=image, cpu=4, memory=8192, timeout=300, volumes={"/vol": volume})
def preflight() -> dict:
    from tam_research.aera_real_language_v22_gpu import validate_protocol

    volume.reload()
    result = validate_protocol(DATA_DIR)
    print("AERA_V22_DEV_PREFLIGHT_JSON=" + json.dumps(result, separators=(",", ":")), flush=True)
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
    import torch
    from tam_research.aera_real_language import TOKEN_BUDGET
    from tam_research.aera_real_language_v22_gpu import train_matched_pair

    volume.reload()
    result_path = Path(RUN_DIR) / "result.json"
    if result_path.exists():
        raise RuntimeError(f"refusing duplicate GPU run: durable result already exists at {result_path}")
    print(
        "AERA_V22_DEV_L4_START_JSON="
        + json.dumps(
            {
                "device": torch.cuda.get_device_name(0),
                "seed": SEED,
                "token_budget_per_model": TOKEN_BUDGET,
                "hard_gpu_timeout_seconds": MAX_GPU_SECONDS,
                "development_seed_only": True,
                "maximum_l4_hours": MAX_GPU_SECONDS / 3600.0,
                "architecture": "v22 interference-corrected dual-delta + frozen conflict-free memory objective",
                "memory_dim": 50,
                "counts_toward_independent_replication": False,
            },
            separators=(",", ":"),
        ),
        flush=True,
    )
    result = train_matched_pair(data_dir=DATA_DIR, run_dir=RUN_DIR, seed=SEED)
    volume.commit()
    print("AERA_V22_DEV_RESULT_JSON=" + json.dumps(result, separators=(",", ":")), flush=True)
    return result


@app.local_entrypoint()
def main() -> None:
    check = preflight.remote()
    print("AERA_V22_DEV_PREFLIGHT_LOCAL=" + json.dumps(check, separators=(",", ":")), flush=True)
    result = run_gate.remote()
    print(json.dumps(result, indent=2), flush=True)
