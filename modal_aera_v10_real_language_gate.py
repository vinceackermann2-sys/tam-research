from __future__ import annotations

import json
from pathlib import Path

import modal

APP_NAME = "tam-research-aera-v10-real-language"
VOLUME_NAME = "tam-research-data"
SEED = 8201
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
    # Keep all PyTorch-heavy imports inside the remote image. The GitHub Actions
    # launcher host intentionally needs only Modal, which prevents a local import
    # failure from being mistaken for a GPU/model failure.
    from tam_research.aera_real_language import validate_protocol

    volume.reload()
    result = validate_protocol(DATA_DIR)
    print(
        "AERA_V10_LANGUAGE_PREFLIGHT_JSON="
        + json.dumps(result, separators=(",", ":")),
        flush=True,
    )
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
    from tam_research.aera_real_language import (
        TOKEN_BUDGET,
        build_aera,
        build_transformer,
        parameter_accounting,
        train_matched_pair,
    )

    volume.reload()
    result_path = Path(RUN_DIR) / "result.json"
    if result_path.exists():
        raise RuntimeError(
            f"refusing duplicate GPU run: durable result already exists at {result_path}"
        )

    # This marker is emitted only from inside the allocated remote GPU function.
    # It is therefore authoritative evidence that the L4 stage, rather than the
    # CPU preflight or GitHub launcher, has actually started.
    device = torch.device("cuda")
    aera = build_aera(device)
    transformer = build_transformer(device)
    counts = parameter_accounting(aera, transformer)
    del aera, transformer
    torch.cuda.empty_cache()
    started = {
        "device": torch.cuda.get_device_name(device),
        "seed": SEED,
        "token_budget_per_model": TOKEN_BUDGET,
        "aera_stored_parameters": counts["aera_stored_parameters"],
        "transformer_parameters": counts["transformer_parameters"],
        "hard_gpu_timeout_seconds": MAX_GPU_SECONDS,
    }
    print(
        "AERA_V10_LANGUAGE_L4_START_JSON="
        + json.dumps(started, separators=(",", ":")),
        flush=True,
    )

    result = train_matched_pair(data_dir=DATA_DIR, run_dir=RUN_DIR, seed=SEED)
    volume.commit()
    print(
        "AERA_V10_LANGUAGE_RESULT_JSON="
        + json.dumps(result, separators=(",", ":")),
        flush=True,
    )
    return result


@app.local_entrypoint()
def main() -> None:
    check = preflight.remote()
    print(
        "AERA_V10_LANGUAGE_PREFLIGHT_LOCAL="
        + json.dumps(check, separators=(",", ":")),
        flush=True,
    )
    result = run_gate.remote()
    print(json.dumps(result, indent=2), flush=True)
