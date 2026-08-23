from __future__ import annotations

import json
from pathlib import Path

import modal

APP_NAME = "tam-research-aera-v14-causal-router-audit"
VOLUME_NAME = "tam-research-data"
DATA_DIR = "/vol/data/tam100m-2b-curated-v1"
RUN_DIR = "/vol/aera-real-language/v14-dev-seed8271"
CHECKPOINT_PATH = f"{RUN_DIR}/aera.pt"
RESULT_PATH = f"{RUN_DIR}/causal_router_audit.json"
MAX_CPU_SECONDS = 1800

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.10,<2.11", "numpy>=2.0,<3")
    .add_local_python_source("tam_research")
)


@app.function(
    image=image,
    cpu=8,
    memory=32768,
    timeout=MAX_CPU_SECONDS,
    volumes={"/vol": volume},
)
def run_audit() -> dict:
    from tam_research.aera_v14_causal_router_audit import audit_checkpoint, save_audit

    volume.reload()
    checkpoint = Path(CHECKPOINT_PATH)
    result_path = Path(RESULT_PATH)
    if not checkpoint.exists():
        raise FileNotFoundError(f"seed8271 checkpoint not found: {checkpoint}")
    if result_path.exists():
        raise RuntimeError(f"refusing duplicate causal audit: {result_path} already exists")

    print(
        "AERA_V14_CAUSAL_AUDIT_START_JSON="
        + json.dumps(
            {
                "cpu_only": True,
                "checkpoint": CHECKPOINT_PATH,
                "gpu_requested": False,
                "hard_timeout_seconds": MAX_CPU_SECONDS,
            },
            separators=(",", ":"),
        ),
        flush=True,
    )
    result = audit_checkpoint(
        checkpoint_path=CHECKPOINT_PATH,
        data_dir=DATA_DIR,
    )
    save_audit(result, RESULT_PATH)
    volume.commit()
    print(
        "AERA_V14_CAUSAL_AUDIT_RESULT_JSON=" + json.dumps(result, separators=(",", ":")),
        flush=True,
    )
    return result


@app.local_entrypoint()
def main() -> None:
    result = run_audit.remote()
    print(json.dumps(result, indent=2), flush=True)
