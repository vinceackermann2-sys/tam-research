from __future__ import annotations

import json
from pathlib import Path

import modal

APP_NAME = "tam-research-aera-v17-compiled-sparse-reasoner-probe"
VOLUME_NAME = "tam-research-data"
RUN_DIR = "/vol/aera-systems/v17-compiled-sparse-reasoner-probe-v1"
MAX_GPU_SECONDS = 600

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.10,<2.11", "numpy>=2.0,<3")
    .add_local_python_source("tam_research")
)


@app.function(image=image, cpu=4, memory=8192, timeout=300)
def preflight() -> dict:
    from tam_research.aera_v17_compiled_reasoner_probe import validate_probe_protocol

    result = validate_probe_protocol()
    print("AERA_V17_COMPILED_REASONER_PREFLIGHT_JSON=" + json.dumps(result, separators=(",", ":")), flush=True)
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
    from tam_research.aera_v17_compiled_reasoner_probe import run_probe

    volume.reload()
    result_path = Path(RUN_DIR) / "result.json"
    if result_path.exists():
        raise RuntimeError(f"refusing duplicate compiled reasoner probe: {result_path}")
    print(
        "AERA_V17_COMPILED_REASONER_L4_START_JSON="
        + json.dumps(
            {
                "measurement_only": True,
                "hard_gpu_timeout_seconds": MAX_GPU_SECONDS,
                "training_performed": False,
                "hard_sparse_semantics_required": True,
            },
            separators=(",", ":"),
        ),
        flush=True,
    )
    result = run_probe()
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2))
    volume.commit()
    print("AERA_V17_COMPILED_REASONER_RESULT_JSON=" + json.dumps(result, separators=(",", ":")), flush=True)
    return result


@app.local_entrypoint()
def main() -> None:
    check = preflight.remote()
    print("AERA_V17_COMPILED_REASONER_PREFLIGHT_LOCAL=" + json.dumps(check, separators=(",", ":")), flush=True)
    result = run_gate.remote()
    print(json.dumps(result, indent=2), flush=True)
