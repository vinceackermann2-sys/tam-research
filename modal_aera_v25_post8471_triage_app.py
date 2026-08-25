from __future__ import annotations

import json
from pathlib import Path

import modal

APP_NAME = "tam-research-aera-v25-post8471-triage-369"
VOLUME_NAME = "tam-research-data"
DATA_DIR = "/vol/data/tam100m-2b-curated-v1"
SOURCE_RUN_DIR = "/vol/aera-real-language/v25-dev-seed8471"
RESULT_PATH = "/vol/aera-real-language/v25-post8471-issue369/result.json"
MAX_GPU_SECONDS = 900

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
    memory=12288,
    timeout=300,
    volumes={"/vol": volume},
)
def preflight() -> dict:
    from tam_research.aera_v25_post8471_triage import source_preflight

    volume.reload()
    result_path = Path(RESULT_PATH)
    if result_path.exists():
        raise RuntimeError(f"refusing duplicate issue369 diagnostic result: {result_path}")
    check = source_preflight(data_dir=DATA_DIR, run_dir=SOURCE_RUN_DIR)
    check["result_path_absent"] = True
    check["result_path"] = RESULT_PATH
    check["gpu_authorized_by_launcher"] = False
    print(
        "AERA_V25_POST8471_TRIAGE_PREFLIGHT_JSON="
        + json.dumps(check, separators=(",", ":")),
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

    from tam_research.aera_v25_post8471_triage import run_checkpoint_triage

    volume.reload()
    result_path = Path(RESULT_PATH)
    if result_path.exists():
        raise RuntimeError(f"refusing duplicate issue369 checkpoint triage: {result_path}")
    print(
        "AERA_V25_POST8471_TRIAGE_L4_START_JSON="
        + json.dumps(
            {
                "device": torch.cuda.get_device_name(0),
                "research_issue": 369,
                "source_result_issue": 368,
                "source_checkpoint_seed": 8471,
                "diagnostic_sampling_seed": 138471,
                "hard_gpu_timeout_seconds": MAX_GPU_SECONDS,
                "training_performed": False,
                "checkpoint_mutation_authorized": False,
                "new_checkpoint_authorized": False,
                "100m_authorized": False,
                "breakthrough_authorized": False,
            },
            separators=(",", ":"),
        ),
        flush=True,
    )
    result = run_checkpoint_triage(
        data_dir=DATA_DIR,
        run_dir=SOURCE_RUN_DIR,
    )
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2))
    volume.commit()
    print(
        "AERA_V25_POST8471_TRIAGE_RESULT_JSON="
        + json.dumps(result, separators=(",", ":")),
        flush=True,
    )
    return result


@app.local_entrypoint()
def main() -> None:
    check = preflight.remote()
    print(
        "AERA_V25_POST8471_TRIAGE_PREFLIGHT_LOCAL="
        + json.dumps(check, separators=(",", ":")),
        flush=True,
    )
    result = run_diagnostic.remote()
    print(json.dumps(result, indent=2), flush=True)
