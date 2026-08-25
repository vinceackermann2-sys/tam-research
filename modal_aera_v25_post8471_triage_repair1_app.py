from __future__ import annotations

import json
from pathlib import Path

import modal

APP_NAME = "tam-research-aera-v25-post8471-triage-repair1"
VOLUME_NAME = "tam-research-data"
DATA_DIR = "/vol/data/tam100m-2b-curated-v1"
SOURCE_RUN_DIR = "/vol/aera-real-language/v25-dev-seed8471"
ORIGINAL_RESULT_PATH = "/vol/aera-real-language/v25-post8471-issue369/result.json"
REPAIR_RESULT_PATH = "/vol/aera-real-language/v25-post8471-issue369-repair1/result.json"
MAX_GPU_SECONDS = 900

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.10,<2.11", "numpy>=2.0,<3")
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_python_source("tam_research")
)


@app.function(image=image, cpu=4, memory=8192, timeout=300, volumes={"/vol": volume})
def preflight() -> dict:
    import torch

    from tam_research.aera_v25_post8471_triage_repair1 import repair_protocol
    from tam_research.aera_real_language_v12 import validate_production_data

    volume.reload()
    source = Path(SOURCE_RUN_DIR)
    aera_path = source / "aera.pt"
    transformer_path = source / "transformer.pt"
    source_result = source / "result.json"
    original_result = Path(ORIGINAL_RESULT_PATH)
    repair_result = Path(REPAIR_RESULT_PATH)
    for path in (aera_path, transformer_path, source_result):
        if not path.exists():
            raise RuntimeError(f"required seed8471 source missing: {path}")
    if original_result.exists():
        raise RuntimeError(
            "refusing repair1: original issue369 diagnostic unexpectedly produced a result"
        )
    if repair_result.exists():
        raise RuntimeError(f"refusing duplicate repair1 result: {repair_result}")
    a_payload = torch.load(aera_path, map_location="cpu", weights_only=False)
    t_payload = torch.load(transformer_path, map_location="cpu", weights_only=False)
    if a_payload.get("seed") != 8471 or t_payload.get("seed") != 8471:
        raise RuntimeError("repair1 source checkpoint seed mismatch")
    source_payload = json.loads(source_result.read_text())
    if int(source_payload.get("seed", -1)) != 8471:
        raise RuntimeError("repair1 source result seed mismatch")
    check = {
        "protocol": repair_protocol(),
        "data": validate_production_data(DATA_DIR),
        "source_checkpoint_seed": 8471,
        "source_result_issue": 368,
        "source_failed_trigger": 371,
        "repair_issue": 372,
        "original_result_path_absent": True,
        "repair_result_path_absent": True,
        "training_performed": False,
        "checkpoint_mutated": False,
        "gpu_authorized_by_preflight": False,
    }
    print(
        "AERA_V25_POST8471_TRIAGE_REPAIR1_PREFLIGHT_JSON="
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

    from tam_research.aera_v25_post8471_triage_repair1 import (
        run_checkpoint_triage_repair1,
    )

    volume.reload()
    original_result = Path(ORIGINAL_RESULT_PATH)
    result_path = Path(REPAIR_RESULT_PATH)
    if original_result.exists():
        raise RuntimeError("original issue369 result appeared; refusing repair1")
    if result_path.exists():
        raise RuntimeError(f"refusing duplicate repair1 diagnostic: {result_path}")
    print(
        "AERA_V25_POST8471_TRIAGE_REPAIR1_L4_START_JSON="
        + json.dumps(
            {
                "device": torch.cuda.get_device_name(0),
                "research_issue": 369,
                "repair_issue": 372,
                "source_failed_trigger": 371,
                "source_checkpoint_seed": 8471,
                "diagnostic_sampling_seed": 138471,
                "loss_time_slice_tokens": 32,
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
    result = run_checkpoint_triage_repair1(
        data_dir=DATA_DIR,
        run_dir=SOURCE_RUN_DIR,
    )
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2))
    volume.commit()
    print(
        "AERA_V25_POST8471_TRIAGE_REPAIR1_RESULT_JSON="
        + json.dumps(result, separators=(",", ":")),
        flush=True,
    )
    return result


@app.local_entrypoint()
def main() -> None:
    check = preflight.remote()
    print(
        "AERA_V25_POST8471_TRIAGE_REPAIR1_PREFLIGHT_LOCAL="
        + json.dumps(check, separators=(",", ":")),
        flush=True,
    )
    result = run_diagnostic.remote()
    print(json.dumps(result, indent=2), flush=True)
