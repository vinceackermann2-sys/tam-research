from __future__ import annotations

import json
from pathlib import Path

import modal

APP_NAME = "tam-research-aera-v25-post8471-triage-repair2"
VOLUME_NAME = "tam-research-data"
DATA_DIR = "/vol/data/tam100m-2b-curated-v1"
SOURCE_RUN_DIR = "/vol/aera-real-language/v25-dev-seed8471"
ORIGINAL_RESULT_PATH = "/vol/aera-real-language/v25-post8471-issue369/result.json"
REPAIR1_RESULT_PATH = "/vol/aera-real-language/v25-post8471-issue369-repair1/result.json"
REPAIR2_RESULT_PATH = "/vol/aera-real-language/v25-post8471-issue369-repair2/result.json"
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

    from tam_research.aera_real_language_v12 import validate_production_data
    from tam_research.aera_v25_post8471_triage_repair2 import (
        repair2_protocol,
        validate_source_result_seed,
    )

    volume.reload()
    source = Path(SOURCE_RUN_DIR)
    aera_path = source / "aera.pt"
    transformer_path = source / "transformer.pt"
    source_result_path = source / "result.json"
    for path in (aera_path, transformer_path, source_result_path):
        if not path.exists():
            raise RuntimeError(f"required seed8471 source missing: {path}")
    for path in (
        Path(ORIGINAL_RESULT_PATH),
        Path(REPAIR1_RESULT_PATH),
        Path(REPAIR2_RESULT_PATH),
    ):
        if path.exists():
            raise RuntimeError(f"refusing repair2 because diagnostic result already exists: {path}")

    a_payload = torch.load(aera_path, map_location="cpu", weights_only=False)
    t_payload = torch.load(transformer_path, map_location="cpu", weights_only=False)
    if a_payload.get("seed") != 8471 or t_payload.get("seed") != 8471:
        raise RuntimeError("repair2 checkpoint seed mismatch")
    source_result = json.loads(source_result_path.read_text())
    result_seed_fields = validate_source_result_seed(source_result)

    check = {
        "protocol": repair2_protocol(),
        "data": validate_production_data(DATA_DIR),
        "checkpoint_seed_aera": int(a_payload["seed"]),
        "checkpoint_seed_transformer": int(t_payload["seed"]),
        "source_result_seed_fields": result_seed_fields,
        "all_prior_diagnostic_result_paths_absent": True,
        "training_performed": False,
        "checkpoint_mutated": False,
        "gpu_authorized_by_preflight": False,
    }
    print(
        "AERA_V25_POST8471_TRIAGE_REPAIR2_PREFLIGHT_JSON="
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

    from tam_research.aera_v25_post8471_triage_repair2 import (
        REPAIR_ISSUE,
        run_checkpoint_triage_repair1,
    )

    volume.reload()
    for path in (
        Path(ORIGINAL_RESULT_PATH),
        Path(REPAIR1_RESULT_PATH),
        Path(REPAIR2_RESULT_PATH),
    ):
        if path.exists():
            raise RuntimeError(f"refusing duplicate/ambiguous repair2 diagnostic: {path}")
    print(
        "AERA_V25_POST8471_TRIAGE_REPAIR2_L4_START_JSON="
        + json.dumps(
            {
                "device": torch.cuda.get_device_name(0),
                "research_issue": 369,
                "repair1_issue": 372,
                "repair2_issue": REPAIR_ISSUE,
                "source_failed_trigger": 376,
                "source_checkpoint_seed": 8471,
                "diagnostic_sampling_seed": 138471,
                "diagnostic_implementation": "repair1_unmodified",
                "hard_gpu_timeout_seconds": MAX_GPU_SECONDS,
                "training_performed": False,
                "checkpoint_mutation_authorized": False,
                "100m_authorized": False,
            },
            separators=(",", ":"),
        ),
        flush=True,
    )
    result = run_checkpoint_triage_repair1(
        data_dir=DATA_DIR,
        run_dir=SOURCE_RUN_DIR,
    )
    result["repair2"] = {
        "repair_issue": REPAIR_ISSUE,
        "source_failed_trigger": 376,
        "semantic_change": "source_result_seed_schema_preflight_only",
        "diagnostic_implementation_changed_from_repair1": False,
    }
    result_path = Path(REPAIR2_RESULT_PATH)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2))
    volume.commit()
    print(
        "AERA_V25_POST8471_TRIAGE_REPAIR2_RESULT_JSON="
        + json.dumps(result, separators=(",", ":")),
        flush=True,
    )
    return result


@app.local_entrypoint()
def main() -> None:
    check = preflight.remote()
    print(
        "AERA_V25_POST8471_TRIAGE_REPAIR2_PREFLIGHT_LOCAL="
        + json.dumps(check, separators=(",", ":")),
        flush=True,
    )
    result = run_diagnostic.remote()
    print(json.dumps(result, indent=2), flush=True)
