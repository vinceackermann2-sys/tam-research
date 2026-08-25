from __future__ import annotations

import json
from pathlib import Path

import modal

APP_NAME = "tam-research-aera-v25-1-issue381-systems"
VOLUME_NAME = "tam-research-data"
SOURCE_RUN_DIR = "/vol/aera-real-language/v25-dev-seed8471"
RESULT_PATH = "/vol/aera-real-language/v25-1-issue381-systems/result.json"
MAX_GPU_SECONDS = 600

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.10,<2.11", "numpy>=2.0,<3")
    .add_local_python_source("tam_research")
)


@app.function(image=image, cpu=4, memory=8192, timeout=300, volumes={"/vol": volume})
def preflight() -> dict:
    import torch

    from tam_research.aera_v25_1_systems import (
        SOURCE_SEED,
        checkpoint_hashes,
        cpu_contract_preflight,
    )
    from tam_research.aera_v25_1_systems_guarded import load_guarded_models
    from tam_research.aera_v25_post8471_triage_repair2 import (
        validate_source_result_seed,
    )

    volume.reload()
    root = Path(SOURCE_RUN_DIR)
    aera_path = root / "aera.pt"
    transformer_path = root / "transformer.pt"
    source_result_path = root / "result.json"
    for path in (aera_path, transformer_path, source_result_path):
        if not path.exists():
            raise RuntimeError(f"required seed8471 source missing: {path}")
    if Path(RESULT_PATH).exists():
        raise RuntimeError(
            f"refusing duplicate issue381 systems run because result exists: {RESULT_PATH}"
        )

    aera_payload = torch.load(aera_path, map_location="cpu", weights_only=False)
    transformer_payload = torch.load(
        transformer_path, map_location="cpu", weights_only=False
    )
    if (
        aera_payload.get("seed") != SOURCE_SEED
        or transformer_payload.get("seed") != SOURCE_SEED
    ):
        raise RuntimeError("issue381 checkpoint seed mismatch")
    source_result = json.loads(source_result_path.read_text())
    source_seed_fields = validate_source_result_seed(source_result)

    # Actual seed8471 checkpoint is strict-loaded into original v25, the final
    # CPU-proven v25.1 candidate, and the matched Transformer on CPU before the
    # sole GPU function can be called.
    original, candidate, transformer = load_guarded_models(
        run_dir=SOURCE_RUN_DIR, device=torch.device("cpu")
    )
    del original, candidate, transformer
    check = {
        "contract": cpu_contract_preflight(),
        "source_checkpoint_seed": SOURCE_SEED,
        "source_result_seed_fields": source_seed_fields,
        "source_checkpoint_hashes": checkpoint_hashes(SOURCE_RUN_DIR),
        "strict_v25_and_final_v25_1_checkpoint_load_cpu": True,
        "result_path_absent": True,
        "gpu_authorized_by_preflight": False,
        "training_performed": False,
        "optimizer_created": False,
        "backward_performed": False,
        "corpus_accessed": False,
        "checkpoint_mutated": False,
    }
    print(
        "AERA_V25_1_ISSUE381_SYSTEMS_PREFLIGHT_JSON="
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
def run_systems() -> dict:
    import torch

    from tam_research.aera_v25_1_systems_guarded import (
        run_guarded_systems_comparison,
    )

    volume.reload()
    result_path = Path(RESULT_PATH)
    if result_path.exists():
        raise RuntimeError(
            f"refusing duplicate issue381 systems run because result exists: {RESULT_PATH}"
        )
    print(
        "AERA_V25_1_ISSUE381_SYSTEMS_L4_START_JSON="
        + json.dumps(
            {
                "device": torch.cuda.get_device_name(0),
                "research_issue": 381,
                "source_checkpoint_seed": 8471,
                "source_checkpoint_mode": "read_only",
                "hard_gpu_timeout_seconds": MAX_GPU_SECONDS,
                "training_performed": False,
                "optimizer_created": False,
                "backward_performed": False,
                "corpus_accessed": False,
                "checkpoint_mutation_authorized": False,
                "100m_authorized": False,
            },
            separators=(",", ":"),
        ),
        flush=True,
    )
    result = run_guarded_systems_comparison(run_dir=SOURCE_RUN_DIR)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2))
    volume.commit()
    print(
        "AERA_V25_1_ISSUE381_SYSTEMS_RESULT_JSON="
        + json.dumps(result, separators=(",", ":")),
        flush=True,
    )
    return result


@app.local_entrypoint()
def main() -> None:
    check = preflight.remote()
    print(
        "AERA_V25_1_ISSUE381_SYSTEMS_PREFLIGHT_LOCAL="
        + json.dumps(check, separators=(",", ":")),
        flush=True,
    )
    result = run_systems.remote()
    print(json.dumps(result, indent=2), flush=True)
