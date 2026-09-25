from __future__ import annotations

import base64
import json
from pathlib import Path
import time
from typing import Any

import modal

JOB_ID = "rlt-scale15m-calibration-modal-20260925-a"
ATTEMPT_SEED = 20_261_014
DATA_SEED = 20_260_927

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.10,<2.11",
        "numpy>=2.0,<3",
        "datasets>=4.0,<5",
        "transformers>=4.55,<5",
        "tokenizers>=0.21,<1",
        "huggingface-hub>=0.34,<1",
    )
    .add_local_python_source("tam_research", "experiments")
)
app = modal.App("tam-rlt-scale15m-calibration-20260925-a")


def _encode(value: dict[str, Any]) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).decode()


@app.function(
    image=image,
    gpu="A100",
    cpu=4.0,
    memory=65536,
    timeout=55 * 60,
    retries=0,
    max_containers=1,
)
def run_calibration_remote(job_json: str) -> str:
    import torch
    from tam_research.data import prepare_fineweb
    from experiments.rlt.scale15m_calibration import (
        ATTEMPT_SEED as CALIBRATION_SEED,
        MICRO_BATCH_SIZE,
        PROBE_STEPS,
        SEQ_LEN,
        run_scale15m_calibration,
    )

    started = time.time()
    job: dict[str, Any] | None = None
    runtime: dict[str, Any] | None = None
    try:
        raw = json.loads(job_json)
        if not isinstance(raw, dict):
            raise ValueError("job JSON must be object")
        job = raw
        expected = {
            "schema": 1,
            "job_id": JOB_ID,
            "task": "scale15m_calibration_only",
            "attempt_seed": ATTEMPT_SEED,
            "requires_c4_job_id": "rlt-c4-balanced-4m-modal-20260925-a",
            "data_seed": DATA_SEED,
            "train_tokens": 500000,
            "val_tokens": 10000,
            "seq_len": SEQ_LEN,
            "micro_batch_size": MICRO_BATCH_SIZE,
            "probe_steps": PROBE_STEPS,
            "automatic_retry": False,
        }
        if job != expected:
            raise RuntimeError(f"scale calibration preregistration mismatch: {job}")
        if CALIBRATION_SEED != ATTEMPT_SEED:
            raise RuntimeError("calibration seed drift")

        runtime = {
            "torch_version": str(torch.__version__),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_runtime": None if torch.version.cuda is None else str(torch.version.cuda),
            "gpu_name": str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else None,
            "bf16_supported": bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        }
        if not runtime["cuda_available"]:
            raise RuntimeError("scale calibration started without CUDA")

        data_dir = Path("/tmp/rlt-scale15m-calibration-data")
        data_dir.mkdir(parents=True, exist_ok=True)
        data_meta = prepare_fineweb(
            str(data_dir),
            train_tokens=job["train_tokens"],
            val_tokens=job["val_tokens"],
            seed=job["data_seed"],
        )
        calibration = run_scale15m_calibration(data_dir=str(data_dir))
        result = {
            "schema": 1,
            "job_id": JOB_ID,
            "status": "complete",
            "started_unix": started,
            "finished_unix": time.time(),
            "worker": "modal",
            "job": job,
            "runtime": runtime,
            "data": data_meta,
            "calibration": calibration,
        }
    except Exception as exc:
        result = {
            "schema": 1,
            "job_id": JOB_ID,
            "status": "failed",
            "started_unix": started,
            "finished_unix": time.time(),
            "worker": "modal",
            "job": job,
            "runtime": runtime,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "scientific_conclusion": None,
        }
    return _encode(result)


@app.local_entrypoint()
def main(job_path: str) -> None:
    result = run_calibration_remote.remote(Path(job_path).read_text())
    if not isinstance(result, str):
        raise TypeError("remote scale calibration result must be base64 text")
    print("RLT_SCALE15M_CALIBRATION_RESULT_B64=" + result)
