from __future__ import annotations

import base64
import json
from pathlib import Path
import time
from typing import Any

import modal


JOB_ID = "transformer-control-64k-modal-20260915-a"
PREREQ_ID = "rlt-tiny-64k-modal-20260915-a"
EXPECTED_SEED = 20260921

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.7,<2.11",
        "numpy>=2.0,<3",
        "datasets>=4.0,<5",
        "transformers>=4.55,<5",
        "tokenizers>=0.21,<1",
        "huggingface-hub>=0.34,<1",
    )
    .add_local_python_source("tam_research", "experiments")
)

app = modal.App("tam-rlt-transformer-control")


def _encode(value: dict[str, Any]) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).decode()


def _validate(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("job JSON must be an object")
    expected = {
        "schema": 1,
        "job_id": JOB_ID,
        "task": "train_transformer_control",
        "requires_job_id": PREREQ_ID,
        "seed": EXPECTED_SEED,
        "token_budget": 65_536,
        "seq_len": 64,
        "micro_batch_size": 2,
        "grad_accum_steps": 4,
        "train_tokens": 1_000_000,
        "val_tokens": 100_000,
    }
    if raw != expected:
        raise ValueError(f"job does not match preregistration: {raw!r}")
    return dict(raw)


@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 60,
    cpu=4.0,
    memory=16384,
    retries=0,
    max_containers=1,
)
def run_control_remote(job_json: str) -> str:
    from tam_research.data import prepare_fineweb
    from experiments.rlt.train_transformer_control import train_transformer_control
    import torch

    started = time.time()
    job: dict[str, Any] | None = None
    runtime: dict[str, Any] | None = None
    try:
        job = _validate(json.loads(job_json))
        runtime = {
            "torch_version": str(torch.__version__),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_runtime": None if torch.version.cuda is None else str(torch.version.cuda),
            "gpu_name": str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else None,
            "bf16_supported": bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        }
        if not runtime["cuda_available"]:
            raise RuntimeError("Modal allocated the control without usable CUDA")

        data_dir = Path("/tmp/rlt-control-data")
        run_root = Path("/tmp/rlt-control-runs") / JOB_ID
        data_dir.mkdir(parents=True, exist_ok=True)
        run_root.mkdir(parents=True, exist_ok=True)
        data_meta = prepare_fineweb(
            str(data_dir),
            train_tokens=job["train_tokens"],
            val_tokens=job["val_tokens"],
            seed=1234,
        )
        training = train_transformer_control(
            seed=job["seed"],
            data_dir=str(data_dir),
            run_root=str(run_root),
            token_budget=job["token_budget"],
            seq_len=job["seq_len"],
            micro_batch_size=job["micro_batch_size"],
            grad_accum_steps=job["grad_accum_steps"],
        )
        result = {
            "schema": 1,
            "job_id": JOB_ID,
            "status": "complete",
            "started_unix": started,
            "finished_unix": time.time(),
            "worker": "modal",
            "job": job,
            "output": {
                "status": "complete",
                "runtime": runtime,
                "data": data_meta,
                "training": training,
            },
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
    raw = Path(job_path).read_text()
    result_b64 = run_control_remote.remote(raw)
    if not isinstance(result_b64, str):
        raise TypeError("remote control result must be a base64 string")
    print("RLT_TRANSFORMER_CONTROL_RESULT_B64=" + result_b64)
