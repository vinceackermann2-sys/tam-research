from __future__ import annotations

import base64
import json
from pathlib import Path
import time
from typing import Any

import modal

JOB_ID = "rlt-compiled-paired-64k-modal-20260915-a"
SCIENTIFIC_SEED = 20_260_926

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
app = modal.App("tam-rlt-compiled-paired-64k-20260915-a")


def _encode(value: dict[str, Any]) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).decode()


@app.function(
    image=image,
    gpu="A100",
    cpu=4.0,
    memory=16384,
    timeout=35 * 60,
    retries=0,
    max_containers=1,
)
def run_pair_remote(job_json: str) -> str:
    from experiments.rlt.train_compiled_pair import SCIENTIFIC_SEED as TRAINER_SEED
    from experiments.rlt.train_compiled_pair import train_compiled_pair
    from tam_research.data import prepare_fineweb
    import torch

    started = time.time()
    job: dict[str, Any] | None = None
    runtime: dict[str, Any] | None = None
    try:
        raw = json.loads(job_json)
        if not isinstance(raw, dict):
            raise ValueError("job JSON must be an object")
        job = raw
        expected = {
            "schema": 1,
            "job_id": JOB_ID,
            "task": "compiled_paired_64k",
            "scientific_seed": SCIENTIFIC_SEED,
            "requires_systems_job_id": "rlt-systems-compile-modal-20260915-d",
            "requires_baseline_comparison_id": "rlt-vs-transformer-64k-comparison-20260915",
            "token_budget": 65536,
            "seq_len": 64,
            "micro_batch_size": 2,
            "grad_accum_steps": 4,
            "train_tokens": 1000000,
            "val_tokens": 100000,
            "automatic_retry": False,
        }
        if job != expected:
            raise RuntimeError(f"paired job preregistration mismatch: {job}")
        if TRAINER_SEED != SCIENTIFIC_SEED:
            raise RuntimeError("trainer scientific seed drift")
        runtime = {
            "torch_version": str(torch.__version__),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_runtime": None if torch.version.cuda is None else str(torch.version.cuda),
            "gpu_name": str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else None,
            "bf16_supported": bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        }
        if not runtime["cuda_available"]:
            raise RuntimeError("Modal paired job started without CUDA")

        data_dir = Path("/tmp/rlt-compiled-pair-data")
        data_dir.mkdir(parents=True, exist_ok=True)
        data_meta = prepare_fineweb(
            str(data_dir),
            train_tokens=job["train_tokens"],
            val_tokens=job["val_tokens"],
            seed=1234,
        )
        pair = train_compiled_pair(
            data_dir=str(data_dir),
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
            "runtime": runtime,
            "data": data_meta,
            "comparison": pair,
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
    result = run_pair_remote.remote(raw)
    if not isinstance(result, str):
        raise TypeError("remote paired result must be base64 text")
    print("RLT_COMPILED_PAIR_RESULT_B64=" + result)
