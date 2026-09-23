from __future__ import annotations

import base64
import json
from pathlib import Path
import time
from typing import Any

import modal

JOB_ID = "rlt-crossdata-eval-4m-c-modal-20260923-a"
CHECKPOINT_VOLUME = "tam-rlt-scientific-checkpoints"

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
checkpoint_volume = modal.Volume.from_name(CHECKPOINT_VOLUME, create_if_missing=False)
app = modal.App("tam-rlt-crossdata-eval-4m-c-20260923-a")


def _encode(value: dict[str, Any]) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).decode()


@app.function(
    image=image,
    gpu="A100",
    cpu=4.0,
    memory=65536,
    timeout=45 * 60,
    retries=0,
    max_containers=1,
    volumes={"/checkpoints": checkpoint_volume},
)
def run_eval_remote(job_json: str) -> str:
    from experiments.rlt.eval_crossdata_4m_c import (
        BATCH_SIZE,
        DATA_SEED,
        EVAL_BATCHES,
        EVAL_TARGET_TOKENS,
        REQUIRED_VAL_TOKENS,
        SEQ_LEN,
        evaluate_crossdata_4m_c,
    )
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
            "task": "crossdata_eval_4m_c",
            "requires_checkpoint_job_ids": [
                "rlt-compiled-paired-4m-modal-20260923-c",
            ],
            "fresh_data_seed": DATA_SEED,
            "eval_target_tokens_per_pair": EVAL_TARGET_TOKENS,
            "eval_batches": EVAL_BATCHES,
            "batch_size": BATCH_SIZE,
            "seq_len": SEQ_LEN,
            "fresh_train_tokens": 100000,
            "fresh_val_tokens": 1100000,
            "checkpoint_volume": CHECKPOINT_VOLUME,
            "automatic_retry": False,
        }
        if job != expected:
            raise RuntimeError(f"cross-data 4M C job preregistration mismatch: {job}")
        if job["fresh_val_tokens"] < REQUIRED_VAL_TOKENS:
            raise RuntimeError("fresh validation allocation too small")

        runtime = {
            "torch_version": str(torch.__version__),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_runtime": None if torch.version.cuda is None else str(torch.version.cuda),
            "gpu_name": str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else None,
            "bf16_supported": bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        }
        if not runtime["cuda_available"]:
            raise RuntimeError("cross-data 4M C evaluation started without CUDA")

        for checkpoint_job_id in job["requires_checkpoint_job_ids"]:
            job_dir = Path("/checkpoints") / checkpoint_job_id
            for name in ("rlt_final.pt", "transformer_final.pt"):
                if not (job_dir / name).exists():
                    raise FileNotFoundError(f"missing durable checkpoint: {job_dir / name}")

        data_dir = Path("/tmp/rlt-crossdata-eval-4m-c")
        data_dir.mkdir(parents=True, exist_ok=True)
        data_meta = prepare_fineweb(
            str(data_dir),
            train_tokens=job["fresh_train_tokens"],
            val_tokens=job["fresh_val_tokens"],
            seed=job["fresh_data_seed"],
        )
        evaluation = evaluate_crossdata_4m_c(
            checkpoint_root="/checkpoints",
            val_path=str(data_dir / "val.bin"),
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
            "fresh_data": data_meta,
            "evaluation": evaluation,
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
    result = run_eval_remote.remote(raw)
    if not isinstance(result, str):
        raise TypeError("remote cross-data 4M C result must be base64 text")
    print("RLT_CROSSDATA_4M_C_RESULT_B64=" + result)
