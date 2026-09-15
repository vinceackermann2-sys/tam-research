from __future__ import annotations

import base64
import json
from pathlib import Path
import time
from typing import Any

import modal

MODAL_JOB_MARKER = "-modal-"

# Modal 1.x does not automatically ship local packages. Package the two Python
# package roots explicitly so imports resolve from /root on the remote worker.
compute_image = (
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

app = modal.App("tam-rlt-publicspec-compute")


def _runtime_metadata() -> dict[str, Any]:
    import torch

    return {
        "torch_version": str(torch.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_runtime": None if torch.version.cuda is None else str(torch.version.cuda),
        "gpu_name": str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else None,
        "bf16_supported": bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
    }


def _encode(value: dict[str, Any]) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).decode()


@app.function(
    image=compute_image,
    gpu="A100",
    timeout=60 * 60,
    cpu=4.0,
    memory=16384,
    retries=0,
    max_containers=1,
)
def run_job_remote(job_json: str) -> str:
    """Execute exactly one job and return only transport-safe base64 text."""
    from experiments.rlt.bridge_worker import validate_job
    from experiments.rlt.smoke import run_smoke
    from experiments.rlt.train_rlt import train_rlt
    from tam_research.data import prepare_fineweb

    started = time.time()
    job_id = "unknown"
    job: dict[str, Any] | None = None
    runtime: dict[str, Any] | None = None
    try:
        raw = json.loads(job_json)
        if not isinstance(raw, dict):
            raise ValueError("job JSON must be an object")
        job_id = str(raw.get("job_id", ""))
        if MODAL_JOB_MARKER not in job_id:
            raise ValueError("refusing a non-Modal job id")
        job = validate_job(raw, f"{job_id}.json")

        runtime = _runtime_metadata()
        if not runtime["cuda_available"]:
            raise RuntimeError("Modal allocated the function without a usable CUDA runtime")

        if job["task"] == "smoke":
            output = dict(run_smoke("cuda", seed=job["seed"]))
            output["runtime"] = runtime
        elif job["task"] == "train":
            data_dir = Path("/tmp/rlt-data")
            run_root = Path("/tmp/rlt-runs") / job_id
            data_dir.mkdir(parents=True, exist_ok=True)
            run_root.mkdir(parents=True, exist_ok=True)
            data_meta = prepare_fineweb(
                str(data_dir),
                train_tokens=job["train_tokens"],
                val_tokens=job["val_tokens"],
                seed=1234,
            )
            training = train_rlt(
                profile=job["profile"],
                seed=job["seed"],
                data_dir=str(data_dir),
                run_root=str(run_root),
                token_budget=job["token_budget"],
                seq_len=job["seq_len"],
                micro_batch_size=job["micro_batch_size"],
                grad_accum_steps=job["grad_accum_steps"],
            )
            output = {
                "status": "complete",
                "runtime": runtime,
                "data": data_meta,
                "training": training,
            }
        else:
            raise ValueError(f"unsupported task {job['task']!r}")

        result = {
            "schema": 1,
            "job_id": job_id,
            "status": "complete",
            "started_unix": started,
            "finished_unix": time.time(),
            "worker": "modal",
            "job": job,
            "output": output,
        }
    except AssertionError as exc:
        result = {
            "schema": 1,
            "job_id": job_id,
            "status": "complete",
            "started_unix": started,
            "finished_unix": time.time(),
            "worker": "modal",
            "job": job,
            "output": {
                "status": "fail",
                "runtime": runtime,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        }
    except Exception as exc:
        result = {
            "schema": 1,
            "job_id": job_id,
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

    # Returning a plain string prevents the local runner from needing Torch or
    # any other remote-only Python class to deserialize the result.
    return _encode(result)


@app.local_entrypoint()
def main(job_path: str) -> None:
    raw = Path(job_path).read_text()
    result_b64 = run_job_remote.remote(raw)
    if not isinstance(result_b64, str):
        raise TypeError("remote result transport must be a base64 string")
    print("RLT_MODAL_RESULT_B64=" + result_b64)
