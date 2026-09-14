from __future__ import annotations

import base64
import json
from pathlib import Path
import sys
import time
from typing import Any

import modal

REPO_ROOT = Path(__file__).resolve().parents[2]
REMOTE_ROOT = "/root/tam-research"
MODAL_JOB_MARKER = "-modal-"

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
    .add_local_dir(
        str(REPO_ROOT / "tam_research"),
        remote_path=f"{REMOTE_ROOT}/tam_research",
        copy=True,
    )
    .add_local_dir(
        str(REPO_ROOT / "experiments"),
        remote_path=f"{REMOTE_ROOT}/experiments",
        copy=True,
    )
)

app = modal.App("tam-rlt-publicspec-compute")


def _runtime_metadata() -> dict[str, Any]:
    import torch

    return {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "bf16_supported": bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
    }


def _encode(value: dict[str, Any]) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).decode()


@app.function(
    image=image,
    timeout=10 * 60,
    cpu=1.0,
    memory=2048,
    retries=0,
    max_containers=1,
)
def preflight_remote() -> dict[str, Any]:
    """Non-GPU Modal control-plane check. It cannot consume a scientific attempt."""
    return {
        "schema": 1,
        "status": "pass",
        "worker": "modal-control",
        "finished_unix": time.time(),
        "scientific_job_executed": False,
        "python_version": sys.version,
    }


@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 60,
    cpu=4.0,
    memory=16384,
    retries=0,
    max_containers=1,
)
def run_job_remote(job_json: str) -> dict[str, Any]:
    if REMOTE_ROOT not in sys.path:
        sys.path.insert(0, REMOTE_ROOT)

    from experiments.rlt.bridge_worker import validate_job
    from experiments.rlt.smoke import run_smoke
    from experiments.rlt.train_rlt import train_rlt
    from tam_research.data import prepare_fineweb

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

    started = time.time()
    try:
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
        return {
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
        return {
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


@app.local_entrypoint()
def preflight() -> None:
    print("RLT_MODAL_PREFLIGHT_B64=" + _encode(preflight_remote.remote()))


@app.local_entrypoint()
def main(job_path: str) -> None:
    raw = Path(job_path).read_text()
    result = run_job_remote.remote(raw)
    print("RLT_MODAL_RESULT_B64=" + _encode(result))
