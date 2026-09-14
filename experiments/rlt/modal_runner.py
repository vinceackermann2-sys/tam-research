from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import modal

REPO_ROOT = Path(__file__).resolve().parents[2]
REMOTE_ROOT = "/root/tam-research"
BRANCH = "exp/rlt-colab"
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

app = modal.App("tam-rlt-publicspec")
github_secret = modal.Secret.from_name(
    "tam-rlt-github",
    required_keys=["GITHUB_TOKEN"],
)


def _runtime_metadata() -> dict[str, Any]:
    import torch

    return {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "bf16_supported": bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
    }


@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 60,
    cpu=4.0,
    memory=16384,
    secrets=[github_secret],
)
def run_job(job_id: str) -> dict[str, Any]:
    if REMOTE_ROOT not in sys.path:
        sys.path.insert(0, REMOTE_ROOT)

    import torch

    from experiments.rlt.bridge_worker import (
        CLAIMS_DIR,
        JOBS_DIR,
        RESULTS_DIR,
        GitHubBridge,
        prerequisite_passed,
        validate_job,
    )
    from experiments.rlt.smoke import run_smoke
    from experiments.rlt.train_rlt import train_rlt
    from tam_research.data import prepare_fineweb

    if MODAL_JOB_MARKER not in job_id:
        raise ValueError(
            f"refusing non-Modal job id {job_id!r}; Modal jobs must contain {MODAL_JOB_MARKER!r}"
        )

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("Modal secret tam-rlt-github must provide GITHUB_TOKEN")

    bridge = GitHubBridge(token=token, branch=BRANCH)
    source_path = f"{JOBS_DIR}/{job_id}.json"
    result_path = f"{RESULTS_DIR}/{job_id}.json"
    claim_path = f"{CLAIMS_DIR}/{job_id}.json"

    if bridge.exists(result_path):
        return bridge.read_json(result_path)

    raw = bridge.read_json(source_path)
    job = validate_job(raw, f"{job_id}.json")
    if job["job_id"] != job_id:
        raise ValueError("job_id mismatch")
    if not prerequisite_passed(bridge, job):
        raise RuntimeError(
            f"prerequisite {job.get('requires_job_id')!r} has not durably passed"
        )
    if bridge.exists(claim_path):
        raise RuntimeError(
            f"{job_id} already has a claim but no terminal result; refusing to rerun"
        )

    started = time.time()
    runtime = _runtime_metadata()
    if not runtime["cuda_available"]:
        raise RuntimeError("Modal allocated the function without a usable CUDA runtime")

    claim = {
        "schema": 1,
        "job_id": job_id,
        "status": "claimed",
        "claimed_unix": started,
        "worker": "modal",
        "branch": BRANCH,
        "runtime": runtime,
        "job": job,
    }
    bridge.create_json(
        claim_path,
        claim,
        f"rlt modal: claim {job_id} before GPU execution",
    )
    print(json.dumps({"event": "claimed", "job_id": job_id, "runtime": runtime}), flush=True)

    try:
        if job["task"] == "smoke":
            output = run_smoke("cuda", seed=job["seed"])
            output = dict(output)
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

        payload: dict[str, Any] = {
            "schema": 1,
            "job_id": job_id,
            "status": "complete",
            "started_unix": started,
            "finished_unix": time.time(),
            "worker": "modal",
            "branch": BRANCH,
            "job": job,
            "output": output,
        }
    except Exception as exc:
        payload = {
            "schema": 1,
            "job_id": job_id,
            "status": "failed",
            "started_unix": started,
            "finished_unix": time.time(),
            "worker": "modal",
            "branch": BRANCH,
            "runtime": runtime,
            "job": job,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

    bridge.create_json(
        result_path,
        payload,
        f"rlt modal: record {job_id} terminal result",
    )
    print(json.dumps({"event": "terminal", "job_id": job_id, "status": payload["status"]}), flush=True)
    return payload


@app.local_entrypoint()
def main(job_id: str) -> None:
    """Run exactly one preregistered Modal job and print its durable result."""
    result = run_job.remote(job_id)
    print(json.dumps(result, indent=2, sort_keys=True))
