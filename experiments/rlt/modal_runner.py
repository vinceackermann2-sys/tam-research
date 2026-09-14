from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import sys
import time
from typing import Any
import urllib.parse

import modal

REPO_ROOT = Path(__file__).resolve().parents[2]
REMOTE_ROOT = "/root/tam-research"
BRANCH = "exp/rlt-colab"
MODAL_JOB_MARKER = "-modal-"
MODAL_ROOT = "experiments/rlt/modal"
JOBS_DIR = f"{MODAL_ROOT}/jobs"
CLAIMS_DIR = f"{MODAL_ROOT}/claims"
RESULTS_DIR = f"{MODAL_ROOT}/results"
DIAGNOSTICS_DIR = f"{MODAL_ROOT}/diagnostics"
DIAGNOSTIC_ID = "modal-auth-writecheck-20260914-a"
DIAGNOSTIC_PATH = f"{DIAGNOSTICS_DIR}/{DIAGNOSTIC_ID}.json"
PREREGISTERED_SMOKE_JOB_ID = "rlt-publicspec-smoke-modal-20260914-a"

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


def _branch_head(bridge: Any) -> str:
    row = bridge._request("GET", f"git/ref/heads/{BRANCH}")
    return str(row["object"]["sha"])


def _read_json_at_ref(bridge: Any, path: str, ref: str) -> tuple[dict[str, Any], str]:
    query = urllib.parse.urlencode({"ref": ref})
    row = bridge._request("GET", f"contents/{path}?{query}")
    raw = base64.b64decode(row["content"]).decode()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value, str(row["sha"])


@app.function(
    image=image,
    timeout=10 * 60,
    cpu=1.0,
    memory=2048,
    secrets=[github_secret],
    retries=0,
    max_containers=1,
)
def preflight_control() -> dict[str, Any]:
    """Prove Modal can read and create an isolated GitHub file before any GPU attempt."""
    if REMOTE_ROOT not in sys.path:
        sys.path.insert(0, REMOTE_ROOT)

    from experiments.rlt.bridge_worker import GitHubBridge, validate_job

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("Modal secret tam-rlt-github must provide GITHUB_TOKEN")

    bridge = GitHubBridge(token=token, branch=BRANCH)
    if bridge.exists(DIAGNOSTIC_PATH):
        existing = bridge.read_json(DIAGNOSTIC_PATH)
        if existing.get("status") != "pass":
            raise RuntimeError("existing Modal preflight diagnostic is not PASS")
        return existing

    head = _branch_head(bridge)
    source_path = f"{JOBS_DIR}/{PREREGISTERED_SMOKE_JOB_ID}.json"
    raw, source_blob_sha = _read_json_at_ref(bridge, source_path, head)
    job = validate_job(raw, f"{PREREGISTERED_SMOKE_JOB_ID}.json")
    if job["task"] != "smoke" or job["job_id"] != PREREGISTERED_SMOKE_JOB_ID:
        raise RuntimeError("preregistered Modal smoke job does not match the expected control job")

    payload = {
        "schema": 1,
        "diagnostic_id": DIAGNOSTIC_ID,
        "status": "pass",
        "worker": "modal-control",
        "finished_unix": time.time(),
        "branch": BRANCH,
        "branch_head": head,
        "job_id_checked": PREREGISTERED_SMOKE_JOB_ID,
        "job_source_blob_sha": source_blob_sha,
        "checks": {
            "github_secret_present": True,
            "branch_read": True,
            "job_read": True,
            "github_create_permission": True,
            "scientific_job_executed": False,
        },
    }
    bridge.create_json(
        DIAGNOSTIC_PATH,
        payload,
        f"rlt modal: record isolated control-plane preflight {DIAGNOSTIC_ID}",
    )
    return payload


@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 60,
    cpu=4.0,
    memory=16384,
    secrets=[github_secret],
    retries=0,
    max_containers=1,
)
def run_job(job_id: str) -> dict[str, Any]:
    if REMOTE_ROOT not in sys.path:
        sys.path.insert(0, REMOTE_ROOT)

    from experiments.rlt.bridge_worker import GitHubBridge, validate_job
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
    if not bridge.exists(DIAGNOSTIC_PATH):
        raise RuntimeError(
            "Modal GitHub preflight has not been durably recorded; run the preflight entrypoint first"
        )
    diagnostic = bridge.read_json(DIAGNOSTIC_PATH)
    if diagnostic.get("status") != "pass":
        raise RuntimeError("Modal GitHub preflight diagnostic is not PASS")

    source_path = f"{JOBS_DIR}/{job_id}.json"
    result_path = f"{RESULTS_DIR}/{job_id}.json"
    claim_path = f"{CLAIMS_DIR}/{job_id}.json"

    if bridge.exists(result_path):
        return bridge.read_json(result_path)
    if bridge.exists(claim_path):
        raise RuntimeError(
            f"{job_id} already has a claim but no terminal result; refusing to rerun"
        )

    head = _branch_head(bridge)
    raw, source_blob_sha = _read_json_at_ref(bridge, source_path, head)
    job = validate_job(raw, f"{job_id}.json")
    if job["job_id"] != job_id:
        raise ValueError("job_id mismatch")

    prerequisite = job.get("requires_job_id")
    if prerequisite:
        prerequisite_path = f"{RESULTS_DIR}/{prerequisite}.json"
        if not bridge.exists(prerequisite_path):
            raise RuntimeError(f"prerequisite {prerequisite!r} has no durable Modal result")
        prerequisite_result = bridge.read_json(prerequisite_path)
        prerequisite_ok = (
            prerequisite_result.get("status") == "complete"
            and isinstance(prerequisite_result.get("output"), dict)
            and prerequisite_result["output"].get("status") == "pass"
        )
        if not prerequisite_ok:
            raise RuntimeError(f"prerequisite {prerequisite!r} has not durably passed")

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
        "branch_head_at_claim": head,
        "job_source_blob_sha": source_blob_sha,
        "runtime": runtime,
        "job": job,
    }
    bridge.create_json(
        claim_path,
        claim,
        f"rlt modal: claim {job_id} before GPU execution",
    )
    print(
        json.dumps(
            {
                "event": "claimed",
                "job_id": job_id,
                "branch_head": head,
                "job_source_blob_sha": source_blob_sha,
                "runtime": runtime,
            }
        ),
        flush=True,
    )

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

        payload: dict[str, Any] = {
            "schema": 1,
            "job_id": job_id,
            "status": "complete",
            "started_unix": started,
            "finished_unix": time.time(),
            "worker": "modal",
            "branch": BRANCH,
            "branch_head_at_claim": head,
            "job_source_blob_sha": source_blob_sha,
            "job": job,
            "output": output,
        }
    except AssertionError as exc:
        payload = {
            "schema": 1,
            "job_id": job_id,
            "status": "complete",
            "started_unix": started,
            "finished_unix": time.time(),
            "worker": "modal",
            "branch": BRANCH,
            "branch_head_at_claim": head,
            "job_source_blob_sha": source_blob_sha,
            "job": job,
            "output": {
                "status": "fail",
                "runtime": runtime,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
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
            "branch_head_at_claim": head,
            "job_source_blob_sha": source_blob_sha,
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
    print(
        json.dumps({"event": "terminal", "job_id": job_id, "status": payload["status"]}),
        flush=True,
    )
    return payload


@app.local_entrypoint()
def preflight() -> None:
    result = preflight_control.remote()
    print(json.dumps(result, indent=2, sort_keys=True))


@app.local_entrypoint()
def main(job_id: str) -> None:
    """Run exactly one preregistered Modal job and print its durable result."""
    result = run_job.remote(job_id)
    print(json.dumps(result, indent=2, sort_keys=True))
