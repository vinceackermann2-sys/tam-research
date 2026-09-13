from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from tam_research.data import prepare_fineweb
from experiments.rlt.smoke import run_smoke
from experiments.rlt.train_rlt import train_rlt

SMOKE_JOB_ID = "rlt-publicspec-smoke-aws-20260913-a"
TRAIN_JOB_ID = "rlt-tiny-64k-aws-20260913-a"
SEED = 20260916


def _write_create_once(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _claim(path: Path, job_id: str, task: str, metadata: dict[str, object]) -> None:
    _write_create_once(
        path,
        {
            "schema": 1,
            "job_id": job_id,
            "status": "claimed",
            "task": task,
            "seed": SEED,
            "claimed_unix": time.time(),
            "worker": "aws-ec2",
            "metadata": metadata,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="/opt/tam-rlt-output")
    parser.add_argument("--data-root", default="/opt/tam-rlt-data")
    parser.add_argument("--run-root", default="/opt/tam-rlt-runs")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("AWS RLT run requires a CUDA GPU instance")

    output_root = Path(args.output_root)
    data_root = Path(args.data_root)
    run_root = Path(args.run_root)
    output_root.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)

    gpu = torch.cuda.get_device_name(0)
    metadata: dict[str, object] = {
        "worker": "aws-ec2",
        "gpu": gpu,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "seed": SEED,
    }

    smoke_claim = output_root / f"{SMOKE_JOB_ID}.claim.json"
    smoke_result = output_root / f"{SMOKE_JOB_ID}.json"
    train_claim = output_root / f"{TRAIN_JOB_ID}.claim.json"
    train_result = output_root / f"{TRAIN_JOB_ID}.json"

    if smoke_claim.exists() or smoke_result.exists():
        raise RuntimeError(
            f"{SMOKE_JOB_ID} already has a local claim/result; refusing to rerun"
        )

    _claim(smoke_claim, SMOKE_JOB_ID, "smoke", metadata)
    smoke_started = time.time()
    try:
        smoke_output = run_smoke("cuda", seed=SEED)
        smoke_payload: dict[str, object] = {
            "schema": 1,
            "job_id": SMOKE_JOB_ID,
            "status": "complete",
            "worker": "aws-ec2",
            "started_unix": smoke_started,
            "finished_unix": time.time(),
            "job": {"task": "smoke", "seed": SEED},
            "metadata": metadata,
            "output": smoke_output,
        }
    except Exception as exc:
        smoke_payload = {
            "schema": 1,
            "job_id": SMOKE_JOB_ID,
            "status": "failed",
            "worker": "aws-ec2",
            "started_unix": smoke_started,
            "finished_unix": time.time(),
            "job": {"task": "smoke", "seed": SEED},
            "metadata": metadata,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

    _write_create_once(smoke_result, smoke_payload)
    print(json.dumps(smoke_payload, indent=2, sort_keys=True), flush=True)

    if (
        smoke_payload.get("status") != "complete"
        or not isinstance(smoke_payload.get("output"), dict)
        or smoke_payload["output"].get("status") != "pass"
    ):
        raise RuntimeError("AWS smoke did not PASS; bounded training remains blocked")

    if train_claim.exists() or train_result.exists():
        raise RuntimeError(
            f"{TRAIN_JOB_ID} already has a local claim/result; refusing to rerun"
        )

    train_job: dict[str, object] = {
        "task": "train",
        "requires_job_id": SMOKE_JOB_ID,
        "seed": SEED,
        "profile": "tiny",
        "token_budget": 65_536,
        "seq_len": 64,
        "micro_batch_size": 2,
        "grad_accum_steps": 4,
        "train_tokens": 1_000_000,
        "val_tokens": 100_000,
    }
    _claim(train_claim, TRAIN_JOB_ID, "train", {**metadata, "job": train_job})
    train_started = time.time()
    try:
        data_meta = prepare_fineweb(
            str(data_root),
            train_tokens=1_000_000,
            val_tokens=100_000,
            seed=1234,
        )
        training = train_rlt(
            profile="tiny",
            seed=SEED,
            data_dir=str(data_root),
            run_root=str(run_root / TRAIN_JOB_ID),
            token_budget=65_536,
            seq_len=64,
            micro_batch_size=2,
            grad_accum_steps=4,
        )
        train_payload: dict[str, object] = {
            "schema": 1,
            "job_id": TRAIN_JOB_ID,
            "status": "complete",
            "worker": "aws-ec2",
            "started_unix": train_started,
            "finished_unix": time.time(),
            "job": train_job,
            "metadata": metadata,
            "output": {"data": data_meta, "training": training},
        }
    except Exception as exc:
        train_payload = {
            "schema": 1,
            "job_id": TRAIN_JOB_ID,
            "status": "failed",
            "worker": "aws-ec2",
            "started_unix": train_started,
            "finished_unix": time.time(),
            "job": train_job,
            "metadata": metadata,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

    _write_create_once(train_result, train_payload)
    print(json.dumps(train_payload, indent=2, sort_keys=True), flush=True)

    if train_payload.get("status") != "complete":
        raise RuntimeError("AWS bounded training failed; see terminal result JSON")

    print("AWS RLT one-shot complete", flush=True)
    print(f"smoke_result={smoke_result}", flush=True)
    print(f"train_result={train_result}", flush=True)


if __name__ == "__main__":
    main()
