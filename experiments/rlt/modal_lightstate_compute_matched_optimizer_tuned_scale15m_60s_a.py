from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import modal

JOB_ID = "rlt-lightstate-compute-matched-optimizer-tuned-scale15m-60s-modal-20261007-a"
CHECKPOINT_VOLUME = "tam-rlt-lightstate-compute-matched-optimizer-tuned-scale15m-account2"
EXPECTED_TRAIN_SHA256 = "0f9ea99680f532f990ca64fabbd02b46bba2c4f8077f868545a03f5e6907c907"
EXPECTED_VAL_SHA256 = "e6b3fb209a80a2cebe22924c3c8af26b30fc4afd322a1a49c71af8b968057db2"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.10,<2.11", "numpy>=2.0,<3", "datasets>=4.0,<5",
        "transformers>=4.55,<5", "tokenizers>=0.21,<1", "huggingface-hub>=0.34,<1",
    )
    .add_local_python_source("tam_research", "experiments")
)
checkpoint_volume = modal.Volume.from_name(CHECKPOINT_VOLUME, create_if_missing=True)
app = modal.App("tam-rlt-lightstate-compute-matched-optimizer-tuned-scale15m-60s-20261007-a")


def _encode(value: dict[str, Any]) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).decode()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=4.0,
    memory=65536,
    timeout=70 * 60,
    retries=0,
    max_containers=1,
    volumes={"/checkpoints": checkpoint_volume},
)
def run_pair_remote(job_json: str) -> str:
    import torch
    from tam_research.data import prepare_fineweb
    from experiments.rlt.train_lightstate_compute_matched_optimizer_tuned_scale15m_60s_a import (
        EXPECTED_PARAMETERS,
        SCIENTIFIC_SEED,
        TIME_BUDGET_SECONDS,
        LIGHTSTATE_TRAIN_BATCH_SIZE,
        TRANSFORMER_TRAIN_BATCH_SIZE,
        LIGHTSTATE_LEARNING_RATE,
        TRANSFORMER_LEARNING_RATE,
        LIGHTSTATE_SCHEDULE,
        TRANSFORMER_SCHEDULE,
        train_lightstate_compute_matched_optimizer_tuned_scale15m_60s_a,
    )

    started = time.time()
    job: dict[str, Any] | None = None
    runtime: dict[str, Any] | None = None
    job_dir = Path("/checkpoints") / JOB_ID
    try:
        raw = json.loads(job_json)
        if not isinstance(raw, dict):
            raise ValueError("job JSON must be an object")
        job = raw
        expected = {
            "schema": 1,
            "job_id": JOB_ID,
            "task": "lightstate_compute_matched_optimizer_tuned_scale15m_60s_a",
            "scientific_seed": SCIENTIFIC_SEED,
            "paired_batch_seed": 20_271_046,
            "paired_eval_seed": 20_291_046,
            "compile_probe_seed": 20_301_046,
            "training_data_seed": 20_260_924,
            "requires_lightstate_profile_job_id": "rlt-systems-lightstate-scale15m-batch768-modal-20261002-ac",
            "requires_pair_a_job_id": "rlt-lightstate-scale15m-paired-4m-modal-20261002-a",
            "requires_pair_b_job_id": "rlt-lightstate-scale15m-paired-4m-modal-20261002-b",
            "requires_crossdata_job_id": "rlt-crossdata-lightstate-4m-ab-modal-20261003-a",
            "requires_batch_calibration_job_id": "rlt-systems-lightstate-qualitysec-batch-modal-20261003-ad",
            "requires_lightstate_optimizer_calibration_job_id": "rlt-systems-lightstate-optimizer-qualitysec-modal-20261007-ae",
            "requires_transformer_optimizer_calibration_job_id": "rlt-systems-scale15m-optimizer-qualitysec-modal-20260927-x",
            "time_budget_seconds_each": TIME_BUDGET_SECONDS,
            "lightstate_train_batch_size": LIGHTSTATE_TRAIN_BATCH_SIZE,
            "transformer_train_batch_size": TRANSFORMER_TRAIN_BATCH_SIZE,
            "lightstate_learning_rate": LIGHTSTATE_LEARNING_RATE,
            "lightstate_schedule": LIGHTSTATE_SCHEDULE,
            "transformer_learning_rate": TRANSFORMER_LEARNING_RATE,
            "transformer_schedule": TRANSFORMER_SCHEDULE,
            "seq_len": 64,
            "train_tokens": 6_000_000,
            "val_tokens": 500_000,
            "expected_parameters_each": EXPECTED_PARAMETERS,
            "checkpoint_volume": CHECKPOINT_VOLUME,
            "automatic_retry": False,
        }
        if job != expected:
            raise RuntimeError(f"light-state optimizer-tuned compute-match preregistration mismatch: {job}")
        if job_dir.exists():
            raise RuntimeError(f"checkpoint namespace already exists: {job_dir}")

        runtime = {
            "torch_version": str(torch.__version__),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_runtime": None if torch.version.cuda is None else str(torch.version.cuda),
            "gpu_name": str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else None,
            "bf16_supported": bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        }
        if not runtime["cuda_available"]:
            raise RuntimeError("light-state optimizer-tuned compute-match started without CUDA")

        data_dir = Path("/tmp/rlt-lightstate-compute-matched-optimizer-tuned-scale15m-data")
        data_meta = prepare_fineweb(
            str(data_dir),
            train_tokens=job["train_tokens"],
            val_tokens=job["val_tokens"],
            seed=job["training_data_seed"],
        )
        data_sha256 = {
            "train.bin": _sha256(data_dir / "train.bin"),
            "val.bin": _sha256(data_dir / "val.bin"),
        }
        if data_sha256["train.bin"] != EXPECTED_TRAIN_SHA256:
            raise RuntimeError("train shard hash mismatch")
        if data_sha256["val.bin"] != EXPECTED_VAL_SHA256:
            raise RuntimeError("val shard hash mismatch")

        job_dir.mkdir(parents=True, exist_ok=False)
        comparison = train_lightstate_compute_matched_optimizer_tuned_scale15m_60s_a(
            data_dir=str(data_dir),
            checkpoint_dir=str(job_dir),
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
            "data_sha256": data_sha256,
            "comparison": comparison,
            "checkpoint_durability": {
                "volume": CHECKPOINT_VOLUME,
                "remote_dir": JOB_ID,
                "lightstate": comparison["lightstate"]["checkpoint"],
                "transformer": comparison["transformer"]["checkpoint"],
                "modal_volume_committed": False,
            },
        }
        (job_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        checkpoint_volume.commit()
        result["checkpoint_durability"]["modal_volume_committed"] = True
        (job_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        checkpoint_volume.commit()
    except Exception as exc:
        failure = {
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
        try:
            if not job_dir.exists():
                job_dir.mkdir(parents=True, exist_ok=False)
            (job_dir / "failure.json").write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n")
            checkpoint_volume.commit()
            failure["checkpoint_failure_record_committed"] = True
        except Exception:
            failure["checkpoint_failure_record_committed"] = False
        result = failure
    return _encode(result)


@app.local_entrypoint()
def main(job_path: str) -> None:
    result = run_pair_remote.remote(Path(job_path).read_text())
    if not isinstance(result, str):
        raise TypeError("remote light-state optimizer-tuned compute-match result must be base64 text")
    print("RLT_LIGHTSTATE_COMPUTE_MATCHED_OPTIMIZER_TUNED_SCALE15M_60S_RESULT_B64=" + result)
