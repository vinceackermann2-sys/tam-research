from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import modal

JOB_ID = "rlt-block-quality-sweep-2m-modal-20260930-ac"
SCIENTIFIC_SEED = 20_261_039
CHECKPOINT_VOLUME = "tam-rlt-block-quality-sweep-ac-account2"
EXPECTED_TRAIN_SHA256 = "0f9ea99680f532f990ca64fabbd02b46bba2c4f8077f868545a03f5e6907c907"
EXPECTED_VAL_SHA256 = "e6b3fb209a80a2cebe22924c3c8af26b30fc4afd322a1a49c71af8b968057db2"

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
checkpoint_volume = modal.Volume.from_name(CHECKPOINT_VOLUME, create_if_missing=True)
app = modal.App("tam-rlt-block-quality-sweep-2m-20260930-ac")


def _encode(value: dict[str, Any]) -> str:
    return base64.urlsafe_b64encode(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).decode()


def _sha256_file(path: Path) -> str:
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
    timeout=120 * 60,
    retries=0,
    max_containers=1,
    volumes={"/checkpoints": checkpoint_volume},
)
def run_sweep_remote(job_json: str) -> str:
    import torch

    from experiments.rlt.train_block_quality_sweep_2m_ac import (
        SCIENTIFIC_SEED as TRAINER_SEED,
        train_block_quality_sweep_2m_ac,
    )
    from tam_research.data import prepare_fineweb

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
            "task": "block_quality_sweep_2m_ac",
            "scientific_seed": SCIENTIFIC_SEED,
            "training_data_seed": 20_260_924,
            "requires_block_sweep_job_id": "rlt-systems-block-sweep-scale15m-batch768-modal-20260929-ab",
            "requires_block8_quality_job_id": "rlt-block8-scale15m-paired-4m-modal-20260929-a",
            "requires_block4_quality_job_id": "rlt-block4-scale15m-paired-4m-modal-20260929-a",
            "requires_block2_quality_job_id": "rlt-block2-scale15m-paired-4m-modal-20260930-a",
            "token_budget": 2_097_152,
            "seq_len": 64,
            "micro_batch_size": 16,
            "grad_accum_steps": 1,
            "train_tokens": 6_000_000,
            "val_tokens": 500_000,
            "expected_parameters_each": 15_129_344,
            "checkpoint_volume": CHECKPOINT_VOLUME,
            "automatic_retry": False,
        }
        if job != expected:
            raise RuntimeError(f"block quality sweep preregistration mismatch: {job}")
        if TRAINER_SEED != SCIENTIFIC_SEED:
            raise RuntimeError("sweep trainer scientific seed drift")
        if job_dir.exists():
            raise RuntimeError(f"checkpoint namespace already exists: {job_dir}")
        if not torch.cuda.is_available():
            raise RuntimeError("block quality sweep started without CUDA")

        runtime = {
            "torch_version": str(torch.__version__),
            "cuda_available": True,
            "cuda_runtime": None if torch.version.cuda is None else str(torch.version.cuda),
            "gpu_name": str(torch.cuda.get_device_name(0)),
            "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        }

        data_dir = Path("/tmp/rlt-block-quality-sweep-ac-data")
        data_dir.mkdir(parents=True, exist_ok=True)
        data_meta = prepare_fineweb(
            str(data_dir),
            train_tokens=job["train_tokens"],
            val_tokens=job["val_tokens"],
            seed=job["training_data_seed"],
        )
        data_sha256 = {
            "train.bin": _sha256_file(data_dir / "train.bin"),
            "val.bin": _sha256_file(data_dir / "val.bin"),
        }
        if data_sha256["train.bin"] != EXPECTED_TRAIN_SHA256:
            raise RuntimeError(f"train hash mismatch: {data_sha256['train.bin']}")
        if data_sha256["val.bin"] != EXPECTED_VAL_SHA256:
            raise RuntimeError(f"val hash mismatch: {data_sha256['val.bin']}")

        job_dir.mkdir(parents=True, exist_ok=False)
        sweep = train_block_quality_sweep_2m_ac(
            data_dir=str(data_dir),
            checkpoint_dir=str(job_dir),
            token_budget=job["token_budget"],
            seq_len=job["seq_len"],
            micro_batch_size=job["micro_batch_size"],
            grad_accum_steps=job["grad_accum_steps"],
        )
        if sweep.get("classification") != "BLOCK_QUALITY_SWEEP_2M_COMPLETE":
            raise RuntimeError(f"unexpected sweep classification: {sweep.get('classification')}")
        if sweep.get("transformer_control_consistency", {}).get("passed") is not True:
            raise RuntimeError("repeated Transformer control consistency failed")

        checkpoint_records: dict[str, Any] = {}
        for block_size in (1, 2, 4, 8):
            pair = sweep["pairs"][str(block_size)]
            checkpoint_records[str(block_size)] = {
                "rlt": pair["rlt"]["checkpoint"],
                "transformer": pair["transformer"]["checkpoint"],
                "remote_subdir": f"block{block_size}",
            }

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
            "comparison": sweep,
            "checkpoint_durability": {
                "volume": CHECKPOINT_VOLUME,
                "remote_dir": JOB_ID,
                "blocks": checkpoint_records,
                "modal_volume_committed": False,
            },
        }
        (job_dir / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
        checkpoint_volume.commit()
        result["checkpoint_durability"]["modal_volume_committed"] = True
        (job_dir / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
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
            (job_dir / "failure.json").write_text(
                json.dumps(failure, indent=2, sort_keys=True) + "\n"
            )
            checkpoint_volume.commit()
            failure["checkpoint_failure_record_committed"] = True
        except Exception:
            failure["checkpoint_failure_record_committed"] = False
        result = failure

    return _encode(result)


@app.local_entrypoint()
def main(job_path: str) -> None:
    result = run_sweep_remote.remote(Path(job_path).read_text())
    if not isinstance(result, str):
        raise TypeError("remote block quality sweep result must be base64 text")
    print("RLT_BLOCK_QUALITY_SWEEP_2M_AC_RESULT_B64=" + result)
