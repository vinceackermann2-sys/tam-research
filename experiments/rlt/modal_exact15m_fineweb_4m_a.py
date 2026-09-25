from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import modal

JOB_ID = "rlt-exact15m-fineweb-4m-modal-20260925-a"
SCIENTIFIC_SEED = 20_261_014
CHECKPOINT_VOLUME = "tam-rlt-exact15m-scientific-checkpoints"
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
app = modal.App("tam-rlt-exact15m-fineweb-4m-20260925-a")


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
    gpu="A100",
    cpu=4.0,
    memory=65536,
    timeout=105 * 60,
    retries=0,
    max_containers=1,
    volumes={"/checkpoints": checkpoint_volume},
)
def run_pair_remote(job_json: str) -> str:
    import torch
    from tam_research.data import prepare_fineweb
    from experiments.rlt.train_exact15m_fineweb_4m_a import (
        SCIENTIFIC_SEED as TRAINER_SEED,
        EXPECTED_PARAMETERS,
        train_exact15m_fineweb_4m_a,
    )

    started = time.time()
    job: dict[str, Any] | None = None
    runtime: dict[str, Any] | None = None
    job_dir = Path("/checkpoints") / JOB_ID
    try:
        raw = json.loads(job_json)
        if not isinstance(raw, dict):
            raise ValueError("job JSON must be object")
        job = raw
        expected = {
            "schema": 1,
            "job_id": JOB_ID,
            "task": "exact15m_fineweb_4m_a",
            "scientific_seed": SCIENTIFIC_SEED,
            "training_data_seed": 20260924,
            "requires_prior_scale_reference_job_id": "rlt-compiled-paired-4m-modal-20260923-d",
            "requires_transfer_job_id": "rlt-transfer-wikitext-ab-modal-20260925-a",
            "requires_systems_job_id": "rlt-systems-decoder-cache768-modal-20260917-l",
            "expected_parameters_per_model": 15129344,
            "token_budget": 4194304,
            "seq_len": 64,
            "micro_batch_size": 16,
            "grad_accum_steps": 1,
            "train_tokens": 6000000,
            "val_tokens": 500000,
            "checkpoint_volume": CHECKPOINT_VOLUME,
            "automatic_retry": False,
        }
        if job != expected:
            raise RuntimeError(f"exact15m preregistration mismatch: {job}")
        if TRAINER_SEED != SCIENTIFIC_SEED:
            raise RuntimeError("scale trainer seed drift")
        if EXPECTED_PARAMETERS != job["expected_parameters_per_model"]:
            raise RuntimeError("scale expected parameter count drift")
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
            raise RuntimeError("exact15m scientific run started without CUDA")

        data_dir = Path("/tmp/rlt-exact15m-fineweb-data")
        data_dir.mkdir(parents=True, exist_ok=True)
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
            raise RuntimeError("exact15m train bytes differ from 7M D reference")
        if data_sha256["val.bin"] != EXPECTED_VAL_SHA256:
            raise RuntimeError("exact15m val bytes differ from 7M D reference")

        job_dir.mkdir(parents=True, exist_ok=False)
        pair = train_exact15m_fineweb_4m_a(
            data_dir=str(data_dir),
            checkpoint_dir=str(job_dir),
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
            "data_sha256": data_sha256,
            "comparison": pair,
            "checkpoint_durability": {
                "volume": CHECKPOINT_VOLUME,
                "remote_dir": JOB_ID,
                "rlt": pair["rlt"]["checkpoint"],
                "transformer": pair["transformer"]["checkpoint"],
                "modal_volume_committed": False,
            },
        }
        (job_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        checkpoint_volume.commit()
        result["checkpoint_durability"]["modal_volume_committed"] = True
        (job_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        checkpoint_volume.commit()
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
        try:
            if not job_dir.exists():
                job_dir.mkdir(parents=True, exist_ok=False)
            (job_dir / "failure.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            checkpoint_volume.commit()
            result["checkpoint_failure_record_committed"] = True
        except Exception:
            result["checkpoint_failure_record_committed"] = False
    return _encode(result)


@app.local_entrypoint()
def main(job_path: str) -> None:
    result = run_pair_remote.remote(Path(job_path).read_text())
    if not isinstance(result, str):
        raise TypeError("exact15m result must be base64 text")
    print("RLT_EXACT15M_FINEWEB_RESULT_B64=" + result)
