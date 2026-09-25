from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import modal

JOB_ID = "rlt-compiled-paired-4m-wikitext-modal-20260925-b"
SCIENTIFIC_SEED = 20_261_013
CHECKPOINT_VOLUME = "tam-rlt-scientific-checkpoints"
DATASET_ID = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-103-raw-v1"
DATASET_REVISION = "00aa25585682d4957f9e86edc73f59be7419af99"
TRAIN_SPLIT = "train"
VAL_SPLIT = "validation"
TOKENIZER_ID = "gpt2"

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
app = modal.App("tam-rlt-compiled-paired-4m-wikitext-20260925-b")


def _encode(value: dict[str, Any]) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).decode()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_split(*, split: str, target_tokens: int, out_path: Path) -> dict[str, Any]:
    from datasets import load_dataset
    import numpy as np
    from transformers import AutoTokenizer

    dataset = load_dataset(
        DATASET_ID,
        name=DATASET_CONFIG,
        split=split,
        revision=DATASET_REVISION,
    )
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, use_fast=True)
    if tokenizer.vocab_size >= 2**16:
        raise RuntimeError("uint16 storage requires vocab < 65536")

    chunks: list[np.ndarray] = []
    total = 0
    source_rows = 0
    for row in dataset:
        text = row.get("text") or ""
        if not text:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        ids.append(tokenizer.eos_token_id)
        arr = np.asarray(ids, dtype=np.uint16)
        chunks.append(arr)
        total += len(arr)
        source_rows += 1
        if total >= target_tokens:
            break
    if total < target_tokens:
        raise RuntimeError(
            f"pinned WikiText {split} split produced {total} tokens; need {target_tokens}"
        )

    tokens = np.concatenate(chunks)[:target_tokens]
    tokens.tofile(out_path)
    return {
        "split": split,
        "target_tokens": target_tokens,
        "tokens_written": int(tokens.size),
        "raw_tokens_before_truncation": total,
        "source_rows_consumed": source_rows,
        "sha256": _sha256_file(out_path),
    }


@app.function(
    image=image,
    gpu="A100",
    cpu=4.0,
    memory=65536,
    timeout=70 * 60,
    retries=0,
    max_containers=1,
    volumes={"/checkpoints": checkpoint_volume},
)
def run_pair_remote(job_json: str) -> str:
    from experiments.rlt.train_compiled_pair_4m_wikitext_b import SCIENTIFIC_SEED as TRAINER_SEED
    from experiments.rlt.train_compiled_pair_4m_wikitext_b import train_compiled_pair_4m_wikitext_b
    import torch

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
            "task": "compiled_paired_4m_wikitext_b",
            "scientific_seed": SCIENTIFIC_SEED,
            "requires_prior_wikitext_job_id": "rlt-compiled-paired-4m-wikitext-modal-20260923-a",
            "requires_crosscorpus_job_id": "rlt-crosscorpus-wikitext-4m-abcd-modal-20260923-a",
            "requires_systems_job_id": "rlt-systems-decoder-cache768-modal-20260917-l",
            "dataset_id": DATASET_ID,
            "dataset_config": DATASET_CONFIG,
            "dataset_revision": DATASET_REVISION,
            "train_split": TRAIN_SPLIT,
            "val_split": VAL_SPLIT,
            "tokenizer_id": TOKENIZER_ID,
            "token_budget": 4194304,
            "seq_len": 64,
            "micro_batch_size": 16,
            "grad_accum_steps": 1,
            "train_tokens": 6000000,
            "val_tokens": 131072,
            "checkpoint_volume": CHECKPOINT_VOLUME,
            "automatic_retry": False,
        }
        if job != expected:
            raise RuntimeError(f"paired 4M WikiText job preregistration mismatch: {job}")
        if TRAINER_SEED != SCIENTIFIC_SEED:
            raise RuntimeError("trainer scientific seed drift")
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
            raise RuntimeError("Modal paired 4M WikiText job started without CUDA")

        data_dir = Path("/tmp/rlt-compiled-pair-4m-wikitext-data-b")
        data_dir.mkdir(parents=True, exist_ok=True)
        train_meta = _write_split(
            split=TRAIN_SPLIT,
            target_tokens=job["train_tokens"],
            out_path=data_dir / "train.bin",
        )
        val_meta = _write_split(
            split=VAL_SPLIT,
            target_tokens=job["val_tokens"],
            out_path=data_dir / "val.bin",
        )
        data_meta = {
            "dataset": DATASET_ID,
            "dataset_config": DATASET_CONFIG,
            "dataset_revision": DATASET_REVISION,
            "tokenizer": TOKENIZER_ID,
            "dtype": "uint16",
            "train": train_meta,
            "validation": val_meta,
        }

        job_dir.mkdir(parents=True, exist_ok=False)
        pair = train_compiled_pair_4m_wikitext_b(
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
    raw = Path(job_path).read_text()
    result = run_pair_remote.remote(raw)
    if not isinstance(result, str):
        raise TypeError("remote paired 4M WikiText result must be base64 text")
    print("RLT_COMPILED_PAIR_4M_WIKITEXT_B_RESULT_B64=" + result)
