from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import modal

JOB_ID = "rlt-c4-balanced-4m-modal-20260925-a"
CHECKPOINT_VOLUME = "tam-rlt-portable-c4-balanced-20260925-a"
C4_DATASET_ID = "allenai/c4"
C4_CONFIG = "en"
C4_REVISION = "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"
C4_SPLIT = "validation"
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
checkpoint_volume = modal.Volume.from_name(CHECKPOINT_VOLUME, create_if_missing=False)
app = modal.App("tam-rlt-c4-balanced-4m-20260925-a")


def _encode(value: dict[str, Any]) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).decode()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _prepare_c4_validation(out_path: Path, required_tokens: int) -> dict[str, Any]:
    from datasets import load_dataset
    import numpy as np
    from transformers import AutoTokenizer

    dataset = load_dataset(
        C4_DATASET_ID,
        name=C4_CONFIG,
        split=C4_SPLIT,
        revision=C4_REVISION,
        streaming=True,
    )
    tok = AutoTokenizer.from_pretrained(TOKENIZER_ID, use_fast=True)
    if tok.vocab_size >= 2**16:
        raise RuntimeError("uint16 token storage requires vocab < 65536")

    chunks: list[np.ndarray] = []
    total = 0
    rows = 0
    for row in dataset:
        text = row.get("text") or ""
        if not text:
            continue
        ids = tok.encode(text, add_special_tokens=False)
        ids.append(tok.eos_token_id)
        arr = np.asarray(ids, dtype=np.uint16)
        chunks.append(arr)
        total += len(arr)
        rows += 1
        if total >= required_tokens:
            break
    if total < required_tokens:
        raise RuntimeError(f"C4 validation produced {total} tokens; need {required_tokens}")

    tokens = np.concatenate(chunks)[:required_tokens]
    tokens.tofile(out_path)
    return {
        "dataset": C4_DATASET_ID,
        "dataset_config": C4_CONFIG,
        "dataset_revision": C4_REVISION,
        "split": C4_SPLIT,
        "tokenizer": TOKENIZER_ID,
        "streaming": True,
        "source_rows_consumed": rows,
        "raw_tokens_before_truncation": total,
        "tokens_written": int(tokens.size),
        "sha256": _sha256(out_path),
    }


@app.function(
    image=image,
    gpu="A100",
    cpu=4.0,
    memory=65536,
    timeout=50 * 60,
    retries=0,
    max_containers=1,
    volumes={"/checkpoints": checkpoint_volume},
)
def run_eval_remote(job_json: str) -> str:
    import torch
    from experiments.rlt.eval_c4_balanced_4m import (
        REQUIRED_VAL_TOKENS,
        evaluate_c4_balanced_4m,
    )

    started = time.time()
    job: dict[str, Any] | None = None
    runtime: dict[str, Any] | None = None
    try:
        raw = json.loads(job_json)
        if not isinstance(raw, dict):
            raise ValueError("job JSON must be object")
        job = raw
        expected = {
            "schema": 1,
            "job_id": JOB_ID,
            "task": "c4_balanced_4m",
            "checkpoint_job_ids": [
                "rlt-compiled-paired-4m-modal-20260920-a",
                "rlt-compiled-paired-4m-modal-20260922-b",
                "rlt-compiled-paired-4m-wikitext-modal-20260923-a",
                "rlt-compiled-paired-4m-wikitext-modal-20260925-b",
            ],
            "artifact_ids": [10606528515, 10707207499, 10774574552, 10853054614],
            "artifact_digests": [
                "sha256:684886d8dbeaf09bb25406d760bb4356f1ee72899ba1303a72960a90efee03fb",
                "sha256:a1747a57e8bf96537e01729b0164bea9db9551415c5ff174bdaa4ee265f3264e",
                "sha256:2e033a74f4db0888941ce569f7d822e81a62109cc275277fd7acfd4ce76df028",
                "sha256:0bc1a55c340a647fd81e0cc5d51a579f2396b0bdaf0e51e10924f70f352f10e4",
            ],
            "checkpoint_volume": CHECKPOINT_VOLUME,
            "c4_dataset_id": C4_DATASET_ID,
            "c4_config": C4_CONFIG,
            "c4_revision": C4_REVISION,
            "c4_split": C4_SPLIT,
            "tokenizer_id": TOKENIZER_ID,
            "eval_target_tokens_per_pair": 1048576,
            "automatic_retry": False,
        }
        if job != expected:
            raise RuntimeError(f"C4 balanced job preregistration mismatch: {job}")

        runtime = {
            "torch_version": str(torch.__version__),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_runtime": None if torch.version.cuda is None else str(torch.version.cuda),
            "gpu_name": str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else None,
            "bf16_supported": bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        }
        if not runtime["cuda_available"]:
            raise RuntimeError("C4 balanced evaluation started without CUDA")

        for jid in job["checkpoint_job_ids"]:
            d = Path("/checkpoints") / jid
            for name in ("rlt_final.pt", "transformer_final.pt"):
                if not (d / name).exists():
                    raise FileNotFoundError(f"missing portable checkpoint {d / name}")

        data_dir = Path("/tmp/rlt-c4-balanced")
        data_dir.mkdir(parents=True, exist_ok=True)
        c4_path = data_dir / "c4_validation.bin"
        c4_meta = _prepare_c4_validation(c4_path, REQUIRED_VAL_TOKENS)

        evaluation = evaluate_c4_balanced_4m(
            checkpoint_root="/checkpoints",
            c4_val_path=str(c4_path),
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
            "c4_data": c4_meta,
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
    result = run_eval_remote.remote(Path(job_path).read_text())
    if not isinstance(result, str):
        raise TypeError("remote C4 balanced result must be base64 text")
    print("RLT_C4_BALANCED_4M_RESULT_B64=" + result)
