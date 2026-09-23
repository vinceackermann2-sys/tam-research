from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import modal

JOB_ID = "rlt-crosscorpus-wikitext-4m-abcd-modal-20260923-a"
CHECKPOINT_VOLUME = "tam-rlt-scientific-checkpoints"
DATASET_ID = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-103-raw-v1"
DATASET_REVISION = "00aa25585682d4957f9e86edc73f59be7419af99"
DATASET_SPLIT = "test"
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
app = modal.App("tam-rlt-crosscorpus-wikitext-4m-abcd-20260923-a")


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
    timeout=45 * 60,
    retries=0,
    max_containers=1,
    volumes={"/checkpoints": checkpoint_volume},
)
def run_eval_remote(job_json: str) -> str:
    from datasets import load_dataset
    import numpy as np
    import torch
    from transformers import AutoTokenizer

    from experiments.rlt.eval_crosscorpus_wikitext_4m_abcd import (
        BATCH_SIZE,
        DATASET_CONFIG as EVAL_DATASET_CONFIG,
        DATASET_ID as EVAL_DATASET_ID,
        DATASET_REVISION as EVAL_DATASET_REVISION,
        DATASET_SPLIT as EVAL_DATASET_SPLIT,
        EVAL_BATCHES,
        EVAL_TARGET_TOKENS,
        REQUIRED_TEST_TOKENS,
        SEQ_LEN,
        TOKENIZER_ID as EVAL_TOKENIZER_ID,
        evaluate_wikitext_4m_abcd,
    )

    started = time.time()
    job: dict[str, Any] | None = None
    runtime: dict[str, Any] | None = None
    try:
        raw = json.loads(job_json)
        if not isinstance(raw, dict):
            raise ValueError("job JSON must be an object")
        job = raw
        expected = {
            "schema": 1,
            "job_id": JOB_ID,
            "task": "crosscorpus_wikitext_4m_abcd",
            "requires_checkpoint_job_ids": [
                "rlt-compiled-paired-4m-modal-20260920-a",
                "rlt-compiled-paired-4m-modal-20260922-b",
                "rlt-compiled-paired-4m-modal-20260923-c",
                "rlt-compiled-paired-4m-modal-20260923-d",
            ],
            "requires_crossdata_analysis_id": "rlt-crossdata-4m-abc-aggregate-20260923-a",
            "dataset_id": DATASET_ID,
            "dataset_config": DATASET_CONFIG,
            "dataset_revision": DATASET_REVISION,
            "dataset_split": DATASET_SPLIT,
            "tokenizer_id": TOKENIZER_ID,
            "eval_target_tokens_per_pair": EVAL_TARGET_TOKENS,
            "eval_batches": EVAL_BATCHES,
            "batch_size": BATCH_SIZE,
            "seq_len": SEQ_LEN,
            "checkpoint_volume": CHECKPOINT_VOLUME,
            "automatic_retry": False,
        }
        if job != expected:
            raise RuntimeError(f"WikiText cross-corpus job preregistration mismatch: {job}")
        if (
            EVAL_DATASET_ID != DATASET_ID
            or EVAL_DATASET_CONFIG != DATASET_CONFIG
            or EVAL_DATASET_REVISION != DATASET_REVISION
            or EVAL_DATASET_SPLIT != DATASET_SPLIT
            or EVAL_TOKENIZER_ID != TOKENIZER_ID
        ):
            raise RuntimeError("evaluator dataset constants drift")

        runtime = {
            "torch_version": str(torch.__version__),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_runtime": None if torch.version.cuda is None else str(torch.version.cuda),
            "gpu_name": str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else None,
            "bf16_supported": bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        }
        if not runtime["cuda_available"]:
            raise RuntimeError("WikiText cross-corpus evaluation started without CUDA")

        for checkpoint_job_id in job["requires_checkpoint_job_ids"]:
            job_dir = Path("/checkpoints") / checkpoint_job_id
            for name in ("rlt_final.pt", "transformer_final.pt"):
                if not (job_dir / name).exists():
                    raise FileNotFoundError(f"missing durable checkpoint: {job_dir / name}")

        dataset = load_dataset(
            DATASET_ID,
            name=DATASET_CONFIG,
            split=DATASET_SPLIT,
            revision=DATASET_REVISION,
        )
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, use_fast=True)
        if tokenizer.vocab_size >= 2**16:
            raise RuntimeError("uint16 token storage requires vocab < 65536")

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
            if total >= REQUIRED_TEST_TOKENS:
                break
        if total < REQUIRED_TEST_TOKENS:
            raise RuntimeError(
                f"pinned WikiText test split produced {total} tokens; need {REQUIRED_TEST_TOKENS}"
            )

        data_dir = Path("/tmp/rlt-crosscorpus-wikitext-4m-abcd")
        data_dir.mkdir(parents=True, exist_ok=True)
        test_path = data_dir / "test.bin"
        tokens = np.concatenate(chunks)[:REQUIRED_TEST_TOKENS]
        tokens.tofile(test_path)
        data_meta = {
            "dataset": DATASET_ID,
            "dataset_config": DATASET_CONFIG,
            "dataset_revision": DATASET_REVISION,
            "split": DATASET_SPLIT,
            "tokenizer": TOKENIZER_ID,
            "source_rows_consumed": source_rows,
            "raw_tokens_before_truncation": total,
            "tokens_written": int(tokens.size),
            "dtype": "uint16",
            "test_bin_sha256": _sha256(test_path),
        }

        evaluation = evaluate_wikitext_4m_abcd(
            checkpoint_root="/checkpoints",
            test_path=str(test_path),
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
            "cross_corpus_data": data_meta,
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
    raw = Path(job_path).read_text()
    result = run_eval_remote.remote(raw)
    if not isinstance(result, str):
        raise TypeError("remote WikiText cross-corpus result must be base64 text")
    print("RLT_CROSSCORPUS_WIKITEXT_4M_ABCD_RESULT_B64=" + result)
