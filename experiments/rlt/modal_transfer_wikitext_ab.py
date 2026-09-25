from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import modal

JOB_ID = "rlt-transfer-wikitext-ab-modal-20260925-a"
CHECKPOINT_VOLUME = "tam-rlt-portable-transfer-wikitext-ab-20260925-a"
WIKITEXT_DATASET_ID = "Salesforce/wikitext"
WIKITEXT_CONFIG = "wikitext-103-raw-v1"
WIKITEXT_REVISION = "00aa25585682d4957f9e86edc73f59be7419af99"
FINEWEB_SEED = 20_260_926

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
app = modal.App("tam-rlt-transfer-wikitext-ab-20260925-a")


def _encode(value: dict[str, Any]) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).decode()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _prepare_wikitext_test(out_path: Path) -> dict[str, Any]:
    from datasets import load_dataset
    import numpy as np
    from transformers import AutoTokenizer
    from experiments.rlt.eval_crosscorpus_wikitext_4m_abcd import REQUIRED_TEST_TOKENS

    dataset = load_dataset(
        WIKITEXT_DATASET_ID,
        name=WIKITEXT_CONFIG,
        split="test",
        revision=WIKITEXT_REVISION,
    )
    tok = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
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
        if total >= REQUIRED_TEST_TOKENS:
            break
    if total < REQUIRED_TEST_TOKENS:
        raise RuntimeError(f"WikiText test too short: {total} < {REQUIRED_TEST_TOKENS}")
    tokens = np.concatenate(chunks)[:REQUIRED_TEST_TOKENS]
    tokens.tofile(out_path)
    return {
        "dataset": WIKITEXT_DATASET_ID,
        "dataset_config": WIKITEXT_CONFIG,
        "dataset_revision": WIKITEXT_REVISION,
        "split": "test",
        "tokenizer": "gpt2",
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
    from tam_research.data import prepare_fineweb
    from experiments.rlt.eval_transfer_wikitext_ab import evaluate_transfer_wikitext_ab

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
            "task": "transfer_wikitext_ab",
            "requires_wikitext_job_ids": [
                "rlt-compiled-paired-4m-wikitext-modal-20260923-a",
                "rlt-compiled-paired-4m-wikitext-modal-20260925-b",
            ],
            "artifact_ids": [10774574552, 10853054614],
            "artifact_digests": [
                "sha256:2e033a74f4db0888941ce569f7d822e81a62109cc275277fd7acfd4ce76df028",
                "sha256:0bc1a55c340a647fd81e0cc5d51a579f2396b0bdaf0e51e10924f70f352f10e4",
            ],
            "checkpoint_volume": CHECKPOINT_VOLUME,
            "wikitext_dataset_revision": WIKITEXT_REVISION,
            "fineweb_seed": FINEWEB_SEED,
            "fineweb_train_tokens": 100000,
            "fineweb_val_tokens": 1100000,
            "automatic_retry": False,
        }
        if job != expected:
            raise RuntimeError(f"transfer job preregistration mismatch: {job}")

        runtime = {
            "torch_version": str(torch.__version__),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_runtime": None if torch.version.cuda is None else str(torch.version.cuda),
            "gpu_name": str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else None,
            "bf16_supported": bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        }
        if not runtime["cuda_available"]:
            raise RuntimeError("transfer evaluation started without CUDA")

        for jid in job["requires_wikitext_job_ids"]:
            d = Path("/checkpoints") / jid
            for name in ("rlt_final.pt", "transformer_final.pt"):
                if not (d / name).exists():
                    raise FileNotFoundError(f"missing portable checkpoint {d / name}")

        data_dir = Path("/tmp/rlt-transfer-wikitext-ab")
        data_dir.mkdir(parents=True, exist_ok=True)
        wiki_path = data_dir / "wikitext_test.bin"
        wiki_meta = _prepare_wikitext_test(wiki_path)

        fine_dir = data_dir / "fineweb"
        fine_meta = prepare_fineweb(
            str(fine_dir),
            train_tokens=job["fineweb_train_tokens"],
            val_tokens=job["fineweb_val_tokens"],
            seed=job["fineweb_seed"],
        )
        fine_path = fine_dir / "val.bin"
        data_meta = {
            "wikitext_test": wiki_meta,
            "fineweb": fine_meta,
            "fineweb_val_sha256": _sha256(fine_path),
        }

        evaluation = evaluate_transfer_wikitext_ab(
            checkpoint_root="/checkpoints",
            wikitext_test_path=str(wiki_path),
            fineweb_val_path=str(fine_path),
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
        raise TypeError("remote transfer result must be base64 text")
    print("RLT_TRANSFER_WIKITEXT_AB_RESULT_B64=" + result)
