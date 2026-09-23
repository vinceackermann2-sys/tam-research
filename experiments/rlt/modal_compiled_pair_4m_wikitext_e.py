from __future__ import annotations

import base64
import json
from pathlib import Path
import time
from typing import Any

import modal

JOB_ID = "rlt-compiled-paired-4m-wikitext-modal-20260923-e"
SCIENTIFIC_SEED = 20_261_012
CHECKPOINT_VOLUME = "tam-rlt-scientific-checkpoints"

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
app = modal.App("tam-rlt-compiled-paired-4m-wikitext-20260923-e")


def _encode(value: dict[str, Any]) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).decode()


@app.function(
    image=image,
    gpu="A100",
    cpu=4.0,
    memory=65536,
    timeout=55 * 60,
    retries=0,
    max_containers=1,
    volumes={"/checkpoints": checkpoint_volume},
)
def run_pair_remote(job_json: str) -> str:
    from experiments.rlt.train_compiled_pair_4m_wikitext_e import SCIENTIFIC_SEED as TRAINER_SEED
    from experiments.rlt.train_compiled_pair_4m_wikitext_e import train_compiled_pair_4m_wikitext_e
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
            "task": "compiled_paired_4m_wikitext_e",
            "scientific_seed": SCIENTIFIC_SEED,
                        "requires_crosscorpus_job_id": "rlt-crosscorpus-wikitext-4m-abcd-modal-20260923-a",
            "dataset_id": "Salesforce/wikitext",
            "dataset_config": "wikitext-103-raw-v1",
            "dataset_revision": "00aa25585682d4957f9e86edc73f59be7419af99",
            "tokenizer_id": "gpt2",
            "requires_systems_job_id": "rlt-systems-decoder-cache768-modal-20260917-l",
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
            raise RuntimeError(f"paired 4M WikiText E job preregistration mismatch: {job}")
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
            raise RuntimeError("Modal paired 4M WikiText E job started without CUDA")

        from datasets import load_dataset
        from transformers import AutoTokenizer
        import numpy as np
        import hashlib

        data_dir = Path("/tmp/rlt-compiled-pair-4m-wikitext-e")
        data_dir.mkdir(parents=True, exist_ok=True)
        tokenizer = AutoTokenizer.from_pretrained(job["tokenizer_id"], use_fast=True)
        if tokenizer.vocab_size >= 2**16:
            raise RuntimeError("uint16 token storage requires vocab < 65536")

        def _sha256_file(path: Path) -> str:
            h = hashlib.sha256()
            with path.open("rb") as f:
                for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                    h.update(chunk)
            return h.hexdigest()

        def _write_split(split: str, target_tokens: int, out_path: Path) -> dict[str, Any]:
            dataset = load_dataset(
                job["dataset_id"],
                name=job["dataset_config"],
                split=split,
                revision=job["dataset_revision"],
            )
            chunks: list[np.ndarray] = []
            total = 0
            rows = 0
            for row in dataset:
                text = row.get("text") or ""
                if not text:
                    continue
                ids = tokenizer.encode(text, add_special_tokens=False)
                ids.append(tokenizer.eos_token_id)
                arr = np.asarray(ids, dtype=np.uint16)
                chunks.append(arr)
                total += len(arr)
                rows += 1
                if total >= target_tokens:
                    break
            if total < target_tokens:
                raise RuntimeError(
                    f"WikiText {split} produced {total} tokens; need {target_tokens}"
                )
            tokens = np.concatenate(chunks)[:target_tokens]
            tokens.tofile(out_path)
            return {
                "split": split,
                "source_rows_consumed": rows,
                "raw_tokens_before_truncation": total,
                "tokens_written": int(tokens.size),
                "sha256": _sha256_file(out_path),
            }

        train_meta = _write_split("train", job["train_tokens"], data_dir / "train.bin")
        val_meta = _write_split("validation", job["val_tokens"], data_dir / "val.bin")
        data_meta = {
            "dataset": job["dataset_id"],
            "dataset_config": job["dataset_config"],
            "dataset_revision": job["dataset_revision"],
            "tokenizer": job["tokenizer_id"],
            "train_tokens": job["train_tokens"],
            "val_tokens": job["val_tokens"],
            "dtype": "uint16",
            "train": train_meta,
            "validation": val_meta,
        }
        data_sha256 = {
            "train.bin": train_meta["sha256"],
            "val.bin": val_meta["sha256"],
        }

        job_dir.mkdir(parents=True, exist_ok=False)
        pair = train_compiled_pair_4m_wikitext_e(
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
        raise TypeError("remote paired 4M WikiText E result must be base64 text")
    print("RLT_COMPILED_PAIR_4M_WIKITEXT_E_RESULT_B64=" + result)
