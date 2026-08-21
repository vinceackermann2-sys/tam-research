from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

DATASET_ID = "HuggingFaceFW/fineweb-edu"
DATASET_CONFIG = "sample-10BT"
TOKENIZER_ID = "gpt2"


def prepare_fineweb(out_dir: str, train_tokens: int = 110_000_000, val_tokens: int = 2_000_000, seed: int = 1234) -> dict[str, object]:
    """Stream FineWeb-Edu once and persist identical uint16 token shards for every model."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_path, val_path, meta_path = out / "train.bin", out / "val.bin", out / "meta.json"
    if train_path.exists() and val_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("train_tokens", 0) >= train_tokens and meta.get("val_tokens", 0) >= val_tokens:
            return meta

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, use_fast=True)
    if tokenizer.vocab_size >= 2**16:
        raise ValueError("uint16 storage requires vocab < 65536")
    stream = load_dataset(DATASET_ID, name=DATASET_CONFIG, split="train", streaming=True)
    stream = stream.shuffle(seed=seed, buffer_size=10_000)

    target = train_tokens + val_tokens
    chunks: list[np.ndarray] = []
    total = 0
    for row in stream:
        text = row.get("text") or ""
        if not text:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        ids.append(tokenizer.eos_token_id)
        arr = np.asarray(ids, dtype=np.uint16)
        chunks.append(arr)
        total += len(arr)
        if total >= target:
            break
    if total < target:
        raise RuntimeError(f"dataset stream ended at {total:,} tokens; wanted {target:,}")

    tokens = np.concatenate(chunks)[:target]
    train = tokens[:train_tokens]
    val = tokens[train_tokens:train_tokens + val_tokens]
    train.tofile(train_path)
    val.tofile(val_path)
    meta = {
        "dataset": DATASET_ID,
        "dataset_config": DATASET_CONFIG,
        "tokenizer": TOKENIZER_ID,
        "seed": seed,
        "train_tokens": int(train.size),
        "val_tokens": int(val.size),
        "dtype": "uint16",
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta


class TokenBin:
    def __init__(self, path: str):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")

    def batch(self, batch_size: int, seq_len: int, generator: torch.Generator, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        hi = len(self.data) - seq_len - 1
        starts = torch.randint(0, hi, (batch_size,), generator=generator).tolist()
        chunks = [np.asarray(self.data[s:s + seq_len + 1], dtype=np.int64) for s in starts]
        x_np = np.stack([c[:-1] for c in chunks])
        y_np = np.stack([c[1:] for c in chunks])
        return (
            torch.from_numpy(x_np).to(device=device, non_blocking=True),
            torch.from_numpy(y_np).to(device=device, non_blocking=True),
        )
