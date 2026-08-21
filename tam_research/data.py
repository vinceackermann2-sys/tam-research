from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

DATASET_ID = "HuggingFaceFW/fineweb-edu"
DATASET_CONFIG = "sample-10BT"
TOKENIZER_ID = "gpt2"


def prepare_fineweb(
    out_dir: str,
    train_tokens: int = 110_000_000,
    val_tokens: int = 2_000_000,
    seed: int = 1234,
) -> dict[str, object]:
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
    val = tokens[train_tokens : train_tokens + val_tokens]
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
    """Memory-mapped token shard with an optional CUDA-resident gather cache.

    Sampling remains driven by the same CPU ``torch.Generator`` and therefore the
    same random start offsets. On CUDA, only the data movement implementation
    changes: the shard is cached once as int32 on the GPU and each batch is gathered
    there instead of Python-slicing NumPy arrays and copying every micro-batch from
    host memory. Model inputs are still returned as int64 token IDs.
    """

    def __init__(self, path: str):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self._device_cache: dict[str, torch.Tensor] = {}

    def _device_tokens(self, device: torch.device) -> torch.Tensor:
        key = str(device)
        cached = self._device_cache.get(key)
        if cached is None:
            # int32 halves persistent HBM relative to int64 while still covering the
            # GPT-2 vocabulary. We cast only the selected batch to int64 for embedding.
            host = np.asarray(self.data, dtype=np.int32)
            cached = torch.from_numpy(host).to(device=device)
            self._device_cache[key] = cached
        return cached

    def batch(
        self,
        batch_size: int,
        seq_len: int,
        generator: torch.Generator,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hi = len(self.data) - seq_len - 1
        starts_cpu = torch.randint(0, hi, (batch_size,), generator=generator)

        if device.type == "cuda":
            source = self._device_tokens(device)
            starts = starts_cpu.to(device=device, dtype=torch.long)
            offsets = torch.arange(seq_len + 1, device=device, dtype=torch.long)
            chunks = source[starts[:, None] + offsets[None, :]].long()
            return chunks[:, :-1], chunks[:, 1:]

        starts = starts_cpu.tolist()
        chunks = [
            np.asarray(self.data[s : s + seq_len + 1], dtype=np.int64)
            for s in starts
        ]
        x_np = np.stack([c[:-1] for c in chunks])
        y_np = np.stack([c[1:] for c in chunks])
        return (
            torch.from_numpy(x_np).to(device=device, non_blocking=True),
            torch.from_numpy(y_np).to(device=device, non_blocking=True),
        )
