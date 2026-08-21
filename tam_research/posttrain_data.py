from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

DATASET_ID = "HuggingFaceH4/ultrafeedback_binarized"
TOKENIZER_ID = "gpt2"


def encode_supervised_pair(
    tokenizer: Any,
    prompt: str,
    response: str,
    *,
    seq_len: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode one prompt/assistant response with assistant-only causal labels.

    No tokenizer vocabulary changes are made. Human-readable ``User:`` / ``Assistant:``
    markers are ordinary GPT-2 tokens, so a pretrained checkpoint can be post-trained
    without resizing embeddings.
    """
    if seq_len < 8:
        raise ValueError("seq_len must be at least 8")
    eos = int(tokenizer.eos_token_id)
    prefix = f"User:\n{prompt.strip()}\nAssistant:\n"
    prefix_ids = list(tokenizer.encode(prefix, add_special_tokens=False))
    response_ids = list(tokenizer.encode(response.strip(), add_special_tokens=False))
    response_ids.append(eos)

    # Keep at most half the context for a very long prompt so an assistant target is
    # always present. This is deterministic and identical for chosen/rejected pairs.
    max_total = seq_len + 1
    max_prefix = min(len(prefix_ids), max_total // 2, max_total - 2)
    prefix_ids = prefix_ids[:max_prefix]
    room = max_total - len(prefix_ids)
    response_ids = response_ids[:room]
    if not response_ids:
        response_ids = [eos]

    actual = prefix_ids + response_ids
    actual_len = len(actual)
    padded = actual + [eos] * (max_total - actual_len)
    x = np.asarray(padded[:-1], dtype=np.uint16)
    shifted = np.asarray(padded[1:], dtype=np.int32)
    labels = np.full(seq_len, -100, dtype=np.int32)

    # x[prefix_len-1] predicts the first assistant token.
    assistant_target_start = max(len(prefix_ids) - 1, 0)
    valid_target_end = max(actual_len - 1, assistant_target_start)
    labels[assistant_target_start:valid_target_end] = shifted[
        assistant_target_start:valid_target_end
    ]
    return x, labels


def _assistant_text(messages: Iterable[dict[str, Any]]) -> str:
    for message in reversed(list(messages)):
        if message.get("role") == "assistant" and message.get("content"):
            return str(message["content"])
    return ""


def _save_supervised(
    rows: Iterable[dict[str, Any]],
    tokenizer: Any,
    out: Path,
    prefix: str,
    count: int,
    seq_len: int,
) -> int:
    inputs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for row in rows:
        prompt = str(row.get("prompt") or "")
        response = _assistant_text(row.get("messages") or row.get("chosen") or [])
        if not prompt or not response:
            continue
        x, y = encode_supervised_pair(tokenizer, prompt, response, seq_len=seq_len)
        if int((y != -100).sum()) < 2:
            continue
        inputs.append(x)
        labels.append(y)
        if len(inputs) >= count:
            break
    if len(inputs) < count:
        raise RuntimeError(f"only collected {len(inputs)} supervised rows; wanted {count}")
    np.save(out / f"{prefix}_inputs.npy", np.stack(inputs))
    np.save(out / f"{prefix}_labels.npy", np.stack(labels))
    return len(inputs)


def _save_preferences(
    rows: Iterable[dict[str, Any]],
    tokenizer: Any,
    out: Path,
    prefix: str,
    count: int,
    seq_len: int,
) -> int:
    chosen_inputs: list[np.ndarray] = []
    chosen_labels: list[np.ndarray] = []
    rejected_inputs: list[np.ndarray] = []
    rejected_labels: list[np.ndarray] = []
    for row in rows:
        prompt = str(row.get("prompt") or "")
        chosen = _assistant_text(row.get("chosen") or [])
        rejected = _assistant_text(row.get("rejected") or [])
        if not prompt or not chosen or not rejected or chosen == rejected:
            continue
        ci, cl = encode_supervised_pair(tokenizer, prompt, chosen, seq_len=seq_len)
        ri, rl = encode_supervised_pair(tokenizer, prompt, rejected, seq_len=seq_len)
        if int((cl != -100).sum()) < 2 or int((rl != -100).sum()) < 2:
            continue
        chosen_inputs.append(ci)
        chosen_labels.append(cl)
        rejected_inputs.append(ri)
        rejected_labels.append(rl)
        if len(chosen_inputs) >= count:
            break
    if len(chosen_inputs) < count:
        raise RuntimeError(f"only collected {len(chosen_inputs)} preference rows; wanted {count}")
    np.save(out / f"{prefix}_chosen_inputs.npy", np.stack(chosen_inputs))
    np.save(out / f"{prefix}_chosen_labels.npy", np.stack(chosen_labels))
    np.save(out / f"{prefix}_rejected_inputs.npy", np.stack(rejected_inputs))
    np.save(out / f"{prefix}_rejected_labels.npy", np.stack(rejected_labels))
    return len(chosen_inputs)


def prepare_posttrain_data(
    out_dir: str,
    *,
    seq_len: int = 512,
    sft_train_rows: int = 20_000,
    sft_eval_rows: int = 1_000,
    preference_train_rows: int = 5_000,
    preference_eval_rows: int = 500,
    seed: int = 7400,
) -> dict[str, Any]:
    """Prepare fixed-shape GPT-2-tokenized SFT and preference arrays.

    UltraFeedback Binarized supplies both the chosen assistant responses used for
    SFT and chosen/rejected pairs used for DPO. The dataset is streamed to avoid a
    large local download, then persisted to the shared Modal Volume so the H100 job
    performs no network-bound preprocessing.
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "meta.json"
    requested = {
        "seq_len": seq_len,
        "sft_train_rows": sft_train_rows,
        "sft_eval_rows": sft_eval_rows,
        "preference_train_rows": preference_train_rows,
        "preference_eval_rows": preference_eval_rows,
        "seed": seed,
    }
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if all(meta.get(k) == v for k, v in requested.items()):
            expected = [
                "sft_train_inputs.npy",
                "sft_train_labels.npy",
                "sft_eval_inputs.npy",
                "sft_eval_labels.npy",
                "pref_train_chosen_inputs.npy",
                "pref_train_chosen_labels.npy",
                "pref_train_rejected_inputs.npy",
                "pref_train_rejected_labels.npy",
                "pref_eval_chosen_inputs.npy",
                "pref_eval_chosen_labels.npy",
                "pref_eval_rejected_inputs.npy",
                "pref_eval_rejected_labels.npy",
            ]
            if all((out / name).exists() for name in expected):
                return meta

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, use_fast=True)
    train_sft = load_dataset(DATASET_ID, split="train_sft", streaming=True).shuffle(
        seed=seed, buffer_size=10_000
    )
    test_sft = load_dataset(DATASET_ID, split="test_sft", streaming=True)
    train_prefs = load_dataset(DATASET_ID, split="train_prefs", streaming=True).shuffle(
        seed=seed + 1, buffer_size=10_000
    )
    test_prefs = load_dataset(DATASET_ID, split="test_prefs", streaming=True)

    _save_supervised(train_sft, tokenizer, out, "sft_train", sft_train_rows, seq_len)
    _save_supervised(test_sft, tokenizer, out, "sft_eval", sft_eval_rows, seq_len)
    _save_preferences(
        train_prefs, tokenizer, out, "pref_train", preference_train_rows, seq_len
    )
    _save_preferences(
        test_prefs, tokenizer, out, "pref_eval", preference_eval_rows, seq_len
    )

    meta = {
        "dataset": DATASET_ID,
        "tokenizer": TOKENIZER_ID,
        **requested,
        "format": "User:\\n{prompt}\\nAssistant:\\n{response}<eos>",
        "label_policy": "assistant tokens only",
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta
