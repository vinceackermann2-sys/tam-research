from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .posttrain_data import encode_supervised_pair

SFT_DATASET_ID = "HuggingFaceTB/smol-smoltalk"
PREFERENCE_DATASET_ID = "HuggingFaceH4/ultrafeedback_binarized"
TOKENIZER_ID = "gpt2"


def _last_user_assistant(messages: Iterable[dict[str, Any]]) -> tuple[str, str]:
    items = list(messages or [])
    assistant_index = -1
    response = ""
    for index in range(len(items) - 1, -1, -1):
        message = items[index]
        if message.get("role") == "assistant" and message.get("content"):
            assistant_index = index
            response = str(message["content"])
            break
    if assistant_index < 0:
        return "", ""
    prompt = ""
    for index in range(assistant_index - 1, -1, -1):
        message = items[index]
        if message.get("role") == "user" and message.get("content"):
            prompt = str(message["content"])
            break
    return prompt, response


def _assistant_text(messages: Iterable[dict[str, Any]]) -> str:
    for message in reversed(list(messages or [])):
        if message.get("role") == "assistant" and message.get("content"):
            return str(message["content"])
    return ""


def _save_smol_sft(
    rows: Iterable[dict[str, Any]],
    tokenizer: Any,
    out: Path,
    *,
    train_count: int,
    eval_count: int,
    seq_len: int,
) -> tuple[int, int]:
    train_inputs: list[np.ndarray] = []
    train_labels: list[np.ndarray] = []
    eval_inputs: list[np.ndarray] = []
    eval_labels: list[np.ndarray] = []

    for row in rows:
        prompt, response = _last_user_assistant(row.get("messages") or [])
        if not prompt or not response:
            continue
        x, y = encode_supervised_pair(tokenizer, prompt, response, seq_len=seq_len)
        if int((y != -100).sum()) < 2:
            continue
        if len(eval_inputs) < eval_count:
            eval_inputs.append(x)
            eval_labels.append(y)
        elif len(train_inputs) < train_count:
            train_inputs.append(x)
            train_labels.append(y)
        if len(eval_inputs) >= eval_count and len(train_inputs) >= train_count:
            break

    if len(train_inputs) < train_count or len(eval_inputs) < eval_count:
        raise RuntimeError(
            f"Smol-SmolTalk ended early: train={len(train_inputs)}/{train_count}, "
            f"eval={len(eval_inputs)}/{eval_count}"
        )
    np.save(out / "sft_train_inputs.npy", np.stack(train_inputs))
    np.save(out / "sft_train_labels.npy", np.stack(train_labels))
    np.save(out / "sft_eval_inputs.npy", np.stack(eval_inputs))
    np.save(out / "sft_eval_labels.npy", np.stack(eval_labels))
    return len(train_inputs), len(eval_inputs)


def _save_preferences(
    rows: Iterable[dict[str, Any]],
    tokenizer: Any,
    out: Path,
    prefix: str,
    count: int,
    seq_len: int,
) -> int:
    ci: list[np.ndarray] = []
    cl: list[np.ndarray] = []
    ri: list[np.ndarray] = []
    rl: list[np.ndarray] = []
    for row in rows:
        prompt = str(row.get("prompt") or "")
        chosen = _assistant_text(row.get("chosen") or [])
        rejected = _assistant_text(row.get("rejected") or [])
        if not prompt or not chosen or not rejected or chosen == rejected:
            continue
        chosen_x, chosen_y = encode_supervised_pair(
            tokenizer, prompt, chosen, seq_len=seq_len
        )
        rejected_x, rejected_y = encode_supervised_pair(
            tokenizer, prompt, rejected, seq_len=seq_len
        )
        if int((chosen_y != -100).sum()) < 2 or int((rejected_y != -100).sum()) < 2:
            continue
        ci.append(chosen_x)
        cl.append(chosen_y)
        ri.append(rejected_x)
        rl.append(rejected_y)
        if len(ci) >= count:
            break
    if len(ci) < count:
        raise RuntimeError(f"UltraFeedback ended early: {len(ci)}/{count}")
    np.save(out / f"{prefix}_chosen_inputs.npy", np.stack(ci))
    np.save(out / f"{prefix}_chosen_labels.npy", np.stack(cl))
    np.save(out / f"{prefix}_rejected_inputs.npy", np.stack(ri))
    np.save(out / f"{prefix}_rejected_labels.npy", np.stack(rl))
    return len(ci)


def prepare_100m_posttrain_data(
    out_dir: str,
    *,
    seq_len: int = 512,
    sft_train_rows: int = 120_000,
    sft_eval_rows: int = 2_000,
    preference_train_rows: int = 20_000,
    preference_eval_rows: int = 1_000,
    seed: int = 8100,
) -> dict[str, Any]:
    """Prepare post-training data sized for a ~100M model.

    SFT uses Smol-SmolTalk because Hugging Face explicitly curated that subset for
    135M/360M models. DPO uses the current UltraFeedback Binarized preference split.
    Both are converted to the existing GPT-2-vocabulary assistant-only label format.
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
    if meta_path.exists() and all((out / name).exists() for name in expected):
        meta = json.loads(meta_path.read_text())
        if all(meta.get(k) == v for k, v in requested.items()):
            return meta

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, use_fast=True)
    sft_stream = load_dataset(SFT_DATASET_ID, split="train", streaming=True).shuffle(
        seed=seed, buffer_size=20_000
    )
    _save_smol_sft(
        sft_stream,
        tokenizer,
        out,
        train_count=sft_train_rows,
        eval_count=sft_eval_rows,
        seq_len=seq_len,
    )

    pref_train = load_dataset(
        PREFERENCE_DATASET_ID, split="train_prefs", streaming=True
    ).shuffle(seed=seed + 1, buffer_size=20_000)
    pref_eval = load_dataset(
        PREFERENCE_DATASET_ID, split="test_prefs", streaming=True
    )
    _save_preferences(
        pref_train,
        tokenizer,
        out,
        "pref_train",
        preference_train_rows,
        seq_len,
    )
    _save_preferences(
        pref_eval,
        tokenizer,
        out,
        "pref_eval",
        preference_eval_rows,
        seq_len,
    )

    meta = {
        "sft_dataset": SFT_DATASET_ID,
        "preference_dataset": PREFERENCE_DATASET_ID,
        "tokenizer": TOKENIZER_ID,
        **requested,
        "label_policy": "assistant tokens only",
        "sft_policy": "last user/assistant turn from Smol-SmolTalk",
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta
