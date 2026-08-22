from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import shutil
from typing import Any, Iterator

import numpy as np

TOKENIZER_ID = "gpt2"


@dataclass(frozen=True)
class PretrainSource:
    name: str
    dataset_id: str
    config: str | None
    train_tokens: int
    val_tokens: int
    text_field: str = "text"
    min_score: float | None = None
    license_note: str = ""


# Exact 2.0B-token train mixture + 5M-token held-out mixture. The mixture intentionally
# balances broad educational language with worked mathematics/reasoning, openly
# licensed code, synthetic textbook-style science, and real open-license research text.
PRETRAIN_SOURCES: tuple[PretrainSource, ...] = (
    PretrainSource(
        "fineweb_edu",
        "HuggingFaceFW/fineweb-edu",
        "sample-10BT",
        900_000_000,
        2_250_000,
        license_note="ODC-BY 1.0 dataset; individual source-page rights remain upstream",
    ),
    PretrainSource(
        "finemath_4plus",
        "HuggingFaceTB/finemath",
        "finemath-4plus",
        350_000_000,
        875_000,
        license_note="ODC-BY; high-quality educational mathematics subset",
    ),
    PretrainSource(
        "stackv2_edu_open",
        "common-pile/stackv2_edu_filtered",
        None,
        300_000_000,
        750_000,
        min_score=3.0,
        license_note="Common-Pile open-license Stack v2 education-filtered subset; score>=3",
    ),
    PretrainSource(
        "cosmopedia_openstax",
        "HuggingFaceTB/cosmopedia",
        "openstax",
        150_000_000,
        375_000,
        license_note="Apache-2.0 synthetic educational text seeded from OpenStax",
    ),
    PretrainSource(
        "cosmopedia_stanford",
        "HuggingFaceTB/cosmopedia",
        "stanford",
        150_000_000,
        375_000,
        license_note="Apache-2.0 synthetic educational text seeded from Stanford material",
    ),
    PretrainSource(
        "arxiv_open_papers",
        "common-pile/arxiv_papers",
        None,
        150_000_000,
        375_000,
        license_note="Common-Pile ArXiv papers restricted upstream to CC BY/CC BY-SA/CC0",
    ),
)

TOTAL_TRAIN_TOKENS = sum(source.train_tokens for source in PRETRAIN_SOURCES)
TOTAL_VAL_TOKENS = sum(source.val_tokens for source in PRETRAIN_SOURCES)
assert TOTAL_TRAIN_TOKENS == 2_000_000_000
assert TOTAL_VAL_TOKENS == 5_000_000


def _source_is_complete(root: Path, source: PretrainSource, seed: int) -> bool:
    train_path = root / f"{source.name}.train.bin"
    val_path = root / f"{source.name}.val.bin"
    meta_path = root / f"{source.name}.json"
    if not train_path.exists() or not val_path.exists() or not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text())
    except Exception:
        return False
    return (
        meta.get("seed") == seed
        and meta.get("source") == asdict(source)
        and train_path.stat().st_size == source.train_tokens * np.dtype(np.uint16).itemsize
        and val_path.stat().st_size == source.val_tokens * np.dtype(np.uint16).itemsize
    )


def _iter_documents(source: PretrainSource, tokenizer: Any, seed: int) -> Iterator[np.ndarray]:
    from datasets import load_dataset

    kwargs: dict[str, Any] = {
        "path": source.dataset_id,
        "split": "train",
        "streaming": True,
    }
    if source.config:
        kwargs["name"] = source.config
    stream = load_dataset(**kwargs).shuffle(seed=seed, buffer_size=20_000)

    eos = int(tokenizer.eos_token_id)
    text_batch: list[str] = []

    def flush() -> Iterator[np.ndarray]:
        nonlocal text_batch
        if not text_batch:
            return iter(())
        encoded = tokenizer(text_batch, add_special_tokens=False, truncation=False)["input_ids"]
        text_batch = []
        arrays = []
        for ids in encoded:
            if not ids:
                continue
            ids = list(ids)
            ids.append(eos)
            arrays.append(np.asarray(ids, dtype=np.uint16))
        return iter(arrays)

    for row in stream:
        if source.min_score is not None:
            score = row.get("score")
            if score is None or float(score) < source.min_score:
                continue
        text = str(row.get(source.text_field) or "").strip()
        if len(text) < 32:
            continue
        text_batch.append(text)
        if len(text_batch) >= 64:
            yield from flush()
    yield from flush()


def _write_exact(handle: Any, array: np.ndarray, remaining: int) -> int:
    take = min(int(array.size), remaining)
    if take <= 0:
        return 0
    np.asarray(array[:take], dtype=np.uint16).tofile(handle)
    return take


def _prepare_source(root: Path, source: PretrainSource, seed: int, tokenizer: Any) -> dict[str, Any]:
    if _source_is_complete(root, source, seed):
        return json.loads((root / f"{source.name}.json").read_text())

    train_final = root / f"{source.name}.train.bin"
    val_final = root / f"{source.name}.val.bin"
    train_tmp = root / f"{source.name}.train.tmp.bin"
    val_tmp = root / f"{source.name}.val.tmp.bin"
    for path in (train_tmp, val_tmp):
        if path.exists():
            path.unlink()

    val_written = 0
    train_written = 0
    documents = 0
    phase = "val"
    with val_tmp.open("wb") as val_handle, train_tmp.open("wb") as train_handle:
        for arr in _iter_documents(source, tokenizer, seed):
            documents += 1
            # Never feed the remainder of a document used for validation into train.
            # This prevents a single boundary document from leaking across the split.
            if phase == "val":
                val_written += _write_exact(val_handle, arr, source.val_tokens - val_written)
                if val_written >= source.val_tokens:
                    phase = "train"
                continue

            train_written += _write_exact(train_handle, arr, source.train_tokens - train_written)
            if train_written >= source.train_tokens:
                break

    if val_written != source.val_tokens or train_written != source.train_tokens:
        raise RuntimeError(
            f"{source.name} stream ended early: val={val_written:,}/{source.val_tokens:,}, "
            f"train={train_written:,}/{source.train_tokens:,}"
        )

    train_tmp.replace(train_final)
    val_tmp.replace(val_final)
    meta = {
        "source": asdict(source),
        "seed": seed,
        "tokenizer": TOKENIZER_ID,
        "dtype": "uint16",
        "documents_streamed": documents,
        "train_tokens": train_written,
        "val_tokens": val_written,
    }
    (root / f"{source.name}.json").write_text(json.dumps(meta, indent=2))
    return meta


def _concat(paths: list[Path], destination: Path) -> None:
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    with tmp.open("wb") as out:
        for path in paths:
            with path.open("rb") as src:
                shutil.copyfileobj(src, out, length=16 * 1024 * 1024)
    tmp.replace(destination)


def prepare_pretrain_mixture(out_dir: str, *, seed: int = 8100) -> dict[str, Any]:
    """Build the production 100M TAM 2B-token pretraining mixture.

    Every component is prepared as an exact, resumable uint16 shard before the final
    train/validation files are assembled. A failed CPU preparation run can therefore
    resume at source granularity without re-tokenizing completed sources.
    """
    from transformers import AutoTokenizer

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    final_meta = root / "meta.json"
    train_path = root / "train.bin"
    val_path = root / "val.bin"

    if final_meta.exists() and train_path.exists() and val_path.exists():
        meta = json.loads(final_meta.read_text())
        if (
            meta.get("train_tokens") == TOTAL_TRAIN_TOKENS
            and meta.get("val_tokens") == TOTAL_VAL_TOKENS
            and train_path.stat().st_size == TOTAL_TRAIN_TOKENS * 2
            and val_path.stat().st_size == TOTAL_VAL_TOKENS * 2
        ):
            return meta

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, use_fast=True)
    if tokenizer.vocab_size >= 2**16:
        raise ValueError("uint16 storage requires tokenizer vocabulary < 65536")

    source_meta = []
    for index, source in enumerate(PRETRAIN_SOURCES):
        source_meta.append(_prepare_source(root, source, seed + index * 101, tokenizer))

    _concat([root / f"{s.name}.train.bin" for s in PRETRAIN_SOURCES], train_path)
    _concat([root / f"{s.name}.val.bin" for s in PRETRAIN_SOURCES], val_path)

    if train_path.stat().st_size != TOTAL_TRAIN_TOKENS * 2:
        raise RuntimeError("assembled train.bin has unexpected size")
    if val_path.stat().st_size != TOTAL_VAL_TOKENS * 2:
        raise RuntimeError("assembled val.bin has unexpected size")

    meta = {
        "name": "TAM-100M-2B-curated-v1",
        "tokenizer": TOKENIZER_ID,
        "dtype": "uint16",
        "seed": seed,
        "train_tokens": TOTAL_TRAIN_TOKENS,
        "val_tokens": TOTAL_VAL_TOKENS,
        "sources": source_meta,
        "mixture_fractions": {
            s.name: s.train_tokens / TOTAL_TRAIN_TOKENS for s in PRETRAIN_SOURCES
        },
    }
    final_meta.write_text(json.dumps(meta, indent=2))
    return meta
