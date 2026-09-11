from __future__ import annotations

"""Zero-credit corpus provenance guard for CHM-v1 / EIEM gate #854.

This module does not launch training, allocate GPU resources, or consume a
scientific seed.  It verifies the frozen logical corpus contract and records
cryptographic fingerprints for the exact bytes that a future separately
authorized scientific runner intends to use.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

RESEARCH_ISSUE = 854
RESOURCE_FREEZE_SHA = "28c96f329070edfe32b8f506d7517315f4356aa7"

EXPECTED_META: dict[str, Any] = {
    "assembly_version": 3,
    "train_tokens": 2_000_000_000,
    "val_tokens": 5_000_000,
    "seed": 8100,
    "tokenizer": "gpt2",
    "dtype": "uint16",
}
EXPECTED_TRAIN_BYTES = 4_000_000_000
EXPECTED_VAL_BYTES = 10_000_000


@dataclass(frozen=True)
class FileFingerprint:
    name: str
    bytes: int
    sha256: str


def sha256_file(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Return SHA-256 for one file using bounded memory."""
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _require_frozen_layout(data_dir: str | Path) -> tuple[Path, Path, Path, dict[str, Any]]:
    root = Path(data_dir)
    train = root / "train.bin"
    val = root / "val.bin"
    meta_path = root / "meta.json"
    required = (train, val, meta_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"CHM-v1 corpus files missing: {missing}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    for key, expected in EXPECTED_META.items():
        actual = meta.get(key)
        if actual != expected:
            raise RuntimeError(
                f"data metadata mismatch {key}: {actual!r} != {expected!r}"
            )

    train_size = train.stat().st_size
    val_size = val.stat().st_size
    if train_size != EXPECTED_TRAIN_BYTES:
        raise RuntimeError(
            f"train.bin byte-size mismatch: {train_size} != {EXPECTED_TRAIN_BYTES}"
        )
    if val_size != EXPECTED_VAL_BYTES:
        raise RuntimeError(
            f"val.bin byte-size mismatch: {val_size} != {EXPECTED_VAL_BYTES}"
        )
    return train, val, meta_path, meta


def fingerprint_frozen_corpus(data_dir: str | Path) -> dict[str, Any]:
    """Validate the frozen corpus contract, then hash the exact corpus bytes.

    This function deliberately records fingerprints rather than silently
    treating a path or same-size file as corpus identity.  A later run
    authorization can freeze the returned hashes and require exact equality
    before any GPU allocation.
    """
    train, val, meta_path, meta = _require_frozen_layout(data_dir)
    files = (
        FileFingerprint("train.bin", train.stat().st_size, sha256_file(train)),
        FileFingerprint("val.bin", val.stat().st_size, sha256_file(val)),
        FileFingerprint("meta.json", meta_path.stat().st_size, sha256_file(meta_path)),
    )
    return {
        "classification": "ZERO_CREDIT_CORPUS_PROVENANCE_ONLY",
        "research_issue": RESEARCH_ISSUE,
        "resource_freeze_sha": RESOURCE_FREEZE_SHA,
        "logical_contract": dict(EXPECTED_META),
        "metadata": meta,
        "files": [asdict(item) for item in files],
        "gpu_authorized": False,
        "scientific_seed_consumed": False,
    }


def assert_fingerprint_matches(
    actual: dict[str, Any], expected: dict[str, Any]
) -> None:
    """Fail closed if a previously frozen fingerprint record changes."""
    if actual.get("logical_contract") != expected.get("logical_contract"):
        raise RuntimeError("corpus logical contract differs from frozen fingerprint")

    def by_name(payload: dict[str, Any]) -> dict[str, tuple[int, str]]:
        result: dict[str, tuple[int, str]] = {}
        for item in payload.get("files", []):
            result[str(item["name"])] = (int(item["bytes"]), str(item["sha256"]))
        return result

    actual_files = by_name(actual)
    expected_files = by_name(expected)
    if actual_files != expected_files:
        raise RuntimeError(
            "corpus byte fingerprint differs from frozen fingerprint: "
            f"actual={actual_files} expected={expected_files}"
        )
