from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tam_research.chm_v1_corpus_fingerprint import (
    EXPECTED_META,
    EXPECTED_TRAIN_BYTES,
    EXPECTED_VAL_BYTES,
    _require_frozen_layout,
    assert_fingerprint_matches,
    sha256_file,
)


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    payload = (b"chm-v1-corpus-fingerprint\n" * 37) + bytes(range(64))
    path = tmp_path / "sample.bin"
    path.write_bytes(payload)
    assert sha256_file(path, chunk_bytes=17) == hashlib.sha256(payload).hexdigest()


def test_frozen_layout_rejects_wrong_byte_sizes(tmp_path: Path) -> None:
    (tmp_path / "train.bin").write_bytes(b"tiny")
    (tmp_path / "val.bin").write_bytes(b"tiny")
    (tmp_path / "meta.json").write_text(json.dumps(EXPECTED_META), encoding="utf-8")
    with pytest.raises(RuntimeError, match="train.bin byte-size mismatch"):
        _require_frozen_layout(tmp_path)


def test_frozen_layout_rejects_metadata_drift_before_hashing(tmp_path: Path) -> None:
    (tmp_path / "train.bin").write_bytes(b"")
    (tmp_path / "val.bin").write_bytes(b"")
    drifted = dict(EXPECTED_META)
    drifted["seed"] = 8101
    (tmp_path / "meta.json").write_text(json.dumps(drifted), encoding="utf-8")
    with pytest.raises(RuntimeError, match="data metadata mismatch seed"):
        _require_frozen_layout(tmp_path)


def _record(train_hash: str = "a" * 64) -> dict:
    return {
        "logical_contract": dict(EXPECTED_META),
        "files": [
            {"name": "train.bin", "bytes": EXPECTED_TRAIN_BYTES, "sha256": train_hash},
            {"name": "val.bin", "bytes": EXPECTED_VAL_BYTES, "sha256": "b" * 64},
            {"name": "meta.json", "bytes": 123, "sha256": "c" * 64},
        ],
    }


def test_fingerprint_match_accepts_exact_record() -> None:
    expected = _record()
    actual = json.loads(json.dumps(expected))
    assert_fingerprint_matches(actual, expected)


def test_fingerprint_match_rejects_same_size_different_bytes() -> None:
    expected = _record()
    actual = _record(train_hash="d" * 64)
    with pytest.raises(RuntimeError, match="byte fingerprint differs"):
        assert_fingerprint_matches(actual, expected)
