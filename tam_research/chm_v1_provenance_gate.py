from __future__ import annotations

"""Zero-credit provenance wrapper for the CHM-v1 / EIEM scientific gate.

This module adds evidence identity checks only.  It does not launch training,
authorize paid compute, alter scientific thresholds, consume a reserved seed,
or change the underlying scientific-gate arithmetic.

A future authorized runner must bind every per-seed record to the exact Git
commit that executed it and to the exact corpus fingerprint produced by
``chm_v1_corpus_fingerprint``.  All three reserved-seed records must agree on
both identities before the existing scientific gate is allowed to classify the
evidence.
"""

from collections.abc import Mapping, Sequence
import re
from typing import Any

from .chm_v1_corpus_fingerprint import (
    EXPECTED_META,
    EXPECTED_TRAIN_BYTES,
    EXPECTED_VAL_BYTES,
)
from .chm_v1_scientific_gate import evaluate_scientific_gate

_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_FILES = {
    "train.bin": EXPECTED_TRAIN_BYTES,
    "val.bin": EXPECTED_VAL_BYTES,
}


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping")
    return value


def validate_execution_code_sha(value: Any) -> str:
    """Require a canonical lowercase full Git SHA."""
    if not isinstance(value, str) or _SHA40.fullmatch(value) is None:
        raise ValueError("execution_code_sha must be a lowercase 40-hex Git SHA")
    return value


def validate_corpus_fingerprint_record(value: Any) -> tuple[tuple[str, Any], ...]:
    """Validate and canonicalize one frozen-corpus fingerprint record.

    The actual hash values are intentionally not hard-coded here: the future
    explicit run authorization must freeze the observed corpus hashes before GPU
    allocation.  This function ensures the record has the frozen logical corpus
    contract, exact train/validation byte sizes, canonical hash shapes, and no
    missing/duplicate/extra source files.
    """
    record = _mapping(value, "corpus_fingerprint")
    logical = _mapping(record.get("logical_contract"), "corpus_fingerprint.logical_contract")
    if dict(logical) != EXPECTED_META:
        raise ValueError("corpus_fingerprint logical contract drifted from CHM-v1 freeze")

    files = record.get("files")
    if not isinstance(files, Sequence) or isinstance(files, (str, bytes)):
        raise ValueError("corpus_fingerprint.files must be a sequence")
    if len(files) != 3:
        raise ValueError("corpus_fingerprint must contain exactly train.bin, val.bin, meta.json")

    canonical_files: dict[str, tuple[int, str]] = {}
    for index, raw in enumerate(files):
        item = _mapping(raw, f"corpus_fingerprint.files[{index}]")
        name = item.get("name")
        byte_count = item.get("bytes")
        sha256 = item.get("sha256")
        if not isinstance(name, str):
            raise ValueError(f"corpus_fingerprint.files[{index}].name must be a string")
        if name in canonical_files:
            raise ValueError(f"duplicate corpus fingerprint file {name!r}")
        if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 0:
            raise ValueError(f"corpus_fingerprint file {name!r} bytes must be a nonnegative integer")
        if not isinstance(sha256, str) or _SHA256.fullmatch(sha256) is None:
            raise ValueError(f"corpus_fingerprint file {name!r} sha256 must be lowercase 64-hex")
        canonical_files[name] = (byte_count, sha256)

    if set(canonical_files) != {"train.bin", "val.bin", "meta.json"}:
        raise ValueError("corpus_fingerprint filenames must be exactly train.bin, val.bin, meta.json")
    for name, expected_bytes in _REQUIRED_FILES.items():
        if canonical_files[name][0] != expected_bytes:
            raise ValueError(
                f"corpus_fingerprint {name} bytes drifted: "
                f"{canonical_files[name][0]} != {expected_bytes}"
            )
    if canonical_files["meta.json"][0] <= 0:
        raise ValueError("corpus_fingerprint meta.json must be non-empty")

    logical_tuple = tuple(sorted((str(key), value) for key, value in logical.items()))
    files_tuple = tuple(
        (name, canonical_files[name][0], canonical_files[name][1])
        for name in sorted(canonical_files)
    )
    return (
        ("logical_contract", logical_tuple),
        ("files", files_tuple),
    )


def evaluate_scientific_gate_with_provenance(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Refuse classification unless all seed records share exact provenance.

    After provenance validation, delegates unchanged to
    :func:`evaluate_scientific_gate`.  Provenance therefore cannot rescue or
    weaken any preregistered scientific criterion.
    """
    code_shas: list[str] = []
    fingerprints: list[tuple[tuple[str, Any], ...]] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"scientific record {index} must be a mapping")
        code_shas.append(validate_execution_code_sha(record.get("execution_code_sha")))
        fingerprints.append(validate_corpus_fingerprint_record(record.get("corpus_fingerprint")))

    if len(set(code_shas)) != 1:
        raise ValueError("scientific seed records disagree on execution_code_sha")
    if len(set(fingerprints)) != 1:
        raise ValueError("scientific seed records disagree on corpus_fingerprint")

    result = dict(evaluate_scientific_gate(records))
    result["provenance"] = {
        "execution_code_sha": code_shas[0],
        "corpus_fingerprint": fingerprints[0],
        "all_seed_records_match": True,
    }
    return result
