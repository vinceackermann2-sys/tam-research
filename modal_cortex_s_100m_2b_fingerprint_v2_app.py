from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import modal

APP_NAME = "cortex-s-v0-100m-2b-fingerprint-v2"
VOLUME_NAME = "tam-research-data"
DATA_DIR = "/vol/data/tam100m-2b-curated-v1"
FINGERPRINT_ROOT = "/vol/cortex-s-v0/100m-2b/fingerprint-v2"
TRAIN_TOKENS = 2_000_000_000
VAL_TOKENS = 5_000_000

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.11")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(32 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _inspect_corpus() -> dict:
    root = Path(DATA_DIR)
    train_path = root / "train.bin"
    val_path = root / "val.bin"
    meta_path = root / "meta.json"
    missing = [str(path) for path in (train_path, val_path, meta_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"historical Transformer corpus missing: {missing}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    expected_meta = {
        "assembly_version": 3,
        "train_tokens": TRAIN_TOKENS,
        "val_tokens": VAL_TOKENS,
        "seed": 8100,
        "tokenizer": "gpt2",
        "dtype": "uint16",
    }
    observed_meta = {key: meta.get(key) for key in expected_meta}
    metadata_matches = observed_meta == expected_meta

    train_bytes = train_path.stat().st_size
    val_bytes = val_path.stat().st_size
    expected_train_bytes = TRAIN_TOKENS * 2
    expected_val_bytes = VAL_TOKENS * 2
    sizes_match = train_bytes == expected_train_bytes and val_bytes == expected_val_bytes

    # This diagnostic deliberately observes fingerprints. It does NOT compare them
    # against the erroneous preregistered hashes from the consumed preflight #761.
    train_sha256 = _sha256(train_path)
    val_sha256 = _sha256(val_path)
    meta_sha256 = _sha256(meta_path)

    return {
        "data_dir": DATA_DIR,
        "metadata_matches_historical_builder_contract": metadata_matches,
        "sizes_match_exact_token_counts": sizes_match,
        "observed_metadata": observed_meta,
        "expected_metadata": expected_meta,
        "train_bytes": train_bytes,
        "expected_train_bytes": expected_train_bytes,
        "val_bytes": val_bytes,
        "expected_val_bytes": expected_val_bytes,
        "train_sha256_observed": train_sha256,
        "val_sha256_observed": val_sha256,
        "meta_sha256_observed": meta_sha256,
    }


@app.function(
    image=image,
    cpu=4,
    memory=4096,
    timeout=30 * 60,
    volumes={"/vol": volume},
)
def fingerprint_v2(source_sha: str) -> dict:
    """Single-use CPU-only observation of the historical Transformer's corpus bytes."""

    volume.reload()
    root = Path(FINGERPRINT_ROOT) / source_sha
    marker = root / "ATTEMPT_STARTED.json"
    result_path = root / "RESULT.json"
    if marker.exists() or result_path.exists():
        raise RuntimeError(
            "fingerprint-v2 namespace already consumed for this source SHA; refusing redispatch"
        )

    root.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps(
            {
                "source_sha": source_sha,
                "started_unix": time.time(),
                "phase": "fingerprint-v2",
                "gpu_allocated": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    volume.commit()

    observed = _inspect_corpus()
    result = {
        "status": "PASS" if (
            observed["metadata_matches_historical_builder_contract"]
            and observed["sizes_match_exact_token_counts"]
        ) else "FAIL",
        "scientific_status": "ZERO_GPU_FINGERPRINT_DIAGNOSTIC_ONLY",
        "source_sha": source_sha,
        "gpu_allocated": False,
        "authorizes_h100": False,
        "authorizes_full_2b": False,
        "prior_preflight_issue_761_consumed": True,
        "note": (
            "This phase only establishes the observed SHA-256 fingerprints of the existing "
            "historical Transformer corpus. It cannot authorize paid calibration or training."
        ),
        "observed": observed,
        "finished_unix": time.time(),
    }
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    volume.commit()
    return result


@app.local_entrypoint()
def main(source_sha: str = ""):
    if not source_sha or len(source_sha) != 40:
        raise ValueError("source_sha must be the exact 40-character checked-out commit SHA")
    result = fingerprint_v2.remote(source_sha)
    print("FINGERPRINT_V2_JSON=" + json.dumps(result, separators=(",", ":")), flush=True)
    if result.get("status") != "PASS":
        raise RuntimeError("fingerprint-v2 metadata/size contract failed")
