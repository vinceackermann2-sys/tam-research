from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import modal


APP_NAME = "cortex-attention8-v2-primary-preflight-v1"
VOLUME_NAME = "tam-research-data"
DATA_DIR = Path("/vol/data/tam100m-2b-curated-v1")
ENGINEERING_RESULT = Path(
    "/vol/cortex-s-v0/100m-200m/reduced-attention-v2-panel-v1/RESULT.json"
)
SUCCESSOR_RESULT_ROOT = Path(
    "/vol/cortex-s-v0/100m-2b/reduced-attention-v2-attention8-pair1-successor-v1"
)

TRAIN_SHA256 = "93e9cb0b7076a4ddd855fc696f657ea62592b8a03a05220be99c402f9043265b"
VAL_SHA256 = "ae0bc5adf36d0aa8e55e5e3903401d3f114b93037f43221944b4e88c5d1a5760"
META_SHA256 = "14bbbcf0ab0b8cba374074ef8ccb80a04ecead06c74780f35a1becd5aef1b8f3"

ENGINEERING_SEED = 2_026_092_901
EXPECTED_ENGINEERING_NLL_DELTA = -0.01449833869934114
EXPECTED_ENGINEERING_TPS_RATIO = 1.0651255526571075

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.11")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_corpus() -> dict[str, Any]:
    train = DATA_DIR / "train.bin"
    val = DATA_DIR / "val.bin"
    meta_path = DATA_DIR / "meta.json"
    missing = [str(path) for path in (train, val, meta_path) if not path.exists()]
    if missing:
        return {"ok": False, "missing": missing}

    sizes = {
        "train_bytes": train.stat().st_size,
        "val_bytes": val.stat().st_size,
        "meta_bytes": meta_path.stat().st_size,
    }
    hashes = {
        "train_sha256": _sha256(train),
        "val_sha256": _sha256(val),
        "meta_sha256": _sha256(meta_path),
    }
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    semantic_ok = all(
        (
            meta.get("assembly_version") == 3,
            meta.get("train_tokens") == 2_000_000_000,
            meta.get("val_tokens") == 5_000_000,
            meta.get("seed") == 8100,
            meta.get("tokenizer") == "gpt2",
            meta.get("dtype") == "uint16",
        )
    )
    exact_ok = (
        sizes["train_bytes"] == 4_000_000_000
        and sizes["val_bytes"] == 10_000_000
        and hashes["train_sha256"] == TRAIN_SHA256
        and hashes["val_sha256"] == VAL_SHA256
        and hashes["meta_sha256"] == META_SHA256
        and semantic_ok
    )
    return {
        "ok": exact_ok,
        "sizes": sizes,
        "hashes": hashes,
        "metadata": meta,
        "semantic_ok": semantic_ok,
    }


def _check_engineering_result() -> dict[str, Any]:
    if not ENGINEERING_RESULT.exists():
        return {"ok": False, "missing": str(ENGINEERING_RESULT)}

    result = json.loads(ENGINEERING_RESULT.read_text(encoding="utf-8"))
    comparisons = result.get("comparisons") or {}
    attention8 = comparisons.get("attention_8") or {}
    gates = result.get("gates") or {}

    try:
        nll_delta = float(attention8["nll_delta"])
        tps_ratio = float(attention8["throughput_ratio"])
    except (KeyError, TypeError, ValueError):
        nll_delta = float("nan")
        tps_ratio = float("nan")

    ok = all(
        (
            result.get("status") == "ENGINEERING_PANEL_COMPLETE",
            result.get("classification")
            == "ENGINEERING_PANEL_CANDIDATE_SCREEN_PASS",
            result.get("engineering_seed") == ENGINEERING_SEED,
            result.get("engineering_seed_consumed") is True,
            result.get("progression_candidates") == ["attention_8"],
            gates.get("v1_replicate_seed_sensitivity_guard") is False,
            attention8.get("combined_gate_pass") is True,
            attention8.get("quality_gate_pass") is True,
            attention8.get("throughput_gate_pass") is True,
            abs(nll_delta - EXPECTED_ENGINEERING_NLL_DELTA) <= 1e-12,
            abs(tps_ratio - EXPECTED_ENGINEERING_TPS_RATIO) <= 1e-12,
            result.get("scientific_seed_consumed") is False,
            result.get("replication_authorized") is False,
            result.get("250m_5b_authorized") is False,
            result.get("breakthrough_claim_allowed") is False,
        )
    )
    return {
        "ok": ok,
        "path": str(ENGINEERING_RESULT),
        "status": result.get("status"),
        "classification": result.get("classification"),
        "engineering_seed": result.get("engineering_seed"),
        "progression_candidates": result.get("progression_candidates"),
        "attention_8": attention8,
        "gates": gates,
    }


@app.function(
    image=image,
    cpu=1,
    memory=1024,
    timeout=20 * 60,
    retries=0,
    volumes={"/vol": volume},
)
def inspect_primary_read_only() -> str:
    volume.reload()
    corpus = _check_corpus()
    engineering = _check_engineering_result()
    successor_absent = not SUCCESSOR_RESULT_ROOT.exists()
    passed = corpus.get("ok") is True and engineering.get("ok") is True and successor_absent

    payload = {
        "status": "PASS" if passed else "FAIL",
        "classification": "ZERO_GPU_ATTENTION8_V2_PRIMARY_PREFLIGHT_V1",
        "volume_name": VOLUME_NAME,
        "corpus": corpus,
        "engineering_prerequisite": engineering,
        "successor_result_root": str(SUCCESSOR_RESULT_ROOT),
        "successor_result_root_absent": successor_absent,
        "gpu_allocated": False,
        "writes_performed": False,
        "seed_consumed": False,
        "scientific_training_authorized": False,
    }
    return json.dumps(payload, sort_keys=True)


@app.local_entrypoint()
def main() -> None:
    print(f"PRIMARY_PREFLIGHT={inspect_primary_read_only.remote()}")
