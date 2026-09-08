from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import modal

from architectures.cortex_s.experiments.scale100m_2b.protocol import (
    DATA_DIR,
    HARD_FULL_TIMEOUT_SECONDS,
    MAX_PROJECTED_FULL_SECONDS,
    TRAIN_SHA256,
    TRAIN_TOKENS,
    VAL_SHA256,
    VAL_TOKENS,
    protocol_snapshot,
)

APP_NAME = "cortex-s-v0-100m-2b"
VOLUME_NAME = "tam-research-data"
PREFLIGHT_ROOT = "/vol/cortex-s-v0/100m-2b/preflight"
RUN_ROOT = "/vol/cortex-s-v0/100m-2b/paired-seed8100"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
github_secret = modal.Secret.from_name("github-secret")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.7,<2.11",
        "numpy>=2.0,<3",
        "PyGithub>=2.3,<3",
    )
    .add_local_python_source("tam_research")
    .add_local_python_source("architectures")
)


def _comment(repo_full_name: str, issue_number: int, body: str) -> None:
    if not repo_full_name or not issue_number:
        print(f"[status] {body}", flush=True)
        return
    import os

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print(f"[status] {body}", flush=True)
        return
    try:
        import github

        client = github.Github(auth=github.Auth.Token(token))
        client.get_repo(repo_full_name).get_issue(number=issue_number).create_comment(body)
    except Exception as exc:
        print(f"[status-report-nonfatal] {type(exc).__name__}: {exc}; body={body}", flush=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(16 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _verify_frozen_data() -> dict:
    root = Path(DATA_DIR)
    train_path = root / "train.bin"
    val_path = root / "val.bin"
    meta_path = root / "meta.json"
    missing = [str(path) for path in (train_path, val_path, meta_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"frozen 100M/2B corpus missing: {missing}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    expected_meta = {
        "assembly_version": 3,
        "train_tokens": TRAIN_TOKENS,
        "val_tokens": VAL_TOKENS,
        "seed": 8100,
        "tokenizer": "gpt2",
        "dtype": "uint16",
    }
    for key, expected in expected_meta.items():
        if meta.get(key) != expected:
            raise RuntimeError(f"data metadata mismatch for {key}: {meta.get(key)!r} != {expected!r}")
    if train_path.stat().st_size != TRAIN_TOKENS * 2:
        raise RuntimeError("train.bin is not exactly 2B uint16 tokens")
    if val_path.stat().st_size != VAL_TOKENS * 2:
        raise RuntimeError("val.bin is not exactly 5M uint16 tokens")
    train_hash = _sha256(train_path)
    val_hash = _sha256(val_path)
    if train_hash != TRAIN_SHA256:
        raise RuntimeError(f"train corpus fingerprint mismatch: {train_hash}")
    if val_hash != VAL_SHA256:
        raise RuntimeError(f"validation corpus fingerprint mismatch: {val_hash}")
    return {
        "meta": meta,
        "train_sha256": train_hash,
        "val_sha256": val_hash,
        "train_bytes": train_path.stat().st_size,
        "val_bytes": val_path.stat().st_size,
    }


@app.function(
    image=image,
    cpu=4,
    memory=8192,
    timeout=30 * 60,
    volumes={"/vol": volume},
    secrets=[github_secret],
)
def verify_data_zero_gpu(
    source_sha: str,
    repo_full_name: str = "",
    issue_number: int = 0,
) -> dict:
    """Hash all 4GB+ of frozen training bytes before any GPU allocation."""

    volume.reload()
    target = Path(PREFLIGHT_ROOT) / "data.json"
    if target.exists():
        previous = json.loads(target.read_text(encoding="utf-8"))
        if previous.get("source_sha") != source_sha:
            raise RuntimeError("existing data preflight belongs to another source SHA")
        return previous
    data = _verify_frozen_data()
    result = {
        "status": "PASS",
        "scientific_status": "ZERO_GPU_DATA_PREFLIGHT",
        "source_sha": source_sha,
        "data": data,
        "protocol": protocol_snapshot(),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    volume.commit()
    _comment(
        repo_full_name,
        issue_number,
        "🟩 **CORTEX-S 100M/2B zero-GPU data gate passed** — exact 2B train and 5M validation bytes, metadata, sizes and SHA-256 fingerprints match the historical Transformer control. No GPU allocated.",
    )
    return result


@app.function(
    image=image,
    gpu="H100!",
    cpu=8,
    memory=32768,
    timeout=30 * 60,
    volumes={"/vol": volume},
    secrets=[github_secret],
)
def h100_preflight(
    source_sha: str,
    repo_full_name: str = "",
    issue_number: int = 0,
) -> dict:
    """Small paid calibration only; never starts the 2B run itself."""

    from architectures.cortex_s.experiments.scale100m_2b.train import run_h100_calibration

    volume.reload()
    data_path = Path(PREFLIGHT_ROOT) / "data.json"
    if not data_path.exists():
        raise RuntimeError("zero-GPU data gate is missing")
    data_gate = json.loads(data_path.read_text(encoding="utf-8"))
    if data_gate.get("status") != "PASS" or data_gate.get("source_sha") != source_sha:
        raise RuntimeError("zero-GPU data gate is not valid for this source SHA")

    dispatch_marker = Path(PREFLIGHT_ROOT) / "H100_PREFLIGHT_DISPATCHED.json"
    result_path = Path(PREFLIGHT_ROOT) / "h100.json"
    if result_path.exists() or dispatch_marker.exists():
        raise RuntimeError("H100 preflight namespace is already consumed; refusing duplicate spend")
    dispatch_marker.write_text(
        json.dumps({"source_sha": source_sha, "dispatched_unix": time.time()}, indent=2),
        encoding="utf-8",
    )
    volume.commit()

    _comment(
        repo_full_name,
        issue_number,
        "🟨 **CORTEX-S 100M H100 calibration started** — engineering seed only; 2B training is still locked. This run measures the exact production graph before spending the full budget.",
    )
    try:
        result = run_h100_calibration(
            data_dir=DATA_DIR,
            output_path=str(result_path),
            source_sha=source_sha,
        )
        volume.commit()
    except Exception:
        volume.commit()
        raise

    cost = result["projected_cost"]["total_usd_conservative"]
    if result["status"] == "PASS":
        _comment(
            repo_full_name,
            issue_number,
            "🟩 **CORTEX-S H100 calibration PASS** — "
            f"{result['training_tokens_per_second']:.0f} tok/s, peak VRAM={result['peak_vram_gb']:.2f} GiB, "
            f"projected full wall/compute envelope={result['projected_full_seconds']:.0f}s, conservative projected compute=${cost:.2f}. "
            "The full 2B run is now eligible but has not been launched by this preflight.",
        )
    else:
        _comment(
            repo_full_name,
            issue_number,
            "🟥 **CORTEX-S H100 calibration ABORTED full progression** — "
            f"projection={result['projected_full_seconds']:.0f}s vs {MAX_PROJECTED_FULL_SECONDS}s gate, "
            f"peak VRAM={result['peak_vram_gb']:.2f} GiB. No 2B training was launched.",
        )
    return result


@app.function(
    image=image,
    gpu="H100!",
    cpu=8,
    memory=32768,
    timeout=HARD_FULL_TIMEOUT_SECONDS,
    volumes={"/vol": volume},
    secrets=[github_secret],
)
def full_2b(
    source_sha: str,
    repo_full_name: str = "",
    issue_number: int = 0,
) -> dict:
    """One fail-closed 100M/2B paired attempt after a PASS calibration."""

    from architectures.cortex_s.experiments.scale100m_2b.train import train_full_2b

    volume.reload()
    preflight_path = Path(PREFLIGHT_ROOT) / "h100.json"
    if not preflight_path.exists():
        raise RuntimeError("H100 preflight result is missing")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("status") != "PASS" or preflight.get("source_sha") != source_sha:
        raise RuntimeError("full run is not authorized by exact-source preflight")
    if float(preflight.get("projected_full_seconds", float("inf"))) > MAX_PROJECTED_FULL_SECONDS:
        raise RuntimeError("full run projection exceeds budget gate")

    dispatch_marker = Path(RUN_ROOT) / "FULL_DISPATCH_CONSUMED.json"
    if dispatch_marker.exists() or (Path(RUN_ROOT) / "SUCCESS.json").exists():
        raise RuntimeError("full 2B namespace already consumed; refusing duplicate H100 spend")
    Path(RUN_ROOT).mkdir(parents=True, exist_ok=True)
    dispatch_marker.write_text(
        json.dumps({"source_sha": source_sha, "dispatched_unix": time.time()}, indent=2),
        encoding="utf-8",
    )
    volume.commit()

    _comment(
        repo_full_name,
        issue_number,
        "🔥 **CORTEX-S 100M / 2B paired run started** — exact source and H100 preflight locked; seed=8100, 2B token exposures, context=512, global batch=128. Hard timeout prevents unbounded credit use.",
    )
    try:
        result = train_full_2b(
            data_dir=DATA_DIR,
            output_dir=RUN_ROOT,
            preflight_path=str(preflight_path),
            source_sha=source_sha,
        )
        volume.commit()
    except Exception:
        # Persist the consumed marker and any checkpoints that were durably written.
        volume.commit()
        raise

    comparison = result["comparison"]
    _comment(
        repo_full_name,
        issue_number,
        "✅ **CORTEX-S 100M / 2B paired run complete** — "
        f"NLL={result['final_eval']['nll']:.6f} vs Transformer {result['historical_transformer']['final_nll']:.6f}; "
        f"PPL={result['final_eval']['perplexity']:.3f}; throughput={result['training_tokens_per_second']:.0f} tok/s; "
        f"peak VRAM={result['peak_vram_gb']:.2f} GiB; equal-token win={comparison['equal_token_nll_cortex_better']}. "
        "Classification remains paired historical-control evidence, not a breakthrough claim.",
    )
    print(json.dumps(result, indent=2), flush=True)
    return result


@app.local_entrypoint()
def main(
    phase: str = "preflight",
    source_sha: str = "",
    repo_full_name: str = "",
    issue_number: int = 0,
):
    if not source_sha:
        raise ValueError("source_sha is required and must be the exact checked-out commit")
    phase = phase.strip().lower()
    if phase == "preflight":
        data = verify_data_zero_gpu.remote(source_sha, repo_full_name, issue_number)
        print(json.dumps({"zero_gpu": data}, indent=2), flush=True)
        result = h100_preflight.remote(source_sha, repo_full_name, issue_number)
        print(json.dumps({"h100_preflight": result}, indent=2), flush=True)
        return
    if phase == "full":
        result = full_2b.remote(source_sha, repo_full_name, issue_number)
        print(json.dumps({"full": result}, indent=2), flush=True)
        return
    raise ValueError("phase must be 'preflight' or 'full'")
