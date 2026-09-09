from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import modal


DATA_DIR = "/vol/data/tam100m-2b-curated-v1"
TRAIN_TOKENS = 2_000_000_000
VAL_TOKENS = 5_000_000
TRAIN_SHA256 = "93e9cb0b7076a4ddd855fc696f657ea62592b8a03a05220be99c402f9043265b"
VAL_SHA256 = "ae0bc5adf36d0aa8e55e5e3903401d3f114b93037f43221944b4e88c5d1a5760"
META_SHA256 = "14bbbcf0ab0b8cba374074ef8ccb80a04ecead06c74780f35a1becd5aef1b8f3"
ENGINEERING_SEED = 2_026_090_901
APP_NAME = "cortex-s-v0-100m-systems-microbench-v1"
VOLUME_NAME = "tam-research-data"
RESULT_ROOT = "/vol/cortex-s-v0/100m-systems-microbench-v1"
H100_TIMEOUT_SECONDS = 15 * 60

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
github_secret = modal.Secret.from_name("github-secret")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.10,<2.11",
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
        # Status reporting must never turn a completed engineering measurement into
        # an apparent compute failure.
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
    observed = {
        "train_sha256": _sha256(train_path),
        "val_sha256": _sha256(val_path),
        "meta_sha256": _sha256(meta_path),
    }
    expected = {
        "train_sha256": TRAIN_SHA256,
        "val_sha256": VAL_SHA256,
        "meta_sha256": META_SHA256,
    }
    for key, value in expected.items():
        if observed[key] != value:
            raise RuntimeError(f"{key} mismatch: {observed[key]}")
    return {
        **observed,
        "train_bytes": train_path.stat().st_size,
        "val_bytes": val_path.stat().st_size,
        "meta": meta,
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
    """Re-hash the immutable corpus before the single H100 allocation."""

    from architectures.cortex_s.experiments.scale100m_2b.systems_microbench_v1 import (
        ENGINEERING_SEED as REMOTE_ENGINEERING_SEED,
        validate_microbench_protocol,
    )

    if REMOTE_ENGINEERING_SEED != ENGINEERING_SEED:
        raise RuntimeError("launcher/remote engineering seed drift")
    protocol = validate_microbench_protocol()
    volume.reload()
    root = Path(RESULT_ROOT)
    data_path = root / "DATA_GATE.json"
    dispatch_path = root / "H100_DISPATCH_CONSUMED.json"
    result_path = root / "RESULT.json"
    if data_path.exists() or dispatch_path.exists() or result_path.exists():
        raise RuntimeError("systems microbenchmark v1 namespace is already consumed")
    data = _verify_frozen_data()
    result = {
        "status": "PASS",
        "scientific_status": "ZERO_GPU_SYSTEMS_DATA_GATE_ONLY",
        "source_sha": source_sha,
        "engineering_seed": ENGINEERING_SEED,
        "data": data,
        "protocol": protocol,
    }
    root.mkdir(parents=True, exist_ok=True)
    data_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    volume.commit()
    _comment(
        repo_full_name,
        issue_number,
        "🟩 **CORTEX-S systems microbench v1 zero-GPU gate PASS** — exact corpus hashes and production-shape protocol verified. No H100 allocated yet.",
    )
    return result


@app.function(
    image=image,
    gpu="H100!",
    cpu=8,
    memory=32768,
    timeout=H100_TIMEOUT_SECONDS,
    volumes={"/vol": volume},
    secrets=[github_secret],
)
def h100_microbenchmark(
    source_sha: str,
    repo_full_name: str = "",
    issue_number: int = 0,
) -> dict:
    """One single-use engineering H100 systems measurement; never a 2B run."""

    from architectures.cortex_s.experiments.scale100m_2b.systems_microbench_v1 import (
        ENGINEERING_SEED as REMOTE_ENGINEERING_SEED,
        run_h100_systems_microbenchmark,
    )

    if REMOTE_ENGINEERING_SEED != ENGINEERING_SEED:
        raise RuntimeError("launcher/remote engineering seed drift")
    volume.reload()
    root = Path(RESULT_ROOT)
    data_path = root / "DATA_GATE.json"
    dispatch_path = root / "H100_DISPATCH_CONSUMED.json"
    result_path = root / "RESULT.json"
    if not data_path.exists():
        raise RuntimeError("zero-GPU data gate is missing")
    data_gate = json.loads(data_path.read_text(encoding="utf-8"))
    if data_gate.get("status") != "PASS" or data_gate.get("source_sha") != source_sha:
        raise RuntimeError("zero-GPU data gate does not authorize this exact source SHA")
    if dispatch_path.exists() or result_path.exists():
        raise RuntimeError("systems H100 microbenchmark namespace already consumed")

    dispatch_path.write_text(
        json.dumps(
            {
                "source_sha": source_sha,
                "engineering_seed": ENGINEERING_SEED,
                "dispatched_unix": time.time(),
                "scientific_seed_used": False,
                "full_training_authorized": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    volume.commit()
    _comment(
        repo_full_name,
        issue_number,
        "🟨 **CORTEX-S systems microbench v1 H100 started** — one engineering seed only, <=15-minute hard timeout. No paired/scientific training function exists in this app.",
    )

    try:
        result = run_h100_systems_microbenchmark(source_sha=source_sha, data_dir=DATA_DIR)
    except Exception as exc:
        failure = {
            "status": "SYSTEMS_MICROBENCH_RUNTIME_FAIL",
            "scientific_status": "ENGINEERING_SYSTEMS_MICROBENCH_ONLY",
            "source_sha": source_sha,
            "engineering_seed": ENGINEERING_SEED,
            "error": f"{type(exc).__name__}: {exc}",
            "full_training_authorized": False,
            "next_stage_authorized": False,
        }
        result_path.write_text(json.dumps(failure, indent=2), encoding="utf-8")
        volume.commit()
        _comment(
            repo_full_name,
            issue_number,
            f"🟥 **CORTEX-S systems microbench v1 failed safely** — {failure['error']}. Engineering seed/namespace are consumed; no retry and no full training.",
        )
        raise

    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    volume.commit()
    speedup = result.get("grouped_full_model_speedup")
    speedup_text = "n/a" if speedup is None else f"{float(speedup):.3f}x"
    _comment(
        repo_full_name,
        issue_number,
        "🧪 **CORTEX-S systems microbench v1 complete** — "
        f"status={result['status']}, grouped/legacy full-model speedup={speedup_text}. "
        "This is engineering evidence only; it does not authorize the 2B run or another H100 dispatch.",
    )
    print(json.dumps(result, indent=2), flush=True)
    return result


@app.local_entrypoint()
def main(
    phase: str = "microbench-v1",
    source_sha: str = "",
    repo_full_name: str = "",
    issue_number: int = 0,
):
    if phase.strip().lower() != "microbench-v1":
        raise ValueError("only phase='microbench-v1' is supported")
    if len(source_sha) != 40 or any(char not in "0123456789abcdef" for char in source_sha.lower()):
        raise ValueError("source_sha must be a full 40-character lowercase/hex commit SHA")
    data = verify_data_zero_gpu.remote(source_sha.lower(), repo_full_name, issue_number)
    print(json.dumps({"zero_gpu": data}, indent=2), flush=True)
    result = h100_microbenchmark.remote(source_sha.lower(), repo_full_name, issue_number)
    print(json.dumps({"h100_microbenchmark": result}, indent=2), flush=True)
