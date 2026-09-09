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
ENGINEERING_SEED = 2_026_090_906
APP_NAME = "cortex-s-v0-100m-systems-profile-v1"
VOLUME_NAME = "tam-research-data"
RESULT_ROOT = "/vol/cortex-s-v0/100m-systems-profile-v1"
H100_TIMEOUT_SECONDS = 15 * 60

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)

# No GitHub credential is mounted. GitHub Actions authenticates the handoff while
# this app writes only its source-bound single-use evidence namespace.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.10,<2.11",
        "numpy>=2.0,<3",
    )
    .add_local_python_source("tam_research")
    .add_local_python_source("architectures")
)


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


def _json_safe(value):
    return json.loads(json.dumps(value))


@app.function(
    image=image,
    cpu=4,
    memory=8192,
    timeout=30 * 60,
    volumes={"/vol": volume},
)
def verify_data_zero_gpu(source_sha: str) -> dict:
    from architectures.cortex_s.experiments.scale100m_2b.systems_profile_v1 import (
        ENGINEERING_SEED as REMOTE_ENGINEERING_SEED,
        validate_profile_protocol,
        zero_gpu_profile_contract_probe,
    )

    if REMOTE_ENGINEERING_SEED != ENGINEERING_SEED:
        raise RuntimeError("profile-v1 launcher/core engineering seed drift")
    protocol = validate_profile_protocol()
    if protocol.get("full_training_authorized") is not False:
        raise RuntimeError("profile-v1 unexpectedly authorizes full training")
    if protocol.get("next_stage_authorized") is not False:
        raise RuntimeError("profile-v1 unexpectedly authorizes a next stage")

    volume.reload()
    root = Path(RESULT_ROOT)
    data_path = root / "DATA_GATE.json"
    dispatch_path = root / "H100_DISPATCH_CONSUMED.json"
    result_path = root / "RESULT.json"
    if data_path.exists() or dispatch_path.exists() or result_path.exists():
        raise RuntimeError("systems profile-v1 namespace is already consumed")

    data = _verify_frozen_data()
    contract = zero_gpu_profile_contract_probe()
    if contract.get("status") != "PASS":
        raise RuntimeError("profile-v1 zero-GPU production contract did not pass")

    result = _json_safe(
        {
            "status": "PASS",
            "scientific_status": "ZERO_GPU_SYSTEMS_PROFILE_GATE_ONLY",
            "profile_namespace": "systems-profile-v1",
            "source_sha": source_sha,
            "engineering_seed": ENGINEERING_SEED,
            "data": data,
            "production_contract": contract,
            "core_protocol": protocol,
            "gpu_allocated": False,
            "full_training_authorized": False,
            "next_stage_authorized": False,
            "scientific_evidence": False,
        }
    )
    root.mkdir(parents=True, exist_ok=True)
    data_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    volume.commit()
    print(json.dumps({"profile_v1_zero_gpu": result}, indent=2), flush=True)
    return result


@app.function(
    image=image,
    gpu="H100!",
    cpu=8,
    memory=32768,
    timeout=H100_TIMEOUT_SECONDS,
    volumes={"/vol": volume},
)
def h100_profile(source_sha: str) -> dict:
    """Exactly one bounded H100 diagnostic profile; never a 2B training run."""

    from architectures.cortex_s.experiments.scale100m_2b.systems_profile_v1 import (
        ENGINEERING_SEED as REMOTE_ENGINEERING_SEED,
        run_h100_systems_profile,
    )

    if REMOTE_ENGINEERING_SEED != ENGINEERING_SEED:
        raise RuntimeError("profile-v1 launcher/core engineering seed drift")

    volume.reload()
    root = Path(RESULT_ROOT)
    data_path = root / "DATA_GATE.json"
    dispatch_path = root / "H100_DISPATCH_CONSUMED.json"
    result_path = root / "RESULT.json"
    if not data_path.exists():
        raise RuntimeError("profile-v1 zero-GPU data/production gate is missing")
    data_gate = json.loads(data_path.read_text(encoding="utf-8"))
    if data_gate.get("status") != "PASS" or data_gate.get("source_sha") != source_sha:
        raise RuntimeError("profile-v1 zero-GPU gate does not authorize this exact source SHA")
    if dispatch_path.exists() or result_path.exists():
        raise RuntimeError("systems profile-v1 H100 namespace is already consumed")

    dispatch = {
        "profile_namespace": "systems-profile-v1",
        "source_sha": source_sha,
        "engineering_seed": ENGINEERING_SEED,
        "dispatched_unix": time.time(),
        "scientific_seed_used": False,
        "full_training_authorized": False,
        "next_stage_authorized": False,
        "scientific_evidence": False,
    }
    root.mkdir(parents=True, exist_ok=True)
    dispatch_path.write_text(json.dumps(dispatch, indent=2), encoding="utf-8")
    volume.commit()
    print(json.dumps({"profile_v1_h100_started": dispatch}, indent=2), flush=True)

    try:
        result = _json_safe(run_h100_systems_profile(source_sha=source_sha, data_dir=DATA_DIR))
    except Exception as exc:
        failure = _json_safe(
            {
                "status": "SYSTEMS_PROFILE_RUNTIME_FAIL",
                "scientific_status": "ENGINEERING_SYSTEMS_PROFILE_ONLY",
                "profile_namespace": "systems-profile-v1",
                "source_sha": source_sha,
                "engineering_seed": ENGINEERING_SEED,
                "error": f"{type(exc).__name__}: {exc}",
                "full_training_authorized": False,
                "next_stage_authorized": False,
                "scientific_evidence": False,
            }
        )
        result_path.write_text(json.dumps(failure, indent=2), encoding="utf-8")
        volume.commit()
        print(json.dumps({"profile_v1_failure": failure}, indent=2), flush=True)
        raise

    result["profile_namespace"] = "systems-profile-v1"
    result["full_training_authorized"] = False
    result["next_stage_authorized"] = False
    result["scientific_evidence"] = False
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    volume.commit()
    print(json.dumps({"profile_v1_result": result}, indent=2), flush=True)
    return result


@app.local_entrypoint()
def main(phase: str = "systems-profile-v1", source_sha: str = ""):
    if phase.strip().lower() != "systems-profile-v1":
        raise ValueError("only phase='systems-profile-v1' is supported")
    source_sha = source_sha.strip().lower()
    if len(source_sha) != 40 or any(char not in "0123456789abcdef" for char in source_sha):
        raise ValueError("source_sha must be a full 40-character lowercase/hex commit SHA")
    data = verify_data_zero_gpu.remote(source_sha)
    print(json.dumps({"zero_gpu": data}, indent=2), flush=True)
    result = h100_profile.remote(source_sha)
    print(json.dumps({"h100_profile": result}, indent=2), flush=True)
