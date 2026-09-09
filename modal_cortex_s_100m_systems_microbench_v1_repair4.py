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
ENGINEERING_SEED = 2_026_090_904
APP_NAME = "cortex-s-v0-100m-systems-microbench-v1-repair4"
VOLUME_NAME = "tam-research-data"
RESULT_ROOT = "/vol/cortex-s-v0/100m-systems-microbench-v1-repair4"
H100_TIMEOUT_SECONDS = 15 * 60

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.10,<2.11", "numpy>=2.0,<3")
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
        raise FileNotFoundError(f"frozen corpus missing: {missing}")
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
            raise RuntimeError(f"metadata mismatch for {key}: {meta.get(key)!r} != {expected!r}")
    if train_path.stat().st_size != TRAIN_TOKENS * 2 or val_path.stat().st_size != VAL_TOKENS * 2:
        raise RuntimeError("frozen corpus byte-size contract failed")
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


@app.function(image=image, cpu=4, memory=8192, timeout=30 * 60, volumes={"/vol": volume})
def verify_data_zero_gpu(source_sha: str) -> str:
    from architectures.cortex_s.experiments.scale100m_2b.systems_microbench_v1_repair4 import (
        ENGINEERING_SEED as REMOTE_ENGINEERING_SEED,
        validate_microbench_protocol,
        zero_gpu_repair4_contract_probe,
    )

    if REMOTE_ENGINEERING_SEED != ENGINEERING_SEED:
        raise RuntimeError("repair4 launcher/remote seed drift")
    protocol = validate_microbench_protocol()
    if protocol["full_training_authorized"] is not False or protocol["next_stage_authorized"] is not False:
        raise RuntimeError("repair4 unexpectedly authorizes later stage")

    volume.reload()
    root = Path(RESULT_ROOT)
    data_path = root / "DATA_GATE.json"
    dispatch_path = root / "H100_DISPATCH_CONSUMED.json"
    result_path = root / "RESULT.json"
    if data_path.exists() or dispatch_path.exists() or result_path.exists():
        raise RuntimeError("repair4 namespace already consumed")

    result = {
        "status": "PASS",
        "scientific_status": "ZERO_GPU_SYSTEMS_DATA_AND_LOGICAL_PADDING_GATE_ONLY",
        "repair_namespace": "microbench-v1-repair4",
        "source_sha": source_sha,
        "engineering_seed": ENGINEERING_SEED,
        "data": _verify_frozen_data(),
        "repair4_contract": zero_gpu_repair4_contract_probe(),
        "protocol": protocol,
        "full_training_authorized": False,
        "next_stage_authorized": False,
    }
    root.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result, indent=2)
    data_path.write_text(payload, encoding="utf-8")
    volume.commit()
    print(json.dumps({"repair4_zero_gpu": result}, indent=2), flush=True)
    return payload


@app.function(
    image=image,
    gpu="H100!",
    cpu=8,
    memory=32768,
    timeout=H100_TIMEOUT_SECONDS,
    volumes={"/vol": volume},
)
def h100_microbenchmark(source_sha: str) -> str:
    from architectures.cortex_s.experiments.scale100m_2b.systems_microbench_v1_repair4 import (
        ENGINEERING_SEED as REMOTE_ENGINEERING_SEED,
        run_h100_systems_microbenchmark,
    )

    if REMOTE_ENGINEERING_SEED != ENGINEERING_SEED:
        raise RuntimeError("repair4 launcher/remote seed drift")
    volume.reload()
    root = Path(RESULT_ROOT)
    data_path = root / "DATA_GATE.json"
    dispatch_path = root / "H100_DISPATCH_CONSUMED.json"
    result_path = root / "RESULT.json"
    if not data_path.exists():
        raise RuntimeError("repair4 zero-GPU gate missing")
    gate = json.loads(data_path.read_text(encoding="utf-8"))
    if gate.get("status") != "PASS" or gate.get("source_sha") != source_sha:
        raise RuntimeError("repair4 zero-GPU gate does not authorize exact source")
    if dispatch_path.exists() or result_path.exists():
        raise RuntimeError("repair4 H100 namespace already consumed")

    dispatch = {
        "repair_namespace": "microbench-v1-repair4",
        "source_sha": source_sha,
        "engineering_seed": ENGINEERING_SEED,
        "dispatched_unix": time.time(),
        "scientific_seed_used": False,
        "full_training_authorized": False,
        "next_stage_authorized": False,
    }
    dispatch_path.write_text(json.dumps(dispatch, indent=2), encoding="utf-8")
    volume.commit()
    print(json.dumps({"repair4_h100_started": dispatch}, indent=2), flush=True)

    try:
        result = run_h100_systems_microbenchmark(source_sha=source_sha, data_dir=DATA_DIR)
    except Exception as exc:
        failure = {
            "status": "SYSTEMS_MICROBENCH_RUNTIME_FAIL",
            "scientific_status": "ENGINEERING_SYSTEMS_MICROBENCH_ONLY",
            "repair_namespace": "microbench-v1-repair4",
            "source_sha": source_sha,
            "engineering_seed": ENGINEERING_SEED,
            "error": f"{type(exc).__name__}: {exc}",
            "full_training_authorized": False,
            "next_stage_authorized": False,
        }
        payload = json.dumps(failure, indent=2)
        result_path.write_text(payload, encoding="utf-8")
        volume.commit()
        print(json.dumps({"repair4_failure": failure}, indent=2), flush=True)
        raise

    result = dict(result)
    result["full_training_authorized"] = False
    result["next_stage_authorized"] = False
    payload = json.dumps(result, indent=2)
    result_path.write_text(payload, encoding="utf-8")
    volume.commit()
    print(json.dumps({"repair4_result": result}, indent=2), flush=True)
    # Return only a plain JSON string so the local runner never needs torch to
    # deserialize a TorchVersion/Tensor-derived object.
    return payload


@app.local_entrypoint()
def main(phase: str = "microbench-v1-repair4", source_sha: str = ""):
    if phase.strip().lower() != "microbench-v1-repair4":
        raise ValueError("only phase='microbench-v1-repair4' is supported")
    source_sha = source_sha.strip().lower()
    if len(source_sha) != 40 or any(char not in "0123456789abcdef" for char in source_sha):
        raise ValueError("source_sha must be a full lowercase hex commit SHA")
    data_json = verify_data_zero_gpu.remote(source_sha)
    print(json.dumps({"zero_gpu": json.loads(data_json)}, indent=2), flush=True)
    result_json = h100_microbenchmark.remote(source_sha)
    print(json.dumps({"h100_microbenchmark": json.loads(result_json)}, indent=2), flush=True)
