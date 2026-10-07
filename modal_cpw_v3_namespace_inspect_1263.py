from __future__ import annotations

import json
from pathlib import Path

import modal

APP_NAME = "tam-research-cpw-v3-namespace-inspect-1263"
VOLUME_NAME = "tam-research-data"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.11")


@app.function(
    image=image,
    cpu=0.125,
    memory=128,
    timeout=60,
    retries=0,
    volumes={"/vol": volume},
)
def inspect() -> dict:
    volume.reload()

    result_root = Path("/vol/cpw-v3/sparse-world-panel-v1")
    data_root = Path("/vol/data/fineweb-edu-gpt2")
    meta_path = data_root / "meta.json"
    train_path = data_root / "train.bin"
    val_path = data_root / "val.bin"

    result = {
        "issue": 1263,
        "scientific_issue": 1261,
        "writes_performed": False,
        "gpu_requested": False,
        "result_root": str(result_root),
        "root_exists": result_root.exists(),
        "attempt": (result_root / "ATTEMPT.json").exists(),
        "result": (result_root / "RESULT.json").exists(),
        "failure": (result_root / "FAILURE.json").exists(),
        "data_meta_exists": meta_path.exists(),
        "train_exists": train_path.exists(),
        "val_exists": val_path.exists(),
        "data_meta": None,
        "train_bytes": train_path.stat().st_size if train_path.exists() else None,
        "val_bytes": val_path.stat().st_size if val_path.exists() else None,
    }

    if meta_path.exists():
        try:
            result["data_meta"] = json.loads(meta_path.read_text())
        except Exception as exc:
            result["data_meta_error"] = f"{type(exc).__name__}: {exc}"

    for name in ("ATTEMPT.json", "RESULT.json", "FAILURE.json"):
        path = result_root / name
        if path.exists():
            try:
                result[name] = json.loads(path.read_text())
            except Exception as exc:
                result[name] = {"read_error": f"{type(exc).__name__}: {exc}"}

    expected_meta = {
        "dataset": "HuggingFaceFW/fineweb-edu",
        "dataset_config": "sample-10BT",
        "tokenizer": "gpt2",
        "seed": 1234,
        "train_tokens": 25_000_000,
        "val_tokens": 2_000_000,
        "dtype": "uint16",
    }
    result["data_guard_ok"] = (
        result["data_meta"] == expected_meta
        and result["train_bytes"] == 25_000_000 * 2
        and result["val_bytes"] == 2_000_000 * 2
    )
    result["namespace_clean"] = not (
        result["attempt"] or result["result"] or result["failure"]
    )

    print("CPW_V3_NAMESPACE_INSPECT=" + json.dumps(result, sort_keys=True), flush=True)
    return result


@app.local_entrypoint()
def main() -> None:
    result = inspect.remote()
    print("CPW_V3_NAMESPACE_INSPECT=" + json.dumps(result, sort_keys=True), flush=True)
