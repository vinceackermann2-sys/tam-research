from __future__ import annotations

import json
from pathlib import Path

import modal

APP_NAME = "tam-research-cpw-v2-sequence-inspect-1254"
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
    root = Path("/vol/cpw-v2/sequence-core-long-horizon-v1")
    data_meta = Path("/vol/data/fineweb-edu-gpt2/meta.json")
    result = {
        "writes_performed": False,
        "root_exists": root.exists(),
        "attempt": (root / "ATTEMPT.json").exists(),
        "result": (root / "RESULT.json").exists(),
        "failure": (root / "FAILURE.json").exists(),
        "data_meta_exists": data_meta.exists(),
        "data_meta": None,
    }
    if data_meta.exists():
        try:
            result["data_meta"] = json.loads(data_meta.read_text())
        except Exception as exc:
            result["data_meta_error"] = f"{type(exc).__name__}: {exc}"
    for name in ("ATTEMPT.json", "RESULT.json", "FAILURE.json"):
        p = root / name
        if p.exists():
            try:
                result[name] = json.loads(p.read_text())
            except Exception as exc:
                result[name] = {"read_error": f"{type(exc).__name__}: {exc}"}
    print("CPW_V2_SEQUENCE_INSPECT=" + json.dumps(result, sort_keys=True), flush=True)
    return result

@app.local_entrypoint()
def main() -> None:
    result = inspect.remote()
    print("CPW_V2_SEQUENCE_INSPECT=" + json.dumps(result, sort_keys=True), flush=True)
