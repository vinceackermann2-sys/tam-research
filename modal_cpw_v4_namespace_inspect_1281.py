from __future__ import annotations

import json
from pathlib import Path

import modal

APP_NAME = "tam-research-cpw-v4-namespace-inspect-1281"
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

    result_root = Path("/vol/cpw-v4/delayed-associative-recall-v1")
    result = {
        "issue": 1281,
        "scientific_issue": 1275,
        "writes_performed": False,
        "gpu_requested": False,
        "scientific_seed_consumed": False,
        "result_root": str(result_root),
        "root_exists": result_root.exists(),
        "attempt": (result_root / "ATTEMPT.json").exists(),
        "result": (result_root / "RESULT.json").exists(),
        "failure": (result_root / "FAILURE.json").exists(),
    }

    for name in ("ATTEMPT.json", "RESULT.json", "FAILURE.json"):
        path = result_root / name
        if path.exists():
            try:
                result[name] = json.loads(path.read_text())
            except Exception as exc:
                result[name] = {"read_error": f"{type(exc).__name__}: {exc}"}

    result["namespace_clean"] = not (
        result["attempt"] or result["result"] or result["failure"]
    )

    print(
        "CPW_V4_NAMESPACE_INSPECT=" + json.dumps(result, sort_keys=True),
        flush=True,
    )
    return result


@app.local_entrypoint()
def main() -> None:
    result = inspect.remote()
    print(
        "CPW_V4_NAMESPACE_INSPECT=" + json.dumps(result, sort_keys=True),
        flush=True,
    )
