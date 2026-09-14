from __future__ import annotations

import base64
import json
import sys
import time
from typing import Any

import modal

# This app is intentionally standalone. It does not import or register the
# heavyweight RLT compute app and cannot request a GPU.
image = modal.Image.debian_slim(python_version="3.11")
app = modal.App("tam-rlt-publicspec-preflight")


def _encode(value: dict[str, Any]) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).decode()


@app.function(
    image=image,
    timeout=5 * 60,
    cpu=1.0,
    memory=512,
    retries=0,
    max_containers=1,
)
def preflight_remote() -> dict[str, Any]:
    return {
        "schema": 1,
        "status": "pass",
        "worker": "modal-control",
        "finished_unix": time.time(),
        "scientific_job_executed": False,
        "python_version": sys.version,
    }


@app.local_entrypoint()
def main() -> None:
    print("RLT_MODAL_PREFLIGHT_B64=" + _encode(preflight_remote.remote()))
