from __future__ import annotations

"""Zero-GPU Modal runtime-admission probe for #1067.

This exists only to prove a workspace can actually admit a remote function
under its current spend-limit/runtime state. It performs no writes.
"""

import json
import modal

APP_NAME = "tam-research-runtime-admission-probe-1067-v1"
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
def runtime_admission_probe() -> str:
    return json.dumps(
        {
            "runtime_admission_ok": True,
            "gpu_allocated": False,
            "writes_performed": False,
            "volume_name": VOLUME_NAME,
        },
        sort_keys=True,
    )


@app.local_entrypoint()
def main() -> None:
    payload = runtime_admission_probe.remote()
    print(f"MODAL_RUNTIME_ADMISSION={payload}")
