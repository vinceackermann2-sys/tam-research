from __future__ import annotations

"""Zero-GPU Modal runtime-admission probe (#1067).

A successful return proves the selected workspace can admit and execute a
minimal remote CPU function while seeing the already-existing research volume.
The probe is read-only and intentionally carries no research payload.
"""

import json
from pathlib import Path

import modal


APP_NAME = "tam-modal-runtime-admission-probe-v1"
VOLUME_NAME = "tam-research-data"
MARKER = "TAM_MODAL_RUNTIME_ADMISSION_PROBE="

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
    payload = {
        "status": "PASS",
        "classification": "TAM_MODAL_RUNTIME_ADMISSION_CPU_ONLY_PASS",
        "volume_mount_visible": Path("/vol").is_dir(),
        "gpu_allocated": False,
        "writes_performed": False,
    }
    if not payload["volume_mount_visible"]:
        raise RuntimeError("required Modal volume mount is not visible")
    return json.dumps(payload, sort_keys=True)


@app.local_entrypoint()
def main() -> None:
    payload = runtime_admission_probe.remote()
    print(f"{MARKER}{payload}")
