from __future__ import annotations

import importlib
import json

import modal

APP_NAME = "tam-research-cpw-v2-sequence-import-probe-1257"

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.7,<2.11",
        "numpy>=2.0,<3",
    )
    .add_local_python_source("tam_research")
    .add_local_python_source("architectures")
)

IMPORTS = (
    "architectures.cortex_s.affine_scan_triton_candidate",
    "tam_research.cpw_v2_sequence.protocol",
    "tam_research.cpw_v2_sequence.train",
)


@app.function(
    image=image,
    cpu=1,
    memory=2048,
    timeout=5 * 60,
)
def probe() -> dict[str, object]:
    loaded: list[str] = []
    for module_name in IMPORTS:
        importlib.import_module(module_name)
        loaded.append(module_name)

    result = {
        "issue": 1257,
        "imports_ok": True,
        "loaded": loaded,
        "gpu_requested": False,
        "volume_mounted": False,
        "scientific_seed_consumed": False,
        "writes_performed": False,
    }
    print(
        "CPW_V2_SEQUENCE_IMPORT_PROBE="
        + json.dumps(result, sort_keys=True),
        flush=True,
    )
    return result


@app.local_entrypoint()
def main() -> None:
    result = probe.remote()
    print(
        "CPW_V2_SEQUENCE_IMPORT_PROBE="
        + json.dumps(result, sort_keys=True),
        flush=True,
    )
