from __future__ import annotations

import json

import modal

APP_NAME = "tam-research-aera-v23-sparse-systems"
MAX_GPU_SECONDS = 600

app = modal.App(APP_NAME)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.10,<2.11", "numpy>=2.0,<3")
    .add_local_python_source("tam_research")
    .add_local_file(
        "aera_v23_sparse_systems_l4.py",
        "/root/aera_v23_sparse_systems_l4.py",
    )
)


@app.function(image=image, cpu=4, memory=16384, timeout=300)
def preflight() -> dict:
    from tam_research.aera_real_language_v23_efficiency import cpu_preflight

    result = cpu_preflight()
    print(
        "AERA_V23_SPARSE_SYSTEMS_PREFLIGHT_JSON="
        + json.dumps(result, separators=(",", ":")),
        flush=True,
    )
    return result


@app.function(
    image=image,
    gpu="L4",
    cpu=8,
    memory=32768,
    timeout=MAX_GPU_SECONDS,
)
def run_benchmark() -> dict:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "aera_v23_sparse_systems_l4",
        "/root/aera_v23_sparse_systems_l4.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load v23 sparse systems benchmark")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.run_l4_benchmark()
    print(
        "AERA_V23_SPARSE_SYSTEMS_L4_RESULT_JSON="
        + json.dumps(result, separators=(",", ":")),
        flush=True,
    )
    return result


@app.local_entrypoint()
def main() -> None:
    check = preflight.remote()
    print(
        "AERA_V23_SPARSE_SYSTEMS_PREFLIGHT_LOCAL="
        + json.dumps(check, separators=(",", ":")),
        flush=True,
    )
    result = run_benchmark.remote()
    print(json.dumps(result, indent=2), flush=True)
