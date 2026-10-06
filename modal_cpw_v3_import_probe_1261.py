from __future__ import annotations

import json

import modal

APP_NAME = "tam-research-cpw-v3-import-probe-1261"

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


@app.function(
    image=image,
    cpu=1,
    memory=2048,
    timeout=5 * 60,
    retries=0,
)
def probe() -> dict:
    import architectures.cortex_s.affine_scan_triton_candidate
    from tam_research.cpw_v3_sparse_world.model import (
        ARM_WORLD_LAYERS,
        SparseWorldCPWResearchLM,
        sparse_world_parameter_count,
    )
    from tam_research.cpw_v3_sparse_world.protocol import CPW_V3_PARAMETERS

    params = {
        arm: sparse_world_parameter_count(arm)
        for arm in ARM_WORLD_LAYERS
    }
    layouts = {}
    for arm in ARM_WORLD_LAYERS:
        model = SparseWorldCPWResearchLM(arm)
        layouts[arm] = [
            i for i, block in enumerate(model.blocks)
            if block.mixer.world is not None
        ]
    result = {
        "issue": 1261,
        "imports_ok": True,
        "gpu_requested": False,
        "volume_mounted": False,
        "writes_performed": False,
        "scientific_seed_consumed": False,
        "expected_parameters": CPW_V3_PARAMETERS,
        "parameters": params,
        "world_layer_layouts": layouts,
        "loaded": [
            "architectures.cortex_s.affine_scan_triton_candidate",
            "tam_research.cpw_v3_sparse_world.model",
            "tam_research.cpw_v3_sparse_world.protocol",
        ],
    }
    if any(value != CPW_V3_PARAMETERS for value in params.values()):
        raise RuntimeError(f"CPW-v3 parameter drift: {params}")
    print("CPW_V3_IMPORT_PROBE=" + json.dumps(result, sort_keys=True), flush=True)
    return result


@app.local_entrypoint()
def main() -> None:
    result = probe.remote()
    print("CPW_V3_IMPORT_PROBE=" + json.dumps(result, sort_keys=True), flush=True)
