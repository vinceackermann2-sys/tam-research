from __future__ import annotations

import json
import modal

APP_NAME = "tam-research-cpw-v4-import-probe-1275"
app = modal.App(APP_NAME)
image = (
    modal.Image.debian_slim(python_version='3.11')
    .pip_install(
        'torch>=2.7,<2.11',
        'numpy>=2.0,<3',
    )
    .add_local_python_source('tam_research')
    .add_local_python_source('architectures')
)

@app.function(image=image, cpu=1.0, memory=4096, timeout=300, retries=0)
def probe() -> dict:
    import torch
    import architectures.cortex_s.affine_scan_triton_candidate
    from tam_research.cpw_v4_memory.protocol import (
        CPW_PARAMETERS,
        ISSUE,
        TRANSFORMER_PARAMETERS,
    )
    from tam_research.cpw_v4_memory.task import make_associative_batch
    from tam_research.cpw_v4_memory.train import build_memory_model
    from tam_research.models import parameter_count

    batch = make_associative_batch(
        batch_size=4,
        generator=torch.Generator().manual_seed(1275),
        delay=64,
    )
    params = {
        arm: parameter_count(build_memory_model(arm))
        for arm in ('transformer', 'sequence_only', 'world_last1')
    }
    result = {
        'issue': ISSUE,
        'imports_ok': True,
        'generator_ok': bool(
            torch.all(batch.query_position - batch.source_value_positions == 64)
        ),
        'parameters': params,
        'parameter_contract_ok': (
            params['transformer'] == TRANSFORMER_PARAMETERS
            and params['sequence_only'] == CPW_PARAMETERS
            and params['world_last1'] == CPW_PARAMETERS
        ),
        'gpu_requested': False,
        'volume_mounted': False,
        'scientific_seed_consumed': False,
        'writes_performed': False,
    }
    print('CPW_V4_IMPORT_PROBE=' + json.dumps(result, sort_keys=True), flush=True)
    return result

@app.local_entrypoint()
def main() -> None:
    result = probe.remote()
    print('CPW_V4_IMPORT_PROBE=' + json.dumps(result, sort_keys=True), flush=True)
