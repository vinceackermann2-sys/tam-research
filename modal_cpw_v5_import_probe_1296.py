from __future__ import annotations

import json
import modal

APP_NAME = "tam-research-cpw-v5-import-probe-1296"
app = modal.App(APP_NAME)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.7,<2.11", "numpy>=2.0,<3")
    .add_local_python_source("tam_research")
    .add_local_python_source("architectures")
)


@app.function(image=image, cpu=1, memory=4096, timeout=300, retries=0)
def probe() -> dict:
    import torch
    import architectures.cortex_s.affine_scan_triton_candidate
    from tam_research.cpw_v4_memory.task import make_associative_batch
    from tam_research.cpw_v5_afm.train import build_model
    from tam_research.cpw_v5_afm.protocol import (
        AFM_PARAMETERS,
        ARMS,
        ISSUE,
        SEQUENCE_PARAMETERS,
        TRANSFORMER_PARAMETERS,
    )
    from tam_research.models import parameter_count

    batch = make_associative_batch(
        batch_size=4,
        generator=torch.Generator().manual_seed(1296),
        delay=64,
    )
    params = {arm: parameter_count(build_model(arm)) for arm in ARMS}
    expected = {
        "transformer": TRANSFORMER_PARAMETERS,
        "sequence_only": SEQUENCE_PARAMETERS,
        "afm_first1": AFM_PARAMETERS,
    }

    result = {
        "issue": ISSUE,
        "imports_ok": True,
        "task_generator_ok": bool(
            torch.all(batch.query_position - batch.source_value_positions == 64)
        ),
        "parameters": params,
        "parameter_contract_ok": params == expected,
        "gpu_requested": False,
        "volume_mounted": False,
        "scientific_seed_consumed": False,
        "writes_performed": False,
    }
    print("CPW_V5_IMPORT_PROBE=" + json.dumps(result, sort_keys=True), flush=True)
    return result


@app.local_entrypoint()
def main() -> None:
    result = probe.remote()
    print("CPW_V5_IMPORT_PROBE=" + json.dumps(result, sort_keys=True), flush=True)
