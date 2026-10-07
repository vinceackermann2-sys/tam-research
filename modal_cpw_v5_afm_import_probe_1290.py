from __future__ import annotations

import json

import modal

APP_NAME = "tam-research-cpw-v5-afm-import-probe-1290"

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
    cpu=2,
    memory=4096,
    timeout=180,
    retries=0,
)
def probe() -> dict:
    import torch

    import architectures.cortex_s.affine_scan_triton_candidate  # noqa: F401
    from tam_research.cpw_v4_memory.task import make_associative_batch
    from tam_research.cpw_v5_afm.model import AFMCPWResearchLM, afm_parameter_count
    from tam_research.cpw_v5_afm.protocol import EXPECTED_AFM_PARAMETERS
    from tam_research.cpw_v5_afm.train import query_logits

    model = AFMCPWResearchLM().eval()
    parameters = afm_parameter_count()
    if parameters != EXPECTED_AFM_PARAMETERS:
        raise RuntimeError(
            f"AFM parameter mismatch: {parameters} != {EXPECTED_AFM_PARAMETERS}"
        )

    generator = torch.Generator(device="cpu").manual_seed(1_290_999)
    batch = make_associative_batch(
        batch_size=1,
        generator=generator,
        delay=32,
    )
    with torch.no_grad():
        logits = query_logits(model, batch.tokens)
    if logits.shape != (1, model.cfg.vocab_size):
        raise RuntimeError(f"unexpected query logits shape: {tuple(logits.shape)}")
    if not torch.isfinite(logits).all():
        raise RuntimeError("non-finite CPU import-probe logits")

    result = {
        "issue": 1290,
        "imports_ok": True,
        "generator_ok": True,
        "query_forward_ok": True,
        "parameter_contract_ok": True,
        "parameters": parameters,
        "gpu_requested": False,
        "volume_mounted": False,
        "scientific_seed_consumed": False,
        "writes_performed": False,
    }
    print("CPW_V5_AFM_IMPORT_PROBE=" + json.dumps(result, sort_keys=True), flush=True)
    return result


@app.local_entrypoint()
def main() -> None:
    result = probe.remote()
    print("CPW_V5_AFM_IMPORT_PROBE=" + json.dumps(result, sort_keys=True), flush=True)
