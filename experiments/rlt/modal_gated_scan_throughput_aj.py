from __future__ import annotations

import base64
import json
import math
import time
from typing import Any

import modal

JOB_ID = "rlt-systems-gated-scan-throughput-modal-20261008-aj"
ENGINEERING_SEED = 20_261_052
BATCH_SEED = 20_271_052
COMPILE_PROBE_SEED = 20_301_052
BATCH_SIZE = 64
SEQ_LEN = 64
BANK_SIZE = 8
TIME_BUDGET_SECONDS = 12.0
EXPECTED_PARAMETERS = 15_129_344
VARIANTS = ("lightstate_full", "gated_scan", "transformer_control")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.10,<2.11", "numpy>=2.0,<3")
    .add_local_python_source("tam_research", "experiments")
)
app = modal.App("tam-rlt-systems-gated-scan-throughput-20261008-aj")


def _encode(value: dict[str, Any]) -> str:
    return base64.urlsafe_b64encode(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).decode()


def _configs():
    from experiments.rlt.model import RLTConfig
    from tam_research.models import ModelConfig

    return (
        RLTConfig(
            vocab_size=50_257, d_model=256, n_heads=8, n_stages=2,
            max_seq_len=128, ff_mult=4, swa_window=32,
        ),
        ModelConfig(
            vocab_size=50_257, d_model=256, n_layers=3, n_heads=8,
            max_seq_len=128, ff_mult=4, ff_inner=938,
            architecture="transformer",
        ),
    )


@app.function(image=image, cpu=2, timeout=300, retries=0, max_containers=1)
def cpu_preflight_remote() -> str:
    import numpy
    import torch

    from experiments.rlt.model import parameter_count
    from experiments.rlt.model_gated_scan import (
        GatedScanLightStateRLT, associative_affine_scan, sequential_affine_reference,
    )
    from experiments.rlt.model_light_state import LightStateRecurrentTransformer
    from tam_research.models import ResearchLM
    from tam_research.train import seed_all

    torch.set_num_threads(2)
    rlt_cfg, transformer_cfg = _configs()
    parameter_counts = {}
    for name, cls, cfg in [
        ("lightstate_full", LightStateRecurrentTransformer, rlt_cfg),
        ("gated_scan", GatedScanLightStateRLT, rlt_cfg),
        ("transformer_control", ResearchLM, transformer_cfg),
    ]:
        seed_all(ENGINEERING_SEED)
        model = cls(cfg)
        count = parameter_count(model)
        if count != EXPECTED_PARAMETERS:
            raise RuntimeError(f"{name} parameter mismatch: {count}")
        parameter_counts[name] = count
        x = torch.tensor([[5, 6, 7]], dtype=torch.long)
        with torch.no_grad():
            y = model(x)
        if y.shape != (1, 3, 50_257) or not torch.isfinite(y).all():
            raise RuntimeError(f"{name} CPU forward failure")

    gen = torch.Generator().manual_seed(COMPILE_PROBE_SEED)
    a = torch.sigmoid(torch.randn(2, 17, 8, generator=gen))
    b = torch.randn(2, 17, 8, generator=gen) * 0.1
    initial = torch.randn(8, generator=gen) * 0.1
    torch.testing.assert_close(
        associative_affine_scan(a, b, initial),
        sequential_affine_reference(a, b, initial),
        atol=2e-6, rtol=2e-6,
    )
    return _encode({
        "status": "pass", "gpu_requested": False,
        "cpu_forward_shapes_passed": True,
        "scan_vs_serial_passed": True,
        "parameter_counts": parameter_counts,
        "numpy_version": str(numpy.__version__),
        "torch_version": str(torch.__version__),
    })


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=4.0,
    memory=65536,
    timeout=80 * 60,
    retries=0,
    max_containers=1,
)
def run_profile_remote(job_json: str) -> str:
    import torch
    import torch.nn.functional as F

    from experiments.rlt.model import parameter_count
    from experiments.rlt.model_gated_scan import (
        GatedScanLightStateRLT, associative_affine_scan, sequential_affine_reference,
    )
    from experiments.rlt.model_light_state import LightStateRecurrentTransformer
    from tam_research.models import ResearchLM
    from tam_research.train import seed_all

    job = json.loads(job_json)
    expected = {
        "schema": 1,
        "job_id": JOB_ID,
        "task": "systems_gated_scan_throughput_aj",
        "engineering_seed": ENGINEERING_SEED,
        "batch_seed": BATCH_SEED,
        "compile_probe_seed": COMPILE_PROBE_SEED,
        "requires_attribution_job_id": "rlt-systems-lightstate-component-attribution-modal-20261008-ai",
        "requires_gated_scan_cpu_workflow_run_id": 37750663298,
        "variants": list(VARIANTS),
        "batch_size": BATCH_SIZE,
        "seq_len": SEQ_LEN,
        "synthetic_bank_size": BANK_SIZE,
        "time_budget_seconds_each": TIME_BUDGET_SECONDS,
        "expected_parameters_each": EXPECTED_PARAMETERS,
        "automatic_retry": False,
    }
    if job != expected:
        raise RuntimeError(f"AJ job preregistration mismatch: {job}")
    if not torch.cuda.is_available():
        raise RuntimeError("AJ engineering attribution requires a CUDA GPU")
    torch.set_float32_matmul_precision("high")
    start = time.time()
    device = torch.device("cuda")
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    rlt_cfg, transformer_cfg = _configs()

    # Gated scan identity is verified independently of the language model.
    gen = torch.Generator(device="cuda").manual_seed(COMPILE_PROBE_SEED)
    gates = torch.sigmoid(torch.randn(2, 17, 8, generator=gen, device=device))
    writes = torch.randn(2, 17, 8, generator=gen, device=device) * 0.1
    initial = torch.randn(8, generator=gen, device=device) * 0.1
    torch.testing.assert_close(
        associative_affine_scan(gates, writes, initial),
        sequential_affine_reference(gates, writes, initial),
        atol=2e-6, rtol=2e-6,
    )

    generator = torch.Generator(device="cuda").manual_seed(BATCH_SEED)
    bank = [
        (
            torch.randint(0, 50_257, (BATCH_SIZE, SEQ_LEN), generator=generator, device=device),
            torch.randint(0, 50_257, (BATCH_SIZE, SEQ_LEN), generator=generator, device=device),
        )
        for _ in range(BANK_SIZE)
    ]

    model_classes = {
        "lightstate_full": (LightStateRecurrentTransformer, rlt_cfg),
        "gated_scan": (GatedScanLightStateRLT, rlt_cfg),
        "transformer_control": (ResearchLM, transformer_cfg),
    }

    def measure(name: str) -> dict[str, Any]:
        seed_all(ENGINEERING_SEED)
        cls, cfg = model_classes[name]
        model = cls(cfg).to(device)
        actual_params = parameter_count(model)
        if actual_params != EXPECTED_PARAMETERS:
            raise RuntimeError(f"{name} parameters {actual_params} != {EXPECTED_PARAMETERS}")
        compiled = torch.compile(model, fullgraph=True, dynamic=False, mode="default")
        opt = torch.optim.AdamW(
            model.parameters(), lr=1e-3, betas=(0.9, 0.95),
            weight_decay=0.1, fused=True,
        )

        # One forward/backward compile probe, no optimizer update before the timer.
        x, y = bank[COMPILE_PROBE_SEED % BANK_SIZE]
        model.train()
        opt.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        c0 = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            logits = compiled(x)
            loss = F.cross_entropy(
                logits.float().reshape(-1, 50_257), y.reshape(-1)
            )
        loss.backward()
        torch.cuda.synchronize(device)
        compile_seconds = time.perf_counter() - c0
        compile_loss = float(loss.detach().cpu())
        model.zero_grad(set_to_none=True)

        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        steps = 0
        last_loss = float("nan")
        while True:
            if time.perf_counter() - t0 >= TIME_BUDGET_SECONDS and steps > 0:
                break
            x, y = bank[steps % BANK_SIZE]
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                logits = compiled(x)
                loss = F.cross_entropy(
                    logits.float().reshape(-1, 50_257), y.reshape(-1)
                )
            # Deliberately retain the identical synchronization protocol used in AI.
            last_loss = float(loss.detach().cpu())
            if not math.isfinite(last_loss):
                raise RuntimeError(f"nonfinite loss in {name}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            steps += 1

        torch.cuda.synchronize(device)
        seconds = time.perf_counter() - t0
        tokens = steps * BATCH_SIZE * SEQ_LEN
        peak_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
        result = {
            "name": name, "parameters": actual_params,
            "compile_probe_seconds": compile_seconds,
            "compile_probe_loss": compile_loss,
            "timed_seconds": seconds, "steps": steps, "tokens": tokens,
            "tokens_per_second": tokens / max(seconds, 1e-9),
            "optimizer_steps_per_second": steps / max(seconds, 1e-9),
            "peak_vram_gb": peak_gb, "last_loss": last_loss,
        }
        del compiled, opt, model
        torch.cuda.empty_cache()
        torch._dynamo.reset()
        return result

    results = [measure(name) for name in VARIANTS]
    by_name = {row["name"]: row for row in results}
    full = by_name["lightstate_full"]["tokens_per_second"]
    scan = by_name["gated_scan"]["tokens_per_second"]
    transformer = by_name["transformer_control"]["tokens_per_second"]
    return _encode({
        "schema": 1, "job_id": JOB_ID, "status": "complete",
        "classification": "PASS_GATED_SCAN_ENGINEERING_THROUGHPUT_ATTRIBUTION",
        "engineering_only": True, "scientific_execution": False,
        "breakthrough_claim_supported": False,
        "engineering_seed": ENGINEERING_SEED,
        "batch_seed": BATCH_SEED,
        "compile_probe_seed": COMPILE_PROBE_SEED,
        "started_unix": start, "finished_unix": time.time(),
        "runtime": {
            "torch_version": str(torch.__version__),
            "cuda_runtime": str(torch.version.cuda),
            "gpu_name": str(torch.cuda.get_device_name(0)),
            "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        },
        "protocol": {
            "variants": list(VARIANTS),
            "batch_size": BATCH_SIZE, "seq_len": SEQ_LEN,
            "synthetic_bank_size": BANK_SIZE,
            "time_budget_seconds_each": TIME_BUDGET_SECONDS,
            "compile_time_excluded": True,
            "same_gpu_same_run": True, "same_engineering_seed": True,
            "same_synthetic_token_bank": True,
            "optimizer": "AdamW(lr=1e-3,betas=(0.9,0.95),weight_decay=0.1,fused=True)",
            "gradient_clip_norm": 1.0, "timing_only_no_quality_claim": True,
            "model_architecture_differs_intentionally": True,
        },
        "gpu_scan_vs_serial_passed": True,
        "results": results,
        "derived": {
            "gated_scan_over_lightstate_tps": scan / max(full, 1e-9),
            "gated_scan_over_transformer_tps": scan / max(transformer, 1e-9),
            "transformer_over_lightstate_tps": transformer / max(full, 1e-9),
        },
        "interpretation_ceiling": (
            "Single-device, engineering-only, synthetic-token throughput screening. "
            "The gated scan changes the recurrence equation while reusing every parameter. "
            "Neither lower synthetic loss nor higher throughput establishes real data quality. "
            "A positive result only justifies a separately preregistered real-data comparison."
        ),
    })


@app.local_entrypoint()
def verify_dependencies() -> None:
    result = cpu_preflight_remote.remote()
    if not isinstance(result, str):
        raise TypeError("AJ CPU smoke test must return base64 text")
    print("RLT_SYSTEMS_GATED_SCAN_AJ_PREFLIGHT_B64=" + result)


@app.local_entrypoint()
def main(job_path: str) -> None:
    from pathlib import Path

    result = run_profile_remote.remote(Path(job_path).read_text())
    if not isinstance(result, str):
        raise TypeError("AJ remote result must return base64 text")
    print("RLT_SYSTEMS_GATED_SCAN_AJ_RESULT_B64=" + result)
