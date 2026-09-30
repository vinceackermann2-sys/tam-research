from __future__ import annotations

import base64
import json
from pathlib import Path
import time
from typing import Any

import modal

JOB_ID = "rlt-systems-onestage-scale15m-batch768-modal-20260930-ac"
ENGINEERING_SEED = 20_261_040
SEQ_LEN = 64
BATCH_SIZE = 768
CHECK_BATCH = 2
CAUSAL_MUTATION_INDEX = 37
WARMUP_STEPS = 1
MEASURED_STEPS = 2
EXPECTED_PARAMETERS = 15_129_344

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.10,<2.11", "numpy>=2.0,<3")
    .add_local_python_source("tam_research", "experiments")
)
app = modal.App("tam-rlt-systems-onestage-scale15m-batch768-20260930-ac")


def _encode(value: dict[str, Any]) -> str:
    return base64.urlsafe_b64encode(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).decode()


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=4.0,
    memory=65536,
    timeout=60 * 60,
    retries=0,
    max_containers=1,
)
def run_profile_remote(job_json: str) -> str:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from experiments.rlt.model import RLTConfig, parameter_count
    from experiments.rlt.model_one_stage import (
        ONE_STAGE_FF_INNER,
        OneStageRecurrentLoopedTransformer,
    )
    from tam_research.models import (
        ModelConfig,
        ResearchLM,
        parameter_count as transformer_parameter_count,
    )

    started = time.time()
    job = json.loads(job_json)
    expected = {
        "schema": 1,
        "job_id": JOB_ID,
        "task": "systems_onestage_scale15m_batch768_ac",
        "engineering_seed": ENGINEERING_SEED,
        "requires_reference_pair_job_id": "rlt-compiled-paired-4m-modal-20260923-d",
        "requires_block4_replication_job_id": "rlt-block4-scale15m-paired-4m-modal-20260930-b",
        "requires_reference_batch768_job_id": "rlt-systems-scale15m-batch768-modal-20260926-n",
        "batch_size": BATCH_SIZE,
        "seq_len": SEQ_LEN,
        "expected_parameters_each": EXPECTED_PARAMETERS,
        "automatic_retry": False,
    }
    if job != expected:
        raise RuntimeError(f"one-stage profile preregistration mismatch: {job}")
    if not torch.cuda.is_available():
        raise RuntimeError("one-stage systems profile requires CUDA")

    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    rlt_cfg = RLTConfig(
        vocab_size=50_257, d_model=256, n_heads=8, n_stages=1,
        max_seq_len=128, ff_mult=4, swa_window=32,
    )
    tr_cfg = ModelConfig(
        vocab_size=50_257, d_model=256, n_layers=3, n_heads=8,
        max_seq_len=128, ff_mult=4, ff_inner=938, architecture="transformer",
    )

    def seed_all(seed: int) -> None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def new_rlt() -> OneStageRecurrentLoopedTransformer:
        seed_all(ENGINEERING_SEED)
        m = OneStageRecurrentLoopedTransformer(rlt_cfg).to(device)
        n = parameter_count(m)
        if n != EXPECTED_PARAMETERS:
            raise RuntimeError(f"one-stage parameter mismatch: {n}")
        return m

    def new_transformer() -> ResearchLM:
        seed_all(ENGINEERING_SEED)
        m = ResearchLM(tr_cfg).to(device)
        n = transformer_parameter_count(m)
        if n != EXPECTED_PARAMETERS:
            raise RuntimeError(f"Transformer parameter mismatch: {n}")
        return m

    # Causality + active recurrent gate check.
    rlt_check = new_rlt()
    gen = torch.Generator(device="cpu").manual_seed(ENGINEERING_SEED + 91)
    x = torch.randint(
        0, rlt_cfg.vocab_size, (CHECK_BATCH, SEQ_LEN),
        generator=gen, dtype=torch.long,
    ).to(device)
    y = torch.randint(
        0, rlt_cfg.vocab_size, (CHECK_BATCH, SEQ_LEN),
        generator=gen, dtype=torch.long,
    ).to(device)
    altered = x.clone()
    altered[:, CAUSAL_MUTATION_INDEX] = (
        altered[:, CAUSAL_MUTATION_INDEX] + 1
    ) % rlt_cfg.vocab_size

    rlt_check.eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        a = rlt_check(x)
        b = rlt_check(altered)
    torch.cuda.synchronize(device)
    prefix_abs = float(
        (a[:, :CAUSAL_MUTATION_INDEX] - b[:, :CAUSAL_MUTATION_INDEX])
        .abs().max().item()
    )
    changed_abs = float(
        (a[:, CAUSAL_MUTATION_INDEX] - b[:, CAUSAL_MUTATION_INDEX])
        .abs().max().item()
    )
    causality_passed = prefix_abs == 0.0 and changed_abs > 0.0
    if not causality_passed:
        raise RuntimeError(
            f"one-stage causality failed: prefix={prefix_abs}, changed={changed_abs}"
        )

    rlt_check.train()
    rlt_check.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = rlt_check(x)
        loss = F.cross_entropy(
            logits.float().reshape(-1, rlt_cfg.vocab_size), y.reshape(-1)
        )
    loss.backward()
    torch.cuda.synchronize(device)
    gate_grad = rlt_check.state_gate.grad
    gate_grad_max_abs = 0.0 if gate_grad is None else float(gate_grad.abs().max().item())
    gate_grad_l1 = 0.0 if gate_grad is None else float(gate_grad.abs().sum().item())
    gate_active = gate_grad is not None and gate_grad_l1 > 0.0
    if not gate_active:
        raise RuntimeError("one-stage state_gate has no gradient")
    state_schema = tuple(rlt_check.state_dict().keys())
    del rlt_check, logits, a, b
    torch.cuda.empty_cache()

    gen = torch.Generator(device="cpu").manual_seed(ENGINEERING_SEED + 1)
    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(WARMUP_STEPS + MEASURED_STEPS + 2):
        bx = torch.randint(
            0, rlt_cfg.vocab_size, (BATCH_SIZE, SEQ_LEN),
            generator=gen, dtype=torch.long,
        ).to(device)
        by = torch.randint(
            0, rlt_cfg.vocab_size, (BATCH_SIZE, SEQ_LEN),
            generator=gen, dtype=torch.long,
        ).to(device)
        batches.append((bx, by))

    def one_step(
        callable_model: Any,
        model: nn.Module,
        opt: torch.optim.Optimizer,
        offset: int,
    ) -> float:
        model.train()
        opt.zero_grad(set_to_none=True)
        bx, by = batches[offset % len(batches)]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = callable_model(bx)
            l = F.cross_entropy(out.float().reshape(-1, 50_257), by.reshape(-1))
        l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        return float(l.detach().cpu())

    def benchmark(kind: str) -> dict[str, Any]:
        if kind == "rlt":
            base: nn.Module = new_rlt()
        elif kind == "transformer":
            base = new_transformer()
        else:
            raise ValueError(kind)
        opt = torch.optim.AdamW(
            base.parameters(), lr=3e-4, betas=(0.9, 0.95),
            weight_decay=0.1, fused=True,
        )
        compiled = torch.compile(base, fullgraph=True, dynamic=False, mode="default")
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        first_loss = one_step(compiled, base, opt, 0)
        torch.cuda.synchronize(device)
        first_seconds = time.perf_counter() - t0
        for i in range(WARMUP_STEPS):
            one_step(compiled, base, opt, 1+i)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        t1 = time.perf_counter()
        last_loss = first_loss
        for i in range(MEASURED_STEPS):
            last_loss = one_step(
                compiled, base, opt, 1 + WARMUP_STEPS + i
            )
        torch.cuda.synchronize(device)
        measured = time.perf_counter() - t1
        tokens = MEASURED_STEPS * BATCH_SIZE * SEQ_LEN
        out = {
            "parameters": sum(p.numel() for p in base.parameters()),
            "first_step_seconds_including_compile": first_seconds,
            "measured_seconds": measured,
            "measured_tokens": tokens,
            "tokens_per_second": tokens / max(measured, 1e-9),
            "last_loss": last_loss,
            "peak_vram_gb": torch.cuda.max_memory_allocated(device)/(1024**3),
        }
        del compiled, opt, base
        torch.cuda.empty_cache()
        torch._dynamo.reset()
        return out

    rlt = benchmark("rlt")
    transformer = benchmark("transformer")
    gap = transformer["tokens_per_second"] / max(rlt["tokens_per_second"], 1e-9)
    if gap <= 1.5:
        classification = "PASS_ONESTAGE_SCALE15M_GAP_LE_1_5X"
    elif gap <= 2.0:
        classification = "PASS_ONESTAGE_SCALE15M_GAP_LE_2X"
    else:
        classification = "PASS_ONESTAGE_SCALE15M_GAP_GT_2X"

    return _encode({
        "schema": 1,
        "job_id": JOB_ID,
        "status": "complete",
        "classification": classification,
        "engineering_only": True,
        "scientific_execution": False,
        "breakthrough_claim_supported": False,
        "engineering_seed": ENGINEERING_SEED,
        "started_unix": started,
        "finished_unix": time.time(),
        "runtime": {
            "torch_version": str(torch.__version__),
            "cuda_runtime": None if torch.version.cuda is None else str(torch.version.cuda),
            "gpu_name": str(torch.cuda.get_device_name(0)),
            "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        },
        "architecture": {
            "variant": "one_stage_token_recurrent_rlt",
            "d_model": 256,
            "n_heads": 8,
            "n_stages": 1,
            "ff_inner": ONE_STAGE_FF_INNER,
            "state_gate_parameters": 256,
            "parameters": EXPECTED_PARAMETERS,
            "token_recurrences_per_seq64": 64,
            "decoder_stage_applications_per_seq64": 64,
            "frozen_two_stage_decoder_stage_applications_per_seq64": 128,
            "state_schema_parameter_count": len(state_schema),
        },
        "validation": {
            "future_token_causality": {
                "passed": causality_passed,
                "mutation_index": CAUSAL_MUTATION_INDEX,
                "prefix_logits_max_abs": prefix_abs,
                "mutated_position_logits_max_abs": changed_abs,
            },
            "state_gate_active": {
                "passed": gate_active,
                "grad_max_abs": gate_grad_max_abs,
                "grad_l1": gate_grad_l1,
            },
            "exact_parameter_match": (
                rlt["parameters"] == EXPECTED_PARAMETERS
                and transformer["parameters"] == EXPECTED_PARAMETERS
            ),
        },
        "profile": {
            "batch_size": BATCH_SIZE,
            "seq_len": SEQ_LEN,
            "warmup_steps": WARMUP_STEPS,
            "measured_steps": MEASURED_STEPS,
        },
        "rlt": rlt,
        "transformer": transformer,
        "derived": {
            "transformer_throughput_multiple_vs_rlt": gap,
            "rlt_fraction_of_transformer_throughput": 1.0 / gap,
        },
        "interpretation_ceiling": (
            "Engineering-only throughput/causality profile for an exact-parameter "
            "one-stage token-recurrent RLT. Random-token losses are not quality evidence. "
            "A matched equal-token training run is required before any scientific conclusion."
        ),
    })


@app.local_entrypoint()
def main(job_path: str) -> None:
    result = run_profile_remote.remote(Path(job_path).read_text())
    if not isinstance(result, str):
        raise TypeError("remote one-stage profile result must be base64 text")
    print("RLT_SYSTEMS_ONESTAGE_SCALE15M_BATCH768_AC_RESULT_B64=" + result)
