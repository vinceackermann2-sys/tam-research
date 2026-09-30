from __future__ import annotations

import base64
import json
from pathlib import Path
import time
from typing import Any

import modal

JOB_ID = "rlt-systems-carrier-scale15m-batch768-modal-20260930-ac"
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
app = modal.App("tam-rlt-systems-carrier-scale15m-batch768-20260930-ac")


def _encode(value: dict[str, Any]) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).decode()


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=4.0,
    memory=65536,
    timeout=75 * 60,
    retries=0,
    max_containers=1,
)
def run_profile_remote(job_json: str) -> str:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from experiments.rlt.model import RLTConfig, RecurrentLoopedTransformer, parameter_count
    from experiments.rlt.model_carrier import CarrierRecurrentLoopedTransformer
    from tam_research.models import ModelConfig, ResearchLM, parameter_count as transformer_parameter_count

    started = time.time()
    job = json.loads(job_json)
    expected = {
        "schema": 1,
        "job_id": JOB_ID,
        "task": "systems_carrier_scale15m_batch768_ac",
        "engineering_seed": ENGINEERING_SEED,
        "requires_block_sweep_job_id": "rlt-systems-block-sweep-scale15m-batch768-modal-20260929-ab",
        "requires_block4_a_job_id": "rlt-block4-scale15m-paired-4m-modal-20260929-a",
        "requires_block4_b_job_id": "rlt-block4-scale15m-paired-4m-modal-20260930-b",
        "batch_size": BATCH_SIZE,
        "seq_len": SEQ_LEN,
        "expected_parameters_each": EXPECTED_PARAMETERS,
        "automatic_retry": False,
    }
    if job != expected:
        raise RuntimeError(f"carrier profile preregistration mismatch: {job}")
    if not torch.cuda.is_available():
        raise RuntimeError("carrier systems profile requires CUDA")

    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")

    rlt_cfg = RLTConfig(
        vocab_size=50_257, d_model=256, n_heads=8, n_stages=2,
        max_seq_len=128, ff_mult=4, swa_window=32,
    )
    tr_cfg = ModelConfig(
        vocab_size=50_257, d_model=256, n_layers=3, n_heads=8,
        max_seq_len=128, ff_mult=4, ff_inner=938, architecture="transformer",
    )

    def seed_local(seed: int) -> None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def new_reference() -> RecurrentLoopedTransformer:
        seed_local(ENGINEERING_SEED)
        m = RecurrentLoopedTransformer(rlt_cfg).to(device)
        if parameter_count(m) != EXPECTED_PARAMETERS:
            raise RuntimeError(f"reference parameter drift: {parameter_count(m)}")
        return m

    def new_carrier() -> CarrierRecurrentLoopedTransformer:
        seed_local(ENGINEERING_SEED)
        m = CarrierRecurrentLoopedTransformer(rlt_cfg).to(device)
        if parameter_count(m) != EXPECTED_PARAMETERS:
            raise RuntimeError(f"carrier parameter drift: {parameter_count(m)}")
        return m

    def new_transformer() -> ResearchLM:
        seed_local(ENGINEERING_SEED)
        m = ResearchLM(tr_cfg).to(device)
        if transformer_parameter_count(m) != EXPECTED_PARAMETERS:
            raise RuntimeError(f"Transformer parameter drift: {transformer_parameter_count(m)}")
        return m

    reference = new_reference()
    carrier_check = new_carrier()
    carrier_check.load_state_dict(reference.state_dict(), strict=True)
    schema_equal = tuple(reference.state_dict().keys()) == tuple(carrier_check.state_dict().keys())
    if not schema_equal:
        raise RuntimeError("carrier state schema differs from frozen RLT")
    del reference
    torch.cuda.empty_cache()

    # Future-token causality guard.
    g = torch.Generator(device="cpu").manual_seed(ENGINEERING_SEED + 91)
    x = torch.randint(
        0, rlt_cfg.vocab_size, (CHECK_BATCH, SEQ_LEN),
        generator=g, dtype=torch.long,
    ).to(device)
    x2 = x.clone()
    x2[:, CAUSAL_MUTATION_INDEX] = (
        x2[:, CAUSAL_MUTATION_INDEX] + 1
    ) % rlt_cfg.vocab_size

    carrier_check.eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        a = carrier_check(x)
        b = carrier_check(x2)
    torch.cuda.synchronize(device)

    prefix_abs = float(
        (a[:, :CAUSAL_MUTATION_INDEX, :] - b[:, :CAUSAL_MUTATION_INDEX, :])
        .abs().max().item()
    )
    changed_abs = float(
        (a[:, CAUSAL_MUTATION_INDEX, :] - b[:, CAUSAL_MUTATION_INDEX, :])
        .abs().max().item()
    )
    downstream_abs = float(
        (a[:, CAUSAL_MUTATION_INDEX + 1 :, :] - b[:, CAUSAL_MUTATION_INDEX + 1 :, :])
        .abs().max().item()
    )
    causal_passed = prefix_abs == 0.0 and changed_abs > 0.0 and downstream_abs > 0.0
    if not causal_passed:
        raise RuntimeError(
            "carrier causality/state-propagation guard failed: "
            f"prefix={prefix_abs}, changed={changed_abs}, downstream={downstream_abs}"
        )
    del carrier_check, a, b
    torch.cuda.empty_cache()

    g = torch.Generator(device="cpu").manual_seed(ENGINEERING_SEED + 1)
    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(WARMUP_STEPS + MEASURED_STEPS + 2):
        xb = torch.randint(
            0, rlt_cfg.vocab_size, (BATCH_SIZE, SEQ_LEN),
            generator=g, dtype=torch.long,
        ).to(device)
        yb = torch.randint(
            0, rlt_cfg.vocab_size, (BATCH_SIZE, SEQ_LEN),
            generator=g, dtype=torch.long,
        ).to(device)
        batches.append((xb, yb))

    def one_step(
        callable_model: Any,
        model: nn.Module,
        opt: torch.optim.Optimizer,
        offset: int,
    ) -> float:
        model.train()
        opt.zero_grad(set_to_none=True)
        xb, yb = batches[offset % len(batches)]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = callable_model(xb)
            loss = F.cross_entropy(
                logits.float().reshape(-1, 50_257), yb.reshape(-1)
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        return float(loss.detach().cpu())

    def benchmark(kind: str) -> dict[str, Any]:
        if kind == "carrier":
            base: nn.Module = new_carrier()
        elif kind == "transformer":
            base = new_transformer()
        else:
            raise ValueError(kind)

        opt = torch.optim.AdamW(
            base.parameters(), lr=3e-4, betas=(0.9, 0.95),
            weight_decay=0.1, fused=True,
        )
        compiled = torch.compile(
            base, fullgraph=True, dynamic=False, mode="default"
        )
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        first_loss = one_step(compiled, base, opt, 0)
        torch.cuda.synchronize(device)
        compile_step = time.perf_counter() - t0

        for i in range(WARMUP_STEPS):
            one_step(compiled, base, opt, 1 + i)
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
            "first_step_seconds_including_compile": compile_step,
            "measured_seconds": measured,
            "measured_tokens": tokens,
            "tokens_per_second": tokens / max(measured, 1e-9),
            "last_loss": last_loss,
            "peak_vram_gb": (
                torch.cuda.max_memory_allocated(device) / (1024**3)
            ),
        }
        del compiled, opt, base
        torch.cuda.empty_cache()
        torch._dynamo.reset()
        return out

    carrier = benchmark("carrier")
    transformer = benchmark("transformer")
    gap = (
        float(transformer["tokens_per_second"])
        / max(float(carrier["tokens_per_second"]), 1e-9)
    )

    if gap <= 1.25:
        classification = "PASS_CARRIER_SCALE15M_GAP_LE_1_25X"
    elif gap <= 1.5:
        classification = "PASS_CARRIER_SCALE15M_GAP_LE_1_5X"
    elif gap <= 2.0:
        classification = "PASS_CARRIER_SCALE15M_GAP_LE_2X"
    else:
        classification = "PASS_CARRIER_SCALE15M_GAP_GT_2X"

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
            "variant": "token_carrier_recurrent_rlt",
            "token_recurrent_carrier_updates_per_seq64": SEQ_LEN,
            "expensive_decoder_stages_inside_recurrence": 0,
            "parallel_decoder_stages": rlt_cfg.n_stages,
            "parameters": EXPECTED_PARAMETERS,
            "parameter_neutral": True,
            "state_schema_equal_to_frozen_rlt": schema_equal,
        },
        "implementation_validation": {
            "future_token_causality_and_state_propagation": {
                "passed": causal_passed,
                "mutation_index": CAUSAL_MUTATION_INDEX,
                "prefix_logits_max_abs": prefix_abs,
                "mutated_position_logits_max_abs": changed_abs,
                "downstream_logits_max_abs": downstream_abs,
            },
        },
        "profile": {
            "batch_size": BATCH_SIZE,
            "seq_len": SEQ_LEN,
            "warmup_steps": WARMUP_STEPS,
            "measured_steps": MEASURED_STEPS,
            "expected_parameters_each": EXPECTED_PARAMETERS,
        },
        "carrier": carrier,
        "transformer": transformer,
        "derived": {
            "transformer_throughput_multiple_vs_carrier": gap,
            "carrier_fraction_of_transformer_throughput": 1.0 / gap,
        },
        "interpretation_ceiling": (
            "Engineering-only throughput/causality profile for a parameter-neutral "
            "token-carrier recurrent RLT variant. It preserves one recurrent carrier "
            "update per token while moving the expensive shared decoder stages out of "
            "the serial recurrence. Random-token losses are not quality evidence."
        ),
    })


@app.local_entrypoint()
def main(job_path: str) -> None:
    result = run_profile_remote.remote(Path(job_path).read_text())
    if not isinstance(result, str):
        raise TypeError("remote carrier systems result must be base64 text")
    print("RLT_SYSTEMS_CARRIER_SCALE15M_BATCH768_AC_RESULT_B64=" + result)
