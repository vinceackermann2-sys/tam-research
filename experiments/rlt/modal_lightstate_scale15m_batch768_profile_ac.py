from __future__ import annotations

import base64
import json
from pathlib import Path
import time
from typing import Any

import modal

JOB_ID = "rlt-systems-lightstate-scale15m-batch768-modal-20261002-ac"
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
app = modal.App("tam-rlt-systems-lightstate-scale15m-batch768-20261002-ac")


def _encode(value: dict[str, Any]) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).decode()


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=4.0,
    memory=65536,
    timeout=55 * 60,
    retries=0,
    max_containers=1,
)
def run_profile_remote(job_json: str) -> str:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from experiments.rlt.model import RLTConfig, RecurrentLoopedTransformer, parameter_count
    from experiments.rlt.model_light_state import LightStateRecurrentTransformer
    from tam_research.models import ModelConfig, ResearchLM, parameter_count as transformer_parameter_count

    started = time.time()
    job = json.loads(job_json)
    expected = {
        "schema": 1,
        "job_id": JOB_ID,
        "task": "systems_lightstate_scale15m_batch768_ac",
        "engineering_seed": ENGINEERING_SEED,
        "requires_block_sweep_job_id": "rlt-systems-block-sweep-scale15m-batch768-modal-20260929-ab",
        "requires_block4_replication_job_id": "rlt-block4-scale15m-paired-4m-modal-20260930-b",
        "batch_size": BATCH_SIZE,
        "seq_len": SEQ_LEN,
        "expected_parameters_each": EXPECTED_PARAMETERS,
        "automatic_retry": False,
    }
    if job != expected:
        raise RuntimeError(f"light-state profile preregistration mismatch: {job}")
    if not torch.cuda.is_available():
        raise RuntimeError("light-state systems profile requires CUDA")

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

    def new_light() -> LightStateRecurrentTransformer:
        seed_local(ENGINEERING_SEED)
        m = LightStateRecurrentTransformer(rlt_cfg).to(device)
        if parameter_count(m) != EXPECTED_PARAMETERS:
            raise RuntimeError(f"light-state parameter drift: {parameter_count(m)}")
        return m

    def new_transformer() -> ResearchLM:
        seed_local(ENGINEERING_SEED)
        m = ResearchLM(tr_cfg).to(device)
        if transformer_parameter_count(m) != EXPECTED_PARAMETERS:
            raise RuntimeError(f"Transformer parameter drift: {transformer_parameter_count(m)}")
        return m

    reference = new_reference()
    light = new_light()
    state_schema_equal = tuple(reference.state_dict().keys()) == tuple(light.state_dict().keys())
    parameter_neutral = parameter_count(light) == parameter_count(reference) == EXPECTED_PARAMETERS
    if not state_schema_equal or not parameter_neutral:
        raise RuntimeError("light-state variant changed trainable parameter/state schema")
    del reference
    torch.cuda.empty_cache()

    # Strict future-token causality guard.
    g = torch.Generator(device="cpu").manual_seed(ENGINEERING_SEED + 91)
    x = torch.randint(0, rlt_cfg.vocab_size, (CHECK_BATCH, SEQ_LEN), generator=g, dtype=torch.long).to(device)
    x2 = x.clone()
    x2[:, CAUSAL_MUTATION_INDEX] = (x2[:, CAUSAL_MUTATION_INDEX] + 1) % rlt_cfg.vocab_size
    light.eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        a = light(x)
        b = light(x2)
    torch.cuda.synchronize(device)
    prefix_abs = float((a[:, :CAUSAL_MUTATION_INDEX, :] - b[:, :CAUSAL_MUTATION_INDEX, :]).abs().max().item())
    changed_abs = float((a[:, CAUSAL_MUTATION_INDEX, :] - b[:, CAUSAL_MUTATION_INDEX, :]).abs().max().item())
    causal_passed = prefix_abs == 0.0 and changed_abs > 0.0
    if not causal_passed:
        raise RuntimeError(
            f"light-state causality failed: prefix={prefix_abs}, changed={changed_abs}"
        )
    del light, a, b
    torch.cuda.empty_cache()

    g = torch.Generator(device="cpu").manual_seed(ENGINEERING_SEED + 1)
    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(WARMUP_STEPS + MEASURED_STEPS + 2):
        bx = torch.randint(0, rlt_cfg.vocab_size, (BATCH_SIZE, SEQ_LEN), generator=g, dtype=torch.long).to(device)
        by = torch.randint(0, rlt_cfg.vocab_size, (BATCH_SIZE, SEQ_LEN), generator=g, dtype=torch.long).to(device)
        batches.append((bx, by))

    def one_step(callable_model: Any, model: nn.Module, opt: torch.optim.Optimizer, offset: int) -> float:
        model.train()
        opt.zero_grad(set_to_none=True)
        bx, by = batches[offset % len(batches)]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = callable_model(bx)
            loss = F.cross_entropy(logits.float().reshape(-1, 50_257), by.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        return float(loss.detach().cpu())

    def benchmark(kind: str) -> dict[str, Any]:
        base: nn.Module = new_light() if kind == "light" else new_transformer()
        opt = torch.optim.AdamW(
            base.parameters(), lr=3e-4, betas=(0.9, 0.95),
            weight_decay=0.1, fused=True,
        )
        compiled = torch.compile(base, fullgraph=True, dynamic=False, mode="default")

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
            last_loss = one_step(compiled, base, opt, 1 + WARMUP_STEPS + i)
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
            "peak_vram_gb": torch.cuda.max_memory_allocated(device) / (1024**3),
        }
        del compiled, opt, base
        torch.cuda.empty_cache()
        torch._dynamo.reset()
        return out

    light_result = benchmark("light")
    transformer = benchmark("transformer")
    gap = transformer["tokens_per_second"] / max(light_result["tokens_per_second"], 1e-9)

    classification = (
        "PASS_LIGHTSTATE_SCALE15M_GAP_LE_1_25X" if gap <= 1.25 else
        "PASS_LIGHTSTATE_SCALE15M_GAP_LE_1_5X" if gap <= 1.5 else
        "PASS_LIGHTSTATE_SCALE15M_GAP_GT_1_5X"
    )

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
            "variant": "light_state_recurrent_rlt",
            "token_recurrent_updates_per_seq64": 64,
            "serial_transition": "merge_plus_rmsnorm_only",
            "heavy_decoder_execution": "parallel_causal_full_sequence",
            "parameters": EXPECTED_PARAMETERS,
            "parameter_neutral": parameter_neutral,
            "state_schema_equal_to_frozen_rlt": state_schema_equal,
        },
        "implementation_validation": {
            "future_token_causality": {
                "passed": causal_passed,
                "mutation_index": CAUSAL_MUTATION_INDEX,
                "prefix_logits_max_abs": prefix_abs,
                "mutated_position_logits_max_abs": changed_abs,
            },
            "parameter_neutral_and_state_schema_equal": parameter_neutral and state_schema_equal,
        },
        "profile": {
            "batch_size": BATCH_SIZE,
            "seq_len": SEQ_LEN,
            "warmup_steps": WARMUP_STEPS,
            "measured_steps": MEASURED_STEPS,
            "expected_parameters_each": EXPECTED_PARAMETERS,
        },
        "light_state": light_result,
        "transformer": transformer,
        "derived": {
            "transformer_throughput_multiple_vs_light_state": gap,
            "light_state_fraction_of_transformer_throughput": 1.0 / gap,
        },
        "interpretation_ceiling": (
            "Engineering-only throughput/causality profile for a parameter-neutral architecture "
            "that preserves per-token recurrent state while moving attention/cross-attention/FFN "
            "out of the serial loop. Random-token loss is not quality evidence. A fresh matched "
            "training run is required before any scientific quality conclusion."
        ),
    })


@app.local_entrypoint()
def main(job_path: str) -> None:
    result = run_profile_remote.remote(Path(job_path).read_text())
    if not isinstance(result, str):
        raise TypeError("remote light-state systems result must be base64 text")
    print("RLT_SYSTEMS_LIGHTSTATE_SCALE15M_BATCH768_AC_RESULT_B64=" + result)
