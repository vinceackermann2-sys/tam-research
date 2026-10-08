from __future__ import annotations

# Independent engineering successor to consumed AH: new job/seeds, patched image, CPU preclaim smoke test.

import base64
import json
import math
import time
from typing import Any

import modal

JOB_ID = "rlt-systems-lightstate-component-attribution-modal-20261008-ai"
ENGINEERING_SEED = 20_261_051
BATCH_SEED = 20_271_051
COMPILE_PROBE_SEED = 20_301_051
SEQ_LEN = 64
BATCH_SIZE = 64
BANK_SIZE = 8
TIME_BUDGET_SECONDS = 12.0
EXPECTED_PARAMETERS = 15_129_344
VARIANTS = (
    "lightstate_full",
    "lightstate_no_recurrence",
    "lightstate_encoder_recurrence_only",
    "lightstate_encoder_only",
    "transformer_control",
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.10,<2.11", "numpy>=2.0,<3")
    .add_local_python_source("tam_research", "experiments")
)
app = modal.App("tam-rlt-systems-lightstate-component-attribution-20261008-ai")


def _encode(value: dict[str, Any]) -> str:
    return base64.urlsafe_b64encode(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).decode()


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
    import torch.nn as nn
    import torch.nn.functional as F

    from experiments.rlt.model import RLTConfig, parameter_count as rlt_parameter_count
    from experiments.rlt.model_light_state import LightStateRecurrentTransformer
    from tam_research.models import ModelConfig, ResearchLM, parameter_count as transformer_parameter_count
    from tam_research.train import seed_all

    started = time.time()
    job = json.loads(job_json)
    expected = {
        "schema": 1,
        "job_id": JOB_ID,
        "task": "systems_lightstate_component_attribution_ah",
        "engineering_seed": ENGINEERING_SEED,
        "batch_seed": BATCH_SEED,
        "compile_probe_seed": COMPILE_PROBE_SEED,
        "requires_lowbatch_calibration_job_id": "rlt-systems-lightstate-lowbatch-qualitysec-modal-20261007-ag",
        "requires_compute_match_job_id": "rlt-lightstate-compute-matched-optimizer-tuned-scale15m-60s-modal-20261007-b",
        "requires_compileorder_job_id": "rlt-systems-lightstate-compileorder-modal-20261007-af",
        "variants": list(VARIANTS),
        "batch_size": BATCH_SIZE,
        "seq_len": SEQ_LEN,
        "synthetic_bank_size": BANK_SIZE,
        "time_budget_seconds_each": TIME_BUDGET_SECONDS,
        "expected_parameters_each": EXPECTED_PARAMETERS,
        "automatic_retry": False,
    }
    if job != expected:
        raise RuntimeError(f"component-attribution preregistration mismatch: {job}")
    if not torch.cuda.is_available():
        raise RuntimeError("component attribution requires CUDA")

    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    rlt_cfg = RLTConfig(
        vocab_size=50_257,
        d_model=256,
        n_heads=8,
        n_stages=2,
        max_seq_len=128,
        ff_mult=4,
        swa_window=32,
    )
    transformer_cfg = ModelConfig(
        vocab_size=50_257,
        d_model=256,
        n_layers=3,
        n_heads=8,
        max_seq_len=128,
        ff_mult=4,
        ff_inner=938,
        architecture="transformer",
    )

    class FullLightState(nn.Module):
        def __init__(self, base: LightStateRecurrentTransformer):
            super().__init__()
            self.base = base

        def forward(self, tokens: torch.Tensor) -> torch.Tensor:
            return self.base(tokens)

    class NoRecurrence(nn.Module):
        def __init__(self, base: LightStateRecurrentTransformer):
            super().__init__()
            self.base = base

        def forward(self, tokens: torch.Tensor) -> torch.Tensor:
            m = self.base
            memory = m.encode(tokens)
            hidden = memory
            memory_kv = [stage.cross_attn.precompute(memory) for stage in m.stages]
            for stage_index, stage in enumerate(m.stages):
                hidden = hidden + m._parallel_local_self_attention(
                    stage_index, stage.self_norm(hidden)
                )
                hidden = hidden + m._parallel_causal_cross_attention(
                    stage_index,
                    stage.cross_norm(hidden),
                    memory_kv[stage_index],
                )
                hidden = hidden + stage.ff(stage.ff_norm(hidden))
            return m.lm_head(m.output_norm(hidden))

    def recurrent_hidden(
        m: LightStateRecurrentTransformer, memory: torch.Tensor
    ) -> torch.Tensor:
        b, t, _ = memory.shape
        state = m.start_state.to(memory.dtype).view(1, -1).expand(b, -1)
        states: list[torch.Tensor] = []
        for token_index in range(t):
            merged = torch.cat((memory[:, token_index, :], state), dim=-1)
            state = m.merge_norm(m.merge(merged))
            states.append(state)
        return torch.stack(states, dim=1)

    class EncoderRecurrenceOnly(nn.Module):
        def __init__(self, base: LightStateRecurrentTransformer):
            super().__init__()
            self.base = base

        def forward(self, tokens: torch.Tensor) -> torch.Tensor:
            m = self.base
            memory = m.encode(tokens)
            hidden = recurrent_hidden(m, memory)
            return m.lm_head(m.output_norm(hidden))

    class EncoderOnly(nn.Module):
        def __init__(self, base: LightStateRecurrentTransformer):
            super().__init__()
            self.base = base

        def forward(self, tokens: torch.Tensor) -> torch.Tensor:
            m = self.base
            memory = m.encode(tokens)
            return m.lm_head(m.output_norm(memory))

    def make_rlt(wrapper_cls: type[nn.Module]) -> tuple[nn.Module, nn.Module]:
        seed_all(ENGINEERING_SEED)
        base = LightStateRecurrentTransformer(rlt_cfg).to(device)
        params = rlt_parameter_count(base)
        if params != EXPECTED_PARAMETERS:
            raise RuntimeError(f"RLT parameter drift: {params}")
        return base, wrapper_cls(base)

    def make_transformer() -> tuple[nn.Module, nn.Module]:
        seed_all(ENGINEERING_SEED)
        base = ResearchLM(transformer_cfg).to(device)
        params = transformer_parameter_count(base)
        if params != EXPECTED_PARAMETERS:
            raise RuntimeError(f"Transformer parameter drift: {params}")
        return base, base

    # Deterministic synthetic token bank. This is a systems diagnostic only:
    # no dataset or quality conclusion is permitted.
    generator = torch.Generator(device="cuda").manual_seed(BATCH_SEED)
    bank: list[tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(BANK_SIZE):
        x = torch.randint(
            0, 50_257, (BATCH_SIZE, SEQ_LEN), device=device, generator=generator
        )
        y = torch.randint(
            0, 50_257, (BATCH_SIZE, SEQ_LEN), device=device, generator=generator
        )
        bank.append((x, y))

    # Exact wrapper-equivalence guard for the full light-state path.
    seed_all(ENGINEERING_SEED)
    reference = LightStateRecurrentTransformer(rlt_cfg).to(device)
    seed_all(ENGINEERING_SEED)
    candidate_base = LightStateRecurrentTransformer(rlt_cfg).to(device)
    candidate_base.load_state_dict(reference.state_dict())
    candidate = FullLightState(candidate_base)
    x0 = bank[0][0][:2]
    y0 = bank[0][1][:2]
    reference.zero_grad(set_to_none=True)
    candidate_base.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=amp_dtype):
        ref_logits = reference(x0)
        cand_logits = candidate(x0)
        ref_loss = F.cross_entropy(
            ref_logits.float().reshape(-1, 50_257), y0.reshape(-1)
        )
        cand_loss = F.cross_entropy(
            cand_logits.float().reshape(-1, 50_257), y0.reshape(-1)
        )
    ref_loss.backward()
    cand_loss.backward()
    torch.cuda.synchronize(device)
    logits_abs = float((ref_logits - cand_logits).abs().max().item())
    loss_abs = float((ref_loss - cand_loss).abs().item())
    grad_abs = 0.0
    for rp, cp in zip(reference.parameters(), candidate_base.parameters()):
        if rp.grad is None or cp.grad is None:
            if rp.grad is not cp.grad:
                grad_abs = float("inf")
                break
            continue
        grad_abs = max(grad_abs, float((rp.grad - cp.grad).abs().max().item()))
    wrapper_exact = logits_abs == 0.0 and loss_abs == 0.0 and grad_abs == 0.0
    del reference, candidate_base, candidate, ref_logits, cand_logits
    torch.cuda.empty_cache()
    if not wrapper_exact:
        raise RuntimeError("full light-state wrapper lost exact semantic equivalence")

    factories = {
        "lightstate_full": lambda: make_rlt(FullLightState),
        "lightstate_no_recurrence": lambda: make_rlt(NoRecurrence),
        "lightstate_encoder_recurrence_only": lambda: make_rlt(EncoderRecurrenceOnly),
        "lightstate_encoder_only": lambda: make_rlt(EncoderOnly),
        "transformer_control": make_transformer,
    }

    def benchmark(name: str) -> dict[str, Any]:
        parameter_model, callable_model = factories[name]()
        compiled = torch.compile(
            callable_model, fullgraph=True, dynamic=False, mode="default"
        )
        optimizer = torch.optim.AdamW(
            parameter_model.parameters(),
            lr=1e-3,
            betas=(0.9, 0.95),
            weight_decay=0.1,
            fused=True,
        )

        # Force forward/backward compilation before the timed window.
        x, y = bank[COMPILE_PROBE_SEED % BANK_SIZE]
        parameter_model.train()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        compile_started = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            logits = compiled(x)
            loss = F.cross_entropy(
                logits.float().reshape(-1, 50_257), y.reshape(-1)
            )
        loss.backward()
        torch.cuda.synchronize(device)
        compile_seconds = time.perf_counter() - compile_started
        compile_loss = float(loss.detach().cpu())
        parameter_model.zero_grad(set_to_none=True)

        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        train_started = time.perf_counter()
        steps = 0
        last_loss = float("nan")
        while True:
            elapsed = time.perf_counter() - train_started
            if elapsed >= TIME_BUDGET_SECONDS and steps > 0:
                break
            x, y = bank[steps % BANK_SIZE]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                logits = compiled(x)
                loss = F.cross_entropy(
                    logits.float().reshape(-1, 50_257), y.reshape(-1)
                )
            # Match the existing throughput harness behavior: each step observes
            # the loss on CPU, which synchronizes the GPU before backward/update.
            last_loss = float(loss.detach().cpu())
            if not math.isfinite(last_loss):
                raise RuntimeError(f"non-finite loss for {name}: {last_loss}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameter_model.parameters(), 1.0)
            optimizer.step()
            steps += 1

        torch.cuda.synchronize(device)
        seconds = time.perf_counter() - train_started
        tokens = steps * BATCH_SIZE * SEQ_LEN
        peak = torch.cuda.max_memory_allocated(device) / (1024**3)
        params = sum(p.numel() for p in parameter_model.parameters())
        out = {
            "name": name,
            "parameters": params,
            "compile_probe_seconds": compile_seconds,
            "compile_probe_loss": compile_loss,
            "timed_seconds": seconds,
            "steps": steps,
            "tokens": tokens,
            "tokens_per_second": tokens / max(seconds, 1e-9),
            "optimizer_steps_per_second": steps / max(seconds, 1e-9),
            "last_loss": last_loss,
            "peak_vram_gb": peak,
        }
        del compiled, optimizer, callable_model, parameter_model
        torch.cuda.empty_cache()
        torch._dynamo.reset()
        return out

    rows = [benchmark(name) for name in VARIANTS]
    by_name = {row["name"]: row for row in rows}
    full = by_name["lightstate_full"]["tokens_per_second"]
    no_rec = by_name["lightstate_no_recurrence"]["tokens_per_second"]
    enc_rec = by_name["lightstate_encoder_recurrence_only"]["tokens_per_second"]
    enc = by_name["lightstate_encoder_only"]["tokens_per_second"]
    transformer = by_name["transformer_control"]["tokens_per_second"]

    derived = {
        "transformer_over_full_lightstate_tps": transformer / max(full, 1e-9),
        "remove_recurrence_speedup_on_full_path": no_rec / max(full, 1e-9),
        "remove_heavy_decoder_speedup": enc_rec / max(full, 1e-9),
        "remove_recurrence_speedup_on_encoder_only_path": enc / max(enc_rec, 1e-9),
        "no_recurrence_heavy_path_vs_transformer_tps": no_rec / max(transformer, 1e-9),
        "encoder_recurrence_only_vs_transformer_tps": enc_rec / max(transformer, 1e-9),
        "encoder_only_vs_transformer_tps": enc / max(transformer, 1e-9),
        "approx_seconds_per_token": {
            name: 1.0 / max(row["tokens_per_second"], 1e-9)
            for name, row in by_name.items()
        },
    }

    return _encode(
        {
            "schema": 1,
            "job_id": JOB_ID,
            "status": "complete",
            "classification": "PASS_LIGHTSTATE_COMPONENT_ATTRIBUTION",
            "engineering_only": True,
            "scientific_execution": False,
            "breakthrough_claim_supported": False,
            "engineering_seed": ENGINEERING_SEED,
            "batch_seed": BATCH_SEED,
            "compile_probe_seed": COMPILE_PROBE_SEED,
            "started_unix": started,
            "finished_unix": time.time(),
            "runtime": {
                "torch_version": str(torch.__version__),
                "cuda_runtime": None if torch.version.cuda is None else str(torch.version.cuda),
                "gpu_name": str(torch.cuda.get_device_name(0)),
                "bf16_supported": bool(torch.cuda.is_bf16_supported()),
            },
            "protocol": {
                "variants": list(VARIANTS),
                "batch_size": BATCH_SIZE,
                "seq_len": SEQ_LEN,
                "synthetic_bank_size": BANK_SIZE,
                "time_budget_seconds_each": TIME_BUDGET_SECONDS,
                "compile_time_excluded": True,
                "same_gpu_same_run": True,
                "same_engineering_seed": True,
                "same_synthetic_token_bank": True,
                "optimizer": "AdamW(lr=1e-3,betas=(0.9,0.95),weight_decay=0.1,fused=True)",
                "gradient_clip_norm": 1.0,
                "timing_only_no_quality_claim": True,
            },
            "full_wrapper_semantic_equivalence": {
                "passed": wrapper_exact,
                "logits_max_abs": logits_abs,
                "loss_abs": loss_abs,
                "grad_max_abs": grad_abs,
            },
            "results": rows,
            "derived": derived,
            "interpretation_ceiling": (
                "Engineering-only same-GPU systems attribution. Ablations intentionally change "
                "the computation graph and are not quality comparisons. Use removal speedups only "
                "to decide which architectural compute path deserves redesign; they do not support "
                "a scientific or breakthrough claim."
            ),
        }
    )



@app.function(
    image=image,
    cpu=2.0,
    timeout=300,
    retries=0,
    max_containers=1,
)
def verify_runtime_remote() -> str:
    import numpy
    import torch
    from experiments.rlt.model import RLTConfig, parameter_count as rlt_parameter_count
    from experiments.rlt.model_light_state import LightStateRecurrentTransformer
    from tam_research.models import ModelConfig, ResearchLM, parameter_count as transformer_parameter_count
    from tam_research.train import seed_all

    torch.set_num_threads(1)
    seed_all(ENGINEERING_SEED)
    rlt_cfg = RLTConfig(
        vocab_size=50_257, d_model=256, n_heads=8, n_stages=2,
        max_seq_len=128, ff_mult=4, swa_window=32,
    )
    transformer_cfg = ModelConfig(
        vocab_size=50_257, d_model=256, n_layers=3, n_heads=8,
        max_seq_len=128, ff_mult=4, ff_inner=938,
        architecture="transformer",
    )
    rlt = LightStateRecurrentTransformer(rlt_cfg)
    transformer = ResearchLM(transformer_cfg)
    rlt_params = rlt_parameter_count(rlt)
    transformer_params = transformer_parameter_count(transformer)
    if rlt_params != EXPECTED_PARAMETERS or transformer_params != EXPECTED_PARAMETERS:
        raise RuntimeError(f"parameter mismatch: {rlt_params} vs {transformer_params}")
    x = torch.tensor([[101, 102]], dtype=torch.long)
    with torch.no_grad():
        rlt_y = rlt(x)
        transformer_y = transformer(x)
    if tuple(rlt_y.shape) != (1, 2, 50_257) or tuple(transformer_y.shape) != (1, 2, 50_257):
        raise RuntimeError(f"unexpected forward shapes: {rlt_y.shape} and {transformer_y.shape}")
    return _encode({
        "status": "pass", "gpu_requested": False,
        "numpy_version": numpy.__version__,
        "torch_version": torch.__version__,
        "rlt_parameters": rlt_params,
        "transformer_parameters": transformer_params,
        "cpu_forward_shapes_passed": True,
    })


@app.local_entrypoint()
def verify_dependencies() -> None:
    out = verify_runtime_remote.remote()
    if not isinstance(out, str):
        raise TypeError("AH successor import preflight must return base64 text")
    print("RLT_SYSTEMS_LIGHTSTATE_COMPONENT_ATTRIBUTION_AI_PREFLIGHT_B64=" + out)


@app.local_entrypoint()
def main(job_path: str) -> None:
    result = run_profile_remote.remote(open(job_path).read())
    if not isinstance(result, str):
        raise TypeError("remote component-attribution result must be base64 text")
    print("RLT_SYSTEMS_LIGHTSTATE_COMPONENT_ATTRIBUTION_AI_RESULT_B64=" + result)
