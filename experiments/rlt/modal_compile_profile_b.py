from __future__ import annotations

import base64
import json
import time
from typing import Any

import modal

ENGINEERING_SEED = 20260923
SEQ_LEN = 64
MICRO_BATCH = 2
GRAD_ACCUM = 4
WARMUP_STEPS = 1
MEASURED_STEPS = 8

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.10,<2.11")
    .add_local_python_source("experiments")
)
app = modal.App("tam-rlt-compile-systems-20260915-b")


def _encode(value: dict[str, Any]) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).decode()


@app.function(
    image=image,
    gpu="A100",
    cpu=4.0,
    memory=16384,
    timeout=15 * 60,
    retries=0,
    max_containers=1,
)
def run_profile_remote() -> str:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from experiments.rlt.model import RLTConfig, RecurrentLoopedTransformer, parameter_count

    started = time.time()
    if not torch.cuda.is_available():
        raise RuntimeError("systems profile requires CUDA")
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")

    cfg = RLTConfig(
        vocab_size=50_257,
        d_model=128,
        n_heads=4,
        n_stages=2,
        max_seq_len=128,
        ff_mult=4,
        swa_window=32,
    )

    def seed_local(seed: int) -> None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    class CompilableRLT(nn.Module):
        """Exact RLT forward math without the diagnostic cache-length side effect."""

        def __init__(self, base: RecurrentLoopedTransformer):
            super().__init__()
            self.base = base

        def forward(self, tokens: torch.Tensor) -> torch.Tensor:
            model = self.base
            b, t = tokens.shape
            memory = model.encode(tokens)
            memory_kv = [stage.cross_attn.precompute(memory) for stage in model.stages]
            state = model.start_state.to(memory.dtype).view(1, -1).expand(b, -1)
            caches: list[tuple[torch.Tensor, torch.Tensor] | None] = [
                None for _ in model.stages
            ]
            logits: list[torch.Tensor] = []
            for token_index in range(t):
                merged = torch.cat((memory[:, token_index, :], state), dim=-1)
                hidden = model.merge_norm(model.merge(merged)).unsqueeze(1)
                prefix_len = token_index + 1
                for layer_index, stage in enumerate(model.stages):
                    hidden, caches[layer_index] = stage.decode_step(
                        hidden,
                        caches[layer_index],
                        memory_kv[layer_index],
                        prefix_len,
                        model.cfg.swa_window,
                    )
                state = hidden[:, 0, :]
                logits.append(model.lm_head(model.output_norm(state)))
            return torch.stack(logits, dim=1)

    generator = torch.Generator(device="cpu").manual_seed(ENGINEERING_SEED + 1)
    batch_count = (WARMUP_STEPS + MEASURED_STEPS + 3) * GRAD_ACCUM
    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(batch_count):
        x = torch.randint(
            0, cfg.vocab_size, (MICRO_BATCH, SEQ_LEN), generator=generator, dtype=torch.long
        ).to(device)
        y = torch.randint(
            0, cfg.vocab_size, (MICRO_BATCH, SEQ_LEN), generator=generator, dtype=torch.long
        ).to(device)
        batches.append((x, y))

    def new_base() -> RecurrentLoopedTransformer:
        seed_local(ENGINEERING_SEED)
        return RecurrentLoopedTransformer(cfg).to(device)

    reference = new_base()
    parameter_total = parameter_count(reference)
    candidate_base = new_base()
    candidate_base.load_state_dict(reference.state_dict())
    candidate_wrapper = CompilableRLT(candidate_base)
    x0, y0 = batches[0]
    reference.zero_grad(set_to_none=True)
    candidate_base.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        reference_logits = reference(x0)
        candidate_logits = candidate_wrapper(x0)
        reference_loss = F.cross_entropy(
            reference_logits.float().reshape(-1, cfg.vocab_size), y0.reshape(-1)
        )
        candidate_loss = F.cross_entropy(
            candidate_logits.float().reshape(-1, cfg.vocab_size), y0.reshape(-1)
        )
    reference_loss.backward()
    candidate_loss.backward()
    torch.cuda.synchronize(device)

    logits_max_abs = float((reference_logits - candidate_logits).abs().max().item())
    loss_abs = float((reference_loss - candidate_loss).abs().item())
    grad_max_abs = 0.0
    for ref_p, cand_p in zip(reference.parameters(), candidate_base.parameters()):
        if ref_p.grad is None or cand_p.grad is None:
            if ref_p.grad is not cand_p.grad:
                grad_max_abs = float("inf")
                break
            continue
        grad_max_abs = max(
            grad_max_abs, float((ref_p.grad - cand_p.grad).abs().max().item())
        )
    semantic_equivalence = (
        logits_max_abs <= 1e-3 and loss_abs <= 1e-5 and grad_max_abs <= 2e-3
    )
    del reference, candidate_base, candidate_wrapper, reference_logits, candidate_logits
    torch.cuda.empty_cache()

    def optimizer_for(model: nn.Module) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            model.parameters(),
            lr=3e-4,
            betas=(0.9, 0.95),
            weight_decay=0.1,
            fused=True,
        )

    def one_step(callable_model: Any, parameter_model: nn.Module, optimizer: Any, offset: int) -> float:
        parameter_model.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        for micro in range(GRAD_ACCUM):
            x, y = batches[(offset * GRAD_ACCUM + micro) % len(batches)]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = callable_model(x)
                loss = F.cross_entropy(
                    logits.float().reshape(-1, cfg.vocab_size), y.reshape(-1)
                ) / GRAD_ACCUM
            loss.backward()
            running += float(loss.detach().cpu()) * GRAD_ACCUM
        torch.nn.utils.clip_grad_norm_(parameter_model.parameters(), 1.0)
        optimizer.step()
        return running / GRAD_ACCUM

    eager_model = new_base()
    eager_opt = optimizer_for(eager_model)
    for step in range(WARMUP_STEPS):
        one_step(eager_model, eager_model, eager_opt, step)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    eager_started = time.perf_counter()
    eager_last_loss = 0.0
    for step in range(MEASURED_STEPS):
        eager_last_loss = one_step(eager_model, eager_model, eager_opt, WARMUP_STEPS + step)
    torch.cuda.synchronize(device)
    eager_seconds = time.perf_counter() - eager_started
    eager_peak = int(torch.cuda.max_memory_allocated(device))
    del eager_model, eager_opt
    torch.cuda.empty_cache()

    compiled_base = new_base()
    wrapper = CompilableRLT(compiled_base)
    compiled_opt = optimizer_for(compiled_base)
    compile_strategy = "fullgraph"
    fullgraph_error: str | None = None
    compile_first_step_seconds: float | None = None
    compiled_callable: Any = None
    compile_error: str | None = None
    try:
        compiled_callable = torch.compile(
            wrapper, fullgraph=True, dynamic=False, mode="reduce-overhead"
        )
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        one_step(compiled_callable, compiled_base, compiled_opt, 0)
        torch.cuda.synchronize(device)
        compile_first_step_seconds = time.perf_counter() - t0
    except Exception as exc:
        fullgraph_error = f"{type(exc).__name__}: {exc}"
        del compiled_opt, compiled_base, wrapper
        torch.cuda.empty_cache()
        torch._dynamo.reset()
        compiled_base = new_base()
        wrapper = CompilableRLT(compiled_base)
        compiled_opt = optimizer_for(compiled_base)
        compile_strategy = "graph-break-tolerant"
        try:
            compiled_callable = torch.compile(
                wrapper, fullgraph=False, dynamic=False, mode="reduce-overhead"
            )
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            one_step(compiled_callable, compiled_base, compiled_opt, 0)
            torch.cuda.synchronize(device)
            compile_first_step_seconds = time.perf_counter() - t0
        except Exception as fallback_exc:
            compile_error = f"{type(fallback_exc).__name__}: {fallback_exc}"

    compiled_seconds: float | None = None
    compiled_peak: int | None = None
    compiled_last_loss: float | None = None
    if compiled_callable is not None and compile_error is None:
        one_step(compiled_callable, compiled_base, compiled_opt, 1)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        measured_started = time.perf_counter()
        for step in range(MEASURED_STEPS):
            compiled_last_loss = one_step(
                compiled_callable,
                compiled_base,
                compiled_opt,
                WARMUP_STEPS + 2 + step,
            )
        torch.cuda.synchronize(device)
        compiled_seconds = time.perf_counter() - measured_started
        compiled_peak = int(torch.cuda.max_memory_allocated(device))

    measured_tokens = MEASURED_STEPS * GRAD_ACCUM * MICRO_BATCH * SEQ_LEN
    eager_tps = measured_tokens / max(eager_seconds, 1e-9)
    compiled_tps = None if compiled_seconds is None else measured_tokens / max(compiled_seconds, 1e-9)
    speedup = None if compiled_tps is None else compiled_tps / eager_tps
    if not semantic_equivalence:
        classification = "FAIL_SEMANTIC_EQUIVALENCE"
    elif compiled_tps is None:
        classification = "FAIL_COMPILE_EXECUTION"
    elif speedup is not None and speedup >= 1.25:
        classification = "PASS_MATERIAL_COMPILE_SPEEDUP"
    else:
        classification = "PASS_NO_MATERIAL_COMPILE_SPEEDUP"

    payload = {
        "schema": 1,
        "status": "complete",
        "classification": classification,
        "engineering_only": True,
        "scientific_execution": False,
        "engineering_seed": ENGINEERING_SEED,
        "started_unix": started,
        "finished_unix": time.time(),
        "runtime": {
            "torch_version": str(torch.__version__),
            "cuda_runtime": None if torch.version.cuda is None else str(torch.version.cuda),
            "gpu_name": str(torch.cuda.get_device_name(0)),
            "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        },
        "model": {
            "profile": "tiny",
            "parameters": parameter_total,
            "seq_len": SEQ_LEN,
            "micro_batch_size": MICRO_BATCH,
            "grad_accum_steps": GRAD_ACCUM,
        },
        "semantic_equivalence": {
            "passed": semantic_equivalence,
            "logits_max_abs": logits_max_abs,
            "loss_abs": loss_abs,
            "grad_max_abs": grad_max_abs,
            "candidate_difference": "diagnostic last_cache_lengths side effect omitted only",
        },
        "compile": {
            "strategy": compile_strategy,
            "fullgraph_error": fullgraph_error,
            "fallback_error": compile_error,
            "first_step_seconds_including_compile": compile_first_step_seconds,
            "mode": "reduce-overhead",
            "dynamic": False,
        },
        "benchmark": {
            "warmup_steps": WARMUP_STEPS,
            "measured_steps": MEASURED_STEPS,
            "measured_tokens": measured_tokens,
            "eager_seconds": eager_seconds,
            "eager_tokens_per_second": eager_tps,
            "eager_last_loss": eager_last_loss,
            "eager_peak_vram_gb": eager_peak / (1024**3),
            "compiled_seconds": compiled_seconds,
            "compiled_tokens_per_second": compiled_tps,
            "compiled_last_loss": compiled_last_loss,
            "compiled_peak_vram_gb": None if compiled_peak is None else compiled_peak / (1024**3),
            "compiled_speedup_multiple": speedup,
            "synthetic_fixed_batches": True,
        },
    }
    return _encode(payload)


@app.local_entrypoint()
def main() -> None:
    result = run_profile_remote.remote()
    if not isinstance(result, str):
        raise TypeError("remote systems result must be a base64 string")
    print("RLT_SYSTEMS_RESULT_B64=" + result)
