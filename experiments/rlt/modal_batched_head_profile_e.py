from __future__ import annotations

import base64
import json
import time
from typing import Any

import modal

ENGINEERING_SEED = 20260927
SEQ_LEN = 64
MICRO_BATCH = 2
GRAD_ACCUM = 4
WARMUP_STEPS = 1
MEASURED_STEPS = 8
D_COMPILED_TPS = 923.2555452135044

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.10,<2.11")
    .add_local_python_source("experiments")
)
app = modal.App("tam-rlt-batched-head-systems-20260915-e")


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

    class BatchedHeadRLT(nn.Module):
        """Same recurrent decoder; defer token-independent output projection.

        The state used at token t+1 is exactly the decoder hidden state, not the
        logits. Therefore output_norm + tied lm_head can be applied after all
        recurrent states are collected without changing the recurrence graph.
        """

        def __init__(self, base: RecurrentLoopedTransformer):
            super().__init__()
            self.base = base

        def forward(self, tokens: torch.Tensor) -> torch.Tensor:
            model = self.base
            b, t = tokens.shape
            memory = model.encode(tokens)
            memory_kv = [stage.cross_attn.precompute(memory) for stage in model.stages]
            state = model.start_state.to(memory.dtype).view(1, -1).expand(b, -1)
            caches: list[tuple[torch.Tensor, torch.Tensor] | None] = [None for _ in model.stages]
            states: list[torch.Tensor] = []
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
                states.append(state)
            all_states = torch.stack(states, dim=1)
            return model.lm_head(model.output_norm(all_states))

    generator = torch.Generator(device="cpu").manual_seed(ENGINEERING_SEED + 1)
    batch_count = (WARMUP_STEPS + MEASURED_STEPS + 4) * GRAD_ACCUM
    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(batch_count):
        x = torch.randint(0, cfg.vocab_size, (MICRO_BATCH, SEQ_LEN), generator=generator, dtype=torch.long).to(device)
        y = torch.randint(0, cfg.vocab_size, (MICRO_BATCH, SEQ_LEN), generator=generator, dtype=torch.long).to(device)
        batches.append((x, y))

    def new_base() -> RecurrentLoopedTransformer:
        seed_local(ENGINEERING_SEED)
        return RecurrentLoopedTransformer(cfg).to(device)

    # Exact-semantics gate against the original public-spec prototype.
    reference = new_base()
    parameter_total = parameter_count(reference)
    candidate_base = new_base()
    candidate_base.load_state_dict(reference.state_dict())
    candidate = BatchedHeadRLT(candidate_base)
    x0, y0 = batches[0]
    reference.zero_grad(set_to_none=True)
    candidate_base.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        ref_logits = reference(x0)
        cand_logits = candidate(x0)
        ref_loss = F.cross_entropy(ref_logits.float().reshape(-1, cfg.vocab_size), y0.reshape(-1))
        cand_loss = F.cross_entropy(cand_logits.float().reshape(-1, cfg.vocab_size), y0.reshape(-1))
    ref_loss.backward()
    cand_loss.backward()
    torch.cuda.synchronize(device)

    logits_max_abs = float((ref_logits - cand_logits).abs().max().item())
    loss_abs = float((ref_loss - cand_loss).abs().item())
    grad_max_abs = 0.0
    for ref_p, cand_p in zip(reference.parameters(), candidate_base.parameters()):
        if ref_p.grad is None or cand_p.grad is None:
            if ref_p.grad is not cand_p.grad:
                grad_max_abs = float("inf")
                break
            continue
        grad_max_abs = max(grad_max_abs, float((ref_p.grad - cand_p.grad).abs().max().item()))
    semantic_equivalence = logits_max_abs <= 1e-3 and loss_abs <= 1e-5 and grad_max_abs <= 2e-3
    del reference, candidate_base, candidate, ref_logits, cand_logits
    torch.cuda.empty_cache()

    def optimizer_for(model: nn.Module) -> torch.optim.Optimizer:
        return torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1, fused=True)

    def one_step(callable_model: Any, parameter_model: nn.Module, optimizer: Any, offset: int) -> float:
        parameter_model.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        for micro in range(GRAD_ACCUM):
            x, y = batches[(offset * GRAD_ACCUM + micro) % len(batches)]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = callable_model(x)
                loss = F.cross_entropy(logits.float().reshape(-1, cfg.vocab_size), y.reshape(-1)) / GRAD_ACCUM
            loss.backward()
            running += float(loss.detach().cpu()) * GRAD_ACCUM
        torch.nn.utils.clip_grad_norm_(parameter_model.parameters(), 1.0)
        optimizer.step()
        return running / GRAD_ACCUM

    def benchmark_eager(kind: str) -> tuple[float, float, int]:
        base = new_base()
        callable_model: nn.Module = base if kind == "reference" else BatchedHeadRLT(base)
        opt = optimizer_for(base)
        for step in range(WARMUP_STEPS):
            one_step(callable_model, base, opt, step)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()
        last_loss = 0.0
        for step in range(MEASURED_STEPS):
            last_loss = one_step(callable_model, base, opt, WARMUP_STEPS + step)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - t0
        peak = int(torch.cuda.max_memory_allocated(device))
        del callable_model, base, opt
        torch.cuda.empty_cache()
        return elapsed, last_loss, peak

    ref_eager_seconds, ref_eager_loss, ref_eager_peak = benchmark_eager("reference")
    batched_eager_seconds, batched_eager_loss, batched_eager_peak = benchmark_eager("batched")

    compiled_base = new_base()
    wrapper = BatchedHeadRLT(compiled_base)
    compiled_opt = optimizer_for(compiled_base)
    compiled_callable = torch.compile(wrapper, fullgraph=True, dynamic=False, mode="default")
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    one_step(compiled_callable, compiled_base, compiled_opt, 0)
    torch.cuda.synchronize(device)
    compile_first_step_seconds = time.perf_counter() - t0

    one_step(compiled_callable, compiled_base, compiled_opt, 1)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    measured_started = time.perf_counter()
    compiled_last_loss = 0.0
    for step in range(MEASURED_STEPS):
        compiled_last_loss = one_step(compiled_callable, compiled_base, compiled_opt, WARMUP_STEPS + 2 + step)
    torch.cuda.synchronize(device)
    compiled_seconds = time.perf_counter() - measured_started
    compiled_peak = int(torch.cuda.max_memory_allocated(device))

    measured_tokens = MEASURED_STEPS * GRAD_ACCUM * MICRO_BATCH * SEQ_LEN
    ref_eager_tps = measured_tokens / max(ref_eager_seconds, 1e-9)
    batched_eager_tps = measured_tokens / max(batched_eager_seconds, 1e-9)
    compiled_tps = measured_tokens / max(compiled_seconds, 1e-9)
    versus_d = compiled_tps / D_COMPILED_TPS

    if not semantic_equivalence:
        classification = "FAIL_SEMANTIC_EQUIVALENCE"
    elif versus_d >= 1.25:
        classification = "PASS_MATERIAL_BATCHED_HEAD_SPEEDUP"
    else:
        classification = "PASS_NO_MATERIAL_BATCHED_HEAD_SPEEDUP"

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
            "candidate_change": "deferred batched output_norm + tied lm_head only",
        },
        "compile": {
            "mode": "default",
            "fullgraph": True,
            "dynamic": False,
            "cuda_graphs_requested": False,
            "first_step_seconds_including_compile": compile_first_step_seconds,
        },
        "benchmark": {
            "measured_tokens": measured_tokens,
            "reference_eager_seconds": ref_eager_seconds,
            "reference_eager_tokens_per_second": ref_eager_tps,
            "reference_eager_last_loss": ref_eager_loss,
            "reference_eager_peak_vram_gb": ref_eager_peak / (1024**3),
            "batched_head_eager_seconds": batched_eager_seconds,
            "batched_head_eager_tokens_per_second": batched_eager_tps,
            "batched_head_eager_last_loss": batched_eager_loss,
            "batched_head_eager_peak_vram_gb": batched_eager_peak / (1024**3),
            "batched_head_compiled_seconds": compiled_seconds,
            "batched_head_compiled_tokens_per_second": compiled_tps,
            "batched_head_compiled_last_loss": compiled_last_loss,
            "batched_head_compiled_peak_vram_gb": compiled_peak / (1024**3),
            "batched_head_eager_speedup_vs_reference_eager": batched_eager_tps / ref_eager_tps,
            "batched_head_compiled_speedup_vs_reference_eager": compiled_tps / ref_eager_tps,
            "prior_d_compiled_tokens_per_second": D_COMPILED_TPS,
            "batched_head_compiled_speedup_vs_prior_d": versus_d,
            "synthetic_fixed_batches": True,
        },
    }
    return _encode(payload)


@app.local_entrypoint()
def main() -> None:
    result = run_profile_remote.remote()
    if not isinstance(result, str):
        raise TypeError("remote systems result must be a base64 string")
    print("RLT_BATCHED_HEAD_RESULT_B64=" + result)
