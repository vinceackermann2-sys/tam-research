from __future__ import annotations

import base64
import json
import time
from typing import Any

import modal

ENGINEERING_SEED = 20260928
SEQ_LEN = 64
BATCH_SIZE = 16
WARMUP_STEPS = 1
MEASURED_STEPS = 6
PRIOR_D_TPS = 923.2555452135044

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.10,<2.11")
    .add_local_python_source("tam_research", "experiments")
)
app = modal.App("tam-rlt-batchscale-systems-20260915-f")


def _encode(value: dict[str, Any]) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).decode()


@app.function(
    image=image,
    gpu="A100",
    cpu=4.0,
    memory=24576,
    timeout=20 * 60,
    retries=0,
    max_containers=1,
)
def run_profile_remote() -> str:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from experiments.rlt.model import RLTConfig, RecurrentLoopedTransformer, parameter_count
    from tam_research.models import ModelConfig, ResearchLM

    started = time.time()
    if not torch.cuda.is_available():
        raise RuntimeError("batch-scale systems profile requires CUDA")
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")

    rlt_cfg = RLTConfig(
        vocab_size=50_257,
        d_model=128,
        n_heads=4,
        n_stages=2,
        max_seq_len=128,
        ff_mult=4,
        swa_window=32,
    )
    transformer_cfg = ModelConfig(
        vocab_size=50_257,
        d_model=128,
        n_layers=3,
        n_heads=4,
        max_seq_len=128,
        ff_mult=4,
        architecture="transformer",
    )

    def seed_local(seed: int) -> None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    class ExactCompiledRLT(nn.Module):
        """D-path math: only the diagnostic cache-length side effect is omitted."""

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
    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(WARMUP_STEPS + MEASURED_STEPS + 3):
        x = torch.randint(0, rlt_cfg.vocab_size, (BATCH_SIZE, SEQ_LEN), generator=generator, dtype=torch.long).to(device)
        y = torch.randint(0, rlt_cfg.vocab_size, (BATCH_SIZE, SEQ_LEN), generator=generator, dtype=torch.long).to(device)
        batches.append((x, y))

    def new_rlt() -> RecurrentLoopedTransformer:
        seed_local(ENGINEERING_SEED)
        return RecurrentLoopedTransformer(rlt_cfg).to(device)

    def new_transformer() -> ResearchLM:
        seed_local(ENGINEERING_SEED)
        return ResearchLM(transformer_cfg).to(device)

    # Strict equivalence check for the unchanged D execution path at the larger batch.
    reference = new_rlt()
    candidate_base = new_rlt()
    candidate_base.load_state_dict(reference.state_dict())
    candidate = ExactCompiledRLT(candidate_base)
    x0, y0 = batches[0]
    reference.zero_grad(set_to_none=True)
    candidate_base.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        ref_logits = reference(x0)
        cand_logits = candidate(x0)
        ref_loss = F.cross_entropy(ref_logits.float().reshape(-1, rlt_cfg.vocab_size), y0.reshape(-1))
        cand_loss = F.cross_entropy(cand_logits.float().reshape(-1, rlt_cfg.vocab_size), y0.reshape(-1))
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
    semantic_equivalence = logits_max_abs == 0.0 and loss_abs == 0.0 and grad_max_abs == 0.0
    rlt_parameters = parameter_count(reference)
    del reference, candidate_base, candidate, ref_logits, cand_logits
    torch.cuda.empty_cache()

    def optimizer_for(model: nn.Module) -> torch.optim.Optimizer:
        return torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1, fused=True)

    def one_step(callable_model: Any, parameter_model: nn.Module, optimizer: Any, offset: int, vocab_size: int) -> float:
        parameter_model.train()
        optimizer.zero_grad(set_to_none=True)
        x, y = batches[offset % len(batches)]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = callable_model(x)
            loss = F.cross_entropy(logits.float().reshape(-1, vocab_size), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameter_model.parameters(), 1.0)
        optimizer.step()
        return float(loss.detach().cpu())

    def compile_and_benchmark(kind: str) -> dict[str, Any]:
        if kind == "rlt":
            base = new_rlt()
            callable_module: nn.Module = ExactCompiledRLT(base)
            vocab_size = rlt_cfg.vocab_size
            params = parameter_count(base)
        else:
            base = new_transformer()
            callable_module = base
            vocab_size = transformer_cfg.vocab_size
            params = sum(p.numel() for p in base.parameters())
        opt = optimizer_for(base)
        compiled = torch.compile(callable_module, fullgraph=True, dynamic=False, mode="default")
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        first_loss = one_step(compiled, base, opt, 0, vocab_size)
        torch.cuda.synchronize(device)
        compile_first_step = time.perf_counter() - t0
        for step in range(WARMUP_STEPS):
            one_step(compiled, base, opt, 1 + step, vocab_size)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        measured_started = time.perf_counter()
        last_loss = first_loss
        for step in range(MEASURED_STEPS):
            last_loss = one_step(compiled, base, opt, 1 + WARMUP_STEPS + step, vocab_size)
        torch.cuda.synchronize(device)
        measured_seconds = time.perf_counter() - measured_started
        peak = int(torch.cuda.max_memory_allocated(device))
        measured_tokens = MEASURED_STEPS * BATCH_SIZE * SEQ_LEN
        result = {
            "parameters": params,
            "first_step_seconds_including_compile": compile_first_step,
            "measured_seconds": measured_seconds,
            "measured_tokens": measured_tokens,
            "tokens_per_second": measured_tokens / max(measured_seconds, 1e-9),
            "last_loss": last_loss,
            "peak_vram_gb": peak / (1024**3),
        }
        del compiled, opt, base, callable_module
        torch.cuda.empty_cache()
        torch._dynamo.reset()
        return result

    compile_error: str | None = None
    rlt_result: dict[str, Any] | None = None
    transformer_result: dict[str, Any] | None = None
    try:
        rlt_result = compile_and_benchmark("rlt")
        transformer_result = compile_and_benchmark("transformer")
    except Exception as exc:
        compile_error = f"{type(exc).__name__}: {exc}"

    if not semantic_equivalence:
        classification = "FAIL_SEMANTIC_EQUIVALENCE"
    elif compile_error is not None or rlt_result is None or transformer_result is None:
        classification = "FAIL_COMPILE_EXECUTION"
    else:
        rlt_scale = rlt_result["tokens_per_second"] / PRIOR_D_TPS
        gap = transformer_result["tokens_per_second"] / rlt_result["tokens_per_second"]
        classification = "PASS_BATCH_SCALING_MATERIAL" if rlt_scale >= 1.5 else "PASS_BATCH_SCALING_WEAK"
        rlt_result["speedup_vs_prior_d_batch2"] = rlt_scale
        transformer_result["throughput_multiple_vs_rlt"] = gap

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
        "profile": {
            "seq_len": SEQ_LEN,
            "batch_size": BATCH_SIZE,
            "warmup_steps": WARMUP_STEPS,
            "measured_steps": MEASURED_STEPS,
            "prior_d_batch2_compiled_tokens_per_second": PRIOR_D_TPS,
            "rlt_parameters": rlt_parameters,
        },
        "semantic_equivalence": {
            "passed": semantic_equivalence,
            "logits_max_abs": logits_max_abs,
            "loss_abs": loss_abs,
            "grad_max_abs": grad_max_abs,
            "candidate_difference": "diagnostic last_cache_lengths side effect omitted only",
        },
        "compile": {
            "fullgraph": True,
            "dynamic": False,
            "mode": "default",
            "cuda_graphs_requested": False,
            "error": compile_error,
        },
        "rlt": rlt_result,
        "transformer": transformer_result,
    }
    return _encode(payload)


@app.local_entrypoint()
def main() -> None:
    result = run_profile_remote.remote()
    if not isinstance(result, str):
        raise TypeError("remote systems result must be a base64 string")
    print("RLT_BATCHSCALE_RESULT_B64=" + result)
