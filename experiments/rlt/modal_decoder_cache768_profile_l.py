from __future__ import annotations

import base64
import json
import time
from typing import Any

import modal

ENGINEERING_SEED = 20261004
SEQ_LEN = 64
BATCH_SIZE = 768
EQUIV_BATCH = 2
WARMUP_STEPS = 1
MEASURED_STEPS = 2
PRIOR_J_TPS = 178815.08311774963

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.10,<2.11")
    .add_local_python_source("tam_research", "experiments")
)
app = modal.App("tam-rlt-decoder-cache768-systems-20260917-l")


def _encode(value: dict[str, Any]) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).decode()


@app.function(
    image=image,
    gpu="A100",
    cpu=4.0,
    memory=65536,
    timeout=45 * 60,
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
        raise RuntimeError("decoder-cache768 systems profile requires CUDA")
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")

    cfg = RLTConfig(vocab_size=50_257, d_model=128, n_heads=4, n_stages=2,
                    max_seq_len=128, ff_mult=4, swa_window=32)

    def seed_local(seed: int) -> None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    class CacheFastRLT(nn.Module):
        def __init__(self, base: RecurrentLoopedTransformer):
            super().__init__()
            self.base = base

        @staticmethod
        def self_step_fast(attn: Any, x: torch.Tensor, cache: Any, window: int):
            q, k, v = attn.qkv(x).chunk(3, dim=-1)
            q, k, v = attn._split(q), attn._split(k), attn._split(v)
            if cache is not None:
                if cache[0].size(2) >= window:
                    k = torch.cat((cache[0][:, :, 1:, :], k), dim=2)
                    v = torch.cat((cache[1][:, :, 1:, :], v), dim=2)
                else:
                    k = torch.cat((cache[0], k), dim=2)
                    v = torch.cat((cache[1], v), dim=2)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
            b = x.size(0)
            y = y.transpose(1, 2).contiguous().view(b, 1, attn.d_model)
            return attn.out(y), (k, v)

        def forward(self, tokens: torch.Tensor) -> torch.Tensor:
            model = self.base
            b, t = tokens.shape
            memory = model.encode(tokens)
            memory_kv = [stage.cross_attn.precompute(memory) for stage in model.stages]
            state = model.start_state.to(memory.dtype).view(1, -1).expand(b, -1)
            caches: list[Any] = [None for _ in model.stages]
            logits: list[torch.Tensor] = []
            for token_index in range(t):
                merged = torch.cat((memory[:, token_index, :], state), dim=-1)
                hidden = model.merge_norm(model.merge(merged)).unsqueeze(1)
                prefix_len = token_index + 1
                for layer_index, stage in enumerate(model.stages):
                    local, caches[layer_index] = self.self_step_fast(
                        stage.self_attn, stage.self_norm(hidden), caches[layer_index], model.cfg.swa_window
                    )
                    hidden = hidden + local
                    hidden = hidden + stage.cross_attn.step(
                        stage.cross_norm(hidden), memory_kv[layer_index], prefix_len
                    )
                    hidden = hidden + stage.ff(stage.ff_norm(hidden))
                state = hidden[:, 0, :]
                logits.append(model.lm_head(model.output_norm(state)))
            return torch.stack(logits, dim=1)

    def new_rlt() -> RecurrentLoopedTransformer:
        seed_local(ENGINEERING_SEED)
        return RecurrentLoopedTransformer(cfg).to(device)

    reference = new_rlt()
    candidate_base = new_rlt()
    candidate_base.load_state_dict(reference.state_dict())
    candidate = CacheFastRLT(candidate_base)
    gen = torch.Generator(device="cpu").manual_seed(ENGINEERING_SEED + 99)
    x0 = torch.randint(0, cfg.vocab_size, (EQUIV_BATCH, SEQ_LEN), generator=gen, dtype=torch.long).to(device)
    y0 = torch.randint(0, cfg.vocab_size, (EQUIV_BATCH, SEQ_LEN), generator=gen, dtype=torch.long).to(device)
    reference.zero_grad(set_to_none=True); candidate_base.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        ref_logits = reference(x0); cand_logits = candidate(x0)
        ref_loss = F.cross_entropy(ref_logits.float().reshape(-1, cfg.vocab_size), y0.reshape(-1))
        cand_loss = F.cross_entropy(cand_logits.float().reshape(-1, cfg.vocab_size), y0.reshape(-1))
    ref_loss.backward(); cand_loss.backward(); torch.cuda.synchronize(device)
    logits_max_abs = float((ref_logits - cand_logits).abs().max().item())
    loss_abs = float((ref_loss - cand_loss).abs().item())
    grad_max_abs = 0.0
    for ref_p, cand_p in zip(reference.parameters(), candidate_base.parameters()):
        if ref_p.grad is None or cand_p.grad is None:
            if ref_p.grad is not cand_p.grad:
                grad_max_abs = float("inf"); break
            continue
        grad_max_abs = max(grad_max_abs, float((ref_p.grad - cand_p.grad).abs().max().item()))
    semantic_equivalence = logits_max_abs == 0.0 and loss_abs == 0.0 and grad_max_abs == 0.0
    params = parameter_count(reference)
    del reference, candidate_base, candidate, ref_logits, cand_logits
    torch.cuda.empty_cache()

    generator = torch.Generator(device="cpu").manual_seed(ENGINEERING_SEED + 1)
    batches = []
    for _ in range(WARMUP_STEPS + MEASURED_STEPS + 3):
        x = torch.randint(0, cfg.vocab_size, (BATCH_SIZE, SEQ_LEN), generator=generator, dtype=torch.long).to(device)
        y = torch.randint(0, cfg.vocab_size, (BATCH_SIZE, SEQ_LEN), generator=generator, dtype=torch.long).to(device)
        batches.append((x, y))

    def one_step(callable_model: Any, parameter_model: nn.Module, optimizer: Any, offset: int) -> float:
        parameter_model.train(); optimizer.zero_grad(set_to_none=True)
        x, y = batches[offset % len(batches)]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = callable_model(x)
            loss = F.cross_entropy(logits.float().reshape(-1, cfg.vocab_size), y.reshape(-1))
        loss.backward(); torch.nn.utils.clip_grad_norm_(parameter_model.parameters(), 1.0); optimizer.step()
        return float(loss.detach().cpu())

    benchmark = None
    compile_error = None
    if semantic_equivalence:
        try:
            base = new_rlt(); candidate = CacheFastRLT(base)
            optimizer = torch.optim.AdamW(base.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1, fused=True)
            compiled = torch.compile(candidate, fullgraph=True, dynamic=False, mode="default")
            torch.cuda.synchronize(device); t0 = time.perf_counter()
            first_loss = one_step(compiled, base, optimizer, 0)
            torch.cuda.synchronize(device); first_step_seconds = time.perf_counter() - t0
            for step in range(WARMUP_STEPS): one_step(compiled, base, optimizer, 1 + step)
            torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device)
            measured_started = time.perf_counter(); last_loss = first_loss
            for step in range(MEASURED_STEPS):
                last_loss = one_step(compiled, base, optimizer, 1 + WARMUP_STEPS + step)
            torch.cuda.synchronize(device); measured_seconds = time.perf_counter() - measured_started
            measured_tokens = MEASURED_STEPS * BATCH_SIZE * SEQ_LEN
            peak = int(torch.cuda.max_memory_allocated(device)); tps = measured_tokens / max(measured_seconds, 1e-9)
            benchmark = {
                "variant":"cache_fast_path", "parameters":parameter_count(base),
                "first_step_seconds_including_compile":first_step_seconds,
                "measured_seconds":measured_seconds, "measured_tokens":measured_tokens,
                "tokens_per_second":tps, "speedup_vs_j_batch768":tps / PRIOR_J_TPS,
                "last_loss":last_loss, "peak_vram_gb":peak/(1024**3)
            }
        except Exception as exc:
            compile_error = f"{type(exc).__name__}: {exc}"

    if not semantic_equivalence:
        classification = "FAIL_CACHE_FAST_PATH_EQUIVALENCE"
    elif compile_error is not None or benchmark is None:
        classification = "FAIL_COMPILE_EXECUTION"
    else:
        classification = "PASS_CACHE768_MATERIAL" if benchmark["speedup_vs_j_batch768"] >= 1.05 else "PASS_CACHE768_WEAK"

    payload = {
        "schema":1, "status":"complete", "classification":classification,
        "engineering_only":True, "scientific_execution":False,
        "engineering_seed":ENGINEERING_SEED, "started_unix":started, "finished_unix":time.time(),
        "runtime":{"torch_version":str(torch.__version__), "cuda_runtime":None if torch.version.cuda is None else str(torch.version.cuda), "gpu_name":str(torch.cuda.get_device_name(0)), "bf16_supported":bool(torch.cuda.is_bf16_supported())},
        "profile":{"seq_len":SEQ_LEN,"batch_size":BATCH_SIZE,"equivalence_batch_size":EQUIV_BATCH,"prior_j_batch768_tokens_per_second":PRIOR_J_TPS,"swa_window":cfg.swa_window},
        "semantic_equivalence":{"passed":semantic_equivalence,"logits_max_abs":logits_max_abs,"loss_abs":loss_abs,"grad_max_abs":grad_max_abs,"candidate_change":"saturated SWA cache fast path only"},
        "compile":{"fullgraph":True,"dynamic":False,"mode":"default","cuda_graphs_requested":False,"error":compile_error},
        "benchmark":benchmark,
    }
    return _encode(payload)


@app.local_entrypoint()
def main() -> None:
    result = run_profile_remote.remote()
    if not isinstance(result, str):
        raise TypeError("remote systems result must be a base64 string")
    print("RLT_DECODER_CACHE768_RESULT_B64=" + result)
