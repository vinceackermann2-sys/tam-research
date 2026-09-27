from __future__ import annotations

import base64
import json
from pathlib import Path
import time
from typing import Any

import modal

JOB_ID = "rlt-systems-scale15m-q1-efficient-whole-modal-20260927-r"
ENGINEERING_SEED = 20_261_024
SEQ_LEN = 64
BATCH_SIZE = 768
EQUIV_BATCH = 2
WARMUP_STEPS = 2
MEASURED_STEPS = 3
EXPECTED_PARAMETERS = 15_129_344

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.10,<2.11", "numpy>=2.0,<3")
    .add_local_python_source("tam_research", "experiments")
)
app = modal.App("tam-rlt-systems-scale15m-q1-efficient-whole-20260927-r")


def _encode(value: dict[str, Any]) -> str:
    return base64.urlsafe_b64encode(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).decode()


@app.function(
    image=image, gpu="A100-80GB", cpu=4.0, memory=65536,
    timeout=70 * 60, retries=0, max_containers=1,
)
def run_profile_remote(job_json: str) -> str:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    from experiments.rlt.model import RLTConfig, RecurrentLoopedTransformer, parameter_count
    from experiments.rlt.train_compiled_pair_1m import CompilableRLT

    started = time.time()
    job = json.loads(job_json)
    expected = {
        "schema": 1, "job_id": JOB_ID,
        "task": "systems_scale15m_q1_efficient_whole_r",
        "engineering_seed": ENGINEERING_SEED,
        "requires_q1_profile_job_id": "rlt-systems-scale15m-q1-attention-modal-20260926-p",
        "requires_reduce_overhead_job_id": "rlt-systems-scale15m-reduce-overhead-modal-20260927-q",
        "batch_size": BATCH_SIZE, "seq_len": SEQ_LEN,
        "expected_parameters_each": EXPECTED_PARAMETERS,
        "automatic_retry": False,
    }
    if job != expected:
        raise RuntimeError(f"q1-efficient whole preregistration mismatch: {job}")
    if not torch.cuda.is_available():
        raise RuntimeError("q1-efficient whole profile requires CUDA")

    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    cfg = RLTConfig(
        vocab_size=50_257, d_model=256, n_heads=8, n_stages=2,
        max_seq_len=128, ff_mult=4, swa_window=32,
    )

    def seed_local(seed: int) -> None:
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    def new_base() -> RecurrentLoopedTransformer:
        seed_local(ENGINEERING_SEED)
        m = RecurrentLoopedTransformer(cfg).to(device)
        if parameter_count(m) != EXPECTED_PARAMETERS:
            raise RuntimeError(f"RLT parameter drift: {parameter_count(m)}")
        return m

    class Q1EfficientRLT(nn.Module):
        """Same RLT equations; only q_len=1 decoder SDPA backend is forced efficient."""
        def __init__(self, base: RecurrentLoopedTransformer):
            super().__init__()
            self.base = base

        @staticmethod
        def _split(x: torch.Tensor, heads: int) -> torch.Tensor:
            b, t, d = x.shape
            return x.view(b, t, heads, d // heads).transpose(1, 2)

        @staticmethod
        def _q1(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
                return F.scaled_dot_product_attention(q, k, v, is_causal=False)

        def forward(self, tokens: torch.Tensor) -> torch.Tensor:
            model = self.base
            b, t = tokens.shape
            memory = model.encode(tokens)  # encoder remains on the existing backend path
            memory_kv = [stage.cross_attn.precompute(memory) for stage in model.stages]
            state = model.start_state.to(memory.dtype).view(1, -1).expand(b, -1)
            caches: list[tuple[torch.Tensor, torch.Tensor] | None] = [None for _ in model.stages]
            logits: list[torch.Tensor] = []

            for token_index in range(t):
                merged = torch.cat((memory[:, token_index, :], state), dim=-1)
                hidden = model.merge_norm(model.merge(merged)).unsqueeze(1)
                prefix_len = token_index + 1
                for li, stage in enumerate(model.stages):
                    sx = stage.self_norm(hidden)
                    q, k, v = stage.self_attn.qkv(sx).chunk(3, dim=-1)
                    q = self._split(q, stage.self_attn.n_heads)
                    k = self._split(k, stage.self_attn.n_heads)
                    v = self._split(v, stage.self_attn.n_heads)
                    cache = caches[li]
                    if cache is not None:
                        k = torch.cat((cache[0], k), dim=2)
                        v = torch.cat((cache[1], v), dim=2)
                    if k.size(2) > model.cfg.swa_window:
                        k = k[:, :, -model.cfg.swa_window:, :]
                        v = v[:, :, -model.cfg.swa_window:, :]
                    local = self._q1(q, k, v)
                    local = local.transpose(1, 2).contiguous().view(b, 1, stage.self_attn.d_model)
                    hidden = hidden + stage.self_attn.out(local)
                    caches[li] = (k, v)

                    cx = stage.cross_norm(hidden)
                    cq = self._split(stage.cross_attn.q(cx), stage.cross_attn.n_heads)
                    mk = memory_kv[li][0][:, :, :prefix_len, :]
                    mv = memory_kv[li][1][:, :, :prefix_len, :]
                    cross = self._q1(cq, mk, mv)
                    cross = cross.transpose(1, 2).contiguous().view(b, 1, stage.cross_attn.d_model)
                    hidden = hidden + stage.cross_attn.out(cross)
                    hidden = hidden + stage.ff(stage.ff_norm(hidden))

                state = hidden[:, 0, :]
                logits.append(model.lm_head(model.output_norm(state)))
            return torch.stack(logits, dim=1)

    # Full-model numerical drift versus the exact/default wrapper.
    a_base = new_base(); b_base = new_base(); b_base.load_state_dict(a_base.state_dict())
    exact = CompilableRLT(a_base)
    efficient = Q1EfficientRLT(b_base)
    gen0 = torch.Generator(device="cpu").manual_seed(ENGINEERING_SEED + 99)
    x0 = torch.randint(0, cfg.vocab_size, (EQUIV_BATCH, SEQ_LEN), generator=gen0, dtype=torch.long).to(device)
    y0 = torch.randint(0, cfg.vocab_size, (EQUIV_BATCH, SEQ_LEN), generator=gen0, dtype=torch.long).to(device)
    a_base.zero_grad(set_to_none=True); b_base.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        la = exact(x0)
        lb = efficient(x0)
        loss_a = F.cross_entropy(la.float().reshape(-1, cfg.vocab_size), y0.reshape(-1))
        loss_b = F.cross_entropy(lb.float().reshape(-1, cfg.vocab_size), y0.reshape(-1))
    loss_a.backward(); loss_b.backward(); torch.cuda.synchronize(device)
    logits_abs = float((la - lb).abs().max().item())
    loss_abs = float((loss_a - loss_b).abs().item())
    grad_abs = 0.0
    for ap, bp in zip(a_base.parameters(), b_base.parameters()):
        if ap.grad is None or bp.grad is None:
            if ap.grad is not bp.grad:
                grad_abs = float("inf"); break
            continue
        grad_abs = max(grad_abs, float((ap.grad - bp.grad).abs().max().item()))
    numerical = {
        "bit_exact": logits_abs == 0.0 and loss_abs == 0.0 and grad_abs == 0.0,
        "logits_max_abs": logits_abs, "loss_abs": loss_abs, "grad_max_abs": grad_abs,
    }
    del a_base, b_base, exact, efficient, la, lb
    torch.cuda.empty_cache()

    gen = torch.Generator(device="cpu").manual_seed(ENGINEERING_SEED + 1)
    batches = [
        (
            torch.randint(0, cfg.vocab_size, (BATCH_SIZE, SEQ_LEN), generator=gen, dtype=torch.long).to(device),
            torch.randint(0, cfg.vocab_size, (BATCH_SIZE, SEQ_LEN), generator=gen, dtype=torch.long).to(device),
        )
        for _ in range(WARMUP_STEPS + MEASURED_STEPS + 2)
    ]

    def opt_for(m: nn.Module) -> torch.optim.Optimizer:
        return torch.optim.AdamW(m.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1, fused=True)

    def bench(kind: str) -> dict[str, Any]:
        base = new_base()
        module: nn.Module = CompilableRLT(base) if kind == "default" else Q1EfficientRLT(base)
        opt = opt_for(base)
        compiled = torch.compile(module, fullgraph=True, dynamic=False, mode="default")

        def step(i: int) -> float:
            base.train(); opt.zero_grad(set_to_none=True)
            x, y = batches[i % len(batches)]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = compiled(x)
                loss = F.cross_entropy(logits.float().reshape(-1, cfg.vocab_size), y.reshape(-1))
            loss.backward(); torch.nn.utils.clip_grad_norm_(base.parameters(), 1.0); opt.step()
            return float(loss.detach().cpu())

        torch.cuda.synchronize(device)
        t0=time.perf_counter(); first=step(0); torch.cuda.synchronize(device)
        compile_seconds=time.perf_counter()-t0
        for i in range(WARMUP_STEPS): step(i+1)
        torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device)
        t1=time.perf_counter(); last=first
        for i in range(MEASURED_STEPS): last=step(1+WARMUP_STEPS+i)
        torch.cuda.synchronize(device); elapsed=time.perf_counter()-t1
        tokens=MEASURED_STEPS*BATCH_SIZE*SEQ_LEN
        out={
            "first_step_seconds_including_compile":compile_seconds,
            "measured_seconds":elapsed,"measured_tokens":tokens,
            "tokens_per_second":tokens/max(elapsed,1e-9),"last_loss":last,
            "peak_vram_gb":torch.cuda.max_memory_allocated(device)/(1024**3),
        }
        del compiled,opt,base,module; torch.cuda.empty_cache(); torch._dynamo.reset()
        return out

    default = bench("default")
    efficient = bench("efficient")
    speedup = efficient["tokens_per_second"] / default["tokens_per_second"]

    return _encode({
        "schema":1,"job_id":JOB_ID,"status":"complete",
        "classification":(
            "PASS_SCALE15M_Q1_EFFICIENT_WHOLE_MATERIAL_SPEEDUP"
            if speedup>=1.10 else "PASS_SCALE15M_Q1_EFFICIENT_WHOLE_NO_MATERIAL_SPEEDUP"
        ),
        "engineering_only":True,"scientific_execution":False,
        "breakthrough_claim_supported":False,"engineering_seed":ENGINEERING_SEED,
        "started_unix":started,"finished_unix":time.time(),
        "runtime":{"torch_version":str(torch.__version__),
                   "cuda_runtime":None if torch.version.cuda is None else str(torch.version.cuda),
                   "gpu_name":str(torch.cuda.get_device_name(0)),
                   "bf16_supported":bool(torch.cuda.is_bf16_supported())},
        "profile":{"batch_size":BATCH_SIZE,"seq_len":SEQ_LEN,
                   "warmup_steps":WARMUP_STEPS,"measured_steps":MEASURED_STEPS,
                   "expected_parameters":EXPECTED_PARAMETERS,
                   "decoder_q1_backend":"EFFICIENT_ATTENTION","encoder_backend":"default"},
        "numerical_vs_default":numerical,
        "default":default,"q1_efficient":efficient,
        "derived":{"q1_efficient_speedup_vs_default":speedup},
        "interpretation_ceiling":(
            "Engineering-only approximate-numerics implementation profile. Decoder q_len=1 attention "
            "uses PyTorch memory-efficient SDPA while encoder math is unchanged. Any nonzero full-model "
            "logit/loss/gradient drift makes this ineligible for the exact-semantics scientific lane "
            "until quality is independently revalidated."
        ),
    })


@app.local_entrypoint()
def main(job_path: str) -> None:
    result=run_profile_remote.remote(Path(job_path).read_text())
    if not isinstance(result,str):
        raise TypeError("remote q1-efficient whole result must be base64 text")
    print("RLT_SYSTEMS_SCALE15M_Q1_EFFICIENT_WHOLE_R_RESULT_B64="+result)
