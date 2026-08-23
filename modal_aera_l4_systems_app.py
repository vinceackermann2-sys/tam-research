from __future__ import annotations

import json
import time

import modal

APP_NAME = "tam-research-aera-l4-systems-gate"

app = modal.App(APP_NAME)
github_secret = modal.Secret.from_name("github-secret")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.7,<2.11", "PyGithub>=2.3,<3")
    .add_local_python_source("tam_research")
)


def _comment(repo_full_name: str, issue_number: int, body: str) -> None:
    if not repo_full_name or not issue_number:
        print(f"[status] {body}", flush=True)
        return
    import os

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print(f"[status] {body}", flush=True)
        return
    try:
        import github

        client = github.Github(auth=github.Auth.Token(token))
        client.get_repo(repo_full_name).get_issue(number=issue_number).create_comment(body)
    except Exception as exc:
        print(
            f"[status-report-nonfatal] {type(exc).__name__}: {exc}; body={body}",
            flush=True,
        )


@app.function(image=image, cpu=2, memory=4096, timeout=120)
def preflight() -> dict:
    """Zero-GPU import/config preflight before the L4 is allowed to spawn."""
    import torch

    from tam_research.aera_full import FullAERAConfig
    from tam_research.aera_integrated import IntegratedAERATextLM

    cfg = FullAERAConfig(
        vocab_size=257,
        d_model=64,
        n_stages=1,
        n_heads=4,
        local_window=16,
        max_seq_len=32,
        n_experts=8,
        top_k_experts=2,
        expert_mult=2,
        memory_dim=16,
        max_reason_steps=4,
    )
    model = IntegratedAERATextLM(cfg)
    total = sum(p.numel() for p in model.parameters())
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    with torch.inference_mode():
        out = model(x, hard=True, update_memory=False)
    logits = out["logits"]
    if not isinstance(logits, torch.Tensor) or logits.shape != (2, 16, cfg.vocab_size):
        raise RuntimeError(f"unexpected integrated output shape: {getattr(logits, 'shape', None)}")
    return {"ok": True, "parameters": total, "output_shape": list(logits.shape)}


@app.function(
    image=image,
    gpu="L4",
    cpu=8,
    memory=24576,
    timeout=600,
    secrets=[github_secret],
)
def run_systems_gate(repo_full_name: str = "", issue_number: int = 0) -> dict:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from tam_research.aera import StreamState
    from tam_research.aera_delta_memory import DeltaFastMemory
    from tam_research.aera_full import BudgetedLatentReasoner, FullAERAConfig, LocalCausalAttention
    from tam_research.aera_integrated import TrainableSparseExpertLayer

    if not torch.cuda.is_available():
        raise RuntimeError("AERA systems gate requires CUDA")

    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(6100)
    torch.cuda.manual_seed_all(6100)
    torch.backends.cuda.matmul.allow_tf32 = True

    _comment(
        repo_full_name,
        issue_number,
        "🧪 **AERA L4 systems gate started** — one L4, inference/microbenchmark only; "
        "no optimizer, no training, no model checkpoint. Measuring actual sparse expert, adaptive-depth, "
        "local-attention, recurrent-state, and fast-memory costs. Hard timeout=600s; no automatic retry.",
    )

    def bench(name: str, fn, *, warmup: int, iterations: int, events: int) -> dict:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=dtype):
            for _ in range(warmup):
                fn()
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                fn()
            end.record()
            end.synchronize()
        total_ms = float(start.elapsed_time(end))
        ms = total_ms / iterations
        peak_mb = float(torch.cuda.max_memory_allocated()) / (1024**2)
        return {
            "name": name,
            "milliseconds_per_iteration": ms,
            "events_per_second": float(events / (ms / 1000.0)),
            "peak_allocated_mb": peak_mb,
            "warmup_iterations": warmup,
            "measured_iterations": iterations,
        }

    results: dict[str, object] = {
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "dtype": "bfloat16",
        "scope": "systems_microbenchmark_no_training",
    }

    # ------------------------------------------------------------------
    # 1. Sparse experts vs same stored experts executed densely.
    # ------------------------------------------------------------------
    expert_cfg = FullAERAConfig(
        d_model=512,
        n_experts=8,
        top_k_experts=2,
        expert_mult=4,
        memory_dim=64,
        max_reason_steps=4,
    )
    sparse_layer = TrainableSparseExpertLayer(expert_cfg).to(device=device, dtype=dtype).eval()
    router = nn.Linear(512, 8).to(device=device, dtype=dtype).eval()
    x = torch.randn(4, 512, 512, device=device, dtype=dtype)
    event_count = x.size(0) * x.size(1)

    def sparse_experts():
        logits = router(x)
        return sparse_layer(x, logits)

    def dense_all_experts():
        logits = router(x)
        probs = F.softmax(logits.float(), dim=-1).to(x.dtype)
        out = torch.zeros_like(x)
        for expert_id, expert in enumerate(sparse_layer.experts):
            out = out + expert(x) * probs[..., expert_id : expert_id + 1]
        return out

    sparse_bench = bench("top2_of_8_sparse_experts", sparse_experts, warmup=8, iterations=24, events=event_count)
    dense_bench = bench("all8_dense_experts", dense_all_experts, warmup=8, iterations=24, events=event_count)
    expert_speedup = dense_bench["milliseconds_per_iteration"] / sparse_bench["milliseconds_per_iteration"]
    results["experts"] = {
        "stored_experts": 8,
        "active_experts_per_event": 2,
        "theoretical_active_fraction": 0.25,
        "sparse": sparse_bench,
        "dense_all": dense_bench,
        "measured_sparse_speedup_vs_dense": expert_speedup,
        "target_speedup_at_least_1_30": expert_speedup >= 1.30,
        "note": "Reference PyTorch dispatch; selected experts only execute, but no fused/grouped MoE kernel is used.",
    }
    _comment(
        repo_full_name,
        issue_number,
        f"📊 **Expert systems result** — top2/8 sparse={sparse_bench['milliseconds_per_iteration']:.3f}ms, "
        f"dense-all={dense_bench['milliseconds_per_iteration']:.3f}ms, speedup={expert_speedup:.2f}x.",
    )

    # ------------------------------------------------------------------
    # 2. Adaptive latent depth vs fixed max depth using identical cell.
    # ------------------------------------------------------------------
    reasoner = BudgetedLatentReasoner(512, 4).to(device=device, dtype=dtype).eval()
    z = torch.randn(4, 512, 512, device=device, dtype=dtype)
    n = z.size(0) * z.size(1)
    balanced_depth = torch.full((n, 4), -12.0, device=device, dtype=dtype)
    idx = torch.arange(n, device=device)
    balanced_depth[idx, idx % 4] = 12.0
    balanced_depth = balanced_depth.view(4, 512, 4)
    fixed_depth = torch.full_like(balanced_depth, -12.0)
    fixed_depth[..., 3] = 12.0

    def adaptive_depth():
        return reasoner(z, balanced_depth, hard=True)

    def fixed_max_depth():
        return reasoner(z, fixed_depth, hard=True)

    adaptive_bench = bench("hard_adaptive_depth_mean_2_5", adaptive_depth, warmup=8, iterations=30, events=n)
    fixed_bench = bench("fixed_depth_4", fixed_max_depth, warmup=8, iterations=30, events=n)
    depth_speedup = fixed_bench["milliseconds_per_iteration"] / adaptive_bench["milliseconds_per_iteration"]
    results["adaptive_reasoning"] = {
        "depth_distribution": [1, 2, 3, 4],
        "mean_requested_depth": 2.5,
        "fixed_baseline_depth": 4,
        "adaptive": adaptive_bench,
        "fixed_max": fixed_bench,
        "measured_speedup": depth_speedup,
        "target_speedup_at_least_1_20": depth_speedup >= 1.20,
    }
    _comment(
        repo_full_name,
        issue_number,
        f"📊 **Adaptive-depth systems result** — mean-depth2.5={adaptive_bench['milliseconds_per_iteration']:.3f}ms, "
        f"fixed-depth4={fixed_bench['milliseconds_per_iteration']:.3f}ms, speedup={depth_speedup:.2f}x.",
    )

    # ------------------------------------------------------------------
    # 3. Current local-attention implementation vs full causal SDPA.
    # This deliberately tests whether the bounded window saves wall-clock now.
    # ------------------------------------------------------------------
    local_attn = LocalCausalAttention(512, 8, 64).to(device=device, dtype=dtype).eval()

    class FullCausalAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.n_heads = 8
            self.head_dim = 64
            self.qkv = nn.Linear(512, 1536, bias=False)
            self.out = nn.Linear(512, 512, bias=False)

        def forward(self, inp):
            b, t, d = inp.shape
            q, k, v = self.qkv(inp).chunk(3, dim=-1)
            q = q.view(b, t, 8, 64).transpose(1, 2)
            k = k.view(b, t, 8, 64).transpose(1, 2)
            v = v.view(b, t, 8, 64).transpose(1, 2)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            return self.out(y.transpose(1, 2).contiguous().view(b, t, d))

    full_attn = FullCausalAttention().to(device=device, dtype=dtype).eval()
    with torch.no_grad():
        full_attn.qkv.weight.copy_(local_attn.qkv.weight)
        full_attn.out.weight.copy_(local_attn.out.weight)
    a = torch.randn(4, 512, 512, device=device, dtype=dtype)
    attn_events = a.size(0) * a.size(1)
    local_bench = bench("window64_local_attention", lambda: local_attn(a), warmup=10, iterations=30, events=attn_events)
    full_attn_bench = bench("full_causal_attention", lambda: full_attn(a), warmup=10, iterations=30, events=attn_events)
    local_speedup = full_attn_bench["milliseconds_per_iteration"] / local_bench["milliseconds_per_iteration"]
    results["attention"] = {
        "sequence_length": 512,
        "local_window": 64,
        "local": local_bench,
        "full": full_attn_bench,
        "measured_local_speedup_vs_full": local_speedup,
        "target_speedup_at_least_1_10": local_speedup >= 1.10,
        "note": "Current reference local attention uses an explicit T×T mask with SDPA; this test determines whether it has real hardware savings yet.",
    }
    _comment(
        repo_full_name,
        issue_number,
        f"📊 **Attention systems result** — window64={local_bench['milliseconds_per_iteration']:.3f}ms, "
        f"full={full_attn_bench['milliseconds_per_iteration']:.3f}ms, local/full speedup={local_speedup:.2f}x.",
    )

    # ------------------------------------------------------------------
    # 4. Reference Python GRUCell stream scan vs fused nn.GRU equivalent.
    # ------------------------------------------------------------------
    stream = StreamState(256).to(device=device, dtype=dtype).eval()
    fused_gru = nn.GRU(256, 256, batch_first=True).to(device=device, dtype=dtype).eval()
    with torch.no_grad():
        fused_gru.weight_ih_l0.copy_(stream.cell.weight_ih)
        fused_gru.weight_hh_l0.copy_(stream.cell.weight_hh)
        fused_gru.bias_ih_l0.copy_(stream.cell.bias_ih)
        fused_gru.bias_hh_l0.copy_(stream.cell.bias_hh)
    sx = torch.randn(4, 256, 256, device=device, dtype=dtype)
    s0 = torch.zeros(4, 256, device=device, dtype=dtype)
    stream_events = sx.size(0) * sx.size(1)

    def reference_stream():
        return stream(sx, s0)

    def fused_stream():
        return fused_gru(sx, s0.unsqueeze(0))

    ref_stream_bench = bench("python_grucell_scan", reference_stream, warmup=2, iterations=8, events=stream_events)
    fused_stream_bench = bench("fused_nn_gru", fused_stream, warmup=4, iterations=20, events=stream_events)
    stream_slowdown = ref_stream_bench["milliseconds_per_iteration"] / fused_stream_bench["milliseconds_per_iteration"]
    results["stream_state"] = {
        "reference": ref_stream_bench,
        "fused_equivalent": fused_stream_bench,
        "reference_slowdown_vs_fused": stream_slowdown,
        "target_reference_within_2x_fused": stream_slowdown <= 2.0,
        "note": "AERA semantics do not require GRU specifically; this quantifies the current Python-scan implementation debt.",
    }
    _comment(
        repo_full_name,
        issue_number,
        f"📊 **Stream-state systems result** — reference={ref_stream_bench['milliseconds_per_iteration']:.3f}ms, "
        f"fused GRU={fused_stream_bench['milliseconds_per_iteration']:.3f}ms, slowdown={stream_slowdown:.2f}x.",
    )

    # ------------------------------------------------------------------
    # 5. Fast neural memory read + local write overhead.
    # ------------------------------------------------------------------
    memory = DeltaFastMemory(256, 64, lr=0.2, decay=0.999).to(device=device, dtype=dtype).eval()
    mx = torch.randn(4, 128, 256, device=device, dtype=dtype)
    mstate = memory.empty_state(4, device, dtype)
    strength = torch.full((4, 128, 1), 0.25, device=device, dtype=dtype)
    memory_events = mx.size(0) * mx.size(1)
    memory_read_bench = bench("fast_memory_read", lambda: memory.read(mx, mstate), warmup=6, iterations=24, events=memory_events)
    # Write uses a sequential local update by design in the current reference path.
    memory_write_bench = bench("fast_memory_local_update", lambda: memory.local_update(mx, strength, mstate), warmup=1, iterations=5, events=memory_events)
    results["fast_memory"] = {
        "read": memory_read_bench,
        "local_update": memory_write_bench,
        "note": "Fast-memory updates mutate only session state, not base weights. Current update loops over event time and is expected to need a fused/scan implementation before scale.",
    }
    _comment(
        repo_full_name,
        issue_number,
        f"📊 **Fast-memory systems result** — read={memory_read_bench['milliseconds_per_iteration']:.3f}ms, "
        f"local-update={memory_write_bench['milliseconds_per_iteration']:.3f}ms per 512 events.",
    )

    expert_ok = bool(results["experts"]["target_speedup_at_least_1_30"])
    depth_ok = bool(results["adaptive_reasoning"]["target_speedup_at_least_1_20"])
    attention_ok = bool(results["attention"]["target_speedup_at_least_1_10"])
    stream_ok = bool(results["stream_state"]["target_reference_within_2x_fused"])
    systems_gate_pass = expert_ok and depth_ok and attention_ok and stream_ok
    results["systems_gate"] = {
        "pass": systems_gate_pass,
        "expert_sparse_target_pass": expert_ok,
        "adaptive_depth_target_pass": depth_ok,
        "local_attention_target_pass": attention_ok,
        "stream_reference_target_pass": stream_ok,
        "interpretation": (
            "PASS: current reference paths show enough real hardware benefit to proceed to a tiny real-language GPU smoke."
            if systems_gate_pass
            else "FAIL: one or more current reference paths do not yet convert conditional computation into adequate L4 wall-clock savings; optimize/replace the failed path before language-scale training."
        ),
    }

    _comment(
        repo_full_name,
        issue_number,
        "✅ **AERA L4 systems benchmark complete.**\n\n"
        f"- sparse experts speedup: **{expert_speedup:.2f}x** (target ≥1.30x)\n"
        f"- adaptive depth speedup: **{depth_speedup:.2f}x** (target ≥1.20x)\n"
        f"- local attention speedup vs full: **{local_speedup:.2f}x** (target ≥1.10x)\n"
        f"- reference stream slowdown vs fused GRU: **{stream_slowdown:.2f}x** (target ≤2.0x)\n"
        f"- systems gate: **{'PASS' if systems_gate_pass else 'FAIL'}**\n\n"
        "This benchmark is inference/systems-only and does not establish language quality or authorize 100M training.",
    )
    print(json.dumps(results, indent=2), flush=True)
    return results


@app.local_entrypoint()
def main(repo_full_name: str = "", issue_number: int = 0):
    check = preflight.remote()
    print(json.dumps({"preflight": check}, indent=2), flush=True)
    result = run_systems_gate.remote(repo_full_name, issue_number)
    print(json.dumps(result, indent=2), flush=True)
