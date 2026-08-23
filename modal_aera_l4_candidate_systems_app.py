from __future__ import annotations

import json

import modal

APP_NAME = "tam-research-aera-l4-candidate-systems"

app = modal.App(APP_NAME)
github_secret = modal.Secret.from_name("github-secret")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.7,<2.11", "numpy>=2,<3", "PyGithub>=2.3,<3")
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
        print(f"[status-report-nonfatal] {type(exc).__name__}: {exc}; body={body}", flush=True)


@app.function(image=image, cpu=2, memory=4096, timeout=120)
def preflight() -> dict:
    import torch

    from tam_research.aera import ExpertMLP

    x = torch.randn(4, 32)
    expert = ExpertMLP(32, 2)
    y = expert(x)
    if y.shape != x.shape:
        raise RuntimeError("candidate preflight shape mismatch")
    return {"ok": True, "shape": list(y.shape)}


@app.function(
    image=image,
    gpu="L4",
    cpu=8,
    memory=24576,
    timeout=600,
    secrets=[github_secret],
)
def run_gate(repo_full_name: str = "", issue_number: int = 0) -> dict:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from tam_research.aera import ExpertMLP, StreamState
    from tam_research.aera_delta_memory import DeltaFastMemory
    from tam_research.aera_full import LocalCausalAttention

    if not torch.cuda.is_available():
        raise RuntimeError("candidate systems gate requires CUDA")

    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(6300)
    torch.cuda.manual_seed_all(6300)
    torch.backends.cuda.matmul.allow_tf32 = True

    _comment(
        repo_full_name,
        issue_number,
        "🧪 **Hardware-aware AERA candidate gate started on one L4** — patch-routed experts, "
        "patch-adaptive depth, Flash causal chunk attention, fused stream state, and patch-level fast-memory writes. "
        "Inference/microbenchmark only; no optimizer or model training; hard timeout=600s; no automatic retry.",
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
        return {
            "name": name,
            "milliseconds_per_iteration": ms,
            "events_per_second": float(events / (ms / 1000.0)),
            "peak_allocated_mb": float(torch.cuda.max_memory_allocated()) / (1024**2),
            "warmup_iterations": warmup,
            "measured_iterations": iterations,
        }

    # ------------------------------------------------------------------
    # 1. Patch-routed sparse experts: route groups of 32 events, then execute
    # contiguous expert GEMMs. This preserves top-k conditional computation but
    # changes the hardware unit of routing from token -> event patch.
    # ------------------------------------------------------------------
    d_model = 512
    hidden = 2048
    n_experts = 8
    top_k = 2
    patch_size = 32
    experts = nn.ModuleList(ExpertMLP(d_model, 4) for _ in range(n_experts)).to(device=device, dtype=dtype).eval()
    router = nn.Linear(d_model, n_experts).to(device=device, dtype=dtype).eval()
    x = torch.randn(4, 512, d_model, device=device, dtype=dtype)
    events = x.size(0) * x.size(1)

    def patch_sparse():
        b, t, d = x.shape
        groups = t // patch_size
        logits = router(x).view(b, groups, patch_size, n_experts).mean(dim=2)
        probs = F.softmax(logits.float(), dim=-1).to(x.dtype)
        selected_probs, top_idx = torch.topk(probs, top_k, dim=-1)
        weights = selected_probs / selected_probs.sum(dim=-1, keepdim=True).detach().clamp_min(1e-6)
        flat_groups = x.view(b * groups, patch_size, d)
        flat_idx = top_idx.view(b * groups, top_k)
        flat_weights = weights.view(b * groups, top_k)
        out = torch.zeros_like(flat_groups)
        for expert_id, expert in enumerate(experts):
            assignments = (flat_idx == expert_id).nonzero(as_tuple=False)
            if assignments.numel() == 0:
                continue
            group_idx = assignments[:, 0]
            route_idx = assignments[:, 1]
            selected = flat_groups.index_select(0, group_idx)
            contribution = expert(selected.reshape(-1, d)).view(-1, patch_size, d)
            contribution = contribution * flat_weights[group_idx, route_idx, None, None]
            out = out.index_add(0, group_idx, contribution.to(out.dtype))
        return out.view_as(x)

    def dense_all():
        logits = router(x)
        probs = F.softmax(logits.float(), dim=-1).to(x.dtype)
        out = torch.zeros_like(x)
        for expert_id, expert in enumerate(experts):
            out = out + expert(x) * probs[..., expert_id : expert_id + 1]
        return out

    patch_sparse_bench = bench("patch32_top2_of8", patch_sparse, warmup=8, iterations=30, events=events)
    dense_bench = bench("dense_all8", dense_all, warmup=8, iterations=30, events=events)
    expert_speedup = dense_bench["milliseconds_per_iteration"] / patch_sparse_bench["milliseconds_per_iteration"]
    expert_ok = expert_speedup >= 1.30
    _comment(
        repo_full_name,
        issue_number,
        f"📊 **Patch expert candidate** — patch32 top2/8={patch_sparse_bench['milliseconds_per_iteration']:.3f}ms, "
        f"dense-all={dense_bench['milliseconds_per_iteration']:.3f}ms, speedup={expert_speedup:.2f}x (target ≥1.30x).",
    )

    # ------------------------------------------------------------------
    # 2. Patch-adaptive latent recurrence. Depth is selected once per 32-event
    # patch, then each active patch is processed as one contiguous token batch.
    # ------------------------------------------------------------------
    cell = nn.GRUCell(d_model, d_model).to(device=device, dtype=dtype).eval()
    z = torch.randn(4, 512, d_model, device=device, dtype=dtype)
    b, t, d = z.shape
    groups = t // patch_size
    patch_depth = torch.arange(b * groups, device=device) % 4 + 1
    zg = z.view(b * groups, patch_size, d)

    def patch_adaptive_depth():
        current = zg
        for step in range(1, 5):
            group_idx = (patch_depth >= step).nonzero(as_tuple=False).squeeze(-1)
            if group_idx.numel() == 0:
                break
            selected = current.index_select(0, group_idx)
            updated = cell(selected.reshape(-1, d), selected.reshape(-1, d)).view_as(selected)
            current = current.index_copy(0, group_idx, updated)
        return current

    def fixed_depth4():
        current = zg
        for _ in range(4):
            current = cell(current.reshape(-1, d), current.reshape(-1, d)).view_as(current)
        return current

    adaptive_bench = bench("patch32_mean_depth2_5", patch_adaptive_depth, warmup=8, iterations=30, events=events)
    fixed_bench = bench("fixed_depth4", fixed_depth4, warmup=8, iterations=30, events=events)
    depth_speedup = fixed_bench["milliseconds_per_iteration"] / adaptive_bench["milliseconds_per_iteration"]
    depth_ok = depth_speedup >= 1.20
    _comment(
        repo_full_name,
        issue_number,
        f"📊 **Patch depth candidate** — mean-depth2.5={adaptive_bench['milliseconds_per_iteration']:.3f}ms, "
        f"fixed-depth4={fixed_bench['milliseconds_per_iteration']:.3f}ms, speedup={depth_speedup:.2f}x (target ≥1.20x).",
    )

    # ------------------------------------------------------------------
    # 3. Replace explicit masked local attention with Flash causal attention
    # inside the processing chunk. Long-range continuity comes from stream state;
    # no T×T custom mask is constructed.
    # ------------------------------------------------------------------
    current_local = LocalCausalAttention(d_model, 8, 64).to(device=device, dtype=dtype).eval()

    class FlashChunkAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
            self.out = nn.Linear(d_model, d_model, bias=False)

        def forward(self, inp):
            b0, t0, d0 = inp.shape
            q, k, v = self.qkv(inp).chunk(3, dim=-1)
            q = q.view(b0, t0, 8, 64).transpose(1, 2)
            k = k.view(b0, t0, 8, 64).transpose(1, 2)
            v = v.view(b0, t0, 8, 64).transpose(1, 2)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            return self.out(y.transpose(1, 2).contiguous().view(b0, t0, d0))

    flash = FlashChunkAttention().to(device=device, dtype=dtype).eval()
    with torch.no_grad():
        flash.qkv.weight.copy_(current_local.qkv.weight)
        flash.out.weight.copy_(current_local.out.weight)
    ax = torch.randn(4, 512, d_model, device=device, dtype=dtype)
    local_bench = bench("explicit_window64_mask", lambda: current_local(ax), warmup=10, iterations=40, events=events)
    flash_bench = bench("flash_causal_chunk512", lambda: flash(ax), warmup=10, iterations=40, events=events)
    attention_speedup = local_bench["milliseconds_per_iteration"] / flash_bench["milliseconds_per_iteration"]
    attention_ok = attention_speedup >= 1.50
    _comment(
        repo_full_name,
        issue_number,
        f"📊 **Attention candidate** — old-window-mask={local_bench['milliseconds_per_iteration']:.3f}ms, "
        f"Flash-chunk={flash_bench['milliseconds_per_iteration']:.3f}ms, replacement speedup={attention_speedup:.2f}x (target ≥1.50x).",
    )

    # ------------------------------------------------------------------
    # 4. Fused recurrent stream. nn.GRU is semantically equivalent to the current
    # GRUCell scan for carrying one hidden state across a chunk, but uses fused GPU
    # kernels.
    # ------------------------------------------------------------------
    old_stream = StreamState(256).to(device=device, dtype=dtype).eval()
    fused_stream = nn.GRU(256, 256, batch_first=True).to(device=device, dtype=dtype).eval()
    with torch.no_grad():
        fused_stream.weight_ih_l0.copy_(old_stream.cell.weight_ih)
        fused_stream.weight_hh_l0.copy_(old_stream.cell.weight_hh)
        fused_stream.bias_ih_l0.copy_(old_stream.cell.bias_ih)
        fused_stream.bias_hh_l0.copy_(old_stream.cell.bias_hh)
    sx = torch.randn(4, 256, 256, device=device, dtype=dtype)
    s0 = torch.zeros(4, 256, device=device, dtype=dtype)
    stream_events = sx.size(0) * sx.size(1)
    old_stream_bench = bench("python_grucell_stream", lambda: old_stream(sx, s0), warmup=2, iterations=8, events=stream_events)
    fused_stream_bench = bench("fused_gru_stream", lambda: fused_stream(sx, s0.unsqueeze(0)), warmup=4, iterations=20, events=stream_events)
    stream_speedup = old_stream_bench["milliseconds_per_iteration"] / fused_stream_bench["milliseconds_per_iteration"]
    stream_ok = stream_speedup >= 1.50
    _comment(
        repo_full_name,
        issue_number,
        f"📊 **Stream candidate** — old={old_stream_bench['milliseconds_per_iteration']:.3f}ms, "
        f"fused={fused_stream_bench['milliseconds_per_iteration']:.3f}ms, speedup={stream_speedup:.2f}x (target ≥1.50x).",
    )

    # ------------------------------------------------------------------
    # 5. Patch-level memory writes. Read every event, but compress writes into one
    # weighted summary per 16-event patch before applying the same delta rule.
    # ------------------------------------------------------------------
    memory = DeltaFastMemory(256, 64, lr=0.2, decay=0.999).to(device=device, dtype=dtype).eval()
    mx = torch.randn(4, 128, 256, device=device, dtype=dtype)
    strength = torch.rand(4, 128, 1, device=device, dtype=dtype)
    mstate = memory.empty_state(4, device, dtype)
    memory_events = mx.size(0) * mx.size(1)
    memory_patch = 16

    def full_memory_update():
        return memory.local_update(mx, strength, mstate)

    def patch_memory_update():
        b0, t0, d0 = mx.shape
        g0 = t0 // memory_patch
        gx = mx.view(b0, g0, memory_patch, d0)
        gs = strength.view(b0, g0, memory_patch, 1)
        denom = gs.float().sum(dim=2).clamp_min(1e-4).to(mx.dtype)
        summary = (gx * gs).sum(dim=2) / denom
        patch_strength = gs.amax(dim=2)
        return memory.local_update(summary, patch_strength, mstate)

    full_memory_bench = bench("per_event_memory_write", full_memory_update, warmup=1, iterations=5, events=memory_events)
    patch_memory_bench = bench("patch16_memory_write", patch_memory_update, warmup=2, iterations=12, events=memory_events)
    memory_speedup = full_memory_bench["milliseconds_per_iteration"] / patch_memory_bench["milliseconds_per_iteration"]
    memory_ok = memory_speedup >= 4.0
    _comment(
        repo_full_name,
        issue_number,
        f"📊 **Memory candidate** — per-event={full_memory_bench['milliseconds_per_iteration']:.3f}ms, "
        f"patch16={patch_memory_bench['milliseconds_per_iteration']:.3f}ms, speedup={memory_speedup:.2f}x (target ≥4.0x).",
    )

    passed = expert_ok and depth_ok and attention_ok and stream_ok and memory_ok
    result = {
        "device": torch.cuda.get_device_name(0),
        "torch_version": str(torch.__version__),
        "scope": "hardware_aware_candidate_microbenchmark_no_training",
        "experts": {
            "patch_size": patch_size,
            "sparse": patch_sparse_bench,
            "dense": dense_bench,
            "speedup": expert_speedup,
            "pass": expert_ok,
        },
        "adaptive_depth": {
            "patch_size": patch_size,
            "adaptive": adaptive_bench,
            "fixed": fixed_bench,
            "speedup": depth_speedup,
            "pass": depth_ok,
        },
        "attention": {
            "old_local": local_bench,
            "flash_chunk": flash_bench,
            "replacement_speedup": attention_speedup,
            "pass": attention_ok,
        },
        "stream": {
            "old": old_stream_bench,
            "fused": fused_stream_bench,
            "replacement_speedup": stream_speedup,
            "pass": stream_ok,
        },
        "memory_write": {
            "patch_size": memory_patch,
            "per_event": full_memory_bench,
            "patch": patch_memory_bench,
            "speedup": memory_speedup,
            "pass": memory_ok,
        },
        "candidate_systems_gate": {"pass": passed},
        "claims": {
            "integrated_architecture_updated": False,
            "language_quality_proven": False,
            "100m_authorized": False,
            "breakthrough_proven": False,
        },
    }

    _comment(
        repo_full_name,
        issue_number,
        "✅ **Hardware-aware AERA candidate gate complete.**\n\n"
        f"- patch experts: **{expert_speedup:.2f}x** (target ≥1.30x)\n"
        f"- patch adaptive depth: **{depth_speedup:.2f}x** (target ≥1.20x)\n"
        f"- Flash chunk attention replacement: **{attention_speedup:.2f}x** (target ≥1.50x vs old masked local)\n"
        f"- fused stream replacement: **{stream_speedup:.2f}x** (target ≥1.50x)\n"
        f"- patch memory writes: **{memory_speedup:.2f}x** (target ≥4.0x)\n"
        f"- candidate systems gate: **{'PASS' if passed else 'FAIL'}**\n\n"
        "Candidate benchmark only. A passing result means integrate these mechanisms and rerun CPU/quality gates; it does not authorize 100M training by itself.",
    )
    print(json.dumps(result, indent=2), flush=True)
    return result


@app.local_entrypoint()
def main(repo_full_name: str = "", issue_number: int = 0):
    check = preflight.remote()
    print(json.dumps({"preflight": check}, indent=2), flush=True)
    result = run_gate.remote(repo_full_name, issue_number)
    print(json.dumps(result, indent=2), flush=True)
