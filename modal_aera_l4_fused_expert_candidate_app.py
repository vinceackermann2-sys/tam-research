from __future__ import annotations

import json

import modal

APP_NAME = "tam-research-aera-l4-fused-expert-candidate"

app = modal.App(APP_NAME)
github_secret = modal.Secret.from_name("github-secret")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.7,<2.11", "numpy>=2,<3", "PyGithub>=2.3,<3")
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


def _fused_topk_reference_check() -> dict:
    import torch
    import torch.nn.functional as F

    torch.manual_seed(6500)
    t, d, h, e, k = 7, 8, 16, 4, 2
    x = torch.randn(t, d)
    w1 = torch.randn(e, h, d)
    w2 = torch.randn(e, d, h)
    logits = torch.randn(e)
    probs = F.softmax(logits.float(), dim=-1).to(x.dtype)
    selected_probs, idx = torch.topk(probs, k)
    weights = selected_probs / selected_probs.sum().clamp_min(1e-6)

    sw1 = w1.index_select(0, idx)
    sw2 = w2.index_select(0, idx)
    hidden = torch.einsum("td,khd->kth", x, sw1)
    hidden = F.gelu(hidden)
    candidate = torch.einsum("kth,kdh->ktd", hidden, sw2)
    fused = (candidate * weights[:, None, None]).sum(dim=0)

    explicit = torch.zeros_like(x)
    for route in range(k):
        expert_id = int(idx[route])
        y = F.linear(F.gelu(F.linear(x, w1[expert_id])), w2[expert_id])
        explicit = explicit + y * weights[route]

    max_error = float((fused - explicit).abs().max())
    if max_error > 1e-5:
        raise RuntimeError(f"fused top-k reference mismatch: {max_error}")
    return {"ok": True, "max_abs_error": max_error}


@app.function(image=image, cpu=2, memory=4096, timeout=120)
def preflight() -> dict:
    return _fused_topk_reference_check()


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

    if not torch.cuda.is_available():
        raise RuntimeError("fused expert candidate requires CUDA")

    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(6500)
    torch.cuda.manual_seed_all(6500)
    torch.backends.cuda.matmul.allow_tf32 = True

    _comment(
        repo_full_name,
        issue_number,
        "🧪 **AERA fused batch-1 expert candidate started on one L4** — one routing decision per 512-event chunk; selected top-2 expert weights are gathered as stacked tensors and executed with batched GEMMs. No optimizer, no training, hard timeout=600s, no automatic retry.",
    )

    def bench(name: str, fn, *, warmup: int = 10, iterations: int = 40) -> dict:
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
            "peak_allocated_mb": float(torch.cuda.max_memory_allocated()) / (1024**2),
            "warmup_iterations": warmup,
            "measured_iterations": iterations,
        }

    d_model = 512
    hidden = d_model * 4
    n_experts = 8
    top_k = 2
    seq_len = 512

    # Production-oriented layout: all expert weights live in contiguous stacked
    # tensors. Sparse batch-1 execution gathers only the selected experts, avoiding
    # the 8-expert Python dispatch and token index_select/index_add path that failed
    # the previous latency gate.
    w1 = nn.Parameter(torch.empty(n_experts, hidden, d_model, device=device, dtype=dtype), requires_grad=False)
    w2 = nn.Parameter(torch.empty(n_experts, d_model, hidden, device=device, dtype=dtype), requires_grad=False)
    nn.init.normal_(w1, mean=0.0, std=0.02)
    nn.init.normal_(w2, mean=0.0, std=0.02)
    router = nn.Linear(d_model, n_experts, bias=True).to(device=device, dtype=dtype).eval()
    x = torch.randn(1, seq_len, d_model, device=device, dtype=dtype)

    def route(inp: torch.Tensor):
        logits = router(inp.mean(dim=1))
        probs = F.softmax(logits.float(), dim=-1).to(inp.dtype)
        selected_probs, idx = torch.topk(probs, top_k, dim=-1)
        weights = selected_probs / selected_probs.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return probs, idx.squeeze(0), weights.squeeze(0)

    def fused_sparse(inp: torch.Tensor) -> torch.Tensor:
        _, idx, weights = route(inp)
        sw1 = w1.index_select(0, idx)
        sw2 = w2.index_select(0, idx)
        tokens = inp.squeeze(0)
        h = torch.einsum("td,khd->kth", tokens, sw1)
        h = F.gelu(h)
        y = torch.einsum("kth,kdh->ktd", h, sw2)
        return (y * weights[:, None, None]).sum(dim=0, keepdim=True)

    def fused_dense(inp: torch.Tensor) -> torch.Tensor:
        probs, _, _ = route(inp)
        tokens = inp.squeeze(0)
        h = torch.einsum("td,ehd->eth", tokens, w1)
        h = F.gelu(h)
        y = torch.einsum("eth,edh->etd", h, w2)
        return (y * probs.squeeze(0)[:, None, None]).sum(dim=0, keepdim=True)

    sparse = bench("stacked_top2of8_batch1", lambda: fused_sparse(x))
    dense = bench("stacked_all8_batch1", lambda: fused_dense(x))
    speedup = dense["milliseconds_per_iteration"] / sparse["milliseconds_per_iteration"]

    # Validate numerical semantics against explicit top-k execution using the same
    # stacked weights. This is outside the timed region.
    with torch.inference_mode():
        probs, idx, weights = route(x)
        explicit = torch.zeros_like(x)
        for route_id in range(top_k):
            expert_id = idx[route_id]
            sw1 = w1.index_select(0, expert_id.view(1)).squeeze(0)
            sw2 = w2.index_select(0, expert_id.view(1)).squeeze(0)
            y = F.linear(F.gelu(F.linear(x.squeeze(0), sw1)), sw2).unsqueeze(0)
            explicit = explicit + y * weights[route_id]
        candidate = fused_sparse(x)
        max_abs_error = float((candidate.float() - explicit.float()).abs().max())

    # Same scientific threshold as the chunk-level batch-1 target. We also require
    # numerical agreement; this candidate only clears the remaining systems blocker.
    pass_speed = speedup >= 2.0
    pass_numerics = max_abs_error <= 0.05
    passed = pass_speed and pass_numerics

    _comment(
        repo_full_name,
        issue_number,
        f"📊 **Fused expert candidate** — stacked top2/8={sparse['milliseconds_per_iteration']:.3f}ms, stacked dense-all8={dense['milliseconds_per_iteration']:.3f}ms, speedup={speedup:.2f}x (target ≥2.0x), max BF16 abs error={max_abs_error:.5f}.",
    )

    result = {
        "device": torch.cuda.get_device_name(0),
        "torch_version": str(torch.__version__),
        "scope": "fused_stacked_batch1_expert_candidate_no_training",
        "routing_granularity": "one decision per 512-event chunk",
        "stored_experts": n_experts,
        "active_experts": top_k,
        "sparse": sparse,
        "dense": dense,
        "speedup": speedup,
        "max_bf16_abs_error_vs_explicit": max_abs_error,
        "thresholds": {"speedup_min": 2.0, "max_abs_error_max": 0.05},
        "candidate_gate": {"pass": passed},
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
        "✅ **AERA fused batch-1 expert candidate complete.**\n\n"
        f"- batch1 top2/8 speedup: **{speedup:.2f}x** (target ≥2.0x)\n"
        f"- numerical agreement: max BF16 abs error **{max_abs_error:.5f}** (target ≤0.05)\n"
        f"- fused expert candidate gate: **{'PASS' if passed else 'FAIL'}**\n\n"
        "Passing clears only the remaining expert hardware candidate. Integration plus correctness and real-language gates are still required before 100M training.",
    )
    print(json.dumps(result, indent=2), flush=True)
    return result


@app.local_entrypoint()
def main(repo_full_name: str = "", issue_number: int = 0):
    check = preflight.remote()
    print(json.dumps({"preflight": check}, indent=2), flush=True)
    result = run_gate.remote(repo_full_name, issue_number)
    print(json.dumps(result, indent=2), flush=True)
