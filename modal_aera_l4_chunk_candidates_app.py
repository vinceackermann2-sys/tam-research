from __future__ import annotations

import json

import modal

APP_NAME = "tam-research-aera-l4-chunk-candidates"

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

    x = torch.randn(2, 16)
    y = ExpertMLP(16, 2)(x)
    if y.shape != x.shape:
        raise RuntimeError("chunk-candidate preflight shape mismatch")
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

    from tam_research.aera import ExpertMLP

    if not torch.cuda.is_available():
        raise RuntimeError("chunk candidate gate requires CUDA")

    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(6400)
    torch.cuda.manual_seed_all(6400)
    torch.backends.cuda.matmul.allow_tf32 = True

    _comment(
        repo_full_name,
        issue_number,
        "🧪 **AERA chunk-conditional candidate gate started on one L4** — routing and latent depth are decided per full processing chunk, then executed as large contiguous GPU work. No optimizer, no model training, hard timeout=600s, no automatic retry.",
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

    d_model = 512
    n_experts = 8
    top_k = 2
    seq_len = 512
    experts = nn.ModuleList(ExpertMLP(d_model, 4) for _ in range(n_experts)).to(device=device, dtype=dtype).eval()
    router = nn.Linear(d_model, n_experts).to(device=device, dtype=dtype).eval()

    def routed_experts(inp: torch.Tensor) -> torch.Tensor:
        # One routing decision per independent processing chunk/sequence. Grouping
        # happens at batch granularity, so each selected expert receives whole
        # contiguous [n_chunks, time, d_model] blocks rather than token fragments.
        b, t, d = inp.shape
        logits = router(inp.mean(dim=1))
        probs = F.softmax(logits.float(), dim=-1).to(inp.dtype)
        selected_probs, top_idx = torch.topk(probs, top_k, dim=-1)
        weights = selected_probs / selected_probs.sum(dim=-1, keepdim=True).detach().clamp_min(1e-6)
        out = torch.zeros_like(inp)
        for expert_id, expert in enumerate(experts):
            assignments = (top_idx == expert_id).nonzero(as_tuple=False)
            if assignments.numel() == 0:
                continue
            batch_idx = assignments[:, 0]
            route_idx = assignments[:, 1]
            selected = inp.index_select(0, batch_idx)
            contribution = expert(selected.reshape(-1, d)).view(-1, t, d)
            contribution = contribution * weights[batch_idx, route_idx, None, None]
            out = out.index_add(0, batch_idx, contribution.to(out.dtype))
        return out

    def dense_experts(inp: torch.Tensor) -> torch.Tensor:
        logits = router(inp.mean(dim=1))
        probs = F.softmax(logits.float(), dim=-1).to(inp.dtype)
        out = torch.zeros_like(inp)
        for expert_id, expert in enumerate(experts):
            out = out + expert(inp) * probs[:, expert_id, None, None]
        return out

    # Latency case: one chunk. This is the common interactive-agent case and has
    # no cross-request gather/scatter problem.
    x1 = torch.randn(1, seq_len, d_model, device=device, dtype=dtype)
    expert_single_sparse = bench("chunk_top2of8_batch1", lambda: routed_experts(x1), warmup=8, iterations=30, events=seq_len)
    expert_single_dense = bench("dense_all8_batch1", lambda: dense_experts(x1), warmup=8, iterations=30, events=seq_len)
    expert_single_speedup = expert_single_dense["milliseconds_per_iteration"] / expert_single_sparse["milliseconds_per_iteration"]

    # Throughput case: many independent chunks. Routing remains per chunk, then
    # whole sequences are grouped by selected expert.
    batch = 16
    xb = torch.randn(batch, seq_len, d_model, device=device, dtype=dtype)
    expert_batch_sparse = bench("chunk_grouped_top2of8_batch16", lambda: routed_experts(xb), warmup=6, iterations=20, events=batch * seq_len)
    expert_batch_dense = bench("dense_all8_batch16", lambda: dense_experts(xb), warmup=6, iterations=20, events=batch * seq_len)
    expert_batch_speedup = expert_batch_dense["milliseconds_per_iteration"] / expert_batch_sparse["milliseconds_per_iteration"]

    expert_single_ok = expert_single_speedup >= 2.0
    expert_batch_ok = expert_batch_speedup >= 1.30
    _comment(
        repo_full_name,
        issue_number,
        f"📊 **Chunk expert candidate** — batch1 speedup={expert_single_speedup:.2f}x (target ≥2.0x); grouped batch16 speedup={expert_batch_speedup:.2f}x (target ≥1.30x).",
    )

    # Chunk-level adaptive depth. One depth is chosen for every event in a chunk.
    # At batch=1 this removes all gather/scatter. At batched throughput, complete
    # chunks are grouped at each recurrent step, keeping matrix sizes large.
    cell = nn.GRUCell(d_model, d_model).to(device=device, dtype=dtype).eval()

    def fixed_depth(inp: torch.Tensor, steps: int) -> torch.Tensor:
        current = inp
        b, t, d = current.shape
        for _ in range(steps):
            current = cell(current.reshape(-1, d), current.reshape(-1, d)).view(b, t, d)
        return current

    def mixed_chunk_depth(inp: torch.Tensor, chosen: torch.Tensor) -> torch.Tensor:
        current = inp
        b, t, d = current.shape
        for step in range(1, 5):
            batch_idx = (chosen >= step).nonzero(as_tuple=False).squeeze(-1)
            if batch_idx.numel() == 0:
                break
            selected = current.index_select(0, batch_idx)
            updated = cell(selected.reshape(-1, d), selected.reshape(-1, d)).view_as(selected)
            current = current.index_copy(0, batch_idx, updated)
        return current

    # Latency: depth2 versus fixed4 is the cleanest hard-compute test.
    depth_single_2 = bench("chunk_depth2_batch1", lambda: fixed_depth(x1, 2), warmup=8, iterations=30, events=seq_len)
    depth_single_4 = bench("chunk_depth4_batch1", lambda: fixed_depth(x1, 4), warmup=8, iterations=30, events=seq_len)
    depth_single_speedup = depth_single_4["milliseconds_per_iteration"] / depth_single_2["milliseconds_per_iteration"]

    # Throughput: balanced depths 1/2/3/4, mean=2.5, versus all depth4.
    chosen = (torch.arange(batch, device=device) % 4) + 1
    depth_batch_adaptive = bench("chunk_mixed_depth_mean2_5_batch16", lambda: mixed_chunk_depth(xb, chosen), warmup=6, iterations=20, events=batch * seq_len)
    depth_batch_fixed = bench("chunk_fixed_depth4_batch16", lambda: fixed_depth(xb, 4), warmup=6, iterations=20, events=batch * seq_len)
    depth_batch_speedup = depth_batch_fixed["milliseconds_per_iteration"] / depth_batch_adaptive["milliseconds_per_iteration"]

    depth_single_ok = depth_single_speedup >= 1.70
    depth_batch_ok = depth_batch_speedup >= 1.20
    _comment(
        repo_full_name,
        issue_number,
        f"📊 **Chunk depth candidate** — batch1 depth2-vs4 speedup={depth_single_speedup:.2f}x (target ≥1.70x); mixed batch16 mean2.5 speedup={depth_batch_speedup:.2f}x (target ≥1.20x).",
    )

    passed = expert_single_ok and expert_batch_ok and depth_single_ok and depth_batch_ok
    result = {
        "device": torch.cuda.get_device_name(0),
        "torch_version": str(torch.__version__),
        "scope": "chunk_conditional_candidate_microbenchmark_no_training",
        "experts": {
            "routing_granularity": "one decision per 512-event chunk",
            "active_experts": 2,
            "stored_experts": 8,
            "batch1": {"sparse": expert_single_sparse, "dense": expert_single_dense, "speedup": expert_single_speedup, "pass": expert_single_ok},
            "batch16": {"sparse": expert_batch_sparse, "dense": expert_batch_dense, "speedup": expert_batch_speedup, "pass": expert_batch_ok},
        },
        "adaptive_depth": {
            "routing_granularity": "one depth per 512-event chunk",
            "batch1_depth2_vs4": {"depth2": depth_single_2, "depth4": depth_single_4, "speedup": depth_single_speedup, "pass": depth_single_ok},
            "batch16_balanced_mean2_5_vs4": {"adaptive": depth_batch_adaptive, "fixed": depth_batch_fixed, "speedup": depth_batch_speedup, "pass": depth_batch_ok},
        },
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
        "✅ **AERA chunk-conditional candidate gate complete.**\n\n"
        f"- expert batch1: **{expert_single_speedup:.2f}x** (target ≥2.0x)\n"
        f"- expert grouped batch16: **{expert_batch_speedup:.2f}x** (target ≥1.30x)\n"
        f"- depth batch1 (2 vs 4): **{depth_single_speedup:.2f}x** (target ≥1.70x)\n"
        f"- depth batch16 (mean2.5 vs 4): **{depth_batch_speedup:.2f}x** (target ≥1.20x)\n"
        f"- chunk-conditional candidate gate: **{'PASS' if passed else 'FAIL'}**\n\n"
        "This only tests hardware viability of the remaining conditional paths. It does not authorize 100M training.",
    )
    print(json.dumps(result, indent=2), flush=True)
    return result


@app.local_entrypoint()
def main(repo_full_name: str = "", issue_number: int = 0):
    check = preflight.remote()
    print(json.dumps({"preflight": check}, indent=2), flush=True)
    result = run_gate.remote(repo_full_name, issue_number)
    print(json.dumps(result, indent=2), flush=True)
