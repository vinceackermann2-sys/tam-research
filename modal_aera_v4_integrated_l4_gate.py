from __future__ import annotations

import json

import modal

APP_NAME = "tam-research-aera-v4-integrated-l4"
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

    from tam_research.aera_hardware_core import HardwareAERAConfig
    from tam_research.aera_hardware_core_v4 import HardwareAwareAERATextLMV4

    cfg = HardwareAERAConfig(
        vocab_size=127,
        d_model=32,
        n_stages=1,
        n_heads=4,
        chunk_size=16,
        n_experts=4,
        max_active_experts=2,
        expert_mult=2,
        memory_dim=8,
        max_reason_steps=2,
        block_size=3,
    )
    model = HardwareAwareAERATextLMV4(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (2, 31))
    with torch.no_grad():
        out = model(x, hard=True, update_memory=False)
    logits = out["logits"]
    if logits.shape != (2, 31, cfg.vocab_size):
        raise RuntimeError("integrated L4 preflight shape mismatch")
    return {"ok": True, "shape": list(logits.shape)}


@app.function(
    image=image,
    gpu="L4",
    cpu=8,
    memory=24576,
    timeout=600,
    secrets=[github_secret],
)
def run_gate(repo_full_name: str = "", issue_number: int = 0) -> dict:
    import copy

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from tam_research.aera_hardware_core import HardwareAERAConfig
    from tam_research.aera_hardware_core_v4 import HardwareAwareAERATextLMV4

    if not torch.cuda.is_available():
        raise RuntimeError("integrated AERA gate requires CUDA")

    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(7300)
    torch.cuda.manual_seed_all(7300)
    torch.backends.cuda.matmul.allow_tf32 = True

    _comment(
        repo_full_name,
        issue_number,
        "🧪 **Integrated AERA-v4 L4 gate started** — full chunk-stateful model forward, Flash causal attention, predictive recurrent state, true hard top1/top2 experts, adaptive latent depth, and fast-memory read path. Inference only; no optimizer or weight updates; hard timeout=600s; no automatic retry.",
    )

    cfg = HardwareAERAConfig(
        vocab_size=50257,
        d_model=512,
        n_stages=4,
        n_heads=8,
        chunk_size=128,
        n_experts=8,
        max_active_experts=2,
        expert_mult=4,
        memory_dim=64,
        max_reason_steps=4,
        block_size=4,
    )
    base = HardwareAwareAERATextLMV4(cfg).to(device=device, dtype=dtype).eval()

    class DenseAllExpertBank(nn.Module):
        """Same stacked expert weights, but evaluate all stored experts."""

        def __init__(self, source):
            super().__init__()
            self.w1 = nn.Parameter(source.w1.detach().clone(), requires_grad=False)
            self.w2 = nn.Parameter(source.w2.detach().clone(), requires_grad=False)
            self.n_experts = source.n_experts

        def forward(self, x, expert_logits, count_logits, *, hard):
            probs = F.softmax(expert_logits.float(), dim=-1).to(x.dtype)
            h = torch.einsum("btd,ehd->beth", x, self.w1)
            h = F.gelu(h)
            y = torch.einsum("beth,edh->betd", h, self.w2)
            return (y * probs[:, :, None, None]).sum(dim=1)

        def stats(self):
            return {"stored_experts": self.n_experts, "mean_active_experts": float(self.n_experts)}

    class FixedDepthReasoner(nn.Module):
        def __init__(self, source, max_steps: int):
            super().__init__()
            self.cell = copy.deepcopy(source.cell)
            self.max_steps = max_steps

        def forward(self, summary, depth_logits, *, hard):
            current = summary
            for _ in range(self.max_steps):
                current = self.cell(current, current)
            return current

        def stats(self):
            return {"mode": "fixed", "mean": float(self.max_steps), "min": float(self.max_steps), "max": float(self.max_steps)}

    def force_profile(model, *, expert_count: int, depth: int) -> None:
        if expert_count not in (1, 2):
            raise ValueError("expert_count must be 1 or 2")
        if not 1 <= depth <= cfg.max_reason_steps:
            raise ValueError("invalid depth")
        with torch.no_grad():
            for stage in model.stages:
                proj = stage.controller.proj
                e = cfg.n_experts
                # Preserve expert identity logits; force only count and depth.
                proj.weight[e : e + 2].zero_()
                proj.bias[e : e + 2].fill_(-8.0)
                proj.bias[e + expert_count - 1] = 8.0
                d0 = e + 2
                proj.weight[d0 : d0 + cfg.max_reason_steps].zero_()
                proj.bias[d0 : d0 + cfg.max_reason_steps].fill_(-8.0)
                proj.bias[d0 + depth - 1] = 8.0

    lean = copy.deepcopy(base)
    heavy = copy.deepcopy(base)
    dense = copy.deepcopy(base)
    force_profile(lean, expert_count=1, depth=2)
    force_profile(heavy, expert_count=2, depth=4)
    force_profile(dense, expert_count=2, depth=4)
    for stage in dense.stages:
        stage.experts = DenseAllExpertBank(stage.experts).to(device=device, dtype=dtype).eval()
        stage.reasoner = FixedDepthReasoner(stage.reasoner, cfg.max_reason_steps).to(device=device, dtype=dtype).eval()

    stored_params = sum(p.numel() for p in base.parameters())

    def bench(name: str, model, tokens: torch.Tensor, *, warmup: int, iterations: int) -> dict:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=dtype):
            for _ in range(warmup):
                model(tokens, hard=True, update_memory=False)
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                out = model(tokens, hard=True, update_memory=False)
            end.record()
            end.synchronize()
        logits = out["logits"]
        if not torch.isfinite(logits).all():
            raise RuntimeError(f"non-finite logits in {name}")
        ms = float(start.elapsed_time(end)) / iterations
        return {
            "name": name,
            "milliseconds": ms,
            "tokens_per_second": float(tokens.numel() / (ms / 1000.0)),
            "peak_allocated_mb": float(torch.cuda.max_memory_allocated()) / (1024**2),
            "batch": int(tokens.size(0)),
            "sequence": int(tokens.size(1)),
        }

    seq = 512
    x1 = torch.randint(0, cfg.vocab_size, (1, seq), device=device)
    x8 = torch.randint(0, cfg.vocab_size, (8, seq), device=device)

    results = {}
    for batch_name, tokens, warmup, iterations in (
        ("batch1", x1, 4, 12),
        ("batch8", x8, 3, 6),
    ):
        lean_m = bench(f"lean_{batch_name}", lean, tokens, warmup=warmup, iterations=iterations)
        heavy_m = bench(f"heavy_{batch_name}", heavy, tokens, warmup=warmup, iterations=iterations)
        dense_m = bench(f"dense_{batch_name}", dense, tokens, warmup=warmup, iterations=iterations)
        results[batch_name] = {
            "lean": lean_m,
            "heavy": heavy_m,
            "dense": dense_m,
            "lean_vs_dense_speedup": dense_m["milliseconds"] / lean_m["milliseconds"],
            "heavy_vs_dense_speedup": dense_m["milliseconds"] / heavy_m["milliseconds"],
            "lean_vs_heavy_speedup": heavy_m["milliseconds"] / lean_m["milliseconds"],
        }

    # Confirm the actual hard-compute profiles after timed execution.
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=dtype):
        lean(x8, hard=True, update_memory=False)
        lean_stats = lean.stats()
        heavy(x8, hard=True, update_memory=False)
        heavy_stats = heavy.stats()

    b1 = results["batch1"]
    b8 = results["batch8"]
    profile_ok = True
    for stage in lean_stats["stages"]:
        profile_ok = profile_ok and stage["experts"]["mean_active_experts"] == 1.0
        profile_ok = profile_ok and stage["reasoning"]["mean"] == 2.0
    for stage in heavy_stats["stages"]:
        profile_ok = profile_ok and stage["experts"]["mean_active_experts"] == 2.0
        profile_ok = profile_ok and stage["reasoning"]["mean"] == 4.0

    thresholds = {
        "lean_batch1_vs_dense": 1.15,
        "lean_batch8_vs_dense": 1.30,
        "heavy_batch1_vs_dense": 1.05,
        "heavy_batch8_vs_dense": 1.15,
        "lean_batch1_vs_heavy": 1.05,
        "lean_batch8_vs_heavy": 1.05,
    }
    checks = {
        "lean_batch1_vs_dense": b1["lean_vs_dense_speedup"] >= thresholds["lean_batch1_vs_dense"],
        "lean_batch8_vs_dense": b8["lean_vs_dense_speedup"] >= thresholds["lean_batch8_vs_dense"],
        "heavy_batch1_vs_dense": b1["heavy_vs_dense_speedup"] >= thresholds["heavy_batch1_vs_dense"],
        "heavy_batch8_vs_dense": b8["heavy_vs_dense_speedup"] >= thresholds["heavy_batch8_vs_dense"],
        "lean_batch1_vs_heavy": b1["lean_vs_heavy_speedup"] >= thresholds["lean_batch1_vs_heavy"],
        "lean_batch8_vs_heavy": b8["lean_vs_heavy_speedup"] >= thresholds["lean_batch8_vs_heavy"],
        "hard_profiles_verified": bool(profile_ok),
    }
    passed = all(checks.values())

    _comment(
        repo_full_name,
        issue_number,
        "📊 **Integrated AERA-v4 L4 results**\n\n"
        f"- batch1 lean(top1/depth2) vs dense(all8/depth4): **{b1['lean_vs_dense_speedup']:.2f}x**\n"
        f"- batch1 heavy(top2/depth4) vs dense: **{b1['heavy_vs_dense_speedup']:.2f}x**\n"
        f"- batch8 lean vs dense: **{b8['lean_vs_dense_speedup']:.2f}x**\n"
        f"- batch8 heavy vs dense: **{b8['heavy_vs_dense_speedup']:.2f}x**\n"
        f"- batch1 lean vs heavy: **{b1['lean_vs_heavy_speedup']:.2f}x**\n"
        f"- batch8 lean vs heavy: **{b8['lean_vs_heavy_speedup']:.2f}x**\n"
        f"- stored parameters in benchmark AERA: **{stored_params:,}**\n"
        f"- hard profiles verified: **{profile_ok}**\n"
        f"- integrated systems gate: **{'PASS' if passed else 'FAIL'}**",
    )

    result = {
        "device": torch.cuda.get_device_name(0),
        "torch_version": str(torch.__version__),
        "scope": "AERA_v4_integrated_L4_inference_systems_gate_no_training",
        "stored_parameters": stored_params,
        "results": results,
        "lean_stats": lean_stats,
        "heavy_stats": heavy_stats,
        "thresholds": thresholds,
        "checks": checks,
        "pass": passed,
        "claims": {
            "100m_authorized": False,
            "real_language_superiority_proven": False,
            "breakthrough_proven": False,
        },
    }
    print(json.dumps(result, indent=2), flush=True)
    return result


@app.local_entrypoint()
def main(repo_full_name: str = "", issue_number: int = 0):
    check = preflight.remote()
    print(json.dumps({"preflight": check}, indent=2), flush=True)
    result = run_gate.remote(repo_full_name, issue_number)
    print(json.dumps(result, indent=2), flush=True)
