from __future__ import annotations

import copy
import json

import modal

APP_NAME = "tam-research-aera-v7-grouped-l4"
app = modal.App(APP_NAME)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.11.0", "numpy>=2,<3")
    .add_local_python_source("tam_research")
)


@app.function(image=image, cpu=2, memory=4096, timeout=120)
def preflight() -> dict:
    import torch
    from tam_research.aera_hardware_core import HardwareAERAConfig
    from tam_research.aera_hardware_core_v7 import HardwareAwareAERATextLMV7

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
    model = HardwareAwareAERATextLMV7(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (2, 31))
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        out = model(x, hard=True, update_memory=False)
    if out["logits"].shape != (2, 31, cfg.vocab_size):
        raise RuntimeError("AERA-v7 CPU preflight shape mismatch")
    if not torch.isfinite(out["logits"]).all():
        raise RuntimeError("AERA-v7 CPU preflight non-finite logits")
    return {
        "ok": True,
        "torch_version": str(torch.__version__),
        "cpu_hard_kernel": model.stats()["stages"][0]["experts"]["hard_kernel"],
    }


@app.function(image=image, gpu="L4", cpu=8, memory=24576, timeout=600)
def run_gate() -> dict:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from tam_research.aera_hardware_core import HardwareAERAConfig
    from tam_research.aera_hardware_core_v6 import HardwareAwareAERATextLMV6
    from tam_research.aera_hardware_core_v7 import HardwareAwareAERATextLMV7

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    if getattr(F, "grouped_mm", None) is None:
        raise RuntimeError(f"torch {torch.__version__} does not expose torch.nn.functional.grouped_mm")
    major, minor = torch.cuda.get_device_capability()
    if major < 8:
        raise RuntimeError(f"grouped_mm requires SM80+; got capability {major}.{minor}")

    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(7370)
    torch.cuda.manual_seed_all(7370)
    torch.backends.cuda.matmul.allow_tf32 = True

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
    grouped_base = HardwareAwareAERATextLMV7(cfg).to(device=device, dtype=dtype).eval()
    bmm_base = HardwareAwareAERATextLMV6(cfg).to(device=device, dtype=dtype).eval()
    bmm_base.load_state_dict(grouped_base.state_dict(), strict=True)

    class DenseAllExpertBank(nn.Module):
        def __init__(self, src):
            super().__init__()
            self.w1 = nn.Parameter(src.w1.detach().clone(), requires_grad=False)
            self.w2 = nn.Parameter(src.w2.detach().clone(), requires_grad=False)
            self.n_experts = src.n_experts

        def forward(self, x, expert_logits, count_logits, *, hard):
            p = F.softmax(expert_logits.float(), dim=-1).to(x.dtype)
            h = F.gelu(torch.einsum("btd,ehd->beth", x, self.w1))
            y = torch.einsum("beth,edh->betd", h, self.w2)
            return (y * p[:, :, None, None]).sum(1)

        def stats(self):
            return {
                "stored_experts": self.n_experts,
                "mean_active_experts": float(self.n_experts),
                "hard_kernel": "dense_all_experts",
            }

    class FixedDepthReasoner(nn.Module):
        def __init__(self, src, steps):
            super().__init__()
            self.cell = copy.deepcopy(src.cell)
            self.steps = int(steps)

        def forward(self, summary, depth_logits, *, hard):
            cur = summary
            for _ in range(self.steps):
                cur = self.cell(cur, cur).to(cur.dtype)
            return cur

        def stats(self):
            return {
                "mode": "fixed",
                "mean": float(self.steps),
                "min": float(self.steps),
                "max": float(self.steps),
            }

    def force_profile(model, expert_count: int, depth: int) -> None:
        with torch.no_grad():
            for stage in model.stages:
                proj = stage.controller.proj
                e = cfg.n_experts
                d0 = e + 2
                proj.weight[e : e + 2].zero_()
                proj.bias[e : e + 2].fill_(-8.0)
                proj.bias[e + expert_count - 1] = 8.0
                proj.weight[d0 : d0 + cfg.max_reason_steps].zero_()
                proj.bias[d0 : d0 + cfg.max_reason_steps].fill_(-8.0)
                proj.bias[d0 + depth - 1] = 8.0

    lean = copy.deepcopy(grouped_base)
    heavy = copy.deepcopy(grouped_base)
    bmm_lean = copy.deepcopy(bmm_base)
    dense = copy.deepcopy(grouped_base)
    force_profile(lean, 1, 2)
    force_profile(heavy, 2, 4)
    force_profile(bmm_lean, 1, 2)
    force_profile(dense, 2, 4)
    for stage in dense.stages:
        stage.experts = DenseAllExpertBank(stage.experts).to(device=device, dtype=dtype).eval()
        stage.reasoner = FixedDepthReasoner(stage.reasoner, 4).to(device=device, dtype=dtype).eval()

    def bench(model, tokens, warmup, iters):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
            for _ in range(warmup):
                model(tokens, hard=True, update_memory=False)
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                out = model(tokens, hard=True, update_memory=False)
            end.record()
            end.synchronize()
        if not torch.isfinite(out["logits"]).all():
            raise RuntimeError("non-finite logits")
        ms = float(start.elapsed_time(end)) / iters
        return {
            "ms": ms,
            "tok_s": float(tokens.numel() / (ms / 1000.0)),
            "peak_mb": float(torch.cuda.max_memory_allocated()) / (1024**2),
        }

    results = {}
    for name, batch, warmup, iters in (("batch1", 1, 5, 16), ("batch8", 8, 4, 10)):
        x = torch.randint(0, cfg.vocab_size, (batch, 512), device=device)
        g = bench(lean, x, warmup, iters)
        h = bench(heavy, x, warmup, iters)
        b = bench(bmm_lean, x, warmup, iters)
        d = bench(dense, x, warmup, iters)
        results[name] = {
            "grouped_lean": g,
            "grouped_heavy": h,
            "v6_bmm_lean": b,
            "dense": d,
            "grouped_lean_vs_dense": d["ms"] / g["ms"],
            "grouped_heavy_vs_dense": d["ms"] / h["ms"],
            "grouped_lean_vs_heavy": h["ms"] / g["ms"],
            "grouped_lean_vs_v6_bmm": b["ms"] / g["ms"],
        }

    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
        probe = torch.randint(0, cfg.vocab_size, (8, 512), device=device)
        lean(probe, hard=True, update_memory=False)
        lean_stats = lean.stats()
        heavy(probe, hard=True, update_memory=False)
        heavy_stats = heavy.stats()

    lean_profile = all(
        s["experts"]["mean_active_experts"] == 1.0
        and s["reasoning"]["mean"] == 2.0
        and s["experts"]["hard_kernel"] == "native_grouped_mm"
        for s in lean_stats["stages"]
    )
    heavy_profile = all(
        s["experts"]["mean_active_experts"] == 2.0
        and s["reasoning"]["mean"] == 4.0
        and s["experts"]["hard_kernel"] == "native_grouped_mm"
        for s in heavy_stats["stages"]
    )
    profile_ok = bool(lean_profile and heavy_profile)

    b1 = results["batch1"]
    b8 = results["batch8"]
    thresholds = {
        "batch1_grouped_lean_vs_dense": 1.15,
        "batch8_grouped_lean_vs_dense": 1.30,
        "batch1_grouped_heavy_vs_dense": 1.05,
        "batch8_grouped_heavy_vs_dense": 1.15,
        "batch1_grouped_lean_vs_heavy": 1.05,
        "batch8_grouped_lean_vs_heavy": 1.05,
    }
    checks = {
        "batch1_grouped_lean_vs_dense": b1["grouped_lean_vs_dense"] >= 1.15,
        "batch8_grouped_lean_vs_dense": b8["grouped_lean_vs_dense"] >= 1.30,
        "batch1_grouped_heavy_vs_dense": b1["grouped_heavy_vs_dense"] >= 1.05,
        "batch8_grouped_heavy_vs_dense": b8["grouped_heavy_vs_dense"] >= 1.15,
        "batch1_grouped_lean_vs_heavy": b1["grouped_lean_vs_heavy"] >= 1.05,
        "batch8_grouped_lean_vs_heavy": b8["grouped_lean_vs_heavy"] >= 1.05,
        "native_grouped_profiles_verified": profile_ok,
    }
    passed = all(checks.values())
    result = {
        "device": torch.cuda.get_device_name(0),
        "capability": [major, minor],
        "torch_version": str(torch.__version__),
        "stored_parameters": sum(p.numel() for p in grouped_base.parameters()),
        "results": results,
        "thresholds": thresholds,
        "checks": checks,
        "pass": passed,
        "claims": {
            "100m_authorized": False,
            "small_real_language_gate_passed": False,
            "breakthrough_proven": False,
        },
    }
    print("AERA_V7_RESULT_JSON=" + json.dumps(result, separators=(",", ":")), flush=True)
    return result


@app.local_entrypoint()
def main():
    check = preflight.remote()
    print("AERA_V7_PREFLIGHT_JSON=" + json.dumps(check, separators=(",", ":")), flush=True)
    result = run_gate.remote()
    print(json.dumps(result, indent=2), flush=True)
