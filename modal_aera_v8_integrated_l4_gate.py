from __future__ import annotations

import copy
import json

import modal

APP_NAME = "tam-research-aera-v8-integrated-l4"
app = modal.App(APP_NAME)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.10,<2.11", "numpy>=2,<3")
    .add_local_python_source("tam_research")
)


def _cfg():
    from tam_research.aera_hardware_core import HardwareAERAConfig

    return HardwareAERAConfig(
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


def _force_stage_profile(model, runs: tuple[bool, ...]) -> None:
    if len(runs) != len(model.stage_routers):
        raise ValueError("stage profile length mismatch")
    import torch

    with torch.no_grad():
        for router, run in zip(model.stage_routers, runs):
            router.proj.weight.zero_()
            router.proj.bias.fill_(10.0 if run else -10.0)


def _force_inner_profile(model, *, expert_count: int, depth: int) -> None:
    import torch

    if expert_count not in (1, 2):
        raise ValueError("expert_count must be 1 or 2")
    if not 1 <= depth <= model.cfg.max_reason_steps:
        raise ValueError("invalid depth")
    with torch.no_grad():
        for stage in model.stages:
            p = stage.controller.proj
            e = model.cfg.n_experts
            depth0 = e + 2
            p.weight[e : e + 2].zero_()
            p.bias[e : e + 2].fill_(-10.0)
            p.bias[e + expert_count - 1] = 10.0
            p.weight[depth0 : depth0 + model.cfg.max_reason_steps].zero_()
            p.bias[depth0 : depth0 + model.cfg.max_reason_steps].fill_(-10.0)
            p.bias[depth0 + depth - 1] = 10.0


def _replace_sparse_backend_with_v6_bmm(model) -> None:
    from tam_research.aera_hardware_core_v6 import BMMHardSparseExpertBank

    for stage in model.stages:
        old = stage.experts
        new = BMMHardSparseExpertBank(model.cfg).to(
            device=old.w1.device,
            dtype=old.w1.dtype,
        )
        new.load_state_dict(old.state_dict())
        stage.experts = new


@app.function(image=image, cpu=2, memory=4096, timeout=120)
def preflight() -> dict:
    import torch
    from tam_research.aera_hardware_core_v8 import HardwareAwareAERATextLMV8

    cfg = _cfg()
    # Smaller shape for CPU semantic preflight; same execution logic.
    from dataclasses import replace
    small = replace(
        cfg,
        vocab_size=127,
        d_model=32,
        n_stages=2,
        n_heads=4,
        chunk_size=16,
        n_experts=4,
        expert_mult=2,
        memory_dim=8,
        max_reason_steps=2,
    )
    model = HardwareAwareAERATextLMV8(small).eval()
    _replace_sparse_backend_with_v6_bmm(model)
    _force_stage_profile(model, (True, False))
    _force_inner_profile(model, expert_count=1, depth=1)
    x = torch.randint(0, small.vocab_size, (2, 31))
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        out = model(x, hard=True, route_mode="hard_sparse", update_memory=False)
    logits = out["logits"]
    if logits.shape != (2, 31, small.vocab_size) or not torch.isfinite(logits).all():
        raise RuntimeError("AERA-v8 stage-routing CPU preflight failed")
    execution = model.last_stage_execution
    if execution[0]["mean_executed_fraction"] != 1.0 or execution[1]["mean_executed_fraction"] != 0.0:
        raise RuntimeError(f"stage skip preflight mismatch: {execution}")
    return {"ok": True, "shape": list(logits.shape), "stage_execution": execution}


@app.function(image=image, gpu="L4", cpu=8, memory=24576, timeout=600)
def run_gate() -> dict:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from tam_research.aera_hardware_core_v8 import HardwareAwareAERATextLMV8

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(8800)
    torch.cuda.manual_seed_all(8800)
    torch.backends.cuda.matmul.allow_tf32 = True

    cfg = _cfg()
    base = HardwareAwareAERATextLMV8(cfg).to(device=device, dtype=dtype).eval()
    _replace_sparse_backend_with_v6_bmm(base)

    lean_skip = copy.deepcopy(base)
    lean_all = copy.deepcopy(base)
    heavy = copy.deepcopy(base)
    dense = copy.deepcopy(base)

    _force_stage_profile(lean_skip, (True, False, True, False))
    _force_stage_profile(lean_all, (True, True, True, True))
    _force_stage_profile(heavy, (True, True, True, True))
    _force_stage_profile(dense, (True, True, True, True))
    _force_inner_profile(lean_skip, expert_count=1, depth=2)
    _force_inner_profile(lean_all, expert_count=1, depth=2)
    _force_inner_profile(heavy, expert_count=2, depth=4)
    _force_inner_profile(dense, expert_count=2, depth=4)

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
                "min_active_experts": self.n_experts,
                "max_active_experts": self.n_experts,
            }

    class FixedDepthReasoner(nn.Module):
        def __init__(self, src, steps: int):
            super().__init__()
            self.cell = copy.deepcopy(src.cell)
            self.steps = steps

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

    for stage in dense.stages:
        stage.experts = DenseAllExpertBank(stage.experts).to(device=device, dtype=dtype).eval()
        stage.reasoner = FixedDepthReasoner(stage.reasoner, 4).to(device=device, dtype=dtype).eval()

    def bench(model, tokens, warmup: int, iters: int):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
            for _ in range(warmup):
                model(tokens, hard=True, route_mode="hard_sparse", update_memory=False)
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                out = model(tokens, hard=True, route_mode="hard_sparse", update_memory=False)
            end.record()
            end.synchronize()
        logits = out["logits"]
        if not torch.isfinite(logits).all():
            raise RuntimeError("non-finite logits")
        ms = float(start.elapsed_time(end)) / iters
        return {
            "ms": ms,
            "tok_s": float(tokens.numel() / (ms / 1000.0)),
            "peak_mb": float(torch.cuda.max_memory_allocated()) / (1024**2),
        }

    results = {}
    profiles = {
        "lean_skip": lean_skip,
        "lean_all": lean_all,
        "heavy": heavy,
        "dense": dense,
    }
    for batch_name, batch, warmup, iters in (("batch1", 1, 4, 12), ("batch8", 8, 3, 6)):
        x = torch.randint(0, cfg.vocab_size, (batch, 512), device=device)
        rows = {name: bench(model, x, warmup, iters) for name, model in profiles.items()}
        ls, la, h, d = rows["lean_skip"], rows["lean_all"], rows["heavy"], rows["dense"]
        rows.update(
            {
                "lean_skip_vs_lean_all": la["ms"] / ls["ms"],
                "lean_skip_vs_heavy": h["ms"] / ls["ms"],
                "lean_skip_vs_dense": d["ms"] / ls["ms"],
                "lean_all_vs_dense": d["ms"] / la["ms"],
            }
        )
        results[batch_name] = rows

    # Verify forced execution profiles after a fresh batch8 forward.
    x = torch.randint(0, cfg.vocab_size, (8, 512), device=device)
    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
        lean_skip(x, hard=True, route_mode="hard_sparse", update_memory=False)
        lean_skip_stats = lean_skip.stats()
        lean_all(x, hard=True, route_mode="hard_sparse", update_memory=False)
        lean_all_stats = lean_all.stats()
        heavy(x, hard=True, route_mode="hard_sparse", update_memory=False)
        heavy_stats = heavy.stats()

    lean_skip_exec = [r["mean_executed_fraction"] for r in lean_skip_stats["stage_execution"]]
    lean_all_exec = [r["mean_executed_fraction"] for r in lean_all_stats["stage_execution"]]
    heavy_exec = [r["mean_executed_fraction"] for r in heavy_stats["stage_execution"]]
    stage_profile_ok = lean_skip_exec == [1.0, 0.0, 1.0, 0.0]
    stage_profile_ok &= lean_all_exec == [1.0, 1.0, 1.0, 1.0]
    stage_profile_ok &= heavy_exec == [1.0, 1.0, 1.0, 1.0]

    def active_profile_ok(stats, expert_count: float, depth: float, active_stages: tuple[int, ...]):
        for i, stage in enumerate(stats["stages"]):
            if i not in active_stages:
                continue
            if abs(float(stage["experts"]["mean_active_experts"]) - expert_count) > 1e-6:
                return False
            if abs(float(stage["reasoning"]["mean"]) - depth) > 1e-6:
                return False
        return True

    profile_ok = stage_profile_ok
    profile_ok &= active_profile_ok(lean_skip_stats, 1.0, 2.0, (0, 2))
    profile_ok &= active_profile_ok(lean_all_stats, 1.0, 2.0, (0, 1, 2, 3))
    profile_ok &= active_profile_ok(heavy_stats, 2.0, 4.0, (0, 1, 2, 3))

    thresholds = {
        "batch1_skip_vs_all": 1.10,
        "batch8_skip_vs_all": 1.20,
        "batch1_skip_vs_dense": 1.05,
        "batch8_skip_vs_dense": 1.15,
        "batch8_skip_vs_heavy": 1.25,
    }
    b1, b8 = results["batch1"], results["batch8"]
    checks = {
        "batch1_skip_vs_all": b1["lean_skip_vs_lean_all"] >= thresholds["batch1_skip_vs_all"],
        "batch8_skip_vs_all": b8["lean_skip_vs_lean_all"] >= thresholds["batch8_skip_vs_all"],
        "batch1_skip_vs_dense": b1["lean_skip_vs_dense"] >= thresholds["batch1_skip_vs_dense"],
        "batch8_skip_vs_dense": b8["lean_skip_vs_dense"] >= thresholds["batch8_skip_vs_dense"],
        "batch8_skip_vs_heavy": b8["lean_skip_vs_heavy"] >= thresholds["batch8_skip_vs_heavy"],
        "profiles": profile_ok,
    }
    passed = all(checks.values())

    stored = sum(p.numel() for p in base.parameters())
    result = {
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "stored_parameters": stored,
        "sparse_backend": "aera_v6_bmm",
        "forced_profiles": {
            "lean_skip": {"stages": [1, 0, 1, 0], "experts": 1, "depth": 2},
            "lean_all": {"stages": [1, 1, 1, 1], "experts": 1, "depth": 2},
            "heavy": {"stages": [1, 1, 1, 1], "experts": 2, "depth": 4},
            "dense": {"stages": [1, 1, 1, 1], "experts": 8, "depth": 4},
        },
        "results": results,
        "thresholds": thresholds,
        "checks": checks,
        "pass": passed,
        "claims": {
            "forced_routing_only": True,
            "learned_matched_quality_proven": False,
            "100m_authorized": False,
            "breakthrough_proven": False,
        },
    }
    print("AERA_V8_RESULT_JSON=" + json.dumps(result, separators=(",", ":")), flush=True)
    print(json.dumps(result, indent=2), flush=True)
    return result


@app.local_entrypoint()
def main():
    check = preflight.remote()
    print("AERA_V8_PREFLIGHT_JSON=" + json.dumps(check, separators=(",", ":")), flush=True)
    result = run_gate.remote()
    print(json.dumps(result, indent=2), flush=True)
