from __future__ import annotations

import copy
import json

import modal

APP_NAME = "tam-research-aera-v5-integrated-l4"
app = modal.App(APP_NAME)
github_secret = modal.Secret.from_name("github-secret")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.7,<2.11", "numpy>=2,<3", "PyGithub>=2.3,<3")
    .add_local_python_source("tam_research")
)


def _comment(repo: str, issue: int, body: str) -> None:
    if not repo or not issue:
        print(body, flush=True)
        return
    import os
    try:
        import github
        token = os.environ.get("GITHUB_TOKEN")
        if token:
            github.Github(auth=github.Auth.Token(token)).get_repo(repo).get_issue(number=issue).create_comment(body)
    except Exception as exc:
        print(f"[callback-nonfatal] {type(exc).__name__}: {exc}; {body}", flush=True)


@app.function(image=image, cpu=2, memory=4096, timeout=120)
def preflight() -> dict:
    import torch
    from tam_research.aera_hardware_core import HardwareAERAConfig
    from tam_research.aera_hardware_core_v5 import HardwareAwareAERATextLMV5

    cfg = HardwareAERAConfig(vocab_size=127, d_model=32, n_stages=1, n_heads=4,
        chunk_size=16, n_experts=4, max_active_experts=2, expert_mult=2,
        memory_dim=8, max_reason_steps=2, block_size=3)
    model = HardwareAwareAERATextLMV5(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (2, 31))
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        y = model(x, hard=True, update_memory=False)["logits"]
    if y.shape != (2, 31, cfg.vocab_size) or not torch.isfinite(y).all():
        raise RuntimeError("AERA v5 mixed-precision preflight failed")
    return {"ok": True, "shape": list(y.shape)}


@app.function(image=image, gpu="L4", cpu=8, memory=24576, timeout=600, secrets=[github_secret])
def run_gate(repo_full_name: str = "", issue_number: int = 0) -> dict:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from tam_research.aera_hardware_core import HardwareAERAConfig
    from tam_research.aera_hardware_core_v5 import HardwareAwareAERATextLMV5

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(7350)
    torch.cuda.manual_seed_all(7350)
    torch.backends.cuda.matmul.allow_tf32 = True

    cfg = HardwareAERAConfig(vocab_size=50257, d_model=512, n_stages=4, n_heads=8,
        chunk_size=128, n_experts=8, max_active_experts=2, expert_mult=4,
        memory_dim=64, max_reason_steps=4, block_size=4)
    base = HardwareAwareAERATextLMV5(cfg).to(device=device, dtype=dtype).eval()

    _comment(repo_full_name, issue_number,
        "🧪 **Integrated AERA-v5 L4 gate started** — BF16-safe adaptive depth, Flash chunk attention, predictive recurrent state, truly sparse hard top1/top2 experts, and fast-memory read path. Inference only; timeout=600s; no retry.")

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
            return {"stored_experts": self.n_experts, "mean_active_experts": float(self.n_experts)}

    class FixedDepthReasoner(nn.Module):
        def __init__(self, src, steps):
            super().__init__(); self.cell = copy.deepcopy(src.cell); self.steps = steps
        def forward(self, summary, depth_logits, *, hard):
            cur = summary
            for _ in range(self.steps):
                cur = self.cell(cur, cur).to(cur.dtype)
            return cur
        def stats(self):
            return {"mode":"fixed","mean":float(self.steps),"min":float(self.steps),"max":float(self.steps)}

    def force_profile(model, expert_count: int, depth: int) -> None:
        with torch.no_grad():
            for stage in model.stages:
                p = stage.controller.proj; e = cfg.n_experts; d0 = e + 2
                p.weight[e:e+2].zero_(); p.bias[e:e+2].fill_(-8); p.bias[e+expert_count-1] = 8
                p.weight[d0:d0+cfg.max_reason_steps].zero_(); p.bias[d0:d0+cfg.max_reason_steps].fill_(-8); p.bias[d0+depth-1] = 8

    lean, heavy, dense = copy.deepcopy(base), copy.deepcopy(base), copy.deepcopy(base)
    force_profile(lean, 1, 2); force_profile(heavy, 2, 4); force_profile(dense, 2, 4)
    for stage in dense.stages:
        stage.experts = DenseAllExpertBank(stage.experts).to(device=device, dtype=dtype).eval()
        stage.reasoner = FixedDepthReasoner(stage.reasoner, 4).to(device=device, dtype=dtype).eval()

    def bench(model, tokens, warmup, iters):
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
            for _ in range(warmup): model(tokens, hard=True, update_memory=False)
            torch.cuda.synchronize(); start = torch.cuda.Event(True); end = torch.cuda.Event(True); start.record()
            for _ in range(iters): out = model(tokens, hard=True, update_memory=False)
            end.record(); end.synchronize()
        if not torch.isfinite(out["logits"]).all(): raise RuntimeError("non-finite logits")
        ms = float(start.elapsed_time(end)) / iters
        return {"ms":ms,"tok_s":float(tokens.numel()/(ms/1000)),"peak_mb":float(torch.cuda.max_memory_allocated())/(1024**2)}

    results = {}
    for name, batch, warmup, iters in (("batch1",1,4,12),("batch8",8,3,6)):
        x = torch.randint(0, cfg.vocab_size, (batch,512), device=device)
        l = bench(lean,x,warmup,iters); h = bench(heavy,x,warmup,iters); d = bench(dense,x,warmup,iters)
        results[name] = {"lean":l,"heavy":h,"dense":d,
            "lean_vs_dense":d["ms"]/l["ms"],"heavy_vs_dense":d["ms"]/h["ms"],"lean_vs_heavy":h["ms"]/l["ms"]}

    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
        x = torch.randint(0,cfg.vocab_size,(8,512),device=device)
        lean(x,hard=True,update_memory=False); ls=lean.stats()
        heavy(x,hard=True,update_memory=False); hs=heavy.stats()
    profile_ok = all(s["experts"]["mean_active_experts"]==1.0 and s["reasoning"]["mean"]==2.0 for s in ls["stages"])
    profile_ok &= all(s["experts"]["mean_active_experts"]==2.0 and s["reasoning"]["mean"]==4.0 for s in hs["stages"])

    thresholds = {"b1_ld":1.15,"b8_ld":1.30,"b1_hd":1.05,"b8_hd":1.15,"b1_lh":1.05,"b8_lh":1.05}
    b1,b8=results["batch1"],results["batch8"]
    checks = {"b1_ld":b1["lean_vs_dense"]>=1.15,"b8_ld":b8["lean_vs_dense"]>=1.30,
        "b1_hd":b1["heavy_vs_dense"]>=1.05,"b8_hd":b8["heavy_vs_dense"]>=1.15,
        "b1_lh":b1["lean_vs_heavy"]>=1.05,"b8_lh":b8["lean_vs_heavy"]>=1.05,"profiles":profile_ok}
    passed = all(checks.values())
    stored = sum(p.numel() for p in base.parameters())
    result = {"device":torch.cuda.get_device_name(0),"stored_parameters":stored,"results":results,
        "thresholds":thresholds,"checks":checks,"pass":passed,
        "claims":{"100m_authorized":False,"real_language_superiority_proven":False,"breakthrough_proven":False}}
    _comment(repo_full_name, issue_number,
        "📊 **Integrated AERA-v5 L4 results**\n\n"
        f"- batch1 lean vs dense: **{b1['lean_vs_dense']:.2f}x**\n- batch8 lean vs dense: **{b8['lean_vs_dense']:.2f}x**\n"
        f"- batch1 heavy vs dense: **{b1['heavy_vs_dense']:.2f}x**\n- batch8 heavy vs dense: **{b8['heavy_vs_dense']:.2f}x**\n"
        f"- batch1 lean vs heavy: **{b1['lean_vs_heavy']:.2f}x**\n- batch8 lean vs heavy: **{b8['lean_vs_heavy']:.2f}x**\n"
        f"- hard profiles verified: **{profile_ok}**\n- systems gate: **{'PASS' if passed else 'FAIL'}**")
    print(json.dumps(result, indent=2), flush=True)
    return result


@app.local_entrypoint()
def main(repo_full_name: str = "", issue_number: int = 0):
    print(json.dumps({"preflight": preflight.remote()}, indent=2), flush=True)
    print(json.dumps(run_gate.remote(repo_full_name, issue_number), indent=2), flush=True)
