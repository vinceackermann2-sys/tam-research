from __future__ import annotations

import base64
import json
from pathlib import Path
import time
from typing import Any

import modal

JOB_ID = "rlt-systems-scale15m-reduce-overhead-modal-20260927-q"
ENGINEERING_SEED = 20_261_023
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
app = modal.App("tam-rlt-systems-scale15m-reduce-overhead-20260927-q")


def _encode(value: dict[str, Any]) -> str:
    return base64.urlsafe_b64encode(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).decode()


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=4.0,
    memory=65536,
    timeout=60 * 60,
    retries=0,
    max_containers=1,
)
def run_profile_remote(job_json: str) -> str:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from experiments.rlt.model import RLTConfig, RecurrentLoopedTransformer, parameter_count
    from experiments.rlt.train_compiled_pair_1m import CompilableRLT

    started = time.time()
    job = json.loads(job_json)
    expected = {
        "schema": 1,
        "job_id": JOB_ID,
        "task": "systems_scale15m_reduce_overhead_q",
        "engineering_seed": ENGINEERING_SEED,
        "requires_kernel_profile_job_id": "rlt-systems-scale15m-kernel-profile-modal-20260926-o",
        "requires_q1_profile_job_id": "rlt-systems-scale15m-q1-attention-modal-20260926-p",
        "batch_size": BATCH_SIZE,
        "seq_len": SEQ_LEN,
        "expected_parameters_each": EXPECTED_PARAMETERS,
        "automatic_retry": False,
    }
    if job != expected:
        raise RuntimeError(f"reduce-overhead preregistration mismatch: {job}")
    if not torch.cuda.is_available():
        raise RuntimeError("reduce-overhead profile requires CUDA")

    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    cfg = RLTConfig(
        vocab_size=50_257, d_model=256, n_heads=8, n_stages=2,
        max_seq_len=128, ff_mult=4, swa_window=32,
    )

    def seed_local(seed: int) -> None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def new_base() -> RecurrentLoopedTransformer:
        seed_local(ENGINEERING_SEED)
        m = RecurrentLoopedTransformer(cfg).to(device)
        if parameter_count(m) != EXPECTED_PARAMETERS:
            raise RuntimeError(f"RLT parameter drift: {parameter_count(m)}")
        return m

    # First prove the wrapper itself remains bit-identical to the reference.
    reference = new_base()
    wrapper_base = new_base()
    wrapper_base.load_state_dict(reference.state_dict())
    wrapper = CompilableRLT(wrapper_base)
    gen0 = torch.Generator(device="cpu").manual_seed(ENGINEERING_SEED + 99)
    x0 = torch.randint(0, cfg.vocab_size, (EQUIV_BATCH, SEQ_LEN), generator=gen0, dtype=torch.long).to(device)
    y0 = torch.randint(0, cfg.vocab_size, (EQUIV_BATCH, SEQ_LEN), generator=gen0, dtype=torch.long).to(device)
    reference.zero_grad(set_to_none=True); wrapper_base.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        ref_logits = reference(x0)
        wr_logits = wrapper(x0)
        ref_loss = F.cross_entropy(ref_logits.float().reshape(-1, cfg.vocab_size), y0.reshape(-1))
        wr_loss = F.cross_entropy(wr_logits.float().reshape(-1, cfg.vocab_size), y0.reshape(-1))
    ref_loss.backward(); wr_loss.backward(); torch.cuda.synchronize(device)
    wrapper_logits_abs = float((ref_logits - wr_logits).abs().max().item())
    wrapper_loss_abs = float((ref_loss - wr_loss).abs().item())
    wrapper_grad_abs = 0.0
    for rp, wp in zip(reference.parameters(), wrapper_base.parameters()):
        if rp.grad is None or wp.grad is None:
            if rp.grad is not wp.grad:
                wrapper_grad_abs = float("inf"); break
            continue
        wrapper_grad_abs = max(wrapper_grad_abs, float((rp.grad - wp.grad).abs().max().item()))
    wrapper_exact = wrapper_logits_abs == 0.0 and wrapper_loss_abs == 0.0 and wrapper_grad_abs == 0.0
    del reference, wrapper_base, wrapper, ref_logits, wr_logits
    torch.cuda.empty_cache()
    if not wrapper_exact:
        raise RuntimeError("CompilableRLT wrapper lost exact semantic equivalence")

    gen = torch.Generator(device="cpu").manual_seed(ENGINEERING_SEED + 1)
    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(WARMUP_STEPS + MEASURED_STEPS + 4):
        x = torch.randint(0, cfg.vocab_size, (BATCH_SIZE, SEQ_LEN), generator=gen, dtype=torch.long).to(device)
        y = torch.randint(0, cfg.vocab_size, (BATCH_SIZE, SEQ_LEN), generator=gen, dtype=torch.long).to(device)
        batches.append((x, y))

    def optimizer_for(model: nn.Module) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            model.parameters(), lr=3e-4, betas=(0.9, 0.95),
            weight_decay=0.1, fused=True,
        )

    def one_step(callable_model: Any, model: nn.Module, opt: Any, idx: int, mark: bool) -> float:
        if mark and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()
        model.train()
        opt.zero_grad(set_to_none=True)
        x, y = batches[idx % len(batches)]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = callable_model(x)
            loss = F.cross_entropy(logits.float().reshape(-1, cfg.vocab_size), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        return float(loss.detach().cpu())

    def benchmark(mode: str, mark: bool) -> dict[str, Any]:
        base = new_base()
        module = CompilableRLT(base)
        opt = optimizer_for(base)
        compiled = torch.compile(module, fullgraph=True, dynamic=False, mode=mode)
        error = None
        result = None
        try:
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            first_loss = one_step(compiled, base, opt, 0, mark)
            torch.cuda.synchronize(device)
            first_step_seconds = time.perf_counter() - t0
            for i in range(WARMUP_STEPS):
                one_step(compiled, base, opt, 1 + i, mark)
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            t1 = time.perf_counter()
            last_loss = first_loss
            for i in range(MEASURED_STEPS):
                last_loss = one_step(compiled, base, opt, 1 + WARMUP_STEPS + i, mark)
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - t1
            tokens = MEASURED_STEPS * BATCH_SIZE * SEQ_LEN
            result = {
                "first_step_seconds_including_compile": first_step_seconds,
                "measured_seconds": elapsed,
                "measured_tokens": tokens,
                "tokens_per_second": tokens / max(elapsed, 1e-9),
                "last_loss": last_loss,
                "peak_vram_gb": torch.cuda.max_memory_allocated(device) / (1024**3),
            }
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            torch.cuda.synchronize(device)
        del compiled, opt, base, module
        torch.cuda.empty_cache()
        torch._dynamo.reset()
        return {"mode": mode, "mark_step_begin": mark, "error": error, "result": result}

    default = benchmark("default", False)
    reduced = benchmark("reduce-overhead", True)

    # Separate exact numerical gate: same initialized weights, same fixed batch,
    # compare default vs reduce-overhead compiled forwards/backwards.
    exact = {
        "attempted": False,
        "passed": False,
        "logits_max_abs": None,
        "loss_abs": None,
        "grad_max_abs": None,
        "error": None,
    }
    try:
        a_base = new_base()
        b_base = new_base()
        b_base.load_state_dict(a_base.state_dict())
        a = torch.compile(CompilableRLT(a_base), fullgraph=True, dynamic=False, mode="default")
        b = torch.compile(CompilableRLT(b_base), fullgraph=True, dynamic=False, mode="reduce-overhead")
        gen1 = torch.Generator(device="cpu").manual_seed(ENGINEERING_SEED + 123)
        x = torch.randint(0, cfg.vocab_size, (EQUIV_BATCH, SEQ_LEN), generator=gen1, dtype=torch.long).to(device)
        y = torch.randint(0, cfg.vocab_size, (EQUIV_BATCH, SEQ_LEN), generator=gen1, dtype=torch.long).to(device)
        a_base.zero_grad(set_to_none=True); b_base.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            la = a(x).clone()
            loss_a = F.cross_entropy(la.float().reshape(-1, cfg.vocab_size), y.reshape(-1))
        loss_a.backward(); torch.cuda.synchronize(device)
        if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            lb = b(x).clone()
            loss_b = F.cross_entropy(lb.float().reshape(-1, cfg.vocab_size), y.reshape(-1))
        loss_b.backward(); torch.cuda.synchronize(device)
        lg = float((la - lb).abs().max().item())
        ls = float((loss_a - loss_b).abs().item())
        gg = 0.0
        for ap, bp in zip(a_base.parameters(), b_base.parameters()):
            if ap.grad is None or bp.grad is None:
                if ap.grad is not bp.grad:
                    gg = float("inf"); break
                continue
            gg = max(gg, float((ap.grad - bp.grad).abs().max().item()))
        exact.update({
            "attempted": True,
            "passed": lg == 0.0 and ls == 0.0 and gg == 0.0,
            "logits_max_abs": lg, "loss_abs": ls, "grad_max_abs": gg,
        })
    except Exception as exc:
        exact["attempted"] = True
        exact["error"] = f"{type(exc).__name__}: {exc}"

    d = default.get("result")
    r = reduced.get("result")
    speedup = None
    if d is not None and r is not None:
        speedup = float(r["tokens_per_second"]) / max(float(d["tokens_per_second"]), 1e-9)

    if reduced["error"] is not None:
        classification = "FAIL_SCALE15M_REDUCE_OVERHEAD_EXECUTION"
    elif not exact["passed"]:
        classification = "FAIL_SCALE15M_REDUCE_OVERHEAD_EXACT_EQUIVALENCE"
    elif speedup is not None and speedup >= 1.10:
        classification = "PASS_SCALE15M_REDUCE_OVERHEAD_MATERIAL_SPEEDUP"
    else:
        classification = "PASS_SCALE15M_REDUCE_OVERHEAD_NO_MATERIAL_SPEEDUP"

    return _encode({
        "schema": 1,
        "job_id": JOB_ID,
        "status": "complete",
        "classification": classification,
        "engineering_only": True,
        "scientific_execution": False,
        "breakthrough_claim_supported": False,
        "engineering_seed": ENGINEERING_SEED,
        "started_unix": started,
        "finished_unix": time.time(),
        "runtime": {
            "torch_version": str(torch.__version__),
            "cuda_runtime": None if torch.version.cuda is None else str(torch.version.cuda),
            "gpu_name": str(torch.cuda.get_device_name(0)),
            "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        },
        "profile": {
            "batch_size": BATCH_SIZE, "seq_len": SEQ_LEN,
            "warmup_steps": WARMUP_STEPS, "measured_steps": MEASURED_STEPS,
            "expected_parameters": EXPECTED_PARAMETERS,
        },
        "wrapper_semantic_equivalence": {
            "passed": wrapper_exact,
            "logits_max_abs": wrapper_logits_abs,
            "loss_abs": wrapper_loss_abs,
            "grad_max_abs": wrapper_grad_abs,
        },
        "default": default,
        "reduce_overhead": reduced,
        "default_vs_reduce_overhead_exact_equivalence": exact,
        "derived": {"reduce_overhead_speedup_vs_default": speedup},
        "interpretation_ceiling": (
            "Engineering-only compile-mode test at the exact 15.1M RLT batch-768 operating point. "
            "A candidate is eligible for scientific use only if the compiled default-vs-reduce-overhead "
            "logits, loss, and gradients are exactly equal."
        ),
    })


@app.local_entrypoint()
def main(job_path: str) -> None:
    result = run_profile_remote.remote(Path(job_path).read_text())
    if not isinstance(result, str):
        raise TypeError("remote reduce-overhead profile result must be base64 text")
    print("RLT_SYSTEMS_SCALE15M_REDUCE_OVERHEAD_Q_RESULT_B64=" + result)
