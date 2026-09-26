from __future__ import annotations

import base64
import json
from pathlib import Path
import time
from typing import Any

import modal

JOB_ID = "rlt-systems-scale15m-kernel-profile-modal-20260926-o"
ENGINEERING_SEED = 20_261_021
SEQ_LEN = 64
BATCH_SIZE = 768
EXPECTED_PARAMETERS = 15_129_344

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.10,<2.11", "numpy>=2.0,<3")
    .add_local_python_source("tam_research", "experiments")
)
app = modal.App("tam-rlt-systems-scale15m-kernel-profile-20260926-o")


def _encode(value: dict[str, Any]) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).decode()


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=4.0,
    memory=65536,
    timeout=55 * 60,
    retries=0,
    max_containers=1,
)
def run_profile_remote(job_json: str) -> str:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.profiler import ProfilerActivity, profile

    from experiments.rlt.model import RLTConfig, RecurrentLoopedTransformer, parameter_count
    from experiments.rlt.train_compiled_pair_1m import CompilableRLT
    from tam_research.models import ModelConfig, ResearchLM, parameter_count as transformer_parameter_count

    started = time.time()
    job = json.loads(job_json)
    expected = {
        "schema": 1,
        "job_id": JOB_ID,
        "task": "systems_scale15m_kernel_profile_o",
        "engineering_seed": ENGINEERING_SEED,
        "requires_systems_profile_job_id": "rlt-systems-scale15m-batch768-modal-20260926-n",
        "requires_compute_match_job_id": "rlt-compute-matched-scale15m-60s-modal-20260926-a",
        "batch_size": BATCH_SIZE,
        "seq_len": SEQ_LEN,
        "expected_parameters_each": EXPECTED_PARAMETERS,
        "automatic_retry": False,
    }
    if job != expected:
        raise RuntimeError(f"kernel-profile preregistration mismatch: {job}")
    if not torch.cuda.is_available():
        raise RuntimeError("kernel profile requires CUDA")

    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    rlt_cfg = RLTConfig(
        vocab_size=50_257, d_model=256, n_heads=8, n_stages=2,
        max_seq_len=128, ff_mult=4, swa_window=32,
    )
    tr_cfg = ModelConfig(
        vocab_size=50_257, d_model=256, n_layers=3, n_heads=8,
        max_seq_len=128, ff_mult=4, ff_inner=938, architecture="transformer",
    )

    def seed_local(seed: int) -> None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def new_rlt() -> RecurrentLoopedTransformer:
        seed_local(ENGINEERING_SEED)
        m = RecurrentLoopedTransformer(rlt_cfg).to(device)
        if parameter_count(m) != EXPECTED_PARAMETERS:
            raise RuntimeError(f"RLT parameter drift: {parameter_count(m)}")
        return m

    def new_transformer() -> ResearchLM:
        seed_local(ENGINEERING_SEED)
        m = ResearchLM(tr_cfg).to(device)
        if transformer_parameter_count(m) != EXPECTED_PARAMETERS:
            raise RuntimeError(f"Transformer parameter drift: {transformer_parameter_count(m)}")
        return m

    # Exact semantic-equivalence guard for the compiled RLT wrapper.
    reference = new_rlt()
    candidate_base = new_rlt()
    candidate_base.load_state_dict(reference.state_dict())
    candidate = CompilableRLT(candidate_base)
    gen0 = torch.Generator(device="cpu").manual_seed(ENGINEERING_SEED + 99)
    x0 = torch.randint(0, rlt_cfg.vocab_size, (2, SEQ_LEN), generator=gen0, dtype=torch.long).to(device)
    y0 = torch.randint(0, rlt_cfg.vocab_size, (2, SEQ_LEN), generator=gen0, dtype=torch.long).to(device)
    reference.zero_grad(set_to_none=True)
    candidate_base.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        ref_logits = reference(x0)
        cand_logits = candidate(x0)
        ref_loss = F.cross_entropy(ref_logits.float().reshape(-1, 50_257), y0.reshape(-1))
        cand_loss = F.cross_entropy(cand_logits.float().reshape(-1, 50_257), y0.reshape(-1))
    ref_loss.backward()
    cand_loss.backward()
    torch.cuda.synchronize(device)
    logits_max_abs = float((ref_logits - cand_logits).abs().max().item())
    loss_abs = float((ref_loss - cand_loss).abs().item())
    grad_max_abs = 0.0
    for rp, cp in zip(reference.parameters(), candidate_base.parameters()):
        if rp.grad is None or cp.grad is None:
            if rp.grad is not cp.grad:
                grad_max_abs = float("inf")
                break
            continue
        grad_max_abs = max(grad_max_abs, float((rp.grad - cp.grad).abs().max().item()))
    semantic_equivalence = (
        logits_max_abs == 0.0 and loss_abs == 0.0 and grad_max_abs == 0.0
    )
    del reference, candidate_base, candidate, ref_logits, cand_logits
    torch.cuda.empty_cache()
    if not semantic_equivalence:
        raise RuntimeError("exact CompilableRLT semantic equivalence failed")

    gen = torch.Generator(device="cpu").manual_seed(ENGINEERING_SEED + 1)
    batches = []
    for _ in range(5):
        x = torch.randint(0, 50_257, (BATCH_SIZE, SEQ_LEN), generator=gen, dtype=torch.long).to(device)
        y = torch.randint(0, 50_257, (BATCH_SIZE, SEQ_LEN), generator=gen, dtype=torch.long).to(device)
        batches.append((x, y))

    def optimizer_for(model: nn.Module) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1, fused=True
        )

    def one_step(callable_model: Any, model: nn.Module, opt: Any, idx: int) -> float:
        model.train()
        opt.zero_grad(set_to_none=True)
        x, y = batches[idx % len(batches)]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = callable_model(x)
            loss = F.cross_entropy(logits.float().reshape(-1, 50_257), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        return float(loss.detach().cpu())

    def event_rows(prof: Any, limit: int = 30) -> list[dict[str, Any]]:
        rows = []
        for event in prof.key_averages():
            self_device = float(
                getattr(event, "self_device_time_total",
                    getattr(event, "self_cuda_time_total", 0.0)) or 0.0
            )
            device_total = float(
                getattr(event, "device_time_total",
                    getattr(event, "cuda_time_total", 0.0)) or 0.0
            )
            self_cpu = float(getattr(event, "self_cpu_time_total", 0.0) or 0.0)
            rows.append({
                "key": str(event.key),
                "count": int(event.count),
                "self_device_time_us": self_device,
                "device_time_total_us": device_total,
                "self_cpu_time_us": self_cpu,
            })
        rows.sort(key=lambda r: r["self_device_time_us"], reverse=True)
        return rows[:limit]

    def benchmark_and_profile(kind: str) -> dict[str, Any]:
        if kind == "rlt":
            base = new_rlt()
            module: nn.Module = CompilableRLT(base)
        else:
            base = new_transformer()
            module = base
        opt = optimizer_for(base)
        compiled = torch.compile(module, fullgraph=True, dynamic=False, mode="default")

        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        one_step(compiled, base, opt, 0)
        torch.cuda.synchronize(device)
        first_step_seconds = time.perf_counter() - t0

        # One warm step after compilation, then one profiled step and one clean timing step.
        one_step(compiled, base, opt, 1)
        torch.cuda.synchronize(device)

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        ) as prof:
            profiled_loss = one_step(compiled, base, opt, 2)
            torch.cuda.synchronize(device)

        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        t1 = time.perf_counter()
        timed_loss = one_step(compiled, base, opt, 3)
        torch.cuda.synchronize(device)
        timed_seconds = time.perf_counter() - t1
        peak = torch.cuda.max_memory_allocated(device) / (1024**3)

        out = {
            "parameters": sum(p.numel() for p in base.parameters()),
            "first_step_seconds_including_compile": first_step_seconds,
            "timed_step_seconds": timed_seconds,
            "timed_step_tokens": BATCH_SIZE * SEQ_LEN,
            "tokens_per_second": (BATCH_SIZE * SEQ_LEN) / max(timed_seconds, 1e-9),
            "profiled_loss": profiled_loss,
            "timed_loss": timed_loss,
            "peak_vram_gb": peak,
            "top_events_by_self_device_time": event_rows(prof),
        }
        del compiled, opt, base, module
        torch.cuda.empty_cache()
        torch._dynamo.reset()
        return out

    rlt = benchmark_and_profile("rlt")
    transformer = benchmark_and_profile("transformer")
    gap = transformer["tokens_per_second"] / max(rlt["tokens_per_second"], 1e-9)

    return _encode({
        "schema": 1,
        "job_id": JOB_ID,
        "status": "complete",
        "classification": "PASS_SCALE15M_KERNEL_PROFILE",
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
        "semantic_equivalence": {
            "passed": semantic_equivalence,
            "logits_max_abs": logits_max_abs,
            "loss_abs": loss_abs,
            "grad_max_abs": grad_max_abs,
        },
        "profile": {
            "batch_size": BATCH_SIZE,
            "seq_len": SEQ_LEN,
            "expected_parameters_each": EXPECTED_PARAMETERS,
            "one_profiled_optimizer_step_per_architecture": True,
        },
        "rlt": rlt,
        "transformer": transformer,
        "derived": {
            "transformer_throughput_multiple_vs_rlt": gap,
        },
        "interpretation_ceiling": (
            "Engineering profiler only. It identifies runtime bottlenecks in the exact-semantics "
            "15.1M implementations at batch 768. Random-token losses are not scientific quality evidence."
        ),
    })


@app.local_entrypoint()
def main(job_path: str) -> None:
    result = run_profile_remote.remote(Path(job_path).read_text())
    if not isinstance(result, str):
        raise TypeError("remote kernel-profile result must be base64 text")
    print("RLT_SYSTEMS_SCALE15M_KERNEL_PROFILE_O_RESULT_B64=" + result)
