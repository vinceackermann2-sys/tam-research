from __future__ import annotations

import base64
import json
from pathlib import Path
import time
from typing import Any

import modal

JOB_ID = "rlt-systems-scale15m-batch768-modal-20260926-n"
ENGINEERING_SEED = 20_261_019
SEQ_LEN = 64
BATCH_SIZE = 768
EQUIV_BATCH = 2
WARMUP_STEPS = 1
MEASURED_STEPS = 2
EXPECTED_PARAMETERS = 15_129_344

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.10,<2.11", "numpy>=2.0,<3")
    .add_local_python_source("tam_research", "experiments")
)
app = modal.App("tam-rlt-systems-scale15m-batch768-20260926-n")


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

    from experiments.rlt.model import RLTConfig, RecurrentLoopedTransformer, parameter_count
    from experiments.rlt.train_compiled_pair_1m import CompilableRLT
    from tam_research.models import ModelConfig, ResearchLM, parameter_count as transformer_parameter_count

    started = time.time()
    job = json.loads(job_json)
    expected = {
        "schema": 1,
        "job_id": JOB_ID,
        "task": "systems_scale15m_batch768_n",
        "engineering_seed": ENGINEERING_SEED,
        "requires_scale4m_a_job_id": "rlt-exact15m-fineweb-4m-modal-20260925-a",
        "requires_scale4m_b_job_id": "rlt-scale15m-paired-4m-modal-20260925-b",
        "requires_scale16m_a_job_id": "rlt-scale15m-paired-16m-modal-20260925-a",
        "requires_scale16m_b_job_id": "rlt-scale15m-paired-16m-modal-20260925-b",
        "batch_size": BATCH_SIZE,
        "seq_len": SEQ_LEN,
        "expected_parameters_each": EXPECTED_PARAMETERS,
        "automatic_retry": False,
    }
    if job != expected:
        raise RuntimeError(f"scale15m batch768 job preregistration mismatch: {job}")
    if not torch.cuda.is_available():
        raise RuntimeError("scale15m batch768 systems profile requires CUDA")

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
        model = RecurrentLoopedTransformer(rlt_cfg).to(device)
        if parameter_count(model) != EXPECTED_PARAMETERS:
            raise RuntimeError(f"RLT parameter drift: {parameter_count(model)}")
        return model

    def new_transformer() -> ResearchLM:
        seed_local(ENGINEERING_SEED)
        model = ResearchLM(tr_cfg).to(device)
        if transformer_parameter_count(model) != EXPECTED_PARAMETERS:
            raise RuntimeError(f"Transformer parameter drift: {transformer_parameter_count(model)}")
        return model

    # Exact wrapper equivalence before benchmarking.
    reference = new_rlt()
    candidate_base = new_rlt()
    candidate_base.load_state_dict(reference.state_dict())
    candidate = CompilableRLT(candidate_base)
    gen = torch.Generator(device="cpu").manual_seed(ENGINEERING_SEED + 99)
    x0 = torch.randint(0, rlt_cfg.vocab_size, (EQUIV_BATCH, SEQ_LEN), generator=gen, dtype=torch.long).to(device)
    y0 = torch.randint(0, rlt_cfg.vocab_size, (EQUIV_BATCH, SEQ_LEN), generator=gen, dtype=torch.long).to(device)
    reference.zero_grad(set_to_none=True)
    candidate_base.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        ref_logits = reference(x0)
        cand_logits = candidate(x0)
        ref_loss = F.cross_entropy(ref_logits.float().reshape(-1, rlt_cfg.vocab_size), y0.reshape(-1))
        cand_loss = F.cross_entropy(cand_logits.float().reshape(-1, rlt_cfg.vocab_size), y0.reshape(-1))
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
    semantic_equivalence = logits_max_abs == 0.0 and loss_abs == 0.0 and grad_max_abs == 0.0
    del reference, candidate_base, candidate, ref_logits, cand_logits
    torch.cuda.empty_cache()
    if not semantic_equivalence:
        raise RuntimeError("exact CompilableRLT semantic equivalence failed")

    generator = torch.Generator(device="cpu").manual_seed(ENGINEERING_SEED + 1)
    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(WARMUP_STEPS + MEASURED_STEPS + 2):
        x = torch.randint(0, rlt_cfg.vocab_size, (BATCH_SIZE, SEQ_LEN), generator=generator, dtype=torch.long).to(device)
        y = torch.randint(0, rlt_cfg.vocab_size, (BATCH_SIZE, SEQ_LEN), generator=generator, dtype=torch.long).to(device)
        batches.append((x, y))

    def one_step(callable_model: Any, model: nn.Module, optimizer: Any, offset: int) -> float:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        x, y = batches[offset % len(batches)]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = callable_model(x)
            loss = F.cross_entropy(logits.float().reshape(-1, 50_257), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        return float(loss.detach().cpu())

    def benchmark(kind: str) -> dict[str, Any]:
        if kind == "rlt":
            base = new_rlt()
            module: nn.Module = CompilableRLT(base)
        else:
            base = new_transformer()
            module = base
        optimizer = torch.optim.AdamW(base.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1, fused=True)
        compiled = torch.compile(module, fullgraph=True, dynamic=False, mode="default")
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        first_loss = one_step(compiled, base, optimizer, 0)
        torch.cuda.synchronize(device)
        first_step_seconds = time.perf_counter() - t0
        for i in range(WARMUP_STEPS):
            one_step(compiled, base, optimizer, 1 + i)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        t1 = time.perf_counter()
        last_loss = first_loss
        for i in range(MEASURED_STEPS):
            last_loss = one_step(compiled, base, optimizer, 1 + WARMUP_STEPS + i)
        torch.cuda.synchronize(device)
        measured_seconds = time.perf_counter() - t1
        tokens = MEASURED_STEPS * BATCH_SIZE * SEQ_LEN
        out = {
            "parameters": sum(p.numel() for p in base.parameters()),
            "first_step_seconds_including_compile": first_step_seconds,
            "measured_seconds": measured_seconds,
            "measured_tokens": tokens,
            "tokens_per_second": tokens / max(measured_seconds, 1e-9),
            "last_loss": last_loss,
            "peak_vram_gb": torch.cuda.max_memory_allocated(device) / (1024**3),
        }
        del compiled, optimizer, base, module
        torch.cuda.empty_cache()
        torch._dynamo.reset()
        return out

    rlt = benchmark("rlt")
    transformer = benchmark("transformer")
    gap = transformer["tokens_per_second"] / max(rlt["tokens_per_second"], 1e-9)
    payload = {
        "schema": 1,
        "job_id": JOB_ID,
        "status": "complete",
        "classification": (
            "PASS_SCALE15M_BATCH768_GAP_LE_3X" if gap <= 3.0
            else "PASS_SCALE15M_BATCH768_GAP_GT_3X"
        ),
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
            "batch_size": BATCH_SIZE,
            "seq_len": SEQ_LEN,
            "warmup_steps": WARMUP_STEPS,
            "measured_steps": MEASURED_STEPS,
            "expected_parameters_each": EXPECTED_PARAMETERS,
        },
        "semantic_equivalence": {
            "passed": semantic_equivalence,
            "logits_max_abs": logits_max_abs,
            "loss_abs": loss_abs,
            "grad_max_abs": grad_max_abs,
        },
        "rlt": rlt,
        "transformer": transformer,
        "derived": {
            "transformer_throughput_multiple_vs_rlt": gap,
            "rlt_fraction_of_transformer_throughput": 1.0 / gap,
        },
        "interpretation_ceiling": (
            "Engineering throughput profile only. Random-token losses are not scientific quality evidence. "
            "The result measures whether the exact 15.1M RLT implementation can utilize an A100-80GB at "
            "large batch and how much of the microbatch-16 throughput gap is implementation/utilization overhead."
        ),
    }
    return _encode(payload)


@app.local_entrypoint()
def main(job_path: str) -> None:
    result = run_profile_remote.remote(Path(job_path).read_text())
    if not isinstance(result, str):
        raise TypeError("remote scale15m batch768 systems result must be base64 text")
    print("RLT_SYSTEMS_SCALE15M_BATCH768_N_RESULT_B64=" + result)
