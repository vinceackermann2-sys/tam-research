from __future__ import annotations

import base64
import json
from pathlib import Path
import time
from typing import Any

import modal

JOB_ID = "rlt-systems-block-sweep-scale15m-batch768-modal-20260929-ab"
ENGINEERING_SEED = 20_261_036
SEQ_LEN = 64
BLOCK_SIZES = (2, 4, 8)
BATCH_SIZE = 768
CHECK_BATCH = 2
CAUSAL_MUTATION_INDEX = 37
WARMUP_STEPS = 1
MEASURED_STEPS = 2
EXPECTED_PARAMETERS = 15_129_344

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.10,<2.11", "numpy>=2.0,<3")
    .add_local_python_source("tam_research", "experiments")
)
app = modal.App("tam-rlt-systems-block-sweep-scale15m-batch768-20260929-ab")


def _encode(value: dict[str, Any]) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).decode()


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=4.0,
    memory=65536,
    timeout=90 * 60,
    retries=0,
    max_containers=1,
)
def run_profile_remote(job_json: str) -> str:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from experiments.rlt.model import RLTConfig, RecurrentLoopedTransformer, parameter_count
    from experiments.rlt.model_block import BlockRecurrentLoopedTransformer
    from tam_research.models import ModelConfig, ResearchLM, parameter_count as transformer_parameter_count

    started = time.time()
    job = json.loads(job_json)
    expected = {
        "schema": 1,
        "job_id": JOB_ID,
        "task": "systems_block_sweep_scale15m_batch768_ab",
        "engineering_seed": ENGINEERING_SEED,
        "requires_block8_profile_job_id": "rlt-systems-block8-scale15m-batch768-modal-20260928-aa",
        "requires_block8_quality_job_id": "rlt-block8-scale15m-paired-4m-modal-20260929-a",
        "block_sizes": list(BLOCK_SIZES),
        "batch_size": BATCH_SIZE,
        "seq_len": SEQ_LEN,
        "expected_parameters_each": EXPECTED_PARAMETERS,
        "automatic_retry": False,
    }
    if job != expected:
        raise RuntimeError(f"block sweep preregistration mismatch: {job}")
    if not torch.cuda.is_available():
        raise RuntimeError("block sweep systems profile requires CUDA")

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

    def new_reference() -> RecurrentLoopedTransformer:
        seed_local(ENGINEERING_SEED)
        m = RecurrentLoopedTransformer(rlt_cfg).to(device)
        if parameter_count(m) != EXPECTED_PARAMETERS:
            raise RuntimeError(f"reference parameter drift: {parameter_count(m)}")
        return m

    def new_block(block_size: int) -> BlockRecurrentLoopedTransformer:
        seed_local(ENGINEERING_SEED)
        m = BlockRecurrentLoopedTransformer(rlt_cfg, block_size=block_size).to(device)
        if parameter_count(m) != EXPECTED_PARAMETERS:
            raise RuntimeError(f"block{block_size} parameter drift: {parameter_count(m)}")
        return m

    def new_transformer() -> ResearchLM:
        seed_local(ENGINEERING_SEED)
        m = ResearchLM(tr_cfg).to(device)
        if transformer_parameter_count(m) != EXPECTED_PARAMETERS:
            raise RuntimeError(f"Transformer parameter drift: {transformer_parameter_count(m)}")
        return m

    # Exact implementation anchor: block_size=1 must be the frozen RLT path.
    reference = new_reference()
    block1 = new_block(1)
    block1.load_state_dict(reference.state_dict(), strict=True)
    g = torch.Generator(device="cpu").manual_seed(ENGINEERING_SEED + 90)
    x0 = torch.randint(0, rlt_cfg.vocab_size, (CHECK_BATCH, SEQ_LEN), generator=g, dtype=torch.long).to(device)
    y0 = torch.randint(0, rlt_cfg.vocab_size, (CHECK_BATCH, SEQ_LEN), generator=g, dtype=torch.long).to(device)
    reference.zero_grad(set_to_none=True)
    block1.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        ref_logits = reference(x0)
        b1_logits = block1(x0)
        ref_loss = F.cross_entropy(ref_logits.float().reshape(-1, rlt_cfg.vocab_size), y0.reshape(-1))
        b1_loss = F.cross_entropy(b1_logits.float().reshape(-1, rlt_cfg.vocab_size), y0.reshape(-1))
    ref_loss.backward(); b1_loss.backward(); torch.cuda.synchronize(device)
    logits_abs = float((ref_logits - b1_logits).abs().max().item())
    loss_abs = float((ref_loss - b1_loss).abs().item())
    grad_abs = 0.0
    ref_named = dict(reference.named_parameters()); b1_named = dict(block1.named_parameters())
    if ref_named.keys() != b1_named.keys():
        raise RuntimeError("block_size=1 parameter names differ from frozen RLT")
    for name in ref_named:
        a, b = ref_named[name], b1_named[name]
        if a.grad is None or b.grad is None:
            if a.grad is not b.grad:
                grad_abs = float("inf"); break
            continue
        grad_abs = max(grad_abs, float((a.grad - b.grad).abs().max().item()))
    block1_exact = logits_abs == 0.0 and loss_abs == 0.0 and grad_abs == 0.0
    if not block1_exact:
        raise RuntimeError(
            f"block_size=1 exact anchor failed: logits={logits_abs}, loss={loss_abs}, grad={grad_abs}"
        )
    reference_schema = tuple(reference.state_dict().keys())
    del reference, block1, ref_logits, b1_logits
    torch.cuda.empty_cache()

    # Each candidate must remain future-token causal and parameter/state-schema neutral.
    causality: dict[str, Any] = {}
    for block_size in BLOCK_SIZES:
        m = new_block(block_size)
        if tuple(m.state_dict().keys()) != reference_schema:
            raise RuntimeError(f"block{block_size} state schema drift")
        g = torch.Generator(device="cpu").manual_seed(ENGINEERING_SEED + 100 + block_size)
        x = torch.randint(0, rlt_cfg.vocab_size, (CHECK_BATCH, SEQ_LEN), generator=g, dtype=torch.long).to(device)
        x2 = x.clone()
        x2[:, CAUSAL_MUTATION_INDEX] = (x2[:, CAUSAL_MUTATION_INDEX] + 1) % rlt_cfg.vocab_size
        m.eval()
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            a = m(x); b = m(x2)
        torch.cuda.synchronize(device)
        prefix_abs = float((a[:, :CAUSAL_MUTATION_INDEX, :] - b[:, :CAUSAL_MUTATION_INDEX, :]).abs().max().item())
        changed_abs = float((a[:, CAUSAL_MUTATION_INDEX, :] - b[:, CAUSAL_MUTATION_INDEX, :]).abs().max().item())
        passed = prefix_abs == 0.0 and changed_abs > 0.0
        if not passed:
            raise RuntimeError(
                f"block{block_size} causality failed: prefix={prefix_abs}, changed={changed_abs}"
            )
        causality[str(block_size)] = {
            "passed": True,
            "mutation_index": CAUSAL_MUTATION_INDEX,
            "prefix_logits_max_abs": prefix_abs,
            "mutated_position_logits_max_abs": changed_abs,
        }
        del m, a, b
        torch.cuda.empty_cache()

    g = torch.Generator(device="cpu").manual_seed(ENGINEERING_SEED + 1)
    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(WARMUP_STEPS + MEASURED_STEPS + 2):
        x = torch.randint(0, rlt_cfg.vocab_size, (BATCH_SIZE, SEQ_LEN), generator=g, dtype=torch.long).to(device)
        y = torch.randint(0, rlt_cfg.vocab_size, (BATCH_SIZE, SEQ_LEN), generator=g, dtype=torch.long).to(device)
        batches.append((x, y))

    def one_step(callable_model: Any, model: nn.Module, opt: torch.optim.Optimizer, offset: int) -> float:
        model.train(); opt.zero_grad(set_to_none=True)
        x, y = batches[offset % len(batches)]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = callable_model(x)
            loss = F.cross_entropy(logits.float().reshape(-1, 50_257), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        return float(loss.detach().cpu())

    def benchmark_block(block_size: int) -> dict[str, Any]:
        base = new_block(block_size)
        opt = torch.optim.AdamW(base.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1, fused=True)
        compiled = torch.compile(base, fullgraph=True, dynamic=False, mode="default")
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        first_loss = one_step(compiled, base, opt, 0)
        torch.cuda.synchronize(device)
        compile_step = time.perf_counter() - t0
        for i in range(WARMUP_STEPS):
            one_step(compiled, base, opt, 1+i)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        t1 = time.perf_counter()
        last_loss = first_loss
        for i in range(MEASURED_STEPS):
            last_loss = one_step(compiled, base, opt, 1 + WARMUP_STEPS + i)
        torch.cuda.synchronize(device)
        measured = time.perf_counter() - t1
        tokens = MEASURED_STEPS * BATCH_SIZE * SEQ_LEN
        out = {
            "block_size": block_size,
            "parameters": sum(p.numel() for p in base.parameters()),
            "first_step_seconds_including_compile": compile_step,
            "measured_seconds": measured,
            "measured_tokens": tokens,
            "tokens_per_second": tokens / max(measured, 1e-9),
            "last_loss": last_loss,
            "peak_vram_gb": torch.cuda.max_memory_allocated(device) / (1024**3),
            "block_recurrences_per_seq64": SEQ_LEN // block_size,
        }
        del compiled, opt, base
        torch.cuda.empty_cache(); torch._dynamo.reset()
        return out

    def benchmark_transformer() -> dict[str, Any]:
        base = new_transformer()
        opt = torch.optim.AdamW(base.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1, fused=True)
        compiled = torch.compile(base, fullgraph=True, dynamic=False, mode="default")
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        first_loss = one_step(compiled, base, opt, 0)
        torch.cuda.synchronize(device)
        compile_step = time.perf_counter() - t0
        for i in range(WARMUP_STEPS):
            one_step(compiled, base, opt, 1+i)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        t1 = time.perf_counter()
        last_loss = first_loss
        for i in range(MEASURED_STEPS):
            last_loss = one_step(compiled, base, opt, 1 + WARMUP_STEPS + i)
        torch.cuda.synchronize(device)
        measured = time.perf_counter() - t1
        tokens = MEASURED_STEPS * BATCH_SIZE * SEQ_LEN
        out = {
            "parameters": sum(p.numel() for p in base.parameters()),
            "first_step_seconds_including_compile": compile_step,
            "measured_seconds": measured,
            "measured_tokens": tokens,
            "tokens_per_second": tokens / max(measured, 1e-9),
            "last_loss": last_loss,
            "peak_vram_gb": torch.cuda.max_memory_allocated(device) / (1024**3),
        }
        del compiled, opt, base
        torch.cuda.empty_cache(); torch._dynamo.reset()
        return out

    blocks = [benchmark_block(bs) for bs in BLOCK_SIZES]
    transformer = benchmark_transformer()
    tr_tps = float(transformer["tokens_per_second"])
    for row in blocks:
        row["transformer_throughput_multiple_vs_block"] = tr_tps / max(float(row["tokens_per_second"]), 1e-9)
        row["fraction_of_transformer_throughput"] = float(row["tokens_per_second"]) / max(tr_tps, 1e-9)

    fastest = max(blocks, key=lambda x: float(x["tokens_per_second"]))
    most_recurrent = min(blocks, key=lambda x: int(x["block_size"]))
    return _encode({
        "schema": 1,
        "job_id": JOB_ID,
        "status": "complete",
        "classification": "PASS_BLOCK_SWEEP_SCALE15M_BATCH768",
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
        "implementation_validation": {
            "block_size_1_exact_reference_equivalence": {
                "passed": block1_exact,
                "logits_max_abs": logits_abs,
                "loss_abs": loss_abs,
                "grad_max_abs": grad_abs,
            },
            "future_token_causality_by_block_size": causality,
            "parameter_neutral_and_state_schema_equal": True,
        },
        "profile": {
            "batch_size": BATCH_SIZE,
            "seq_len": SEQ_LEN,
            "block_sizes": list(BLOCK_SIZES),
            "warmup_steps": WARMUP_STEPS,
            "measured_steps": MEASURED_STEPS,
            "expected_parameters_each": EXPECTED_PARAMETERS,
        },
        "blocks": blocks,
        "transformer": transformer,
        "derived": {
            "fastest_block_size": fastest["block_size"],
            "fastest_block_tokens_per_second": fastest["tokens_per_second"],
            "smallest_tested_block_size": most_recurrent["block_size"],
        },
        "interpretation_ceiling": (
            "Engineering-only speed/causality sweep across parameter-neutral block-recurrent RLT variants. "
            "Random-token losses are not quality evidence. Use this only to choose a block size for a fresh "
            "equal-parameter/equal-token quality experiment."
        ),
    })


@app.local_entrypoint()
def main(job_path: str) -> None:
    result = run_profile_remote.remote(Path(job_path).read_text())
    if not isinstance(result, str):
        raise TypeError("remote block sweep result must be base64 text")
    print("RLT_SYSTEMS_BLOCK_SWEEP_SCALE15M_BATCH768_AB_RESULT_B64=" + result)
