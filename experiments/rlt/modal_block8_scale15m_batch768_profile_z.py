from __future__ import annotations

import base64
import json
from pathlib import Path
import time
from typing import Any

import modal

JOB_ID = "rlt-systems-block8-scale15m-batch768-modal-20260928-z"
ENGINEERING_SEED = 20_261_033
SEQ_LEN = 64
BLOCK_SIZE = 8
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
app = modal.App("tam-rlt-systems-block8-scale15m-batch768-20260928-z")


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
    from experiments.rlt.model_block import BlockRecurrentLoopedTransformer
    from tam_research.models import (
        ModelConfig,
        ResearchLM,
        parameter_count as transformer_parameter_count,
    )

    started = time.time()
    job = json.loads(job_json)
    expected = {
        "schema": 1,
        "job_id": JOB_ID,
        "task": "systems_block8_scale15m_batch768_z",
        "engineering_seed": ENGINEERING_SEED,
        "requires_optimizer_tuned_compute_match_job_id": "rlt-compute-matched-optimizer-tuned-scale15m-60s-modal-20260928-a",
        "requires_reference_batch768_job_id": "rlt-systems-scale15m-batch768-modal-20260926-n",
        "block_size": BLOCK_SIZE,
        "batch_size": BATCH_SIZE,
        "seq_len": SEQ_LEN,
        "expected_parameters_each": EXPECTED_PARAMETERS,
        "automatic_retry": False,
    }
    if job != expected:
        raise RuntimeError(f"block8 profile preregistration mismatch: {job}")
    if not torch.cuda.is_available():
        raise RuntimeError("block8 systems profile requires CUDA")

    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")

    rlt_cfg = RLTConfig(
        vocab_size=50_257,
        d_model=256,
        n_heads=8,
        n_stages=2,
        max_seq_len=128,
        ff_mult=4,
        swa_window=32,
    )
    tr_cfg = ModelConfig(
        vocab_size=50_257,
        d_model=256,
        n_layers=3,
        n_heads=8,
        max_seq_len=128,
        ff_mult=4,
        ff_inner=938,
        architecture="transformer",
    )

    def seed_local(seed: int) -> None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def new_reference() -> RecurrentLoopedTransformer:
        seed_local(ENGINEERING_SEED)
        model = RecurrentLoopedTransformer(rlt_cfg).to(device)
        if parameter_count(model) != EXPECTED_PARAMETERS:
            raise RuntimeError(f"reference RLT parameter drift: {parameter_count(model)}")
        return model

    def new_block(block_size: int) -> BlockRecurrentLoopedTransformer:
        seed_local(ENGINEERING_SEED)
        model = BlockRecurrentLoopedTransformer(rlt_cfg, block_size=block_size).to(device)
        if parameter_count(model) != EXPECTED_PARAMETERS:
            raise RuntimeError(f"block RLT parameter drift: {parameter_count(model)}")
        return model

    def new_transformer() -> ResearchLM:
        seed_local(ENGINEERING_SEED)
        model = ResearchLM(tr_cfg).to(device)
        if transformer_parameter_count(model) != EXPECTED_PARAMETERS:
            raise RuntimeError(
                f"Transformer parameter drift: {transformer_parameter_count(model)}"
            )
        return model

    # Implementation guard: block_size=1 must exactly reproduce frozen token recurrence.
    reference = new_reference()
    block1 = new_block(1)
    block1.load_state_dict(reference.state_dict(), strict=True)
    gen = torch.Generator(device="cpu").manual_seed(ENGINEERING_SEED + 90)
    x0 = torch.randint(
        0, rlt_cfg.vocab_size, (CHECK_BATCH, SEQ_LEN),
        generator=gen, dtype=torch.long,
    ).to(device)
    y0 = torch.randint(
        0, rlt_cfg.vocab_size, (CHECK_BATCH, SEQ_LEN),
        generator=gen, dtype=torch.long,
    ).to(device)

    reference.zero_grad(set_to_none=True)
    block1.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        ref_logits = reference(x0)
        block1_logits = block1(x0)
        ref_loss = F.cross_entropy(
            ref_logits.float().reshape(-1, rlt_cfg.vocab_size), y0.reshape(-1)
        )
        block1_loss = F.cross_entropy(
            block1_logits.float().reshape(-1, rlt_cfg.vocab_size), y0.reshape(-1)
        )
    ref_loss.backward()
    block1_loss.backward()
    torch.cuda.synchronize(device)

    b1_logits_abs = float((ref_logits - block1_logits).abs().max().item())
    b1_loss_abs = float((ref_loss - block1_loss).abs().item())
    b1_grad_abs = 0.0
    ref_named = dict(reference.named_parameters())
    b1_named = dict(block1.named_parameters())
    if ref_named.keys() != b1_named.keys():
        raise RuntimeError("block_size=1 parameter names differ from frozen RLT")
    for name in ref_named:
        rp = ref_named[name]
        bp = b1_named[name]
        if rp.grad is None or bp.grad is None:
            if rp.grad is not bp.grad:
                b1_grad_abs = float("inf")
                break
            continue
        b1_grad_abs = max(
            b1_grad_abs, float((rp.grad - bp.grad).abs().max().item())
        )
    block1_exact = (
        b1_logits_abs == 0.0 and b1_loss_abs == 0.0 and b1_grad_abs == 0.0
    )
    if not block1_exact:
        raise RuntimeError(
            "block_size=1 failed exact frozen-RLT equivalence: "
            f"logits={b1_logits_abs}, loss={b1_loss_abs}, grad={b1_grad_abs}"
        )
    del reference, block1, ref_logits, block1_logits
    torch.cuda.empty_cache()

    # Architectural guard: block_size=8 must remain causal inside a block.
    block8_check = new_block(BLOCK_SIZE)
    gen_causal = torch.Generator(device="cpu").manual_seed(ENGINEERING_SEED + 91)
    causal_x = torch.randint(
        0, rlt_cfg.vocab_size, (CHECK_BATCH, SEQ_LEN),
        generator=gen_causal, dtype=torch.long,
    ).to(device)
    altered_x = causal_x.clone()
    altered_x[:, CAUSAL_MUTATION_INDEX] = (
        altered_x[:, CAUSAL_MUTATION_INDEX] + 1
    ) % rlt_cfg.vocab_size

    block8_check.eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        original_logits = block8_check(causal_x)
        altered_logits = block8_check(altered_x)
    torch.cuda.synchronize(device)

    prefix_abs = float(
        (
            original_logits[:, :CAUSAL_MUTATION_INDEX, :]
            - altered_logits[:, :CAUSAL_MUTATION_INDEX, :]
        ).abs().max().item()
    )
    changed_token_abs = float(
        (
            original_logits[:, CAUSAL_MUTATION_INDEX, :]
            - altered_logits[:, CAUSAL_MUTATION_INDEX, :]
        ).abs().max().item()
    )
    causal_passed = prefix_abs == 0.0 and changed_token_abs > 0.0
    if not causal_passed:
        raise RuntimeError(
            "block_size=8 causality guard failed: "
            f"prefix_abs={prefix_abs}, changed_token_abs={changed_token_abs}"
        )

    block8_params = parameter_count(block8_check)
    block8_keys = tuple(block8_check.state_dict().keys())
    reference_keys = tuple(new_reference().state_dict().keys())
    state_schema_equal = block8_keys == reference_keys
    if not state_schema_equal or block8_params != EXPECTED_PARAMETERS:
        raise RuntimeError("block_size=8 changed parameter/state schema")
    del block8_check, original_logits, altered_logits
    torch.cuda.empty_cache()

    generator = torch.Generator(device="cpu").manual_seed(ENGINEERING_SEED + 1)
    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(WARMUP_STEPS + MEASURED_STEPS + 2):
        x = torch.randint(
            0, rlt_cfg.vocab_size, (BATCH_SIZE, SEQ_LEN),
            generator=generator, dtype=torch.long,
        ).to(device)
        y = torch.randint(
            0, rlt_cfg.vocab_size, (BATCH_SIZE, SEQ_LEN),
            generator=generator, dtype=torch.long,
        ).to(device)
        batches.append((x, y))

    def one_step(
        callable_model: Any,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        offset: int,
    ) -> float:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        x, y = batches[offset % len(batches)]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = callable_model(x)
            loss = F.cross_entropy(
                logits.float().reshape(-1, 50_257), y.reshape(-1)
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        return float(loss.detach().cpu())

    def benchmark(kind: str) -> dict[str, Any]:
        if kind == "block8":
            base: nn.Module = new_block(BLOCK_SIZE)
        elif kind == "transformer":
            base = new_transformer()
        else:
            raise ValueError(kind)

        optimizer = torch.optim.AdamW(
            base.parameters(),
            lr=3e-4,
            betas=(0.9, 0.95),
            weight_decay=0.1,
            fused=True,
        )
        compiled = torch.compile(
            base, fullgraph=True, dynamic=False, mode="default"
        )
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
            last_loss = one_step(
                compiled, base, optimizer, 1 + WARMUP_STEPS + i
            )
        torch.cuda.synchronize(device)
        measured_seconds = time.perf_counter() - t1
        measured_tokens = MEASURED_STEPS * BATCH_SIZE * SEQ_LEN

        result = {
            "parameters": sum(p.numel() for p in base.parameters()),
            "first_step_seconds_including_compile": first_step_seconds,
            "measured_seconds": measured_seconds,
            "measured_tokens": measured_tokens,
            "tokens_per_second": measured_tokens / max(measured_seconds, 1e-9),
            "last_loss": last_loss,
            "peak_vram_gb": (
                torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            ),
        }
        del compiled, optimizer, base
        torch.cuda.empty_cache()
        torch._dynamo.reset()
        return result

    block8 = benchmark("block8")
    transformer = benchmark("transformer")
    gap = (
        transformer["tokens_per_second"]
        / max(block8["tokens_per_second"], 1e-9)
    )

    if gap <= 1.5:
        classification = "PASS_BLOCK8_SCALE15M_GAP_LE_1_5X"
    elif gap <= 2.0:
        classification = "PASS_BLOCK8_SCALE15M_GAP_LE_2X"
    else:
        classification = "PASS_BLOCK8_SCALE15M_GAP_GT_2X"

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
            "cuda_runtime": (
                None if torch.version.cuda is None else str(torch.version.cuda)
            ),
            "gpu_name": str(torch.cuda.get_device_name(0)),
            "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        },
        "architecture": {
            "variant": "block_recurrent_rlt",
            "block_size": BLOCK_SIZE,
            "token_recurrences_per_seq64": SEQ_LEN // BLOCK_SIZE,
            "frozen_reference_token_recurrences_per_seq64": SEQ_LEN,
            "parameters": block8_params,
            "state_schema_equal_to_frozen_rlt": state_schema_equal,
            "parameter_neutral": block8_params == EXPECTED_PARAMETERS,
        },
        "implementation_validation": {
            "block_size_1_exact_reference_equivalence": {
                "passed": block1_exact,
                "logits_max_abs": b1_logits_abs,
                "loss_abs": b1_loss_abs,
                "grad_max_abs": b1_grad_abs,
            },
            "block_size_8_future_token_causality": {
                "passed": causal_passed,
                "mutation_index": CAUSAL_MUTATION_INDEX,
                "prefix_logits_max_abs": prefix_abs,
                "mutated_position_logits_max_abs": changed_token_abs,
            },
        },
        "profile": {
            "batch_size": BATCH_SIZE,
            "seq_len": SEQ_LEN,
            "block_size": BLOCK_SIZE,
            "warmup_steps": WARMUP_STEPS,
            "measured_steps": MEASURED_STEPS,
            "expected_parameters_each": EXPECTED_PARAMETERS,
        },
        "block8": block8,
        "transformer": transformer,
        "derived": {
            "transformer_throughput_multiple_vs_block8": gap,
            "block8_fraction_of_transformer_throughput": 1.0 / gap,
        },
        "interpretation_ceiling": (
            "Engineering-only throughput and causality profile for a parameter-neutral "
            "block-recurrent RLT research variant. Random-token losses are not quality "
            "evidence. block_size=1 must exactly reproduce the frozen RLT; block_size=8 "
            "is a deliberate architectural change and requires separate matched training "
            "before any scientific quality conclusion."
        ),
    })


@app.local_entrypoint()
def main(job_path: str) -> None:
    result = run_profile_remote.remote(Path(job_path).read_text())
    if not isinstance(result, str):
        raise TypeError("remote block8 systems result must be base64 text")
    print("RLT_SYSTEMS_BLOCK8_SCALE15M_BATCH768_Z_RESULT_B64=" + result)
