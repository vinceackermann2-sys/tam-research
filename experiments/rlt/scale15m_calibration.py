from __future__ import annotations

from dataclasses import asdict
import math
from pathlib import Path
import time
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.rlt.model import RLTConfig, RecurrentLoopedTransformer, parameter_count as rlt_parameter_count
from experiments.rlt.train_compiled_pair_1m import CompilableRLT
from tam_research.data import TokenBin
from tam_research.models import ModelConfig, ResearchLM, parameter_count as transformer_parameter_count
from tam_research.train import seed_all


ATTEMPT_SEED = 20_261_014
PAIRED_BATCH_SEED = 20_271_014
COMPILE_PROBE_SEED = 20_301_014
SEQ_LEN = 64
MICRO_BATCH_SIZE = 16
PROBE_STEPS = 32
FULL_TOKEN_BUDGET_REFERENCE = 4_194_304
EXPECTED_PARAMETERS = 15_129_344


def rlt_config() -> RLTConfig:
    return RLTConfig(
        vocab_size=50_257,
        d_model=256,
        n_heads=8,
        n_stages=2,
        max_seq_len=128,
        ff_mult=4,
        swa_window=32,
    )


def transformer_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=50_257,
        d_model=256,
        n_layers=3,
        n_heads=8,
        max_seq_len=128,
        ff_mult=4,
        ff_inner=938,
        architecture="transformer",
    )


def _amp_dtype() -> torch.dtype:
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def _prove_rlt_wrapper_equivalence(train_data: TokenBin) -> dict[str, float | bool]:
    device = torch.device("cuda")
    cfg = rlt_config()
    seed_all(ATTEMPT_SEED)
    reference = RecurrentLoopedTransformer(cfg).to(device)
    seed_all(ATTEMPT_SEED)
    candidate_base = RecurrentLoopedTransformer(cfg).to(device)
    candidate_base.load_state_dict(reference.state_dict())
    candidate = CompilableRLT(candidate_base)
    generator = torch.Generator(device="cpu").manual_seed(COMPILE_PROBE_SEED)
    x, y = train_data.batch(2, SEQ_LEN, generator, device)
    reference.zero_grad(set_to_none=True)
    candidate_base.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=_amp_dtype()):
        ref_logits = reference(x)
        cand_logits = candidate(x)
        ref_loss = F.cross_entropy(ref_logits.float().reshape(-1, cfg.vocab_size), y.reshape(-1))
        cand_loss = F.cross_entropy(cand_logits.float().reshape(-1, cfg.vocab_size), y.reshape(-1))
    ref_loss.backward()
    cand_loss.backward()
    torch.cuda.synchronize(device)
    logits_max_abs = float((ref_logits - cand_logits).abs().max().item())
    loss_abs = float((ref_loss - cand_loss).abs().item())
    grad_max_abs = 0.0
    for ref_p, cand_p in zip(reference.parameters(), candidate_base.parameters()):
        if ref_p.grad is None or cand_p.grad is None:
            if ref_p.grad is not cand_p.grad:
                grad_max_abs = float("inf")
                break
            continue
        grad_max_abs = max(grad_max_abs, float((ref_p.grad - cand_p.grad).abs().max().item()))
    passed = logits_max_abs == 0.0 and loss_abs == 0.0 and grad_max_abs == 0.0
    del reference, candidate_base, candidate, ref_logits, cand_logits
    torch.cuda.empty_cache()
    return {
        "passed": passed,
        "logits_max_abs": logits_max_abs,
        "loss_abs": loss_abs,
        "grad_max_abs": grad_max_abs,
    }


def _compile_probe(
    callable_model: Any,
    parameter_model: nn.Module,
    train_data: TokenBin,
    *,
    vocab_size: int,
) -> dict[str, float]:
    device = torch.device("cuda")
    generator = torch.Generator(device="cpu").manual_seed(COMPILE_PROBE_SEED)
    x, y = train_data.batch(MICRO_BATCH_SIZE, SEQ_LEN, generator, device)
    parameter_model.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.autocast(device_type="cuda", dtype=_amp_dtype()):
        logits = callable_model(x)
        loss = F.cross_entropy(logits.float().reshape(-1, vocab_size), y.reshape(-1))
    loss.backward()
    torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated(device) / (1024**3)
    parameter_model.zero_grad(set_to_none=True)
    return {
        "seconds_forward_backward_no_update": seconds,
        "loss": float(loss.detach().cpu()),
        "peak_vram_gb": peak,
    }


def _steady_probe(
    callable_model: Any,
    parameter_model: nn.Module,
    train_data: TokenBin,
    *,
    vocab_size: int,
) -> dict[str, float | int]:
    device = torch.device("cuda")
    optimizer = torch.optim.AdamW(
        parameter_model.parameters(),
        lr=3e-4,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        fused=True,
    )
    generator = torch.Generator(device="cpu").manual_seed(PAIRED_BATCH_SEED)
    amp = _amp_dtype()
    parameter_model.train()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    last_loss = float("nan")
    for _ in range(PROBE_STEPS):
        optimizer.zero_grad(set_to_none=True)
        x, y = train_data.batch(MICRO_BATCH_SIZE, SEQ_LEN, generator, device)
        with torch.autocast(device_type="cuda", dtype=amp):
            logits = callable_model(x)
            loss = F.cross_entropy(logits.float().reshape(-1, vocab_size), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameter_model.parameters(), 1.0)
        optimizer.step()
        last_loss = float(loss.detach().cpu())
    torch.cuda.synchronize(device)
    seconds = max(time.perf_counter() - started, 1e-9)
    tokens = PROBE_STEPS * MICRO_BATCH_SIZE * SEQ_LEN
    peak = torch.cuda.max_memory_allocated(device) / (1024**3)
    return {
        "steps": PROBE_STEPS,
        "tokens": tokens,
        "seconds": seconds,
        "tokens_per_second": tokens / seconds,
        "last_loss_for_runtime_probe_only": last_loss,
        "peak_vram_gb": peak,
    }


def run_scale15m_calibration(*, data_dir: str) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("scale calibration requires CUDA")
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    train_data = TokenBin(str(Path(data_dir) / "train.bin"))

    equivalence = _prove_rlt_wrapper_equivalence(train_data)
    if not equivalence["passed"]:
        raise RuntimeError(f"RLT wrapper equivalence failed: {equivalence}")

    seed_all(ATTEMPT_SEED)
    rlt_base = RecurrentLoopedTransformer(rlt_config()).to(device)
    rlt_params = rlt_parameter_count(rlt_base)
    if rlt_params != EXPECTED_PARAMETERS:
        raise RuntimeError(f"RLT parameter drift: {rlt_params}")
    rlt_compiled = torch.compile(
        CompilableRLT(rlt_base),
        fullgraph=True,
        dynamic=False,
        mode="default",
    )
    rlt_compile = _compile_probe(rlt_compiled, rlt_base, train_data, vocab_size=50_257)
    rlt_steady = _steady_probe(rlt_compiled, rlt_base, train_data, vocab_size=50_257)
    del rlt_compiled, rlt_base
    torch.cuda.empty_cache()
    torch._dynamo.reset()

    seed_all(ATTEMPT_SEED)
    transformer = ResearchLM(transformer_config()).to(device)
    transformer_params = transformer_parameter_count(transformer)
    if transformer_params != EXPECTED_PARAMETERS:
        raise RuntimeError(f"Transformer parameter drift: {transformer_params}")
    if transformer_params != rlt_params:
        raise RuntimeError(f"exact parameter match failed: {rlt_params} vs {transformer_params}")
    transformer_compiled = torch.compile(
        transformer,
        fullgraph=True,
        dynamic=False,
        mode="default",
    )
    transformer_compile = _compile_probe(
        transformer_compiled,
        transformer,
        train_data,
        vocab_size=50_257,
    )
    transformer_steady = _steady_probe(
        transformer_compiled,
        transformer,
        train_data,
        vocab_size=50_257,
    )

    rlt_projected_train = FULL_TOKEN_BUDGET_REFERENCE / max(float(rlt_steady["tokens_per_second"]), 1e-9)
    transformer_projected_train = FULL_TOKEN_BUDGET_REFERENCE / max(float(transformer_steady["tokens_per_second"]), 1e-9)
    rlt_projected_effective = rlt_projected_train + float(rlt_compile["seconds_forward_backward_no_update"])
    transformer_projected_effective = transformer_projected_train + float(transformer_compile["seconds_forward_backward_no_update"])

    return {
        "status": "complete",
        "classification": "SCALE15M_CALIBRATION_PASS",
        "calibration_only": True,
        "scientific_quality_claim_supported": False,
        "breakthrough_claim_supported": False,
        "attempt_seed": ATTEMPT_SEED,
        "paired_batch_seed": PAIRED_BATCH_SEED,
        "compile_probe_seed": COMPILE_PROBE_SEED,
        "semantic_equivalence": equivalence,
        "exact_parameter_match": {
            "passed": rlt_params == transformer_params == EXPECTED_PARAMETERS,
            "rlt_parameters": rlt_params,
            "transformer_parameters": transformer_params,
            "delta_parameters": transformer_params - rlt_params,
        },
        "configs": {
            "rlt": asdict(rlt_config()),
            "transformer": asdict(transformer_config()),
        },
        "probe_conditions": {
            "seq_len": SEQ_LEN,
            "micro_batch_size": MICRO_BATCH_SIZE,
            "probe_steps": PROBE_STEPS,
            "full_token_budget_reference": FULL_TOKEN_BUDGET_REFERENCE,
            "compile_fullgraph": True,
            "compile_dynamic": False,
            "compile_mode": "default",
        },
        "rlt": {
            "compile_probe": rlt_compile,
            "steady_probe": rlt_steady,
            "projected_full_4m_train_seconds_excluding_eval": rlt_projected_train,
            "projected_full_4m_seconds_including_compile_excluding_eval": rlt_projected_effective,
        },
        "transformer": {
            "compile_probe": transformer_compile,
            "steady_probe": transformer_steady,
            "projected_full_4m_train_seconds_excluding_eval": transformer_projected_train,
            "projected_full_4m_seconds_including_compile_excluding_eval": transformer_projected_effective,
        },
        "derived": {
            "steady_transformer_throughput_multiple_vs_rlt": (
                float(transformer_steady["tokens_per_second"]) /
                max(float(rlt_steady["tokens_per_second"]), 1e-9)
            ),
            "projected_rlt_full_4m_hours_including_compile_excluding_eval": rlt_projected_effective / 3600.0,
            "projected_transformer_full_4m_hours_including_compile_excluding_eval": transformer_projected_effective / 3600.0,
        },
        "interpretation_ceiling": (
            "Runtime/memory/semantic calibration only. Loss values from the compile and 32-step "
            "probes are not scientific quality evidence and must not be used to claim an RLT "
            "quality advantage or disadvantage. A separate preregistered full scale run is required."
        ),
    }
