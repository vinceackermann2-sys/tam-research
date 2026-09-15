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
from tam_research.data import TokenBin
from tam_research.models import ModelConfig, ResearchLM, parameter_count as transformer_parameter_count
from tam_research.train import cosine_lr, seed_all


SCIENTIFIC_SEED = 20_260_926
PAIRED_BATCH_SEED = 20_270_926
PAIRED_EVAL_SEED = 20_290_926
COMPILE_PROBE_SEED = 20_300_926
EXPECTED_RLT_PARAMETERS = 7_007_616
EXPECTED_TRANSFORMER_PARAMETERS = 7_040_896
MAX_PARAMETER_DELTA_FRACTION = 0.005


class CompilableRLT(nn.Module):
    """Exact RLT forward math with only the diagnostic cache-length side effect omitted."""

    def __init__(self, base: RecurrentLoopedTransformer):
        super().__init__()
        self.base = base

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        model = self.base
        b, t = tokens.shape
        memory = model.encode(tokens)
        memory_kv = [stage.cross_attn.precompute(memory) for stage in model.stages]
        state = model.start_state.to(memory.dtype).view(1, -1).expand(b, -1)
        caches: list[tuple[torch.Tensor, torch.Tensor] | None] = [None for _ in model.stages]
        logits: list[torch.Tensor] = []
        for token_index in range(t):
            merged = torch.cat((memory[:, token_index, :], state), dim=-1)
            hidden = model.merge_norm(model.merge(merged)).unsqueeze(1)
            prefix_len = token_index + 1
            for layer_index, stage in enumerate(model.stages):
                hidden, caches[layer_index] = stage.decode_step(
                    hidden,
                    caches[layer_index],
                    memory_kv[layer_index],
                    prefix_len,
                    model.cfg.swa_window,
                )
            state = hidden[:, 0, :]
            logits.append(model.lm_head(model.output_norm(state)))
        return torch.stack(logits, dim=1)


def rlt_config(seq_len: int) -> RLTConfig:
    return RLTConfig(
        vocab_size=50_257,
        d_model=128,
        n_heads=4,
        n_stages=2,
        max_seq_len=max(128, seq_len),
        ff_mult=4,
        swa_window=32,
    )


def transformer_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=50_257,
        d_model=128,
        n_layers=3,
        n_heads=4,
        max_seq_len=128,
        ff_mult=4,
        architecture="transformer",
    )


def _amp_dtype() -> torch.dtype:
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def _optimizer(model: nn.Module, learning_rate: float, weight_decay: float) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=weight_decay,
        fused=True,
    )


def _prove_rlt_wrapper_equivalence(train_data: TokenBin, *, seq_len: int, batch_size: int) -> dict[str, float | bool]:
    device = torch.device("cuda")
    cfg = rlt_config(seq_len)
    seed_all(SCIENTIFIC_SEED)
    reference = RecurrentLoopedTransformer(cfg).to(device)
    seed_all(SCIENTIFIC_SEED)
    candidate_base = RecurrentLoopedTransformer(cfg).to(device)
    candidate_base.load_state_dict(reference.state_dict())
    candidate = CompilableRLT(candidate_base)
    generator = torch.Generator(device="cpu").manual_seed(COMPILE_PROBE_SEED)
    x, y = train_data.batch(batch_size, seq_len, generator, device)
    reference.zero_grad(set_to_none=True)
    candidate_base.zero_grad(set_to_none=True)
    amp = _amp_dtype()
    with torch.autocast(device_type="cuda", dtype=amp):
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
    passed = logits_max_abs <= 1e-3 and loss_abs <= 1e-5 and grad_max_abs <= 2e-3
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
    seq_len: int,
    batch_size: int,
    vocab_size: int,
) -> dict[str, float]:
    """Compile forward/backward without changing weights or consuming a training update."""
    device = torch.device("cuda")
    generator = torch.Generator(device="cpu").manual_seed(COMPILE_PROBE_SEED)
    x, y = train_data.batch(batch_size, seq_len, generator, device)
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


def _evaluate(
    callable_model: Any,
    parameter_model: nn.Module,
    val_data: TokenBin,
    *,
    seq_len: int,
    batch_size: int,
    vocab_size: int,
    batches: int = 5,
) -> dict[str, float]:
    parameter_model.eval()
    device = torch.device("cuda")
    generator = torch.Generator(device="cpu").manual_seed(PAIRED_EVAL_SEED)
    losses: list[float] = []
    with torch.no_grad():
        for _ in range(batches):
            x, y = val_data.batch(batch_size, seq_len, generator, device)
            with torch.autocast(device_type="cuda", dtype=_amp_dtype()):
                logits = callable_model(x)
                loss = F.cross_entropy(logits.float().reshape(-1, vocab_size), y.reshape(-1))
            losses.append(float(loss.detach().cpu()))
    torch.cuda.synchronize(device)
    nll = sum(losses) / len(losses)
    return {"nll": nll, "perplexity": math.exp(min(nll, 20.0))}


def _train_one(
    *,
    name: str,
    callable_model: Any,
    parameter_model: nn.Module,
    train_data: TokenBin,
    val_data: TokenBin,
    vocab_size: int,
    token_budget: int,
    seq_len: int,
    micro_batch_size: int,
    grad_accum_steps: int,
    learning_rate: float,
    weight_decay: float,
    warmup_ratio: float,
) -> dict[str, Any]:
    device = torch.device("cuda")
    optimizer = _optimizer(parameter_model, learning_rate, weight_decay)
    tokens_per_step = micro_batch_size * seq_len * grad_accum_steps
    total_steps = math.ceil(token_budget / tokens_per_step)
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    batch_gen = torch.Generator(device="cpu").manual_seed(PAIRED_BATCH_SEED)
    amp = _amp_dtype()
    scaler = torch.cuda.amp.GradScaler(enabled=amp == torch.float16)

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    tokens_seen = 0
    last_loss = float("nan")
    for step in range(total_steps):
        parameter_model.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        for _ in range(grad_accum_steps):
            x, y = train_data.batch(micro_batch_size, seq_len, batch_gen, device)
            with torch.autocast(device_type="cuda", dtype=amp):
                logits = callable_model(x)
                loss = F.cross_entropy(logits.float().reshape(-1, vocab_size), y.reshape(-1)) / grad_accum_steps
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            running += float(loss.detach().cpu()) * grad_accum_steps
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(parameter_model.parameters(), 1.0)
        lr = cosine_lr(step, total_steps, warmup_steps, learning_rate)
        for group in optimizer.param_groups:
            group["lr"] = lr
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        last_loss = running / grad_accum_steps
        tokens_seen = min(token_budget, (step + 1) * tokens_per_step)
        if step == 0 or (step + 1) % 16 == 0 or step + 1 == total_steps:
            print({"architecture": name, "step": step + 1, "steps": total_steps, "tokens_seen": tokens_seen, "train_loss": last_loss, "lr": lr}, flush=True)
    torch.cuda.synchronize(device)
    train_seconds = max(time.perf_counter() - started, 1e-9)
    train_peak = torch.cuda.max_memory_allocated(device) / (1024**3)
    final_eval = _evaluate(
        callable_model,
        parameter_model,
        val_data,
        seq_len=seq_len,
        batch_size=micro_batch_size,
        vocab_size=vocab_size,
    )
    return {
        "status": "complete",
        "tokens_seen": tokens_seen,
        "steps": total_steps,
        "last_train_loss": last_loss,
        "final_eval": final_eval,
        "train_seconds_excluding_compile": train_seconds,
        "tokens_per_second_excluding_compile": tokens_seen / train_seconds,
        "train_peak_vram_gb": train_peak,
    }


def train_compiled_pair(
    *,
    data_dir: str,
    token_budget: int = 65_536,
    seq_len: int = 64,
    micro_batch_size: int = 2,
    grad_accum_steps: int = 4,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.1,
    warmup_ratio: float = 0.02,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("compiled paired comparison requires CUDA")
    if token_budget != 65_536 or seq_len != 64 or micro_batch_size != 2 or grad_accum_steps != 4:
        raise RuntimeError("compiled paired 64K preregistration drift")
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    train_data = TokenBin(str(Path(data_dir) / "train.bin"))
    val_data = TokenBin(str(Path(data_dir) / "val.bin"))

    equivalence = _prove_rlt_wrapper_equivalence(train_data, seq_len=seq_len, batch_size=micro_batch_size)
    if not equivalence["passed"]:
        raise RuntimeError(f"RLT compile wrapper semantic equivalence failed: {equivalence}")

    # RLT: fresh scientific initialization, default-mode fullgraph compile, no CUDA graphs.
    seed_all(SCIENTIFIC_SEED)
    rlt_base = RecurrentLoopedTransformer(rlt_config(seq_len)).to(device)
    rlt_params = rlt_parameter_count(rlt_base)
    if rlt_params != EXPECTED_RLT_PARAMETERS:
        raise RuntimeError(f"RLT parameter drift: {rlt_params}")
    rlt_wrapper = CompilableRLT(rlt_base)
    rlt_compiled = torch.compile(rlt_wrapper, fullgraph=True, dynamic=False, mode="default")
    rlt_compile = _compile_probe(
        rlt_compiled,
        rlt_base,
        train_data,
        seq_len=seq_len,
        batch_size=micro_batch_size,
        vocab_size=50_257,
    )
    rlt_training = _train_one(
        name="compiled-rlt",
        callable_model=rlt_compiled,
        parameter_model=rlt_base,
        train_data=train_data,
        val_data=val_data,
        vocab_size=50_257,
        token_budget=token_budget,
        seq_len=seq_len,
        micro_batch_size=micro_batch_size,
        grad_accum_steps=grad_accum_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
    )
    rlt_effective_seconds = rlt_compile["seconds_forward_backward_no_update"] + rlt_training["train_seconds_excluding_compile"]
    rlt_training["compile"] = rlt_compile
    rlt_training["effective_tokens_per_second_including_compile"] = token_budget / rlt_effective_seconds
    rlt_training["parameters"] = rlt_params
    rlt_training["config"] = asdict(rlt_config(seq_len))
    del rlt_compiled, rlt_wrapper, rlt_base
    torch.cuda.empty_cache()
    torch._dynamo.reset()

    # Transformer: same scientific seed, same data/eval streams, same compile policy.
    seed_all(SCIENTIFIC_SEED)
    transformer = ResearchLM(transformer_config()).to(device)
    transformer_params = transformer_parameter_count(transformer)
    if transformer_params != EXPECTED_TRANSFORMER_PARAMETERS:
        raise RuntimeError(f"Transformer parameter drift: {transformer_params}")
    delta = (transformer_params - rlt_params) / rlt_params
    if abs(delta) > MAX_PARAMETER_DELTA_FRACTION:
        raise RuntimeError(f"parameter match exceeds 0.5%: {delta}")
    transformer_compiled = torch.compile(transformer, fullgraph=True, dynamic=False, mode="default")
    transformer_compile = _compile_probe(
        transformer_compiled,
        transformer,
        train_data,
        seq_len=seq_len,
        batch_size=micro_batch_size,
        vocab_size=50_257,
    )
    transformer_training = _train_one(
        name="compiled-transformer-control",
        callable_model=transformer_compiled,
        parameter_model=transformer,
        train_data=train_data,
        val_data=val_data,
        vocab_size=50_257,
        token_budget=token_budget,
        seq_len=seq_len,
        micro_batch_size=micro_batch_size,
        grad_accum_steps=grad_accum_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
    )
    transformer_effective_seconds = transformer_compile["seconds_forward_backward_no_update"] + transformer_training["train_seconds_excluding_compile"]
    transformer_training["compile"] = transformer_compile
    transformer_training["effective_tokens_per_second_including_compile"] = token_budget / transformer_effective_seconds
    transformer_training["parameters"] = transformer_params
    transformer_training["config"] = asdict(transformer_config())

    rlt_nll = float(rlt_training["final_eval"]["nll"])
    transformer_nll = float(transformer_training["final_eval"]["nll"])
    rlt_ppl = float(rlt_training["final_eval"]["perplexity"])
    transformer_ppl = float(transformer_training["final_eval"]["perplexity"])
    quality_delta_nll = rlt_nll - transformer_nll
    quality_delta_ppl = rlt_ppl - transformer_ppl
    classification = (
        "COMPILED_64K_RLT_LOWER_NLL_SINGLE_RUN_HINT"
        if quality_delta_nll < 0
        else "COMPILED_64K_NO_RLT_QUALITY_ADVANTAGE"
    )
    return {
        "status": "complete",
        "classification": classification,
        "breakthrough_claim_supported": False,
        "scientific_seed": SCIENTIFIC_SEED,
        "paired_batch_seed": PAIRED_BATCH_SEED,
        "paired_eval_seed": PAIRED_EVAL_SEED,
        "compile_probe_seed": COMPILE_PROBE_SEED,
        "semantic_equivalence": equivalence,
        "matched_conditions": {
            "token_budget": token_budget,
            "seq_len": seq_len,
            "micro_batch_size": micro_batch_size,
            "grad_accum_steps": grad_accum_steps,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "compile_mode": "default",
            "compile_fullgraph": True,
            "compile_dynamic": False,
            "cuda_graphs": False,
            "same_model_seed": True,
            "same_batch_stream": True,
            "same_eval_stream": True,
        },
        "rlt": rlt_training,
        "transformer": transformer_training,
        "derived": {
            "rlt_minus_transformer_nll": quality_delta_nll,
            "rlt_minus_transformer_perplexity": quality_delta_ppl,
            "steady_state_transformer_throughput_multiple_vs_rlt": transformer_training["tokens_per_second_excluding_compile"] / max(rlt_training["tokens_per_second_excluding_compile"], 1e-9),
            "effective_transformer_throughput_multiple_vs_rlt_including_compile": transformer_training["effective_tokens_per_second_including_compile"] / max(rlt_training["effective_tokens_per_second_including_compile"], 1e-9),
            "transformer_parameter_delta_fraction": delta,
        },
        "interpretation_ceiling": "Single 64K-token paired run. May indicate direction only; cannot support a breakthrough, scaling, or general capability claim.",
    }
