from __future__ import annotations

"""Training/evaluation protocol binding for preregistered CHM-v1 gate #854.

No launcher or paid-compute trigger lives here. The module fixes the matched
training geometry before any scientific seed is run: 1024-token contiguous
sessions are processed as two independent 512-token local chunks. LOCAL gets
no cross-chunk state; EIEM may retrieve only exact causal representations from
previous chunks. MICRO_BATCH=4 keeps global tokens/optimizer-step identical to
the repository's established 8x512x4 small real-language protocol. EIEM uses
the same frozen two sequential retrieval hops in differentiable training and
exact inference.
"""

from contextlib import nullcontext
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Literal

import torch
import torch.nn.functional as F

from .aera_real_language import TOKEN_BUDGET, VOCAB_SIZE
from .chm_v1_small_lm import (
    LOCAL_WINDOW,
    RETRIEVAL_HOPS,
    SCIENTIFIC_SEEDS,
    CHMV1EIEMLM,
    CHMV1LocalLM,
    EpisodicState,
    _hidden,
    parameter_accounting,
)
from .data import TokenBin
from .train import cosine_lr

SESSION_LEN = 1024
MICRO_BATCH = 4
GRAD_ACCUM = 4
TOKENS_PER_STEP = MICRO_BATCH * SESSION_LEN * GRAD_ACCUM
TOTAL_STEPS = TOKEN_BUDGET // TOKENS_PER_STEP
PEAK_LR = 3e-4
WARMUP_STEPS = max(1, int(TOTAL_STEPS * 0.02))
WEIGHT_DECAY = 0.1
BETAS = (0.9, 0.95)
GRAD_CLIP = 1.0
FLAT_TRAIN_TEMPERATURE = 0.10
COMPILE_ENABLED = False

Kind = Literal["local", "eiem"]


def _autocast(device: torch.device):
    return (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _peak_memory_bytes(device: torch.device) -> int:
    return int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0


def _seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _chunked(tokens: torch.Tensor):
    if tokens.ndim != 2 or tokens.shape[1] != SESSION_LEN:
        raise ValueError(f"expected [batch,{SESSION_LEN}] contiguous sessions")
    for start in range(0, SESSION_LEN, LOCAL_WINDOW):
        yield tokens[:, start : start + LOCAL_WINDOW]


def local_session_logits(model: CHMV1LocalLM, tokens: torch.Tensor) -> torch.Tensor:
    """Two bounded-local chunks, deliberately no cross-chunk model state."""
    return torch.cat([model(chunk) for chunk in _chunked(tokens)], dim=1)


def eiem_flat_training_session_logits(
    model: CHMV1EIEMLM,
    tokens: torch.Tensor,
    *,
    temperature: float = FLAT_TRAIN_TEMPERATURE,
) -> torch.Tensor:
    """Differentiable exhaustive two-hop memory over prior 512-token chunks.

    The memory contains exact hidden values; only retrieval weights are a soft
    exhaustive training surrogate. Each hop addresses the same prior evidence
    bank, and the first retrieved value changes the representation used by the
    second learned query. Current-chunk keys/values are appended only after that
    chunk's logits are formed.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    memory_keys: torch.Tensor | None = None
    memory_values: torch.Tensor | None = None
    logits: list[torch.Tensor] = []

    for chunk in _chunked(tokens):
        hidden = _hidden(model.backbone, chunk)
        keys = model.key_for(hidden)
        query_state = hidden
        if memory_keys is not None:
            assert memory_values is not None
            for _ in range(RETRIEVAL_HOPS):
                queries = model.query_for(query_state)
                score = torch.einsum("btd,bsd->bts", queries, memory_keys) / temperature
                weights = torch.softmax(score.float(), dim=-1).to(hidden.dtype)
                memory = torch.matmul(weights, memory_values)
                query_state = model._integrate(query_state, memory)
        logits.append(model.backbone.lm_head(query_state))
        # Appending after logits prevents self/future episodic access.
        memory_keys = keys if memory_keys is None else torch.cat((memory_keys, keys), dim=1)
        memory_values = hidden if memory_values is None else torch.cat((memory_values, hidden), dim=1)
    return torch.cat(logits, dim=1)


def build_scientific_pair(
    seed: int,
    device: torch.device,
    *,
    paid_run_authorized: bool = False,
) -> tuple[CHMV1LocalLM, CHMV1EIEMLM]:
    """Construct paired models while refusing reserved seeds by default.

    A future separately authorized runner must opt in explicitly. This flag is
    only a programmatic guardrail; setting it is not itself authorization.
    """
    if seed in SCIENTIFIC_SEEDS and not paid_run_authorized:
        raise RuntimeError("reserved scientific seed requires separate paid-run authorization")
    _seed_all(seed)
    local = CHMV1LocalLM().to(device)
    _seed_all(seed)
    eiem = CHMV1EIEMLM().to(device)
    for name, value in local.backbone.state_dict().items():
        if not torch.equal(value, eiem.backbone.state_dict()[name]):
            raise RuntimeError(f"paired backbone init mismatch at {name}")
    return local, eiem


def validate_corpus(data_dir: str) -> dict[str, Any]:
    root = Path(data_dir)
    required = [root / "train.bin", root / "val.bin", root / "meta.json"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"CHM-v1 corpus files missing: {missing}")
    meta = json.loads((root / "meta.json").read_text())
    expected = {
        "assembly_version": 3,
        "train_tokens": 2_000_000_000,
        "val_tokens": 5_000_000,
        "seed": 8100,
        "tokenizer": "gpt2",
        "dtype": "uint16",
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            raise RuntimeError(f"data metadata mismatch {key}: {meta.get(key)!r} != {value!r}")
    if (root / "train.bin").stat().st_size != 4_000_000_000:
        raise RuntimeError("train.bin is not exact 2B uint16 token stream")
    if (root / "val.bin").stat().st_size != 10_000_000:
        raise RuntimeError("val.bin is not exact 5M uint16 validation stream")
    return expected


def protocol_preflight(data_dir: str | None = None) -> dict[str, Any]:
    if SESSION_LEN != 2 * LOCAL_WINDOW:
        raise RuntimeError("small-LM session must remain exactly two local windows")
    if RETRIEVAL_HOPS != 2:
        raise RuntimeError("two-hop gate requires exactly two sequential retrieval hops")
    if TOKEN_BUDGET % TOKENS_PER_STEP:
        raise RuntimeError("token budget must divide exactly by tokens/step")
    if TOKENS_PER_STEP != 8 * 512 * 4:
        raise RuntimeError("global optimizer token accounting drifted from inherited protocol")
    accounting = parameter_accounting()
    if not accounting["within_preregistered_one_percent"]:
        raise RuntimeError(f"parameter fairness failed: {accounting}")
    corpus = validate_corpus(data_dir) if data_dir is not None else None
    return {
        "classification": "IMPLEMENTATION_PREFLIGHT_ONLY_NO_GPU_AUTHORITY",
        "research_issue": 854,
        "scientific_base": "91b02ec575440b42050377351a6e404139632e22",
        "scientific_seeds_reserved": list(SCIENTIFIC_SEEDS),
        "local_window": LOCAL_WINDOW,
        "retrieval_hops": RETRIEVAL_HOPS,
        "session_len": SESSION_LEN,
        "micro_batch": MICRO_BATCH,
        "grad_accum": GRAD_ACCUM,
        "tokens_per_step": TOKENS_PER_STEP,
        "token_budget_per_model": TOKEN_BUDGET,
        "optimizer_steps": TOTAL_STEPS,
        "warmup_steps": WARMUP_STEPS,
        "compile_enabled": COMPILE_ENABLED,
        "optimizer": {
            "name": "AdamW",
            "betas": list(BETAS),
            "weight_decay": WEIGHT_DECAY,
            "peak_lr": PEAK_LR,
            "grad_clip": GRAD_CLIP,
        },
        "parameter_accounting": accounting,
        "corpus": corpus,
        "gpu_authorized": False,
    }


def _optimizer(model: torch.nn.Module, device: torch.device) -> torch.optim.Optimizer:
    kwargs: dict[str, Any] = {
        "lr": PEAK_LR,
        "betas": BETAS,
        "weight_decay": WEIGHT_DECAY,
    }
    if device.type == "cuda":
        kwargs["fused"] = True
    return torch.optim.AdamW(model.parameters(), **kwargs)


def train_one(
    kind: Kind,
    model: CHMV1LocalLM | CHMV1EIEMLM,
    train_data: TokenBin,
    *,
    device: torch.device,
    seed: int,
    max_steps: int | None = None,
    paid_run_authorized: bool = False,
) -> dict[str, Any]:
    """Matched loop; caller owns checkpointing and external run authorization."""
    if kind not in {"local", "eiem"}:
        raise ValueError(kind)
    if seed in SCIENTIFIC_SEEDS and not paid_run_authorized:
        raise RuntimeError("reserved scientific seed requires separate paid-run authorization")
    optimizer = _optimizer(model, device)
    generator = torch.Generator(device="cpu").manual_seed(seed + 10_000)
    steps = TOTAL_STEPS if max_steps is None else min(int(max_steps), TOTAL_STEPS)
    losses: list[float] = []
    tokens_seen = 0

    _reset_peak_memory(device)
    _synchronize(device)
    started = time.perf_counter()
    for step in range(steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        micro_losses: list[float] = []
        for _ in range(GRAD_ACCUM):
            x, y = train_data.batch(MICRO_BATCH, SESSION_LEN, generator, device)
            with _autocast(device):
                if kind == "local":
                    assert isinstance(model, CHMV1LocalLM)
                    logits = local_session_logits(model, x)
                else:
                    assert isinstance(model, CHMV1EIEMLM)
                    logits = eiem_flat_training_session_logits(model, x)
                loss = F.cross_entropy(logits.float().reshape(-1, VOCAB_SIZE), y.reshape(-1))
                scaled = loss / GRAD_ACCUM
            scaled.backward()
            micro_losses.append(float(loss.detach()))
            tokens_seen += x.numel()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        lr = cosine_lr(step, TOTAL_STEPS, WARMUP_STEPS, PEAK_LR)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        losses.append(sum(micro_losses) / len(micro_losses))
    _synchronize(device)
    elapsed = time.perf_counter() - started

    return {
        "kind": kind,
        "seed": seed,
        "steps_completed": steps,
        "tokens_seen": tokens_seen,
        "final_train_nll": losses[-1] if losses else float("nan"),
        "loss_trajectory": losses,
        "wall_seconds": elapsed,
        "tokens_per_second": tokens_seen / max(elapsed, 1e-9),
        "peak_vram_bytes": _peak_memory_bytes(device),
        "compiled": COMPILE_ENABLED,
        "compile_seconds": 0.0,
    }


@torch.no_grad()
def evaluate_local_language(
    model: CHMV1LocalLM,
    val: TokenBin,
    *,
    batches: int,
    batch_size: int,
    seed: int,
) -> dict[str, float]:
    model.eval()
    device = next(model.parameters()).device
    generator = torch.Generator(device="cpu").manual_seed(seed)
    losses: list[float] = []
    tokens_evaluated = batches * batch_size * SESSION_LEN
    _reset_peak_memory(device)
    _synchronize(device)
    started = time.perf_counter()
    for _ in range(batches):
        x, y = val.batch(batch_size, SESSION_LEN, generator, device)
        with _autocast(device):
            logits = local_session_logits(model, x)
        losses.append(float(F.cross_entropy(logits.float().reshape(-1, VOCAB_SIZE), y.reshape(-1))))
    _synchronize(device)
    elapsed = time.perf_counter() - started
    nll = sum(losses) / len(losses)
    return {
        "nll": nll,
        "perplexity": math.exp(min(nll, 20.0)),
        "batch_size": float(batch_size),
        "tokens_evaluated": float(tokens_evaluated),
        "wall_seconds": elapsed,
        "tokens_per_second": tokens_evaluated / max(elapsed, 1e-9),
        "peak_vram_bytes": float(_peak_memory_bytes(device)),
    }


@torch.no_grad()
def evaluate_eiem_language(
    model: CHMV1EIEMLM,
    val: TokenBin,
    *,
    mode: Literal["flat", "indexed"],
    batches: int,
    batch_size: int,
    seed: int,
) -> dict[str, float]:
    model.eval()
    device = next(model.parameters()).device
    generator = torch.Generator(device="cpu").manual_seed(seed)
    losses: list[float] = []
    exact_matches = 0
    calls = 0
    reads = 0
    flat_reads = 0
    nodes = 0
    index_build_seconds = 0.0
    search_seconds = 0.0
    verification_seconds = 0.0
    write_seconds = 0.0
    max_state_payload_bytes = 0
    tokens_evaluated = batches * batch_size * SESSION_LEN

    _reset_peak_memory(device)
    _synchronize(device)
    started = time.perf_counter()
    for batch_no in range(batches):
        x, y = val.batch(batch_size, SESSION_LEN, generator, device)
        states = [EpisodicState(f"eval-{batch_no}-{i}") for i in range(batch_size)]
        chunks = list(_chunked(x))
        logits: list[torch.Tensor] = []
        for chunk in chunks:
            with _autocast(device):
                chunk_logits, stats = model.forward_session_chunk(
                    chunk,
                    states,
                    mode=mode,
                    update_memory=True,
                    verify_indexed_exactness=mode == "indexed",
                )
            logits.append(chunk_logits)
            exact_matches += stats.exact_matches
            calls += stats.calls
            reads += stats.address_vector_reads
            flat_reads += stats.flat_address_vector_reads
            nodes += stats.directory_nodes_visited
            index_build_seconds += stats.index_build_seconds
            search_seconds += stats.search_seconds
            verification_seconds += stats.verification_seconds
            write_seconds += stats.write_seconds
            max_state_payload_bytes = max(max_state_payload_bytes, stats.state_payload_bytes)
        joined = torch.cat(logits, dim=1)
        losses.append(float(F.cross_entropy(joined.float().reshape(-1, VOCAB_SIZE), y.reshape(-1))))
    _synchronize(device)
    elapsed = time.perf_counter() - started
    nll = sum(losses) / len(losses)
    return {
        "nll": nll,
        "perplexity": math.exp(min(nll, 20.0)),
        "batch_size": float(batch_size),
        "tokens_evaluated": float(tokens_evaluated),
        "wall_seconds": elapsed,
        "tokens_per_second": tokens_evaluated / max(elapsed, 1e-9),
        "peak_vram_bytes": float(_peak_memory_bytes(device)),
        "retrieval_calls": float(calls),
        "indexed_flat_exact_match_rate": exact_matches / max(calls, 1),
        "address_vector_read_fraction": reads / max(flat_reads, 1),
        "directory_nodes_per_retrieval": nodes / max(calls, 1),
        "index_build_seconds": index_build_seconds,
        "search_seconds": search_seconds,
        "verification_seconds": verification_seconds,
        "write_seconds": write_seconds,
        "max_state_payload_bytes": float(max_state_payload_bytes),
    }
