from __future__ import annotations

"""CHM-v1 #959 optimized exact-retrieval replication inference harness.

This module is additive. It leaves the frozen #854 training/model implementation
unchanged and substitutes only the validated clipped-query exact traversal from
#957/#958 for indexed inference. It contains no launcher, GPU allocation, paid
compute action, or scientific-seed consumption mechanism.
"""

import math
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .aera_real_language import TOKEN_BUDGET, VOCAB_SIZE
from .chm_v1_clipped_bound_prototype import ClippedBoundPrototype
from .chm_v1_small_lm import (
    ADDRESS_DIM,
    LEAF_SIZE,
    LOCAL_WINDOW,
    RETRIEVAL_HOPS,
    CHMV1EIEMLM,
    EpisodicState,
    RetrievalMode,
    RetrievalStats,
    SearchResult,
    parameter_accounting,
)
from .chm_v1_small_lm_protocol import (
    BETAS,
    COMPILE_ENABLED,
    GRAD_ACCUM,
    GRAD_CLIP,
    MICRO_BATCH,
    PEAK_LR,
    SESSION_LEN,
    TOKENS_PER_STEP,
    TOTAL_STEPS,
    WARMUP_STEPS,
    WEIGHT_DECAY,
    _autocast,
    _chunked,
    _peak_memory_bytes,
    _reset_peak_memory,
    _synchronize,
)
from .data import TokenBin

REPLICATION_ISSUE = 959
REPLICATION_BASE_SHA = "2e1378f92048e2a3a32d0e36bf895c8846c6a20d"
FRESH_SCIENTIFIC_SEEDS = (19_591, 19_592, 19_593)
LEGACY_BLOCKED_SCIENTIFIC_SEEDS = (8_611, 8_612, 8_613)
NON_SCIENTIFIC_SMOKE_SEED = 959_001


def validate_replication_seed(seed: int, *, paid_run_authorized: bool = False) -> int:
    """Guard a future runner without granting execution authority itself.

    Legacy #854 seeds are permanently refused. Fresh #959 seeds require a
    separate explicit run-control authorization before a future caller may opt
    in with ``paid_run_authorized=True``. This function never allocates compute.
    """

    seed = int(seed)
    if seed in LEGACY_BLOCKED_SCIENTIFIC_SEEDS:
        raise RuntimeError(f"legacy scientific seed is permanently blocked: {seed}")
    if seed in FRESH_SCIENTIFIC_SEEDS and not paid_run_authorized:
        raise RuntimeError(
            f"fresh #959 scientific seed requires separate paid/GPU run-control authorization: {seed}"
        )
    return seed


def replication_preflight() -> dict[str, Any]:
    """CPU-safe protocol identity/fairness check; no scientific execution."""

    if SESSION_LEN != 2 * LOCAL_WINDOW:
        raise RuntimeError("replication session geometry drifted")
    if RETRIEVAL_HOPS != 2:
        raise RuntimeError("replication retrieval-hop count drifted")
    if TOKEN_BUDGET != 8_388_608:
        raise RuntimeError("replication token budget drifted")
    if TOKENS_PER_STEP != MICRO_BATCH * SESSION_LEN * GRAD_ACCUM:
        raise RuntimeError("replication optimizer token accounting drifted")
    if TOTAL_STEPS * TOKENS_PER_STEP != TOKEN_BUDGET:
        raise RuntimeError("replication total-step accounting drifted")
    accounting = parameter_accounting()
    if not accounting["within_preregistered_one_percent"]:
        raise RuntimeError(f"parameter fairness failed: {accounting}")
    return {
        "classification": "IMPLEMENTATION_PREFLIGHT_ONLY_NO_GPU_AUTHORITY",
        "research_issue": REPLICATION_ISSUE,
        "replication_base_sha": REPLICATION_BASE_SHA,
        "fresh_scientific_seeds": list(FRESH_SCIENTIFIC_SEEDS),
        "legacy_blocked_scientific_seeds": list(LEGACY_BLOCKED_SCIENTIFIC_SEEDS),
        "local_window": LOCAL_WINDOW,
        "address_dim": ADDRESS_DIM,
        "leaf_size": LEAF_SIZE,
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
        "optimized_index": "ClippedBoundPrototype",
        "gpu_authorized": False,
        "scientific_seed_consumed": False,
    }


class OptimizedEpisodicState(EpisodicState):
    """Frozen episodic state with #958 traversal substituted for indexed search."""

    def __init__(self, session_id: str):
        super().__init__(session_id=session_id)
        self._cached_optimized_index: ClippedBoundPrototype | None = None
        self.optimized_sidecar_build_seconds_total = 0.0
        self.optimized_wrapper_seconds_total = 0.0

    def reset(self) -> None:
        super().reset()
        self._cached_optimized_index = None
        self.optimized_sidecar_build_seconds_total = 0.0
        self.optimized_wrapper_seconds_total = 0.0

    def write(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        super().write(keys, values)
        self._cached_optimized_index = None

    def optimized_index(self) -> ClippedBoundPrototype:
        frozen = self.index()
        if (
            self._cached_optimized_index is None
            or self._cached_optimized_index.frozen is not frozen
        ):
            started = time.perf_counter()
            self._cached_optimized_index = ClippedBoundPrototype(frozen)
            elapsed = time.perf_counter() - started
            self.optimized_sidecar_build_seconds_total += elapsed
            # Sidecar construction is part of practical optimized-index build
            # cost and therefore participates in the inherited build accounting.
            self.index_build_seconds_total += elapsed
        return self._cached_optimized_index

    def retrieve(
        self,
        query: torch.Tensor,
        *,
        mode: RetrievalMode,
        verify_indexed_exactness: bool = True,
    ) -> tuple[torch.Tensor, SearchResult, bool, float, float, float]:
        if mode != "indexed":
            return super().retrieve(
                query,
                mode=mode,
                verify_indexed_exactness=verify_indexed_exactness,
            )
        if query.ndim != 1:
            raise ValueError("query must be one address vector")

        call_started = time.perf_counter()
        _, values, _ = self._arrays()
        q = query.detach().float().cpu().numpy().astype(np.float32, copy=False)
        build_before = self.index_build_seconds_total
        optimized = self.optimized_index()
        build_delta = self.index_build_seconds_total - build_before

        started = time.perf_counter()
        result = optimized.indexed_search(q)
        search_elapsed = time.perf_counter() - started
        self.indexed_search_seconds_total += search_elapsed

        verification_elapsed = 0.0
        if verify_indexed_exactness:
            verify_started = time.perf_counter()
            flat = optimized.frozen.flat_search(q)
            verification_elapsed = time.perf_counter() - verify_started
            self.verification_seconds_total += verification_elapsed
            exact_match = result.item_id == flat.item_id and result.position == flat.position
            if not exact_match:
                raise AssertionError(
                    f"optimized-indexed/flat mismatch in session {self.session_id}: "
                    f"optimized={result} flat={flat}"
                )
        else:
            exact_match = True

        value = torch.as_tensor(values[result.position], device=query.device, dtype=query.dtype)
        total_elapsed = time.perf_counter() - call_started
        wrapper_elapsed = max(
            0.0,
            total_elapsed - build_delta - search_elapsed - verification_elapsed,
        )
        self.optimized_wrapper_seconds_total += wrapper_elapsed
        return (
            value,
            result,
            exact_match,
            build_delta,
            search_elapsed,
            verification_elapsed,
        )


@torch.no_grad()
def evaluate_eiem_optimized_language(
    model: CHMV1EIEMLM,
    val: TokenBin,
    *,
    batches: int,
    batch_size: int,
    seed: int,
) -> dict[str, float]:
    """Inherited #854 language evaluation using only optimized indexed retrieval."""

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
    wrapper_seconds = 0.0
    sidecar_build_seconds = 0.0
    max_state_payload_bytes = 0
    tokens_evaluated = batches * batch_size * SESSION_LEN

    _reset_peak_memory(device)
    _synchronize(device)
    started = time.perf_counter()
    for batch_no in range(batches):
        x, y = val.batch(batch_size, SESSION_LEN, generator, device)
        states = [OptimizedEpisodicState(f"opt-eval-{batch_no}-{i}") for i in range(batch_size)]
        logits: list[torch.Tensor] = []
        for chunk in _chunked(x):
            with _autocast(device):
                chunk_logits, stats = model.forward_session_chunk(
                    chunk,
                    states,
                    mode="indexed",
                    update_memory=True,
                    verify_indexed_exactness=True,
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
        wrapper_seconds += sum(state.optimized_wrapper_seconds_total for state in states)
        sidecar_build_seconds += sum(
            state.optimized_sidecar_build_seconds_total for state in states
        )
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
        "optimized_sidecar_build_seconds": sidecar_build_seconds,
        "search_seconds": search_seconds,
        "verification_seconds": verification_seconds,
        "write_seconds": write_seconds,
        "wrapper_seconds": wrapper_seconds,
        "max_state_payload_bytes": float(max_state_payload_bytes),
    }
