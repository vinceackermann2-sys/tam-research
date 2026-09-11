from __future__ import annotations

"""Preregistered CHM-v1 / EIEM small matched-LM implementation (#854).

This file contains model and inference-memory plumbing only. It does not launch
training, consume scientific seeds, or authorize GPU work.

LOCAL and EIEM share the repository's established ~25M GPT-style Transformer
backbone. Calls are hard-limited to 512 local tokens. EIEM adds only learned
normalized query/key address projections plus a channel-wise integration gate;
full causal hidden states are stored as exact values. Two sequential retrieval
hops are frozen before scientific execution: the first retrieved value changes
the representation used to form the second learned query, so the preregistered
two-hop probe cannot be satisfied merely by a one-read output shortcut. At
inference a chunk retrieves only from evidence written by earlier chunks and
current-chunk writes happen after all logits are formed, preventing self/future
leakage.
"""

from dataclasses import dataclass, field
import hashlib
import time
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .aera_real_language import TOKEN_BUDGET, transformer_25m_config
from .chm_v1_exact_index import ExactEpisodicIndex, SearchResult
from .models import ResearchLM, parameter_count

LOCAL_WINDOW = 512
ADDRESS_DIM = 32
LEAF_SIZE = 16
RETRIEVAL_HOPS = 2
NON_SCIENTIFIC_SMOKE_SEED = 12_345
SCIENTIFIC_SEEDS = (8611, 8612, 8613)

RetrievalMode = Literal["flat", "indexed"]


def _hidden(backbone: ResearchLM, tokens: torch.Tensor) -> torch.Tensor:
    if tokens.ndim != 2:
        raise ValueError("tokens must be [batch, time]")
    _, length = tokens.shape
    if length > LOCAL_WINDOW:
        raise ValueError(
            f"CHM-v1 local path refuses {length} tokens; local window is {LOCAL_WINDOW}"
        )
    pos = torch.arange(length, device=tokens.device)
    x = backbone.token_emb(tokens) + backbone.pos_emb(pos)[None]
    for block in backbone.blocks:
        x = block(x)
    return backbone.norm(x)


class CHMV1LocalLM(nn.Module):
    """Bounded-local Transformer control; no persistent episodic state."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = ResearchLM(transformer_25m_config())

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        h = _hidden(self.backbone, tokens)
        return self.backbone.lm_head(h)


@dataclass
class RetrievalStats:
    calls: int = 0
    address_vector_reads: int = 0
    flat_address_vector_reads: int = 0
    directory_nodes_visited: int = 0
    exact_matches: int = 0
    index_build_seconds: float = 0.0
    search_seconds: float = 0.0
    verification_seconds: float = 0.0
    write_seconds: float = 0.0
    state_payload_bytes: int = 0

    def add(
        self,
        result: SearchResult,
        memory_size: int,
        *,
        exact_match: bool,
        index_build_seconds: float = 0.0,
        search_seconds: float = 0.0,
        verification_seconds: float = 0.0,
    ) -> None:
        self.calls += 1
        self.address_vector_reads += int(result.address_vector_reads)
        self.flat_address_vector_reads += int(memory_size)
        self.directory_nodes_visited += int(result.directory_nodes_visited)
        self.exact_matches += int(exact_match)
        self.index_build_seconds += float(index_build_seconds)
        self.search_seconds += float(search_seconds)
        self.verification_seconds += float(verification_seconds)

    @property
    def exact_match_rate(self) -> float:
        return self.exact_matches / max(self.calls, 1)

    @property
    def address_read_fraction(self) -> float:
        return self.address_vector_reads / max(self.flat_address_vector_reads, 1)


@dataclass
class EpisodicState:
    """Exact evidence owned by one session; no model parameters live here."""

    session_id: str
    keys: list[np.ndarray] = field(default_factory=list)
    values: list[np.ndarray] = field(default_factory=list)
    item_ids: list[int] = field(default_factory=list)
    next_item_id: int = 0
    index_build_seconds_total: float = 0.0
    flat_search_seconds_total: float = 0.0
    indexed_search_seconds_total: float = 0.0
    verification_seconds_total: float = 0.0
    write_seconds_total: float = 0.0
    index_build_count: int = 0
    _cached_index: ExactEpisodicIndex | None = field(default=None, init=False, repr=False)

    def __len__(self) -> int:
        return len(self.item_ids)

    def reset(self) -> None:
        self.keys.clear()
        self.values.clear()
        self.item_ids.clear()
        self.next_item_id = 0
        self.index_build_seconds_total = 0.0
        self.flat_search_seconds_total = 0.0
        self.indexed_search_seconds_total = 0.0
        self.verification_seconds_total = 0.0
        self.write_seconds_total = 0.0
        self.index_build_count = 0
        self._cached_index = None

    def payload_bytes(self) -> int:
        """Exact tensor/ID payload bytes, excluding Python container overhead."""
        key_bytes = sum(int(key.nbytes) for key in self.keys)
        value_bytes = sum(int(value.nbytes) for value in self.values)
        id_bytes = len(self.item_ids) * np.dtype(np.int64).itemsize
        return key_bytes + value_bytes + id_bytes

    def write(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        started = time.perf_counter()
        try:
            if keys.ndim != 2 or values.ndim != 2 or keys.shape[0] != values.shape[0]:
                raise ValueError("keys/values must be aligned [items, dim] matrices")
            k = keys.detach().float().cpu().numpy().astype(np.float32, copy=True)
            v = values.detach().float().cpu().numpy().astype(np.float32, copy=True)
            if not np.isfinite(k).all() or not np.isfinite(v).all():
                raise ValueError("episodic writes must be finite")
            for key, value in zip(k, v):
                self.keys.append(key)
                self.values.append(value)
                self.item_ids.append(self.next_item_id)
                self.next_item_id += 1
            self._cached_index = None
        finally:
            self.write_seconds_total += time.perf_counter() - started

    def _arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.keys:
            raise RuntimeError("cannot retrieve from empty episodic state")
        return (
            np.stack(self.keys, axis=0),
            np.stack(self.values, axis=0),
            np.asarray(self.item_ids, dtype=np.int64),
        )

    def index(self) -> ExactEpisodicIndex:
        if self._cached_index is None:
            started = time.perf_counter()
            keys, _, item_ids = self._arrays()
            self._cached_index = ExactEpisodicIndex(keys, item_ids, leaf_size=LEAF_SIZE)
            self.index_build_seconds_total += time.perf_counter() - started
            self.index_build_count += 1
        return self._cached_index

    def retrieve(
        self,
        query: torch.Tensor,
        *,
        mode: RetrievalMode,
        verify_indexed_exactness: bool = True,
    ) -> tuple[torch.Tensor, SearchResult, bool, float, float, float]:
        """Return value/result/exactness plus build/search/verification seconds.

        `search_seconds` measures the requested retrieval path only. For indexed
        mode, the exhaustive flat call used solely to assert exactness is kept in
        `verification_seconds` so correctness checking cannot masquerade as the
        indexed algorithm's own wall-clock cost.
        """
        if query.ndim != 1:
            raise ValueError("query must be one address vector")
        _, values, _ = self._arrays()
        q = query.detach().float().cpu().numpy().astype(np.float32, copy=False)
        build_before = self.index_build_seconds_total
        index = self.index()
        build_delta = self.index_build_seconds_total - build_before
        verification_elapsed = 0.0
        if mode == "flat":
            started = time.perf_counter()
            result = index.flat_search(q)
            search_elapsed = time.perf_counter() - started
            self.flat_search_seconds_total += search_elapsed
            exact_match = True
        elif mode == "indexed":
            started = time.perf_counter()
            result = index.indexed_search(q)
            search_elapsed = time.perf_counter() - started
            self.indexed_search_seconds_total += search_elapsed
            if verify_indexed_exactness:
                verify_started = time.perf_counter()
                flat = index.flat_search(q)
                verification_elapsed = time.perf_counter() - verify_started
                self.verification_seconds_total += verification_elapsed
                exact_match = result.item_id == flat.item_id and result.position == flat.position
                if not exact_match:
                    raise AssertionError(
                        f"indexed/flat mismatch in session {self.session_id}: "
                        f"indexed={result} flat={flat}"
                    )
            else:
                exact_match = True
        else:
            raise ValueError(f"unknown retrieval mode {mode!r}")
        value = torch.as_tensor(values[result.position], device=query.device, dtype=query.dtype)
        return (
            value,
            result,
            exact_match,
            build_delta,
            search_elapsed,
            verification_elapsed,
        )


class CHMV1EIEMLM(nn.Module):
    """Same local backbone plus exact, indexable persistent episodic evidence."""

    def __init__(self, *, address_dim: int = ADDRESS_DIM) -> None:
        super().__init__()
        cfg = transformer_25m_config()
        self.backbone = ResearchLM(cfg)
        self.address_dim = int(address_dim)
        self.query_address = nn.Linear(cfg.d_model, self.address_dim, bias=False)
        self.key_address = nn.Linear(cfg.d_model, self.address_dim, bias=False)
        # Starts nearly memory-off so paired initialization does not inject a
        # large untrained residual; the gate remains fully trainable.
        self.memory_gate_logit = nn.Parameter(torch.full((cfg.d_model,), -4.0))

    def query_for(self, representation: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.query_address(representation), dim=-1)

    def key_for(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.key_address(hidden), dim=-1)

    def addresses(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.query_for(hidden), self.key_for(hidden)

    def _integrate(self, hidden: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.memory_gate_logit).to(hidden.dtype)
        return hidden + gate * memory

    def forward_local(self, tokens: torch.Tensor) -> torch.Tensor:
        """Memory-disabled ablation on the EIEM checkpoint."""
        hidden = _hidden(self.backbone, tokens)
        return self.backbone.lm_head(hidden)

    def forward_flat_differentiable(self, tokens: torch.Tensor, *, temperature: float = 0.10) -> torch.Tensor:
        """Differentiable exhaustive reference with two causal retrieval hops.

        Only strictly earlier positions can contribute. This O(T^2) helper is
        used by zero-credit gradient/invariant tests; the scientific matched
        training session path lives in `chm_v1_small_lm_protocol.py` and only
        exposes prior chunks as episodic state. Each hop uses the same exact
        causal value bank, while hop 1 changes the learned query for hop 2.
        """
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        hidden = _hidden(self.backbone, tokens)
        keys = self.key_for(hidden)
        length = tokens.shape[1]
        allowed = torch.tril(
            torch.ones(length, length, device=tokens.device, dtype=torch.bool),
            diagonal=-1,
        )
        valid_query_rows = allowed.any(dim=-1).view(1, length, 1)
        query_state = hidden
        for _ in range(RETRIEVAL_HOPS):
            queries = self.query_for(query_state)
            score = torch.einsum("btd,bsd->bts", queries, keys) / temperature
            score = score.masked_fill(~allowed[None], torch.finfo(score.dtype).min)
            weights = torch.softmax(score.float(), dim=-1).to(hidden.dtype)
            # Token zero has no causal predecessor. Mask functionally rather than
            # mutating the softmax output in-place so autograd can reuse it safely.
            weights = weights * valid_query_rows.to(weights.dtype)
            memory = torch.matmul(weights, hidden)
            query_state = self._integrate(query_state, memory)
        return self.backbone.lm_head(query_state)

    @torch.no_grad()
    def forward_session_chunk(
        self,
        tokens: torch.Tensor,
        states: list[EpisodicState],
        *,
        mode: RetrievalMode,
        update_memory: bool = True,
        verify_indexed_exactness: bool = True,
    ) -> tuple[torch.Tensor, RetrievalStats]:
        """Inference/eval path across <=512-token local chunks.

        Retrieval sees only states that existed before this call. Each token can
        make exactly two sequential reads from that same prior state: the first
        retrieved value changes the representation used to form the second
        query. Current hidden states are appended only after all logits are
        complete, so neither hop can read self/future evidence.
        """
        if tokens.shape[0] != len(states):
            raise ValueError("one EpisodicState is required per batch element")
        hidden = _hidden(self.backbone, tokens)
        keys = self.key_for(hidden)
        fused = hidden.clone()
        stats = RetrievalStats()

        for batch_index, state in enumerate(states):
            if len(state) == 0:
                continue
            memory_size = len(state)
            stats.state_payload_bytes = max(stats.state_payload_bytes, state.payload_bytes())
            for token_index in range(tokens.shape[1]):
                query_state = hidden[batch_index, token_index]
                for _ in range(RETRIEVAL_HOPS):
                    query = self.query_for(query_state)
                    (
                        value,
                        result,
                        exact_match,
                        build_seconds,
                        search_seconds,
                        verification_seconds,
                    ) = state.retrieve(
                        query,
                        mode=mode,
                        verify_indexed_exactness=verify_indexed_exactness,
                    )
                    stats.add(
                        result,
                        memory_size,
                        exact_match=exact_match,
                        index_build_seconds=build_seconds,
                        search_seconds=search_seconds,
                        verification_seconds=verification_seconds,
                    )
                    query_state = self._integrate(query_state, value)
                fused[batch_index, token_index] = query_state

        logits = self.backbone.lm_head(fused)
        if update_memory:
            for batch_index, state in enumerate(states):
                write_before = state.write_seconds_total
                state.write(keys[batch_index], hidden[batch_index])
                stats.write_seconds += state.write_seconds_total - write_before
                stats.state_payload_bytes = max(stats.state_payload_bytes, state.payload_bytes())
        return logits, stats


def build_matched_pair(seed: int, device: torch.device) -> tuple[CHMV1LocalLM, CHMV1EIEMLM]:
    if seed in SCIENTIFIC_SEEDS:
        raise RuntimeError("scientific seed refused by implementation-only builder")
    torch.manual_seed(seed)
    local = CHMV1LocalLM().to(device)
    torch.manual_seed(seed)
    eiem = CHMV1EIEMLM().to(device)
    local_state = local.backbone.state_dict()
    eiem_state = eiem.backbone.state_dict()
    if any(not torch.equal(local_state[name], eiem_state[name]) for name in local_state):
        raise RuntimeError("paired LOCAL/EIEM backbone initialization is not bit-identical")
    return local, eiem


def parameter_accounting() -> dict[str, float | int | bool]:
    torch.manual_seed(NON_SCIENTIFIC_SMOKE_SEED)
    local = CHMV1LocalLM()
    torch.manual_seed(NON_SCIENTIFIC_SMOKE_SEED)
    eiem = CHMV1EIEMLM()
    local_count = parameter_count(local)
    eiem_count = parameter_count(eiem)
    delta = (eiem_count - local_count) / local_count
    return {
        "local_trainable_parameters": local_count,
        "eiem_trainable_parameters": eiem_count,
        "eiem_extra_parameters": eiem_count - local_count,
        "delta_fraction": delta,
        "within_preregistered_one_percent": abs(delta) <= 0.01,
        "address_dim": ADDRESS_DIM,
        "local_window": LOCAL_WINDOW,
        "retrieval_hops": RETRIEVAL_HOPS,
        "token_budget_per_model": TOKEN_BUDGET,
    }


def parameter_digest(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()