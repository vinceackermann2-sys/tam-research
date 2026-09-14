from __future__ import annotations

import ast
import inspect

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import tam_research.chm_v1_optimized_batched_eval as optimized_batched
from tam_research.chm_v1_batched_eval import (
    forward_session_chunk_batched_transport,
    retrieve_many_exact,
)
from tam_research.chm_v1_clipped_bound_prototype import ClippedBoundPrototype
from tam_research.chm_v1_optimized_replication import OptimizedEpisodicState
from tam_research.chm_v1_small_lm import ADDRESS_DIM, CHMV1EIEMLM, EpisodicState, _hidden


def _payload(seed: int = 969_001, *, items: int = 64, value_width: int = 12):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    keys = F.normalize(torch.randn(items, ADDRESS_DIM, generator=generator), dim=-1)
    values = torch.randn(items, value_width, generator=generator)
    queries = F.normalize(torch.randn(8, ADDRESS_DIM, generator=generator), dim=-1)
    return keys, values, queries


def test_generic_batched_indexed_path_does_not_dispatch_to_optimized_index(monkeypatch) -> None:
    """Regression proof for the #969 integration discrepancy."""

    keys, values, queries = _payload()
    state = OptimizedEpisodicState("generic-bypass-proof")
    state.write(keys, values)

    def forbidden_optimized_index():
        raise AssertionError("generic batched helper unexpectedly called optimized_index")

    monkeypatch.setattr(state, "optimized_index", forbidden_optimized_index)
    batch = retrieve_many_exact(
        state,
        queries,
        mode="indexed",
        verify_indexed_exactness=True,
    )

    assert len(batch.results) == queries.shape[0]
    assert state._cached_index is not None
    assert state._cached_optimized_index is None


def test_corrected_batched_path_calls_clipped_index_and_preserves_full_indexed_accounting(
    monkeypatch,
) -> None:
    keys, values, queries = _payload(seed=969_002)
    state = OptimizedEpisodicState("corrected-dispatch")
    state.write(keys, values)

    calls = 0
    original = ClippedBoundPrototype.indexed_search

    def counted(self, query):
        nonlocal calls
        calls += 1
        return original(self, query)

    monkeypatch.setattr(ClippedBoundPrototype, "indexed_search", counted)
    batch = optimized_batched.retrieve_many_optimized_exact(
        state,
        queries,
        verify_indexed_exactness=True,
    )

    assert calls == queries.shape[0]
    assert state._cached_optimized_index is not None
    assert state._cached_optimized_index.frozen is state.index()
    assert all(batch.exact_matches)

    query_matrix = queries.detach().float().cpu().numpy().astype(np.float32, copy=False)
    frozen_results = tuple(state.index().indexed_search(query) for query in query_matrix)
    assert batch.results == frozen_results


def test_corrected_batched_path_preserves_duplicate_tie_semantics() -> None:
    state = OptimizedEpisodicState("duplicate-tie")
    key = F.normalize(torch.arange(1, ADDRESS_DIM + 1, dtype=torch.float32), dim=0)
    keys = torch.stack([key, key, -key])
    values = torch.arange(3 * 5, dtype=torch.float32).reshape(3, 5)
    state.write(keys, values)

    batch = optimized_batched.retrieve_many_optimized_exact(
        state,
        key.unsqueeze(0),
        verify_indexed_exactness=True,
    )
    result = batch.results[0]
    frozen = state.index().indexed_search(key.numpy().astype(np.float32, copy=False))
    flat = state.index().flat_search(key.numpy().astype(np.float32, copy=False))

    assert result == frozen
    assert result.item_id == flat.item_id == 0
    assert result.position == flat.position == 0
    assert torch.equal(batch.values[0], values[0])


def _prime_state_from_tokens(model: CHMV1EIEMLM, state: EpisodicState, tokens: torch.Tensor) -> None:
    with torch.no_grad():
        hidden = _hidden(model.backbone, tokens)
        keys = model.key_for(hidden)
    state.write(keys[0], hidden[0])


def test_corrected_two_hop_batched_logits_match_flat_exact_reference() -> None:
    torch.manual_seed(969_003)
    model = CHMV1EIEMLM().cpu().eval()
    generator = torch.Generator(device="cpu").manual_seed(969_004)
    prior = torch.randint(0, 128, (1, 16), generator=generator)
    query_tokens = torch.randint(0, 128, (1, 8), generator=generator)

    flat_state = EpisodicState("flat-reference")
    optimized_state = OptimizedEpisodicState("optimized-reference")
    _prime_state_from_tokens(model, flat_state, prior)
    _prime_state_from_tokens(model, optimized_state, prior)

    flat_logits, flat_stats = forward_session_chunk_batched_transport(
        model,
        query_tokens,
        [flat_state],
        mode="flat",
        update_memory=False,
        verify_indexed_exactness=False,
    )
    optimized_logits, optimized_stats = (
        optimized_batched.forward_session_chunk_optimized_batched_transport(
            model,
            query_tokens,
            [optimized_state],
            update_memory=False,
            verify_indexed_exactness=True,
        )
    )

    assert torch.allclose(flat_logits.float(), optimized_logits.float(), rtol=0.0, atol=1e-5)
    assert optimized_stats.calls == flat_stats.calls == 2 * query_tokens.numel()
    assert optimized_stats.exact_match_rate == 1.0
    assert optimized_state._cached_optimized_index is not None


def test_issue_969_seed_guard_refuses_all_predecessor_scientific_seeds() -> None:
    for seed in optimized_batched.BLOCKED_SCIENTIFIC_SEEDS:
        with pytest.raises(RuntimeError, match="scientific seed refused"):
            optimized_batched.validate_diagnostic_seed(seed)

    for seed in optimized_batched.DIAGNOSTIC_SEEDS:
        assert optimized_batched.validate_diagnostic_seed(seed) == seed


def _dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def test_issue_969_module_exposes_no_gpu_paid_or_training_execution_path() -> None:
    source = inspect.getsource(optimized_batched)
    tree = ast.parse(source)

    imported_roots: set[str] = set()
    called_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call):
            name = _dotted_name(node.func)
            if name:
                called_names.add(name)

    assert "modal" not in imported_roots
    assert all(not name.startswith("modal.") for name in called_names)
    assert all("cuda" not in name.lower() for name in called_names)
    assert all(not name.endswith(".backward") and name != "backward" for name in called_names)
    assert all("optimizer.step" not in name for name in called_names)
    assert all("train_one" not in name for name in called_names)
