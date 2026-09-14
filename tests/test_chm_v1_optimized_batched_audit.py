from __future__ import annotations

import ast
import inspect

import pytest
import torch
import torch.nn.functional as F

import tam_research.chm_v1_optimized_batched_audit as audit
from tam_research.chm_v1_batched_eval import (
    forward_session_chunk_batched_transport,
    retrieve_many_exact,
)
from tam_research.chm_v1_clipped_bound_prototype import ClippedBoundPrototype
from tam_research.chm_v1_optimized_replication import OptimizedEpisodicState
from tam_research.chm_v1_small_lm import (
    ADDRESS_DIM,
    CHMV1EIEMLM,
    EpisodicState,
    parameter_digest,
)


def _memory(seed: int, items: int = 31) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    keys = F.normalize(
        torch.randn(items, ADDRESS_DIM, generator=generator),
        dim=-1,
    )
    values = torch.randn(items, 256, generator=generator)
    return keys, values


def _queries(seed: int, count: int = 9) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return F.normalize(
        torch.randn(count, ADDRESS_DIM, generator=generator),
        dim=-1,
    )


def _optimized_state(
    name: str,
    keys: torch.Tensor,
    values: torch.Tensor,
) -> OptimizedEpisodicState:
    state = OptimizedEpisodicState(name)
    state.write(keys, values)
    return state


def _base_state(name: str, keys: torch.Tensor, values: torch.Tensor) -> EpisodicState:
    state = EpisodicState(name)
    state.write(keys, values)
    return state


def test_generic_batched_indexed_path_bypasses_optimized_index(monkeypatch) -> None:
    keys, values = _memory(969_010)
    queries = _queries(969_011)
    state = _optimized_state("generic-bypass", keys, values)

    index_calls = 0
    original_index = state.index

    def counted_index():
        nonlocal index_calls
        index_calls += 1
        return original_index()

    def forbidden_optimized_index():
        raise AssertionError("generic batched helper unexpectedly used optimized_index()")

    monkeypatch.setattr(state, "index", counted_index)
    monkeypatch.setattr(state, "optimized_index", forbidden_optimized_index)

    batch = retrieve_many_exact(
        state,
        queries,
        mode="indexed",
        verify_indexed_exactness=True,
    )

    assert len(batch.results) == len(queries)
    assert index_calls == 1
    assert state._cached_index is not None
    assert state._cached_optimized_index is None


def test_corrected_batched_path_exercises_clipped_indexed_search(monkeypatch) -> None:
    keys, values = _memory(969_020)
    queries = _queries(969_021, count=7)
    state = _optimized_state("optimized-counted", keys, values)

    calls = 0
    original = ClippedBoundPrototype.indexed_search

    def counted(self, query):
        nonlocal calls
        calls += 1
        return original(self, query)

    monkeypatch.setattr(ClippedBoundPrototype, "indexed_search", counted)

    batch = audit.retrieve_many_optimized_exact(
        state,
        queries,
        verify_indexed_exactness=True,
    )

    assert calls == len(queries)
    assert state._cached_optimized_index is not None
    assert all(batch.exact_matches)


def test_corrected_batched_path_matches_frozen_index_full_search_results() -> None:
    keys, values = _memory(969_030, items=64)
    queries = _queries(969_031, count=13)

    frozen_state = _base_state("frozen-indexed", keys, values)
    optimized_state = _optimized_state("optimized-indexed", keys, values)
    flat_state = _base_state("flat-reference", keys, values)

    frozen = retrieve_many_exact(
        frozen_state,
        queries,
        mode="indexed",
        verify_indexed_exactness=True,
    )
    optimized = audit.retrieve_many_optimized_exact(
        optimized_state,
        queries,
        verify_indexed_exactness=True,
    )
    flat = retrieve_many_exact(
        flat_state,
        queries,
        mode="flat",
        verify_indexed_exactness=True,
    )

    assert optimized.results == frozen.results
    assert torch.equal(optimized.values, frozen.values)
    assert all(optimized.exact_matches)
    assert [
        (result.item_id, result.position) for result in optimized.results
    ] == [
        (result.item_id, result.position) for result in flat.results
    ]


def test_corrected_batched_path_preserves_duplicate_and_equidistant_ties() -> None:
    first = torch.zeros(ADDRESS_DIM)
    first[0] = 1.0
    second = torch.zeros(ADDRESS_DIM)
    second[1] = 1.0
    keys = torch.stack([first, first.clone(), second, first.clone()])
    values = torch.arange(4 * 256, dtype=torch.float32).reshape(4, 256)
    queries = torch.stack([first, F.normalize(first + second, dim=0)])

    frozen_state = _base_state("tie-frozen", keys, values)
    optimized_state = _optimized_state("tie-optimized", keys, values)
    flat_state = _base_state("tie-flat", keys, values)

    frozen = retrieve_many_exact(
        frozen_state,
        queries,
        mode="indexed",
        verify_indexed_exactness=True,
    )
    optimized = audit.retrieve_many_optimized_exact(
        optimized_state,
        queries,
        verify_indexed_exactness=True,
    )
    flat = retrieve_many_exact(
        flat_state,
        queries,
        mode="flat",
    )

    assert optimized.results == frozen.results
    assert [
        (result.item_id, result.position) for result in optimized.results
    ] == [
        (result.item_id, result.position) for result in flat.results
    ]
    assert optimized.results[0].item_id == 0
    assert optimized.results[0].position == 0


def test_corrected_two_hop_batched_logits_match_flat_exact_retrieval() -> None:
    torch.manual_seed(969_040)
    model = CHMV1EIEMLM().eval()
    keys, values = _memory(969_041, items=23)
    flat_state = _base_state("two-hop-flat", keys, values)
    optimized_state = _optimized_state("two-hop-optimized", keys, values)
    generator = torch.Generator(device="cpu").manual_seed(969_042)
    tokens = torch.randint(0, 50_257, (1, 9), generator=generator)

    before = parameter_digest(model)
    flat_logits, flat_stats = forward_session_chunk_batched_transport(
        model,
        tokens,
        [flat_state],
        mode="flat",
        update_memory=False,
        verify_indexed_exactness=True,
    )
    optimized_logits, optimized_stats = (
        audit.forward_session_chunk_optimized_batched_transport(
            model,
            tokens,
            [optimized_state],
            update_memory=False,
            verify_indexed_exactness=True,
        )
    )
    after = parameter_digest(model)

    assert torch.equal(flat_logits, optimized_logits)
    assert before == after
    assert flat_stats.calls == optimized_stats.calls == 18
    assert optimized_stats.exact_matches == optimized_stats.calls
    assert optimized_stats.flat_address_vector_reads == flat_stats.flat_address_vector_reads


def test_corrected_transport_preserves_post_logit_write_boundary() -> None:
    torch.manual_seed(969_050)
    model = CHMV1EIEMLM().eval()
    keys, values = _memory(969_051, items=5)
    flat_state = _base_state("write-flat", keys, values)
    optimized_state = _optimized_state("write-optimized", keys, values)
    generator = torch.Generator(device="cpu").manual_seed(969_052)
    tokens = torch.randint(0, 50_257, (1, 4), generator=generator)

    flat_logits, flat_stats = forward_session_chunk_batched_transport(
        model,
        tokens,
        [flat_state],
        mode="flat",
        update_memory=True,
    )
    optimized_logits, optimized_stats = (
        audit.forward_session_chunk_optimized_batched_transport(
            model,
            tokens,
            [optimized_state],
            update_memory=True,
        )
    )

    assert torch.equal(flat_logits, optimized_logits)
    assert len(flat_state) == len(optimized_state) == 9
    assert flat_stats.calls == optimized_stats.calls == 8
    for flat_key, optimized_key in zip(flat_state.keys, optimized_state.keys):
        assert (flat_key == optimized_key).all()
    for flat_value, optimized_value in zip(flat_state.values, optimized_state.values):
        assert (flat_value == optimized_value).all()
    assert optimized_state._cached_index is None
    assert optimized_state._cached_optimized_index is None


def test_diagnostic_seed_guard_blocks_all_scientific_seed_families() -> None:
    for seed in audit.BLOCKED_SCIENTIFIC_SEEDS:
        with pytest.raises(RuntimeError, match="scientific seed refused"):
            audit.validate_diagnostic_seed(seed)

    for seed in audit.DIAGNOSTIC_SEEDS:
        assert audit.validate_diagnostic_seed(seed) == seed


def _dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def test_audit_module_exposes_no_paid_compute_or_training_path() -> None:
    source = inspect.getsource(audit)
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
    assert all(not name.endswith(".backward") and name != "backward" for name in called_names)
    assert all("optimizer.step" not in name for name in called_names)
    assert all("train_one" not in name for name in called_names)
