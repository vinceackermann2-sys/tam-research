from __future__ import annotations

import ast
import inspect

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import tam_research.chm_v1_optimized_replication as replication
from tam_research.chm_v1_small_lm import ADDRESS_DIM, EpisodicState


def _payload(seed: int = replication.NON_SCIENTIFIC_SMOKE_SEED):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    keys = F.normalize(torch.randn(64, ADDRESS_DIM, generator=generator), dim=-1)
    values = torch.randn(64, 12, generator=generator)
    query = F.normalize(torch.randn(ADDRESS_DIM, generator=generator), dim=-1)
    return keys, values, query


def test_replication_preflight_freezes_issue_959_contract_without_gpu_authority() -> None:
    preflight = replication.replication_preflight()
    assert preflight["classification"] == "IMPLEMENTATION_PREFLIGHT_ONLY_NO_GPU_AUTHORITY"
    assert preflight["research_issue"] == 959
    assert preflight["replication_base_sha"] == replication.REPLICATION_BASE_SHA
    assert preflight["fresh_scientific_seeds"] == [19591, 19592, 19593]
    assert preflight["legacy_blocked_scientific_seeds"] == [8611, 8612, 8613]
    assert preflight["token_budget_per_model"] == 8_388_608
    assert preflight["retrieval_hops"] == 2
    assert preflight["optimized_index"] == "ClippedBoundPrototype"
    assert preflight["gpu_authorized"] is False
    assert preflight["scientific_seed_consumed"] is False


def test_seed_guard_permanently_refuses_legacy_and_requires_fresh_authority() -> None:
    for seed in replication.LEGACY_BLOCKED_SCIENTIFIC_SEEDS:
        with pytest.raises(RuntimeError, match="permanently blocked"):
            replication.validate_replication_seed(seed)
        with pytest.raises(RuntimeError, match="permanently blocked"):
            replication.validate_replication_seed(seed, paid_run_authorized=True)

    for seed in replication.FRESH_SCIENTIFIC_SEEDS:
        with pytest.raises(RuntimeError, match="separate paid/GPU run-control authorization"):
            replication.validate_replication_seed(seed)
        assert replication.validate_replication_seed(seed, paid_run_authorized=True) == seed

    assert (
        replication.validate_replication_seed(replication.NON_SCIENTIFIC_SMOKE_SEED)
        == replication.NON_SCIENTIFIC_SMOKE_SEED
    )


def test_optimized_state_matches_frozen_indexed_result_value_and_accounting() -> None:
    keys, values, query = _payload()
    frozen_state = EpisodicState("frozen")
    optimized_state = replication.OptimizedEpisodicState("optimized")
    frozen_state.write(keys, values)
    optimized_state.write(keys, values)

    frozen_value, frozen_result, frozen_match, *_ = frozen_state.retrieve(
        query,
        mode="indexed",
        verify_indexed_exactness=True,
    )
    optimized_value, optimized_result, optimized_match, build, search, verification = (
        optimized_state.retrieve(
            query,
            mode="indexed",
            verify_indexed_exactness=True,
        )
    )

    assert optimized_result == frozen_result
    assert torch.equal(optimized_value, frozen_value)
    assert optimized_match is True
    assert frozen_match is True
    assert build >= 0.0
    assert search >= 0.0
    assert verification >= 0.0
    assert optimized_state.optimized_sidecar_build_seconds_total >= 0.0
    assert optimized_state.optimized_wrapper_seconds_total >= 0.0


def test_flat_path_is_inherited_unchanged() -> None:
    keys, values, query = _payload(seed=959_002)
    base = EpisodicState("base-flat")
    optimized = replication.OptimizedEpisodicState("optimized-flat")
    base.write(keys, values)
    optimized.write(keys, values)

    base_value, base_result, base_match, *_ = base.retrieve(query, mode="flat")
    optimized_value, optimized_result, optimized_match, *_ = optimized.retrieve(
        query,
        mode="flat",
    )

    assert optimized_result == base_result
    assert torch.equal(optimized_value, base_value)
    assert optimized_match is True
    assert base_match is True
    assert optimized._cached_optimized_index is None


def test_optimized_sidecar_cache_tracks_frozen_index_and_invalidates_on_write_reset() -> None:
    keys, values, _query = _payload(seed=959_003)
    state = replication.OptimizedEpisodicState("cache")
    state.write(keys[:32], values[:32])

    first = state.optimized_index()
    assert first.frozen is state.index()
    assert state._cached_optimized_index is first

    state.write(keys[32:33], values[32:33])
    assert state._cached_index is None
    assert state._cached_optimized_index is None

    second = state.optimized_index()
    assert second is not first
    assert second.frozen is state.index()
    assert len(state) == 33

    state.reset()
    assert len(state) == 0
    assert state._cached_index is None
    assert state._cached_optimized_index is None
    assert state.optimized_sidecar_build_seconds_total == 0.0
    assert state.optimized_wrapper_seconds_total == 0.0


def test_optimized_state_exactly_matches_flat_answers_on_many_queries() -> None:
    keys, values, _query = _payload(seed=959_004)
    state = replication.OptimizedEpisodicState("many")
    state.write(keys, values)
    generator = torch.Generator(device="cpu").manual_seed(959_005)

    for _ in range(25):
        query = F.normalize(torch.randn(ADDRESS_DIM, generator=generator), dim=-1)
        _value, optimized, exact_match, *_ = state.retrieve(
            query,
            mode="indexed",
            verify_indexed_exactness=True,
        )
        q = query.detach().float().cpu().numpy().astype(np.float32, copy=False)
        flat = state.index().flat_search(q)
        assert exact_match is True
        assert optimized.item_id == flat.item_id
        assert optimized.position == flat.position


def _dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def test_replication_module_has_no_modal_launcher_or_training_execution_call() -> None:
    source = inspect.getsource(replication)
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
    assert all("train_one" not in name for name in called_names)
    assert all("optimizer.step" not in name for name in called_names)
