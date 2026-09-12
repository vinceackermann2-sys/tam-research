from __future__ import annotations

import inspect
import json

import numpy as np
import pytest
import torch

from tam_research.chm_v1_contiguous_state_prototype import (
    SCIENTIFIC_SEEDS,
    SYNTHETIC_PROTOTYPE_SEED,
    ContiguousEpisodicStatePrototype,
    benchmark_prototype_once,
    prototype_accounting,
)
from tam_research.chm_v1_small_lm import EpisodicState


def _assert_same_lookup(
    current: EpisodicState,
    prototype: ContiguousEpisodicStatePrototype,
    query: torch.Tensor,
    *,
    mode: str,
) -> None:
    current_value, current_result, current_exact, *_ = current.retrieve(
        query,
        mode=mode,  # type: ignore[arg-type]
        verify_indexed_exactness=True,
    )
    prototype_value, prototype_result, prototype_exact, *_ = prototype.retrieve(
        query,
        mode=mode,  # type: ignore[arg-type]
        verify_indexed_exactness=True,
    )
    assert prototype_exact is current_exact is True
    assert prototype_result.item_id == current_result.item_id
    assert prototype_result.position == current_result.position
    assert prototype_result.squared_distance == current_result.squared_distance
    assert prototype_result.address_vector_reads == current_result.address_vector_reads
    assert prototype_result.directory_nodes_visited == current_result.directory_nodes_visited
    assert torch.equal(prototype_value, current_value)


def test_prototype_preserves_exact_ties_and_returned_values() -> None:
    keys = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
        dtype=torch.float32,
    )
    values = torch.tensor(
        [[10.0, 11.0], [20.0, 21.0], [30.0, 31.0], [40.0, 41.0]],
        dtype=torch.float32,
    )
    current = EpisodicState("current")
    prototype = ContiguousEpisodicStatePrototype("prototype")
    current.write(keys, values)
    prototype.write(keys, values)

    query = torch.tensor([1.0, 0.0], dtype=torch.float32)
    for mode in ("flat", "indexed"):
        _assert_same_lookup(current, prototype, query, mode=mode)
        value, result, *_ = prototype.retrieve(
            query,
            mode=mode,  # type: ignore[arg-type]
            verify_indexed_exactness=True,
        )
        assert result.item_id == 0
        assert result.position == 0
        assert torch.equal(value, values[0])


def test_prototype_matches_current_state_across_incremental_writes() -> None:
    generator = torch.Generator(device="cpu").manual_seed(SYNTHETIC_PROTOTYPE_SEED)
    keys = torch.nn.functional.normalize(
        torch.randn(48, 8, generator=generator), dim=-1
    ).float()
    values = torch.randn(48, 12, generator=generator).float()

    current = EpisodicState("current")
    prototype = ContiguousEpisodicStatePrototype("prototype")
    for start, end in ((0, 11), (11, 29), (29, 48)):
        current.write(keys[start:end], values[start:end])
        prototype.write(keys[start:end], values[start:end])
        assert len(prototype) == len(current) == end
        assert prototype.next_item_id == current.next_item_id == end
        assert prototype.item_ids.tolist() == current.item_ids
        assert prototype.payload_bytes() == current.payload_bytes()

    queries = [keys[i] for i in (0, 7, 10, 11, 28, 47)]
    queries.extend(
        torch.nn.functional.normalize(
            torch.randn(10, 8, generator=generator), dim=-1
        ).float()
    )
    for query in queries:
        _assert_same_lookup(current, prototype, query, mode="flat")
        _assert_same_lookup(current, prototype, query, mode="indexed")


def test_warm_retrieval_uses_persistent_contiguous_storage() -> None:
    generator = torch.Generator(device="cpu").manual_seed(SYNTHETIC_PROTOTYPE_SEED + 1)
    keys = torch.nn.functional.normalize(
        torch.randn(64, 16, generator=generator), dim=-1
    ).float()
    values = torch.randn(64, 24, generator=generator).float()
    state = ContiguousEpisodicStatePrototype("persistent")
    state.write(keys[:32], values[:32])
    first_keys_identity = id(state._keys)
    state.write(keys[32:], values[32:])
    assert id(state._keys) != first_keys_identity
    assert state.keys.flags.c_contiguous
    assert state.values.flags.c_contiguous
    assert state.item_ids.flags.c_contiguous

    state.index()
    assert state.index_build_count == 1
    cached = state._cached_index
    before_keys = state._keys
    before_values = state._values
    before_ids = state._item_ids

    for position in (0, 17, 63):
        state.retrieve(
            keys[position],
            mode="indexed",
            verify_indexed_exactness=False,
        )
        assert state._cached_index is cached
        assert state._keys is before_keys
        assert state._values is before_values
        assert state._item_ids is before_ids
        assert state.index_build_count == 1

    accounting = prototype_accounting(state)
    assert accounting["warm_retrieve_np_stack_calls"] == 0
    assert accounting["warm_retrieve_full_state_materializations"] == 0
    assert accounting["items"] == 64
    assert accounting["payload_bytes"] == state.payload_bytes()

    retrieve_source = inspect.getsource(ContiguousEpisodicStatePrototype.retrieve)
    assert "np.stack" not in retrieve_source
    assert "np.concatenate" not in retrieve_source


def test_reset_isolation_finite_guards_and_payload_accounting() -> None:
    a = ContiguousEpisodicStatePrototype("a")
    b = ContiguousEpisodicStatePrototype("b")
    keys = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    values = torch.tensor([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]], dtype=torch.float32)
    a.write(keys, values)

    assert len(a) == 2
    assert len(b) == 0
    assert a.payload_bytes() == 2 * 2 * 4 + 2 * 3 * 4 + 2 * 8
    assert a.item_ids.tolist() == [0, 1]
    assert a.next_item_id == 2

    with pytest.raises(RuntimeError, match="empty episodic state"):
        b.retrieve(torch.tensor([1.0, 2.0]), mode="indexed")

    bad_keys = keys.clone()
    bad_keys[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        b.write(bad_keys, values)
    assert len(b) == 0

    a.reset()
    assert len(a) == 0
    assert a.next_item_id == 0
    assert a.payload_bytes() == 0
    assert a.index_build_count == 0
    assert a._cached_index is None


def test_width_changes_are_rejected_without_corrupting_state() -> None:
    state = ContiguousEpisodicStatePrototype("width")
    state.write(torch.zeros((2, 4)), torch.zeros((2, 5)))
    before_ids = state.item_ids.copy()
    before_payload = state.payload_bytes()

    with pytest.raises(ValueError, match="key width"):
        state.write(torch.zeros((1, 3)), torch.zeros((1, 5)))
    with pytest.raises(ValueError, match="value width"):
        state.write(torch.zeros((1, 4)), torch.zeros((1, 6)))

    assert np.array_equal(state.item_ids, before_ids)
    assert state.payload_bytes() == before_payload
    assert state.next_item_id == 2


def test_benchmark_refuses_scientific_seeds_and_is_json_serializable() -> None:
    for seed in SCIENTIFIC_SEEDS:
        with pytest.raises(RuntimeError, match="scientific seed refused"):
            benchmark_prototype_once(
                session_length=8,
                query_count=2,
                key_width=4,
                value_width=6,
                seed=seed,
            )

    result = benchmark_prototype_once(
        session_length=32,
        query_count=4,
        key_width=8,
        value_width=12,
        seed=SYNTHETIC_PROTOTYPE_SEED,
    )
    assert result["measurement_kind"] == "engineering_only_cpu"
    assert result["scientific_credit"] is False
    assert result["paid_compute"] is False
    assert result["modal_trigger"] is False
    assert result["accounting"]["warm_retrieve_np_stack_calls"] == 0
    assert result["accounting"]["warm_retrieve_full_state_materializations"] == 0
    assert result["indexed_address_vector_reads"] > 0
    assert result["directory_nodes_visited"] > 0
    assert isinstance(result["elapsed_ns"], int)
    assert result["elapsed_ns"] >= 0
    json.dumps(result, sort_keys=True)


def test_prototype_has_no_cuda_or_modal_execution_path() -> None:
    import tam_research.chm_v1_contiguous_state_prototype as module

    source = inspect.getsource(module).lower()
    assert "torch.cuda" not in source
    assert ".cuda(" not in source
    assert "modal." not in source
    assert "modal run" not in source
