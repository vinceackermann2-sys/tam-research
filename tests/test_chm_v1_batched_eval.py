from __future__ import annotations

import torch
import torch.nn.functional as F

from tam_research.chm_v1_batched_eval import (
    forward_session_chunk_batched_transport,
    retrieve_many_exact,
)
from tam_research.chm_v1_small_lm import (
    NON_SCIENTIFIC_SMOKE_SEED,
    CHMV1EIEMLM,
    EpisodicState,
    parameter_digest,
)


def _memory(seed: int, items: int = 19) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    keys = F.normalize(torch.randn(items, 32, generator=generator), dim=-1)
    values = torch.randn(items, 256, generator=generator)
    return keys, values


def _state(name: str, keys: torch.Tensor, values: torch.Tensor) -> EpisodicState:
    state = EpisodicState(name)
    state.write(keys, values)
    return state


def test_retrieve_many_exact_matches_scalar_reference_for_flat_and_indexed() -> None:
    keys, values = _memory(NON_SCIENTIFIC_SMOKE_SEED)
    generator = torch.Generator(device="cpu").manual_seed(NON_SCIENTIFIC_SMOKE_SEED + 1)
    queries = F.normalize(torch.randn(7, 32, generator=generator), dim=-1)

    for mode in ("flat", "indexed"):
        scalar_state = _state(f"scalar-{mode}", keys, values)
        batch_state = _state(f"batch-{mode}", keys, values)

        scalar_values: list[torch.Tensor] = []
        scalar_results = []
        scalar_matches: list[bool] = []
        for query in queries:
            value, result, exact_match, *_ = scalar_state.retrieve(
                query,
                mode=mode,
                verify_indexed_exactness=True,
            )
            scalar_values.append(value)
            scalar_results.append(result)
            scalar_matches.append(exact_match)

        batched = retrieve_many_exact(
            batch_state,
            queries,
            mode=mode,
            verify_indexed_exactness=True,
        )

        assert torch.equal(torch.stack(scalar_values), batched.values)
        assert [result.item_id for result in scalar_results] == [
            result.item_id for result in batched.results
        ]
        assert [result.position for result in scalar_results] == [
            result.position for result in batched.results
        ]
        assert [result.address_vector_reads for result in scalar_results] == [
            result.address_vector_reads for result in batched.results
        ]
        assert [result.directory_nodes_visited for result in scalar_results] == [
            result.directory_nodes_visited for result in batched.results
        ]
        assert tuple(scalar_matches) == batched.exact_matches


def test_batched_transport_is_bit_identical_to_scalar_chunk_reference() -> None:
    torch.manual_seed(NON_SCIENTIFIC_SMOKE_SEED)
    model = CHMV1EIEMLM().eval()
    keys, values = _memory(NON_SCIENTIFIC_SMOKE_SEED + 2, items=23)
    scalar_state = _state("scalar-forward", keys, values)
    batch_state = _state("batch-forward", keys, values)
    generator = torch.Generator(device="cpu").manual_seed(NON_SCIENTIFIC_SMOKE_SEED + 3)
    tokens = torch.randint(0, 50_257, (1, 9), generator=generator)

    before = parameter_digest(model)
    scalar_logits, scalar_stats = model.forward_session_chunk(
        tokens,
        [scalar_state],
        mode="indexed",
        update_memory=False,
        verify_indexed_exactness=True,
    )
    batched_logits, batched_stats = forward_session_chunk_batched_transport(
        model,
        tokens,
        [batch_state],
        mode="indexed",
        update_memory=False,
        verify_indexed_exactness=True,
    )
    after = parameter_digest(model)

    assert torch.equal(scalar_logits, batched_logits)
    assert before == after
    assert scalar_stats.calls == batched_stats.calls == 18
    assert scalar_stats.exact_matches == batched_stats.exact_matches == 18
    assert scalar_stats.address_vector_reads == batched_stats.address_vector_reads
    assert scalar_stats.flat_address_vector_reads == batched_stats.flat_address_vector_reads
    assert scalar_stats.directory_nodes_visited == batched_stats.directory_nodes_visited
    assert scalar_stats.state_payload_bytes == batched_stats.state_payload_bytes
    # Timings are intentionally not expected to be numerically identical.


def test_batched_transport_preserves_post_logit_write_boundary() -> None:
    torch.manual_seed(NON_SCIENTIFIC_SMOKE_SEED)
    model = CHMV1EIEMLM().eval()
    keys, values = _memory(NON_SCIENTIFIC_SMOKE_SEED + 4, items=5)
    scalar_state = _state("scalar-write", keys, values)
    batch_state = _state("batch-write", keys, values)
    generator = torch.Generator(device="cpu").manual_seed(NON_SCIENTIFIC_SMOKE_SEED + 5)
    tokens = torch.randint(0, 50_257, (1, 4), generator=generator)

    scalar_logits, scalar_stats = model.forward_session_chunk(
        tokens,
        [scalar_state],
        mode="flat",
        update_memory=True,
    )
    batched_logits, batched_stats = forward_session_chunk_batched_transport(
        model,
        tokens,
        [batch_state],
        mode="flat",
        update_memory=True,
    )

    assert torch.equal(scalar_logits, batched_logits)
    assert len(scalar_state) == len(batch_state) == 9
    assert scalar_stats.calls == batched_stats.calls == 8
    assert scalar_stats.flat_address_vector_reads == batched_stats.flat_address_vector_reads
    assert scalar_stats.address_vector_reads == batched_stats.address_vector_reads
    for scalar_key, batch_key in zip(scalar_state.keys, batch_state.keys):
        assert (scalar_key == batch_key).all()
    for scalar_value, batch_value in zip(scalar_state.values, batch_state.values):
        assert (scalar_value == batch_value).all()
