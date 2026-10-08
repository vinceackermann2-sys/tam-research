"""CPU-only, model-free falsification of the frozen CPW delayed-binding benchmark.

Issue #1301. This file changes no generator, model, trainer or scientific result.
"""
from __future__ import annotations

import pytest
import torch

from tam_research.cpw_v4_memory.protocol import (
    DISTANCES,
    KEY_START,
    KEY_COUNT,
    PAIR_COUNT,
    VALUE_START,
    VALUE_COUNT,
)
from tam_research.cpw_v4_memory.task import make_associative_batch


def _key_blind_position_decoder(tokens: torch.Tensor, query_position: int):
    """Read only four fixed offsets and VALUE vocabulary membership.

    It never reads the query-key token, any source key, or a key/value pair.
    """
    candidate_positions = torch.tensor(
        [query_position - d for d in DISTANCES], dtype=torch.long
    )
    candidates = tokens.index_select(1, candidate_positions)
    is_value = (candidates >= VALUE_START) & (
        candidates < VALUE_START + VALUE_COUNT
    )
    # Distinct candidate positions are separated by >=32, while the ONLY
    # block of value tokens spans 24 tokens (8 key/value/separator triples).
    guesses = candidates.masked_fill(~is_value, 0).sum(dim=1)
    return guesses, is_value, candidates, candidate_positions


@pytest.mark.parametrize("seed", [1301001, 1301002, 1301003])
@pytest.mark.parametrize("delay", [None, *DISTANCES])
def test_frozen_generator_has_perfect_key_independent_position_shortcut(
    seed: int, delay: int | None
) -> None:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    batch = make_associative_batch(
        batch_size=128, generator=generator, delay=delay
    )
    guesses, is_value, candidates, candidate_positions = (
        _key_blind_position_decoder(batch.tokens, batch.query_position)
    )

    assert candidate_positions.tolist() == [287, 255, 191, 63]
    assert torch.all(is_value.sum(dim=1) == 1)
    assert torch.equal(guesses, batch.targets)
    assert torch.equal(
        candidate_positions[is_value.long().argmax(dim=1)],
        batch.source_value_positions,
    )
    assert torch.all(batch.tokens[:, batch.query_position] >= KEY_START)
    assert torch.all(
        batch.tokens[:, batch.query_position] < KEY_START + KEY_COUNT
    )

    # Query another genuinely different key from the SAME source block.
    # The semantically correct answer changes, but the model-free decoder
    # sees the same fixed-offset value and remains completely unchanged.
    other_indices = (batch.query_pair_indices + 1) % PAIR_COUNT
    other_keys = batch.keys.gather(1, other_indices[:, None]).squeeze(1)
    other_values = batch.values.gather(1, other_indices[:, None]).squeeze(1)
    assert torch.all(other_keys != batch.tokens[:, batch.query_position])
    assert torch.all(other_values != batch.targets)
    alternate = batch.tokens.clone()
    alternate[:, batch.query_position] = other_keys
    alternate_guesses, alternate_mask, _, _ = _key_blind_position_decoder(
        alternate, batch.query_position
    )
    assert torch.equal(alternate_guesses, guesses)
    assert torch.equal(alternate_mask, is_value)
    assert torch.all(alternate_guesses != other_values)


def test_shortcut_is_independent_of_target_binding_and_unknown_distance() -> None:
    generator = torch.Generator(device="cpu").manual_seed(1301099)
    batch = make_associative_batch(batch_size=1024, generator=generator)
    guesses, mask, _, candidate_positions = _key_blind_position_decoder(
        batch.tokens, batch.query_position
    )
    inferred_distance = (
        batch.query_position - candidate_positions[mask.long().argmax(dim=1)]
    )
    assert torch.equal(inferred_distance, batch.delays)
    assert torch.equal(guesses, batch.targets)
    assert int((guesses == batch.targets).sum()) == len(batch.targets)
