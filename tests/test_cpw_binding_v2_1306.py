"""CPU-only adversarial validity gates for issue #1306's binding-required task."""
from __future__ import annotations

import pytest
import torch

from tam_research.cpw_v4_memory.protocol import (
    DISTANCES,
    KEY_COUNT,
    KEY_START,
    PAIR_COUNT,
    PAIR_TOKEN,
    QUERY_TOKEN,
    SEQ_LEN,
    VALUE_COUNT,
    VALUE_START,
)
from tam_research.cpw_binding_v2.task import (
    counterfactual_anchor_queries,
    make_binding_required_batch,
)


def _source_oracle(tokens: torch.Tensor) -> torch.Tensor:
    """Oracle uses only tokens, parsing the source keys and their next values."""
    answers: list[int] = []
    for sample in tokens:
        bindings: dict[int, int] = {}
        for i in range(sample.numel() - 3):
            key = int(sample[i])
            value = int(sample[i + 1])
            sep = int(sample[i + 2])
            if (
                KEY_START <= key < KEY_START + KEY_COUNT
                and VALUE_START <= value < VALUE_START + VALUE_COUNT
                and sep == PAIR_TOKEN
            ):
                assert key not in bindings
                bindings[key] = value
        assert len(bindings) == PAIR_COUNT
        answers.append(bindings[int(sample[-1])])
    return torch.tensor(answers, dtype=torch.long)


@pytest.mark.parametrize("seed", [1306001, 1306002, 1306003])
@pytest.mark.parametrize("delay", [None, *DISTANCES])
def test_source_contains_four_simultaneous_candidates_and_oracle_wins(
    seed: int, delay: int | None,
) -> None:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    batch = make_binding_required_batch(
        batch_size=32, generator=generator, delay=delay
    )

    candidate_positions = [batch.query_position - d for d in DISTANCES]
    candidates = batch.tokens[:, candidate_positions]
    assert candidate_positions == [287, 255, 191, 63]
    assert torch.all((candidates >= VALUE_START) & (candidates < VALUE_START + VALUE_COUNT))
    assert torch.equal(candidates, batch.anchor_values)
    assert torch.all(batch.anchor_values.unique(dim=1).shape[1] == 4)
    assert torch.equal(_source_oracle(batch.tokens), batch.targets)
    assert torch.all(batch.tokens[:, -2] == QUERY_TOKEN)
    assert torch.all(batch.delays > 15)
    for row in range(batch.tokens.shape[0]):
        target = int(batch.targets[row])
        assert int((batch.tokens[row, :-1] == target).sum()) == 1
        assert target not in batch.tokens[row, -16:-1].tolist()
    if delay is not None:
        assert torch.all(batch.delays == delay)


@pytest.mark.parametrize("seed", [1306021, 1306022, 1306023])
def test_identical_source_four_query_counterfactuals_force_binding(seed: int) -> None:
    batch = make_binding_required_batch(
        batch_size=80,
        generator=torch.Generator(device="cpu").manual_seed(seed),
    )
    variants, targets = counterfactual_anchor_queries(batch)
    assert variants.shape == (80, 4, SEQ_LEN)
    assert targets.shape == (80, 4)
    assert torch.equal(variants[:, 0, :-1], batch.tokens[:, :-1])
    for j in range(4):
        assert torch.equal(variants[:, j, :-1], batch.tokens[:, :-1])
        assert torch.equal(variants[:, j, -1], batch.anchor_keys[:, j])
        assert torch.equal(targets[:, j], batch.anchor_values[:, j])
        assert torch.equal(_source_oracle(variants[:, j]), targets[:, j])
    assert torch.all(targets.unique(dim=1).shape[1] == 4)

    # Same source, no query key: each of four distinct targets is equally
    # likely. Even an optimal deterministic key-blind answer is <=25%.
    masked = variants.clone()
    masked[:, :, -1] = QUERY_TOKEN
    for j in range(1, 4):
        assert torch.equal(masked[:, j], masked[:, 0])
    key_blind_pick_first = targets[:, :1].expand_as(targets)
    assert int((key_blind_pick_first == targets).sum()) == len(batch.tokens)
    assert float((key_blind_pick_first == targets).float().mean()) == 0.25

    # Audit #1301's exact four-distance membership attack is now ambiguous:
    # all four offsets hold a valid (and distinct) VALUE in every example.
    candidate_slots = variants[:, :, [319 - d for d in DISTANCES]]
    value_mask = (candidate_slots >= VALUE_START) & (
        candidate_slots < VALUE_START + VALUE_COUNT
    )
    assert torch.all(value_mask.sum(dim=-1) == 4)


def test_reproducible_and_separate_from_frozen_v4_generator() -> None:
    a = make_binding_required_batch(
        batch_size=64,
        generator=torch.Generator(device="cpu").manual_seed(1306099),
    )
    b = make_binding_required_batch(
        batch_size=64,
        generator=torch.Generator(device="cpu").manual_seed(1306099),
    )
    c = make_binding_required_batch(
        batch_size=64,
        generator=torch.Generator(device="cpu").manual_seed(1306100),
    )
    assert torch.equal(a.tokens, b.tokens)
    assert torch.equal(a.targets, b.targets)
    assert not torch.equal(a.tokens, c.tokens)
    assert torch.all(a.keys.unique(dim=1).shape[1] == PAIR_COUNT)
    assert torch.all(a.values.unique(dim=1).shape[1] == PAIR_COUNT)
    assert torch.equal(a.anchor_keys, a.keys[:, :4])
    assert torch.equal(a.anchor_values, a.values[:, :4])


@pytest.mark.parametrize("bad_delay", [0, 15, 31, 33, 257])
def test_invalid_delay_rejected(bad_delay: int) -> None:
    with pytest.raises(ValueError):
        make_binding_required_batch(
            batch_size=1,
            generator=torch.Generator().manual_seed(1306101),
            delay=bad_delay,
        )
