from __future__ import annotations

import torch

from tam_research.cpw_binding_v2.task import (
    FILLER_COUNT,
    FILLER_START,
    KEY_COUNT,
    KEY_START,
    PAIR_COUNT,
    PAIR_TOKEN,
    QUERY_MARKER_POSITION,
    QUERY_POSITION,
    QUERY_TOKEN,
    SCORED_DELAYS,
    VALUE_COUNT,
    VALUE_DELAYS,
    VALUE_START,
    counterfactual_query,
    explicit_key_lookup,
    make_binding_batch,
    positional_keyblind_guess,
)


def _batch(*, seed: int = 1307, size: int = 64, delay=None):
    return make_binding_batch(
        batch_size=size,
        generator=torch.Generator().manual_seed(seed),
        delay=delay,
    )


def test_exactly_eight_value_slots_always_present() -> None:
    for seed in (1307, 1308, 1309):
        b = _batch(seed=seed)
        assert b.tokens.shape == (64, 320)
        assert b.query_position == 319
        assert b.tokens[:, QUERY_MARKER_POSITION].eq(QUERY_TOKEN).all()
        assert b.delays.unique().numel() == 4
        for offset in VALUE_DELAYS:
            pos = QUERY_POSITION - offset
            assert b.tokens[:, pos].ge(VALUE_START).all()
            assert b.tokens[:, pos].lt(VALUE_START + VALUE_COUNT).all()
            assert b.tokens[:, pos - 1].ge(KEY_START).all()
            assert b.tokens[:, pos - 1].lt(KEY_START + KEY_COUNT).all()
            assert b.tokens[:, pos + 1].eq(PAIR_TOKEN).all()

        for row in b.tokens:
            values = row[[QUERY_POSITION - d for d in VALUE_DELAYS]]
            assert values.unique().numel() == PAIR_COUNT


def test_explicit_query_key_lookup_is_perfect_at_all_distances() -> None:
    for distance in SCORED_DELAYS:
        b = _batch(seed=1311 + distance, size=64, delay=distance)
        assert torch.equal(explicit_key_lookup(b.tokens), b.targets)
        assert torch.equal(
            b.source_value_positions,
            torch.full((64,), QUERY_POSITION - distance),
        )
        for row, target in zip(b.tokens, b.targets):
            assert int((row == target).sum()) == 1
            assert not bool(row[-15:].eq(target).any())


def test_balanced_key_blind_positional_attack_only_hits_one_quarter() -> None:
    for seed in (1307, 1308, 1309, 1310):
        b = _batch(seed=seed, size=128)
        attack = positional_keyblind_guess(b.tokens)
        assert float((attack == b.targets).float().mean()) == 0.25
        assert torch.equal(explicit_key_lookup(b.tokens), b.targets)

        all_four = torch.stack(
            [b.tokens[:, QUERY_POSITION - d] for d in SCORED_DELAYS],
            dim=1,
        )
        assert torch.all(
            (all_four.ge(VALUE_START) & all_four.lt(VALUE_START + VALUE_COUNT))
        )
        assert torch.all((all_four == b.targets[:, None]).sum(dim=1) == 1)


def test_counterfactual_query_changes_answer_with_same_prefix() -> None:
    b = _batch(size=64)
    alternate = (64, 128, 256, 32)
    for current in SCORED_DELAYS:
        rows = torch.nonzero(b.delays == current).flatten()
        if len(rows) == 0:
            continue
        # Switch to another of the four candidates without touching the prefix.
        choice = alternate[SCORED_DELAYS.index(current)]
        sub = _batch(seed=1307 + current, size=8, delay=current)
        tokens2, target2 = counterfactual_query(sub, new_delay=choice)
        assert torch.equal(tokens2[:, :-1], sub.tokens[:, :-1])
        assert torch.all(tokens2[:, -1] != sub.tokens[:, -1])
        assert torch.all(target2 != sub.targets)
        assert torch.equal(explicit_key_lookup(tokens2), target2)
        assert torch.equal(
            positional_keyblind_guess(tokens2),
            positional_keyblind_guess(sub.tokens),
        )


def test_query_key_not_present_in_filler_and_generator_replays() -> None:
    a = _batch(seed=2291)
    b = _batch(seed=2291)
    assert torch.equal(a.tokens, b.tokens)
    assert torch.equal(a.targets, b.targets)
    assert torch.equal(a.slot_to_pair, b.slot_to_pair)

    allowed = torch.zeros(a.tokens.shape, dtype=torch.bool)
    for distance in VALUE_DELAYS:
        position = QUERY_POSITION - distance
        allowed[:, position-1:position+2] = True
    allowed[:, QUERY_MARKER_POSITION:] = True
    fillers = a.tokens[~allowed]
    assert fillers.ge(FILLER_START).all()
    assert fillers.lt(FILLER_START + FILLER_COUNT).all()
    assert torch.equal(explicit_key_lookup(a.tokens), a.targets)


def test_invalid_delay_is_rejected() -> None:
    import pytest
    with pytest.raises(ValueError):
        _batch(delay=24)
    with pytest.raises(ValueError):
        counterfactual_query(_batch(), new_delay=24)
