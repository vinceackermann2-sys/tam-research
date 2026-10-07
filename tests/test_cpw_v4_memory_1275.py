from __future__ import annotations

import torch

from tam_research.cpw_v1.model import CPWV1Config
from tam_research.cpw_v3_sparse_world.model import SparseWorldCPWResearchLM
from tam_research.cpw_v4_memory.protocol import (
    CPW_PARAMETERS,
    DISTANCES,
    FILLER_START,
    FILLER_COUNT,
    KEY_START,
    KEY_COUNT,
    PAIR_COUNT,
    QUERY_TOKEN,
    SEQ_LEN,
    TRANSFORMER_PARAMETERS,
    VALUE_START,
    VALUE_COUNT,
)
from tam_research.cpw_v4_memory.task import make_associative_batch
from tam_research.cpw_v4_memory.train import build_memory_model, query_logits
from tam_research.models import ModelConfig, ResearchLM, parameter_count


def test_generator_is_deterministic_and_mappings_are_fresh() -> None:
    a = make_associative_batch(
        batch_size=16,
        generator=torch.Generator().manual_seed(1275),
    )
    b = make_associative_batch(
        batch_size=16,
        generator=torch.Generator().manual_seed(1275),
    )
    c = make_associative_batch(
        batch_size=16,
        generator=torch.Generator().manual_seed(1276),
    )
    torch.testing.assert_close(a.tokens, b.tokens, rtol=0, atol=0)
    torch.testing.assert_close(a.targets, b.targets, rtol=0, atol=0)
    assert not torch.equal(a.tokens, c.tokens)


def test_generator_enforces_unique_bindings_exact_delays_and_no_target_leakage() -> None:
    for delay in DISTANCES:
        batch = make_associative_batch(
            batch_size=24,
            generator=torch.Generator().manual_seed(20_000 + delay),
            delay=delay,
        )
        assert batch.tokens.shape == (24, SEQ_LEN)
        assert batch.query_position == SEQ_LEN - 1
        assert torch.all(batch.delays == delay)
        assert torch.all(batch.query_position - batch.source_value_positions == delay)
        assert delay > 15

        for row in range(24):
            keys = batch.keys[row].tolist()
            values = batch.values[row].tolist()
            assert len(set(keys)) == PAIR_COUNT
            assert len(set(values)) == PAIR_COUNT
            q_idx = int(batch.query_pair_indices[row])
            assert int(batch.targets[row]) == int(batch.values[row, q_idx])
            target = int(batch.targets[row])
            query_key = int(batch.keys[row, q_idx])
            tokens = batch.tokens[row]
            source = int(batch.source_value_positions[row])

            assert int((tokens == target).sum()) == 1
            assert int(tokens[source]) == target
            assert int((tokens == target)[source + 1 :].sum()) == 0
            assert int((tokens == query_key).sum()) == 2
            assert int(tokens[-2]) == QUERY_TOKEN
            assert int(tokens[-1]) == query_key

            # Vocabulary regions are disjoint by construction.
            filler = tokens[(tokens >= FILLER_START) & (tokens < FILLER_START + FILLER_COUNT)]
            assert not bool((filler == target).any())
            assert KEY_START <= query_key < KEY_START + KEY_COUNT
            assert VALUE_START <= target < VALUE_START + VALUE_COUNT


def test_mixed_training_batch_balances_delay_buckets() -> None:
    batch = make_associative_batch(
        batch_size=64,
        generator=torch.Generator().manual_seed(12750),
    )
    counts = {d: int((batch.delays == d).sum()) for d in DISTANCES}
    assert counts == {32: 16, 64: 16, 128: 16, 256: 16}


def test_frozen_parameter_counts_and_selected_world_layout() -> None:
    transformer = build_memory_model('transformer')
    sequence = build_memory_model('sequence_only')
    world = build_memory_model('world_last1')
    assert parameter_count(transformer) == TRANSFORMER_PARAMETERS
    assert parameter_count(sequence) == CPW_PARAMETERS
    assert parameter_count(world) == CPW_PARAMETERS
    assert tuple(world.world_layers) == (14,)
    assert tuple(sequence.world_layers) == ()

    for model in (sequence, world):
        names = [name.lower() for name, _ in model.named_modules()]
        assert not any('attention' in name for name in names)
        assert not any('router' in name for name in names)
        assert not any('memory' in name for name in names)


def test_sequence_only_cannot_see_beyond_15_but_world_last1_can() -> None:
    torch.manual_seed(12751)
    cfg = CPWV1Config(
        vocab_size=5_000,
        d_model=32,
        n_layers=15,
        max_seq_len=32,
        ff_mult=2,
        world_state_size=8,
        sequence_predictor_rank=12,
        memory_predictor_rank=8,
    )
    a = torch.randint(0, cfg.vocab_size, (1, 17))
    b = a.clone()
    b[:, 0] = (b[:, 0] + 1) % cfg.vocab_size

    sequence = SparseWorldCPWResearchLM('sequence_only', cfg).eval()
    with torch.no_grad():
        sa = sequence(a)[:, -1]
        sb = sequence(b)[:, -1]
    torch.testing.assert_close(sa, sb, rtol=0, atol=0)

    torch.manual_seed(12751)
    world = SparseWorldCPWResearchLM('world_last1', cfg).eval()
    with torch.no_grad():
        wa = world(a)[:, -1]
        wb = world(b)[:, -1]
    assert float((wa - wb).abs().max()) > 0.0


def test_reduced_models_are_causal_and_have_finite_gradients() -> None:
    torch.manual_seed(12752)
    cfg = CPWV1Config(
        vocab_size=5_000,
        d_model=32,
        n_layers=15,
        max_seq_len=32,
        ff_mult=2,
        world_state_size=8,
        sequence_predictor_rank=12,
        memory_predictor_rank=8,
    )
    for arm in ('sequence_only', 'world_last1'):
        model = SparseWorldCPWResearchLM(arm, cfg)
        tokens = torch.randint(0, cfg.vocab_size, (1, 24))
        logits = model(tokens)
        assert torch.isfinite(logits).all()
        logits[:, -1].float().square().mean().backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        assert grads
        assert all(torch.isfinite(g).all() for g in grads)

        model.eval()
        a = torch.randint(0, cfg.vocab_size, (1, 24))
        b = a.clone()
        b[:, 17:] = torch.randint(0, cfg.vocab_size, (1, 7))
        with torch.no_grad():
            la = model(a)
            lb = model(b)
        torch.testing.assert_close(la[:, :17], lb[:, :17], rtol=0, atol=2e-5)


def test_query_only_projection_matches_full_forward_exactly() -> None:
    torch.manual_seed(12753)
    transformer_cfg = ModelConfig(
        vocab_size=5_000,
        d_model=32,
        n_layers=2,
        n_heads=4,
        max_seq_len=32,
        ff_mult=2,
        architecture="transformer",
    )
    transformer = ResearchLM(transformer_cfg).eval()
    tokens = torch.randint(0, 5_000, (2, 20))
    with torch.no_grad():
        full = transformer(tokens)[:, -1]
        query = query_logits(transformer, tokens)
    torch.testing.assert_close(full, query, rtol=1e-6, atol=2e-7)

    cpw_cfg = CPWV1Config(
        vocab_size=5_000,
        d_model=32,
        n_layers=15,
        max_seq_len=32,
        ff_mult=2,
        world_state_size=8,
        sequence_predictor_rank=12,
        memory_predictor_rank=8,
    )
    for arm in ("sequence_only", "world_last1"):
        torch.manual_seed(12753)
        model = SparseWorldCPWResearchLM(arm, cpw_cfg).eval()
        with torch.no_grad():
            full = model(tokens)[:, -1]
            query = query_logits(model, tokens)
        torch.testing.assert_close(full, query, rtol=1e-6, atol=2e-7)
