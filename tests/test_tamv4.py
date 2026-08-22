import torch

from tam_research.models import ModelConfig, ResearchLM, parameter_count as base_parameter_count
from tam_research.tamv4 import TAMV4Config, TAMV4LM, parameter_count


def tiny_cfg() -> TAMV4Config:
    return TAMV4Config(
        vocab_size=257,
        d_model=64,
        n_layers=2,
        n_heads=4,
        max_local_positions=64,
        ff_mult=4,
        state_size=16,
        attention_inner=52,
        attention_window=16,
    )


def test_tamv4_is_near_parameter_matched_at_25m():
    transformer = base_parameter_count(ResearchLM(ModelConfig(architecture="transformer")))
    tamv4 = parameter_count(TAMV4LM())
    assert abs(tamv4 - transformer) / transformer < 0.001


def test_tamv4_forward_is_causal_inside_chunk():
    torch.manual_seed(7)
    cfg = tiny_cfg()
    model = TAMV4LM(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (2, 12))
    y, _ = model(x)
    x2 = x.clone()
    x2[:, 8:] = torch.randint(0, cfg.vocab_size, x2[:, 8:].shape)
    y2, _ = model(x2)
    torch.testing.assert_close(y[:, :8], y2[:, :8], rtol=1e-5, atol=1e-5)


def test_tamv4_state_carries_across_chunks_and_can_reset():
    torch.manual_seed(8)
    cfg = tiny_cfg()
    model = TAMV4LM(cfg).eval()
    first = torch.randint(0, cfg.vocab_size, (1, 10))
    second = torch.randint(0, cfg.vocab_size, (1, 10))

    _, state = model(first)
    carried, _ = model(second, state)
    reset, _ = model(second, None)
    assert not torch.allclose(carried, reset)


def test_forward_stream_matches_manual_chunking():
    torch.manual_seed(9)
    cfg = tiny_cfg()
    model = TAMV4LM(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (1, 21))

    streamed, streamed_state = model.forward_stream(x, chunk_size=7)
    state = None
    pieces = []
    for start in range(0, x.size(1), 7):
        logits, state = model(x[:, start : start + 7], state)
        pieces.append(logits)
    manual = torch.cat(pieces, dim=1)

    torch.testing.assert_close(streamed, manual)
    assert len(streamed_state.layers) == cfg.n_layers


def test_router_and_write_stats_are_logged():
    torch.manual_seed(10)
    cfg = tiny_cfg()
    model = TAMV4LM(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (2, 12))
    model(x)
    stats = model.router_stats()
    assert stats is not None
    assert 0.0 <= stats["mean"]["attention"] <= 1.0
    assert 0.0 <= stats["mean"]["world"] <= 1.0
    assert 0.0 <= stats["mean"]["write_rate"] <= 1.0
