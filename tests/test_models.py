import torch

from tam_research.models import ModelConfig, ResearchLM, diagonal_affine_scan, parameter_count


TAMV2_VARIANTS = ("tamv2", "tamv2_nomem", "tamv2_noworld", "tamv2_fixed")
ALL_ARCHITECTURES = ("transformer", "tam", *TAMV2_VARIANTS)


def sequential_scan(a, b):
    state = torch.zeros_like(a[:, 0])
    states = []
    for t in range(a.size(1)):
        state = a[:, t] * state + b[:, t]
        states.append(state)
    return torch.stack(states, dim=1)


def test_parallel_scan_matches_sequential():
    torch.manual_seed(1)
    a = torch.sigmoid(torch.randn(2, 17, 5))
    b = torch.randn(2, 17, 5)
    torch.testing.assert_close(diagonal_affine_scan(a, b), sequential_scan(a, b), rtol=1e-5, atol=1e-6)


def test_parameter_matching_25m():
    counts = {arch: parameter_count(ResearchLM(ModelConfig(architecture=arch))) for arch in ALL_ARCHITECTURES}
    assert min(counts.values()) > 24_000_000
    assert max(counts.values()) < 26_000_000
    assert (max(counts.values()) - min(counts.values())) / counts["transformer"] < 0.01
    for arch in TAMV2_VARIANTS:
        assert counts[arch] == counts["tamv2"]


def test_forward_shapes_and_causality():
    torch.manual_seed(2)
    for arch in ALL_ARCHITECTURES:
        cfg = ModelConfig(
            vocab_size=257,
            d_model=64,
            n_layers=2,
            n_heads=4,
            max_seq_len=32,
            tamv2_branch_inner=28,
            tamv2_state_size=16,
            architecture=arch,
        )
        model = ResearchLM(cfg).eval()
        x = torch.randint(0, cfg.vocab_size, (2, 12))
        y = model(x)
        assert y.shape == (2, 12, cfg.vocab_size)
        x2 = x.clone()
        x2[:, 8:] = torch.randint(0, cfg.vocab_size, x2[:, 8:].shape)
        y2 = model(x2)
        torch.testing.assert_close(y[:, :8], y2[:, :8], rtol=1e-5, atol=1e-5)


def test_tamv2_ablation_routes():
    torch.manual_seed(3)
    expected_zero = {
        "tamv2_nomem": "memory",
        "tamv2_noworld": "world",
    }
    for arch, zero_key in expected_zero.items():
        cfg = ModelConfig(
            vocab_size=257,
            d_model=64,
            n_layers=1,
            n_heads=4,
            max_seq_len=16,
            tamv2_branch_inner=28,
            tamv2_state_size=16,
            architecture=arch,
        )
        model = ResearchLM(cfg).eval()
        model(torch.randint(0, cfg.vocab_size, (2, 10)))
        stats = model.router_stats()["mean"]
        assert abs(stats[zero_key]) < 1e-7

    cfg = ModelConfig(
        vocab_size=257,
        d_model=64,
        n_layers=1,
        n_heads=4,
        max_seq_len=16,
        tamv2_branch_inner=28,
        tamv2_state_size=16,
        architecture="tamv2_fixed",
    )
    model = ResearchLM(cfg).eval()
    model(torch.randint(0, cfg.vocab_size, (2, 10)))
    stats = model.router_stats()["mean"]
    for value in stats.values():
        assert abs(value - 1.0 / 3.0) < 1e-3
