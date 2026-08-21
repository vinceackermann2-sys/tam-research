import torch

from tam_research.models import ModelConfig, ResearchLM, parameter_count


def test_tamv3_is_nearly_exact_parameter_match():
    transformer = parameter_count(ResearchLM(ModelConfig(architecture="transformer")))
    tamv3 = parameter_count(ResearchLM(ModelConfig(architecture="tamv3")))
    tamv3_fixed = parameter_count(ResearchLM(ModelConfig(architecture="tamv3_fixed")))
    assert tamv3 == tamv3_fixed
    assert abs(tamv3 - transformer) / transformer < 0.0001


def test_tamv3_forward_causality_and_fixed_gate():
    torch.manual_seed(11)
    for arch in ("tamv3", "tamv3_fixed"):
        cfg = ModelConfig(
            vocab_size=257,
            d_model=64,
            n_layers=2,
            n_heads=4,
            max_seq_len=32,
            tamv2_state_size=16,
            tamv3_attn_inner=52,
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

        stats = model.router_stats()["mean"]
        assert abs(stats["memory"]) < 1e-8
        if arch == "tamv3_fixed":
            assert abs(stats["attention"] - 0.5) < 1e-6
            assert abs(stats["world"] - 0.5) < 1e-6
