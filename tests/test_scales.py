from tam_research.models import ResearchLM, parameter_count
from tam_research.scales import SCALE_SPECS, model_config_for_scale


def test_scaling_family_is_nearly_exact_parameter_match():
    expected = {
        "25m": (24_940_288, 24_941_263),
        "50m": (49_799_808, 49_801_457),
        "100m": (101_803_520, 101_806_616),
    }
    for scale, (expected_transformer, expected_tamv3) in expected.items():
        t_cfg = model_config_for_scale("transformer", scale, max_seq_len=1024)
        v3_cfg = model_config_for_scale("tamv3", scale, max_seq_len=1024)
        t_count = parameter_count(ResearchLM(t_cfg))
        v3_count = parameter_count(ResearchLM(v3_cfg))
        assert t_count == expected_transformer
        assert v3_count == expected_tamv3
        assert abs(v3_count - t_count) / t_count < 0.0001


def test_scaling_family_preserves_architecture_ratios():
    for name, spec in SCALE_SPECS.items():
        cfg = model_config_for_scale("tamv3", name, max_seq_len=1024)
        assert cfg.tamv2_state_size == cfg.d_model // 4
        assert cfg.tamv3_attn_inner == 13 * cfg.d_model // 16
        assert cfg.tamv3_attn_inner % cfg.n_heads == 0
