from tam_research.models import ResearchLM, parameter_count
from tam_research.scales import (
    SCALE_SPECS,
    analytical_parameter_count,
    model_config_for_scale,
)


EXPECTED_COUNTS = {
    "25m": (24_940_288, 24_941_263),
    "50m": (49_799_808, 49_801_457),
    "100m": (101_803_520, 101_806_616),
    "300m": (294_299_904, 294_306_852),
    "1b": (1_009_606_400, 1_009_621_808),
}


def test_scaling_family_is_nearly_exact_parameter_match():
    for scale, (expected_transformer, expected_tamv3) in EXPECTED_COUNTS.items():
        t_count = analytical_parameter_count("transformer", scale)
        v3_count = analytical_parameter_count("tamv3", scale)
        assert t_count == expected_transformer
        assert v3_count == expected_tamv3
        assert abs(v3_count - t_count) / t_count < 0.0001


def test_analytical_counts_match_instantiated_models_through_100m():
    # Do not instantiate 300M/1B models in CI just to count parameters.
    for scale in ("25m", "50m", "100m"):
        t_cfg = model_config_for_scale("transformer", scale, max_seq_len=1024)
        v3_cfg = model_config_for_scale("tamv3", scale, max_seq_len=1024)
        assert parameter_count(ResearchLM(t_cfg)) == analytical_parameter_count("transformer", scale)
        assert parameter_count(ResearchLM(v3_cfg)) == analytical_parameter_count("tamv3", scale)


def test_scaling_family_preserves_architecture_ratios():
    for name, spec in SCALE_SPECS.items():
        cfg = model_config_for_scale("tamv3", name, max_seq_len=1024)
        assert cfg.tamv2_state_size == cfg.d_model // 4
        assert cfg.tamv3_attn_inner == 13 * cfg.d_model // 16
        assert cfg.tamv3_attn_inner % cfg.n_heads == 0
        assert cfg.d_model % cfg.n_heads == 0
