from tam_research.future_scales import FUTURE_SCALE_SPECS, analytical_parameter_count


EXPECTED = {
    "300m": (294_299_904, 294_306_852),
    "1b": (1_009_606_400, 1_009_621_808),
}


def test_future_scale_counts_are_nearly_exact_matches():
    for scale, (expected_t, expected_tam) in EXPECTED.items():
        t = analytical_parameter_count("transformer", scale)
        tam = analytical_parameter_count("tamv3", scale)
        assert t == expected_t
        assert tam == expected_tam
        assert abs(tam - t) / t < 0.0001


def test_future_scales_preserve_tam_v3_ratios():
    for spec in FUTURE_SCALE_SPECS.values():
        assert spec.state_size == spec.d_model // 4
        assert spec.tamv3_attention_inner == 13 * spec.d_model // 16
        assert spec.d_model % spec.n_heads == 0
        assert spec.tamv3_attention_inner % spec.n_heads == 0
