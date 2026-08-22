from tam_research.models import ResearchLM, parameter_count
from tam_research.scales import model_config_for_scale


def test_100m_transformer_is_parameter_matched_to_tamv3():
    # train_language_model uses max(1024, seq_len), so a 512-token training run
    # still has 1024 learned positional embeddings in both architectures.
    transformer_cfg = model_config_for_scale("transformer", "100m", max_seq_len=1024)
    tam_cfg = model_config_for_scale("tamv3", "100m", max_seq_len=1024)

    assert transformer_cfg.d_model == tam_cfg.d_model == 512
    assert transformer_cfg.n_layers == tam_cfg.n_layers == 24
    assert transformer_cfg.n_heads == tam_cfg.n_heads == 16
    assert transformer_cfg.max_seq_len == tam_cfg.max_seq_len == 1024

    transformer_params = parameter_count(ResearchLM(transformer_cfg))
    tam_params = parameter_count(ResearchLM(tam_cfg))
    assert transformer_params == 101_803_520
    assert tam_params == 101_806_616
    assert abs(transformer_params - tam_params) == 3_096
    assert abs(transformer_params - tam_params) / tam_params < 0.0001


def test_2b_protocol_step_count_matches_tam_run():
    token_budget = 2_000_000_000
    micro_batch = 64
    seq_len = 512
    grad_accum = 2
    tokens_per_step = micro_batch * seq_len * grad_accum
    steps = (token_budget + tokens_per_step - 1) // tokens_per_step
    assert tokens_per_step == 65_536
    assert steps == 30_518
