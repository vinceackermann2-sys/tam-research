from __future__ import annotations

import torch

from tam_research.cpw_v1.model import CPWV1Config
from tam_research.cpw_v5_afm.model import (
    AFMCPWResearchLM,
    AssociativeFastMemoryPredictor,
    afm_mixer_parameter_count,
    afm_parameter_count,
    exclusive_fast_weight_read,
    exclusive_fast_weight_read_reference,
)
from tam_research.cpw_v5_afm.protocol import (
    AFM_MIXER_PARAMETERS,
    AFM_PARAMETERS,
    SEQUENCE_PARAMETERS,
)


def test_afm_parameter_match_is_exact() -> None:
    assert afm_mixer_parameter_count() == AFM_MIXER_PARAMETERS
    assert afm_parameter_count() == AFM_PARAMETERS
    assert afm_parameter_count() == SEQUENCE_PARAMETERS


def test_vectorized_exclusive_prefix_matches_reference() -> None:
    torch.manual_seed(1296)
    q = torch.rand(2, 9, 5) + 0.1
    k = torch.rand(2, 9, 5) + 0.1
    v = torch.randn(2, 9, 7)
    gate = torch.sigmoid(torch.randn(2, 9, 1))
    gate[:, 0] = 0.0

    fast = exclusive_fast_weight_read(q, k, v, gate)
    slow = exclusive_fast_weight_read_reference(q, k, v, gate)
    torch.testing.assert_close(fast, slow, rtol=2e-5, atol=2e-5)


def test_current_write_is_not_visible_to_current_read() -> None:
    q = torch.ones(1, 2, 3)
    k = torch.ones(1, 2, 3)
    v = torch.zeros(1, 2, 4)
    v[:, 1] = 10_000.0
    gate = torch.zeros(1, 2, 1)
    gate[:, 1] = 1.0

    read = exclusive_fast_weight_read(q, k, v, gate)
    assert torch.equal(read[:, 0], torch.zeros_like(read[:, 0]))
    assert torch.equal(read[:, 1], torch.zeros_like(read[:, 1]))


def test_future_tokens_do_not_change_earlier_logits() -> None:
    torch.manual_seed(96)
    cfg = CPWV1Config(n_layers=3, max_seq_len=32)
    model = AFMCPWResearchLM(cfg, afm_layer=0).eval()
    a = torch.randint(0, cfg.vocab_size, (1, 24))
    b = a.clone()
    b[:, 16:] = torch.randint(0, cfg.vocab_size, (1, 8))

    with torch.no_grad():
        la = model(a)
        lb = model(b)

    torch.testing.assert_close(la[:, :16], lb[:, :16], rtol=0, atol=3e-5)


def test_forward_backward_reaches_all_afm_parameters() -> None:
    torch.manual_seed(1296001)
    cfg = CPWV1Config(n_layers=2, max_seq_len=32)
    model = AFMCPWResearchLM(cfg, afm_layer=0)
    tokens = torch.randint(0, cfg.vocab_size, (2, 16))
    logits = model(tokens)
    assert torch.isfinite(logits).all()
    logits.float().square().mean().backward()

    names = {
        "blocks.0.mixer.afm.q_proj.weight",
        "blocks.0.mixer.afm.k_proj.weight",
        "blocks.0.mixer.afm.v_proj.weight",
        "blocks.0.mixer.afm.out_proj.weight",
        "blocks.0.mixer.afm.gate_prev.weight",
        "blocks.0.mixer.afm.gate_cur.weight",
    }
    params = dict(model.named_parameters())
    for name in names:
        grad = params[name].grad
        assert grad is not None, name
        assert torch.isfinite(grad).all(), name


def test_candidate_contains_no_attention_or_world_module() -> None:
    names = [name.lower() for name, _ in AFMCPWResearchLM().named_modules()]
    assert not any("attention" in name for name in names)
    assert not any(".world" in name for name in names)


def test_afm_predictor_starts_with_sparse_writes() -> None:
    torch.manual_seed(7)
    predictor = AssociativeFastMemoryPredictor(256)
    x = torch.randn(2, 12, 256)
    _ = predictor(x)
    assert predictor.last_gate_mean is not None
    gate_mean = float(predictor.last_gate_mean)
    assert 0.0 < gate_mean < 0.5
