from __future__ import annotations

from pathlib import Path

import torch

from tam_research.cpw_v1.model import CPWV1Config
from tam_research.cpw_v3_sparse_world.model import SparseWorldCPWResearchLM
from tam_research.cpw_v5_afm.model import (
    AFMCPWResearchLM,
    AssociativeFastMemoryPredictor,
    afm_parameter_count,
    associative_scan_reference,
    associative_scan_vectorized,
)
from tam_research.cpw_v5_afm.protocol import (
    AFM_LAYER,
    EXPECTED_AFM_PARAMETERS,
    MEMORY_RANK,
    TRANSFORMER_PARAMETERS,
)
from tam_research.cpw_v5_afm.train import query_logits


def _tiny_cfg() -> CPWV1Config:
    return CPWV1Config(
        vocab_size=5_000,
        d_model=32,
        n_layers=15,
        max_seq_len=64,
        ff_mult=2,
        world_state_size=8,
        sequence_predictor_rank=12,
        memory_predictor_rank=8,
    )


def test_parameter_budget_and_sparse_layout() -> None:
    model = AFMCPWResearchLM()
    assert afm_parameter_count() == EXPECTED_AFM_PARAMETERS
    assert afm_parameter_count() < TRANSFORMER_PARAMETERS
    assert tuple(model.afm_layers) == (AFM_LAYER,)
    for i, block in enumerate(model.blocks):
        if i == AFM_LAYER:
            assert block.afm is not None
            assert block.afm.rank == MEMORY_RANK
            assert block.afm.state_size == MEMORY_RANK * MEMORY_RANK + MEMORY_RANK
            assert block.sequence is None
            assert not hasattr(block.afm, "query_proj")
        else:
            assert block.afm is None
            assert block.sequence is not None

    names = [name.lower() for name, _ in model.named_modules()]
    assert not any("attention" in name for name in names)
    assert not any("router" in name for name in names)
    assert not any(".world" in name for name, _ in model.named_modules())


def test_vectorized_scan_matches_literal_recurrence() -> None:
    torch.manual_seed(1290)
    keys = torch.rand(2, 11, 7) + 0.1
    values = torch.randn(2, 11, 7)
    gates = torch.sigmoid(torch.randn(2, 11, 1))
    fast_read, fast_state = associative_scan_vectorized(keys, values, gates)
    ref_read, ref_state = associative_scan_reference(keys, values, gates)
    torch.testing.assert_close(fast_read, ref_read, rtol=1e-5, atol=2e-6)
    torch.testing.assert_close(fast_state, ref_state, rtol=1e-5, atol=2e-6)


def test_finite_forward_backward_and_afm_gradients() -> None:
    torch.manual_seed(1291)
    cfg = _tiny_cfg()
    model = AFMCPWResearchLM(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (2, 24))
    logits = query_logits(model, tokens)
    assert logits.shape == (2, cfg.vocab_size)
    assert torch.isfinite(logits).all()
    logits.float().square().mean().backward()

    grads = [
        p.grad
        for name, p in model.named_parameters()
        if ".afm." in name and p.requires_grad
    ]
    assert grads
    assert all(g is not None and torch.isfinite(g).all() for g in grads)


def test_afm_has_distant_dependency_beyond_sequence_horizon() -> None:
    torch.manual_seed(1292)
    cfg = _tiny_cfg()
    a = torch.randint(0, cfg.vocab_size, (1, 40))
    b = a.clone()
    b[:, 0] = (b[:, 0] + 1) % cfg.vocab_size

    sequence = SparseWorldCPWResearchLM("sequence_only", cfg).eval()
    with torch.no_grad():
        sa = query_logits(sequence, a)
        sb = query_logits(sequence, b)
    torch.testing.assert_close(sa, sb, rtol=0, atol=0)

    torch.manual_seed(1292)
    afm = AFMCPWResearchLM(cfg).eval()
    with torch.no_grad():
        aa = query_logits(afm, a)
        ab = query_logits(afm, b)
    assert float((aa - ab).abs().max()) > 0.0


def test_strict_future_token_causality() -> None:
    torch.manual_seed(1293)
    cfg = _tiny_cfg()
    model = AFMCPWResearchLM(cfg).eval()
    a = torch.randint(0, cfg.vocab_size, (1, 36))
    b = a.clone()
    b[:, 24:] = torch.randint(0, cfg.vocab_size, (1, 12))
    with torch.no_grad():
        la = model(a)
        lb = model(b)
    torch.testing.assert_close(la[:, :24], lb[:, :24], rtol=0, atol=3e-5)


def test_shared_key_query_projection_and_generic_gate() -> None:
    module = AssociativeFastMemoryPredictor(32, rank=MEMORY_RANK)
    assert isinstance(module.key_proj, torch.nn.Linear)
    assert not hasattr(module, "query_proj")
    assert module.gate_prev.in_features == 32
    assert module.gate_cur.in_features == 32
    assert module.gate_prev.out_features == 1
    assert module.gate_cur.out_features == 1


def test_cpw_v4_task_generator_is_not_modified_on_this_track() -> None:
    text = Path("tam_research/cpw_v4_memory/task.py").read_text()
    assert "make_associative_batch" in text
    assert "query_position = seq_len - 1" in text
    assert "query_position - source_pos != d" in text
