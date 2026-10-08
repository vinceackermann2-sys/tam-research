"""CPW #1313: zero-GPU structural, causal and training-path controls.

Do not interpret these tests as scientific evidence of learned binding.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from tam_research.cpw_binding_placement.model import (
    EARLY_AFM_LAYER,
    LATE_AFM_LAYER,
    EarlyAFMResearchLM,
    placement_parameter_counts,
)
from tam_research.cpw_binding_v2.task import (
    SCORED_DELAYS,
    counterfactual_query,
    explicit_key_lookup,
    make_binding_batch,
)
from tam_research.cpw_v1.model import CPWV1Config
from tam_research.cpw_v3_sparse_world.model import SparseWorldCPWResearchLM
from tam_research.cpw_v5_afm.model import AFMCPWResearchLM
from tam_research.cpw_v5_afm.train import query_logits
from tam_research.models import parameter_count


def _tiny_cfg() -> CPWV1Config:
    return CPWV1Config(
        vocab_size=5000,
        d_model=32,
        n_layers=15,
        max_seq_len=320,
        ff_mult=2,
        world_state_size=8,
        sequence_predictor_rank=12,
        memory_predictor_rank=8,
    )


def _state_equal(a: torch.nn.Module, b: torch.nn.Module) -> None:
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys()
    for name in sa:
        torch.testing.assert_close(sa[name], sb[name], rtol=0, atol=0)


def test_exact_frozen_parameter_budget_and_single_afm_location() -> None:
    counts = placement_parameter_counts()
    assert counts == {"early_afm": 21721344, "late_afm": 21721344}
    assert counts["early_afm"] < 24940288
    for cls, expected in ((EarlyAFMResearchLM, 0), (AFMCPWResearchLM, 14)):
        model = cls(_tiny_cfg())
        assert model.afm_layers == (expected,)
        assert len(model.blocks) == 15
        assert parameter_count(model) > 0
        for i, block in enumerate(model.blocks):
            assert (block.afm is not None) == (i == expected)
            assert (block.sequence is None) == (i == expected)
        memory = model.blocks[expected].afm
        assert memory.rank == 32
        assert memory.state_size == 1056
        assert not hasattr(memory, "query_proj")
        names = [name.lower() for name, _ in model.named_modules()]
        assert not any("attention" in name or "router" in name for name in names)


def test_identical_initial_parameters_modulo_exact_block_swap() -> None:
    torch.manual_seed(1313)
    late = AFMCPWResearchLM(_tiny_cfg())
    torch.manual_seed(1313)
    early = EarlyAFMResearchLM(_tiny_cfg())
    _state_equal(late.token_emb, early.token_emb)
    _state_equal(late.pos_emb, early.pos_emb)
    _state_equal(late.norm, early.norm)
    _state_equal(late.lm_head, early.lm_head)
    _state_equal(late.blocks[LATE_AFM_LAYER], early.blocks[EARLY_AFM_LAYER])
    _state_equal(late.blocks[EARLY_AFM_LAYER], early.blocks[LATE_AFM_LAYER])
    for idx in range(1, LATE_AFM_LAYER):
        _state_equal(late.blocks[idx], early.blocks[idx])


def test_strict_future_token_causality_and_distant_source_path() -> None:
    torch.manual_seed(1314)
    cfg = _tiny_cfg()
    early = EarlyAFMResearchLM(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (1, 44))
    changed_future = x.clone()
    changed_future[:, 29:] = torch.randint(0, cfg.vocab_size, (1, 15))
    with torch.no_grad():
        a = early(x)
        b = early(changed_future)
    torch.testing.assert_close(a[:, :29], b[:, :29], rtol=0, atol=3e-5)

    changed_source = x.clone()
    changed_source[:, 0] = (changed_source[:, 0] + 1) % cfg.vocab_size
    local = SparseWorldCPWResearchLM("sequence_only", cfg).eval()
    with torch.no_grad():
        ea = query_logits(early, x)
        eb = query_logits(early, changed_source)
        la = query_logits(local, x)
        lb = query_logits(local, changed_source)
    assert torch.isfinite(ea).all() and torch.isfinite(eb).all()
    assert float((ea - eb).abs().max()) > 0.0
    torch.testing.assert_close(la, lb, rtol=0, atol=0)


def test_corrected_task_counterfactuals_and_cpu_train_step() -> None:
    torch.manual_seed(1315)
    batch = make_binding_batch(
        batch_size=2, generator=torch.Generator().manual_seed(1315)
    )
    assert torch.equal(explicit_key_lookup(batch.tokens), batch.targets)
    assert all(int(d) in SCORED_DELAYS for d in batch.delays)

    changed, changed_targets = counterfactual_query(batch, new_delay=256)
    assert torch.equal(changed[:, :-1], batch.tokens[:, :-1])
    assert torch.equal(explicit_key_lookup(changed), changed_targets)

    model = EarlyAFMResearchLM(_tiny_cfg()).train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1
    )
    optimizer.zero_grad(set_to_none=True)
    logits = query_logits(model, batch.tokens)
    loss = F.cross_entropy(logits.float(), batch.targets)
    assert torch.isfinite(loss)
    loss.backward()
    afm_grads = [
        p.grad for name, p in model.named_parameters()
        if ".afm." in name and p.requires_grad
    ]
    assert afm_grads
    assert all(g is not None and torch.isfinite(g).all() for g in afm_grads)
    assert any(float(g.abs().sum()) > 0 for g in afm_grads)
    optimizer.step()
    stats = model.memory_stats()
    assert stats["afm_layer_index"] == 0
    assert stats["memory_state_scalars"] == 1056
    assert 0.0 <= stats["mean_write_gate"] <= 1.0
    # No performance threshold or claims: a single CPU optimizer step is
    # only an integrity smoke, not a learned associative-recall result.
