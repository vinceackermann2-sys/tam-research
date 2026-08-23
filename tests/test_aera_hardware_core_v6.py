from __future__ import annotations

import torch

from tam_research.aera_hardware_core import HardwareAERAConfig
from tam_research.aera_hardware_core_v4 import TrulySparseStackedChunkExpertBank
from tam_research.aera_hardware_core_v6 import BMMHardSparseExpertBank


def cfg() -> HardwareAERAConfig:
    return HardwareAERAConfig(
        d_model=32,
        n_stages=1,
        n_heads=4,
        chunk_size=8,
        n_experts=4,
        max_active_experts=2,
        expert_mult=2,
        memory_dim=8,
        max_reason_steps=2,
    )


def test_bmm_hard_experts_match_reference_top1_and_top2():
    torch.manual_seed(61)
    c = cfg()
    ref = TrulySparseStackedChunkExpertBank(c)
    opt = BMMHardSparseExpertBank(c)
    opt.load_state_dict(ref.state_dict())
    x = torch.randn(5, 8, c.d_model)
    logits = torch.randn(5, c.n_experts)

    for count in (1, 2):
        count_logits = torch.full((5, 2), -8.0)
        count_logits[:, count - 1] = 8.0
        with torch.no_grad():
            a = ref(x, logits, count_logits, hard=True)
            b = opt(x, logits, count_logits, hard=True)
        assert torch.allclose(a, b, atol=2e-5, rtol=2e-5)


def test_all_top1_executes_only_one_selected_expert_pass():
    torch.manual_seed(62)
    c = cfg()
    bank = BMMHardSparseExpertBank(c)
    calls: list[int] = []
    original = bank._run_selected

    def counted(x, ids):
        calls.append(int(ids.numel()))
        return original(x, ids)

    bank._run_selected = counted  # type: ignore[method-assign]
    x = torch.randn(7, 8, c.d_model)
    logits = torch.randn(7, c.n_experts)
    count_logits = torch.tensor([[8.0, -8.0]]).expand(7, -1)
    with torch.no_grad():
        bank(x, logits, count_logits, hard=True)
    assert calls == [7]
    stats = bank.stats()
    assert stats is not None
    assert stats["mean_active_experts"] == 1.0


def test_bmm_bank_mixed_precision_cpu_autocast():
    torch.manual_seed(63)
    c = cfg()
    bank = BMMHardSparseExpertBank(c)
    x = torch.randn(4, 8, c.d_model)
    logits = torch.randn(4, c.n_experts)
    count_logits = torch.tensor([[8.0, -8.0], [-8.0, 8.0], [8.0, -8.0], [-8.0, 8.0]])
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        out = bank(x, logits, count_logits, hard=True)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
