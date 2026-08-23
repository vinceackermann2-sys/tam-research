from __future__ import annotations

import torch

from tam_research.aera_hardware_core import HardwareAERAConfig, StackedChunkExpertBank
from tam_research.aera_hardware_core_v4 import TrulySparseStackedChunkExpertBank


def cfg() -> HardwareAERAConfig:
    return HardwareAERAConfig(
        vocab_size=31,
        d_model=16,
        n_stages=1,
        n_heads=4,
        chunk_size=7,
        n_experts=4,
        max_active_experts=2,
        expert_mult=2,
        memory_dim=8,
        max_reason_steps=2,
        block_size=3,
    )


def test_v4_hard_sparse_bank_matches_reference_hard_outputs():
    torch.manual_seed(31)
    c = cfg()
    reference = StackedChunkExpertBank(c)
    sparse = TrulySparseStackedChunkExpertBank(c)
    sparse.load_state_dict(reference.state_dict())

    x = torch.randn(6, c.chunk_size, c.d_model)
    expert_logits = torch.randn(6, c.n_experts)
    count_logits = torch.tensor(
        [[3.0, -3.0], [-3.0, 3.0], [2.0, -2.0], [-2.0, 2.0], [4.0, -4.0], [-4.0, 4.0]]
    )
    with torch.no_grad():
        expected = reference(x, expert_logits, count_logits, hard=True)
        actual = sparse(x, expert_logits, count_logits, hard=True)
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
    assert sparse.last_second_batch_size == 3
    assert sparse.last_executed_expert_slots == 9  # 6 first + 3 second


def test_v4_top1_hard_path_executes_no_second_expert_slots():
    torch.manual_seed(32)
    c = cfg()
    sparse = TrulySparseStackedChunkExpertBank(c)
    x = torch.randn(5, c.chunk_size, c.d_model)
    expert_logits = torch.randn(5, c.n_experts)
    count_logits = torch.tensor([[5.0, -5.0]]).repeat(5, 1)
    with torch.no_grad():
        sparse(x, expert_logits, count_logits, hard=True)
    assert sparse.last_second_batch_size == 0
    assert sparse.last_executed_expert_slots == 5
    stats = sparse.stats()
    assert stats is not None
    assert stats["mean_active_experts"] == 1.0
