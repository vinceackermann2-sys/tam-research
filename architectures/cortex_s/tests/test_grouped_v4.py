from __future__ import annotations

import inspect

import torch

from architectures.cortex_s.grouped_moe import PhysicalPaddedGroupedSparseMoE
from architectures.cortex_s.grouped_moe_v4 import (
    MemoryLeanPhysicalPaddedGroupedSparseMoE,
    SYSTEMS_VARIANT,
    pack_topk_assignments_lean,
)
from architectures.cortex_s.language_model import TrulySparseMoE, parameter_count
from architectures.cortex_s.systems_optimization import pack_topk_assignments


def test_v4_lean_packing_is_exactly_the_same_routing_plan() -> None:
    torch.manual_seed(2026090906)
    tokens = 97
    top_k = 2
    experts = 8
    logits = torch.randn(tokens, experts)
    top_values, top_indices = logits.topk(top_k, dim=-1)
    weights = torch.softmax(top_values.float(), dim=-1)

    legacy = pack_topk_assignments(top_indices, weights, num_experts=experts)
    lean = pack_topk_assignments_lean(top_indices, weights, num_experts=experts)

    assert torch.equal(lean.token_indices, legacy.token_indices)
    assert torch.equal(lean.assignment_weights, legacy.assignment_weights)
    assert torch.equal(lean.counts, legacy.counts)
    assert torch.equal(lean.offsets, legacy.offsets)


def test_v4_cpu_forward_and_gradients_match_v3() -> None:
    torch.manual_seed(2026090906)
    legacy = TrulySparseMoE(d_model=16, num_experts=4, top_k=2, hidden=10)
    v3 = PhysicalPaddedGroupedSparseMoE(d_model=16, num_experts=4, top_k=2, hidden=10)
    v4 = MemoryLeanPhysicalPaddedGroupedSparseMoE(
        d_model=16, num_experts=4, top_k=2, hidden=10
    )
    v3.copy_from_legacy(legacy)
    v4.copy_from_legacy(legacy)

    assert parameter_count(v4) == parameter_count(v3) == parameter_count(legacy)

    x3 = torch.randn(3, 7, 16, requires_grad=True)
    x4 = x3.detach().clone().requires_grad_(True)
    target = torch.randn(3, 7, 16)
    out3 = v3(x3)
    out4 = v4(x4)
    assert torch.equal(out4, out3)

    loss3 = (out3 - target).square().mean()
    loss4 = (out4 - target).square().mean()
    loss3.backward()
    loss4.backward()
    assert torch.equal(x4.grad, x3.grad)
    assert torch.isfinite(x4.grad).all()


def test_v4_bf16_physical_matrices_equal_v3_eventual_grouped_inputs() -> None:
    torch.manual_seed(2026090906)
    legacy = TrulySparseMoE(d_model=32, num_experts=8, top_k=2, hidden=338)
    v3 = PhysicalPaddedGroupedSparseMoE(d_model=32, num_experts=8, top_k=2, hidden=338)
    v4 = MemoryLeanPhysicalPaddedGroupedSparseMoE(
        d_model=32, num_experts=8, top_k=2, hidden=338
    )
    v3.copy_from_legacy(legacy)
    v4.copy_from_legacy(legacy)

    old_w1, old_w2 = v3._physical_matrices()
    new_w1, new_w2 = v4._physical_matrices_bf16()
    assert old_w1.shape == new_w1.shape == (8, 32, 344)
    assert old_w2.shape == new_w2.shape == (8, 344, 32)
    assert torch.equal(new_w1, old_w1.to(torch.bfloat16))
    assert torch.equal(new_w2, old_w2.to(torch.bfloat16))
    assert new_w1.stride(-1) == 1
    assert new_w2.stride(-1) == 1


def test_v4_source_casts_before_gather_and_before_padding() -> None:
    source = inspect.getsource(MemoryLeanPhysicalPaddedGroupedSparseMoE)
    flat_cast = source.index("flat.to(CUDA_COMPUTE_DTYPE)")
    gather = source.index("compute_flat.index_select")
    w1_cast = source.index("self.expert_w1.transpose(-2, -1).to(dtype=CUDA_COMPUTE_DTYPE)")
    w1_pad = source.index("F.pad(w1, (0, pad))")
    w2_cast = source.index("self.expert_w2.to(dtype=CUDA_COMPUTE_DTYPE)")
    w2_pad = source.index("F.pad(w2, (0, 0, 0, pad))")

    assert flat_cast < gather
    assert w1_cast < w1_pad
    assert w2_cast < w2_pad
    assert SYSTEMS_VARIANT == "memory_lean_cast_before_gather_pad_v4"
