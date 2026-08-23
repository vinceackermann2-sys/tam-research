from __future__ import annotations

import types

import torch

from tam_research.aera import AERAState, FastMemoryState
from tam_research.aera_hardware_core import HardwareAERAConfig
from tam_research.aera_hardware_core_v6 import BMMHardSparseExpertBank
from tam_research.aera_hardware_core_v9 import HardwareAwareAERATextLMV9


def cfg() -> HardwareAERAConfig:
    return HardwareAERAConfig(
        vocab_size=47,
        d_model=32,
        n_stages=1,
        n_heads=4,
        chunk_size=8,
        n_experts=4,
        max_active_experts=2,
        expert_mult=2,
        memory_dim=8,
        max_reason_steps=2,
        block_size=2,
    )


def test_v9_uses_verified_bmm_expert_backend_by_default():
    model = HardwareAwareAERATextLMV9(cfg())
    assert isinstance(model.stages[0].experts, BMMHardSparseExpertBank)


def test_hard_sparse_merge_restores_residual_and_state_dtypes():
    torch.manual_seed(91)
    model = HardwareAwareAERATextLMV9(cfg()).eval()
    stage = model.stages[0]
    router = model.stage_routers[0]

    # Force only example 0 to execute the expensive stage.
    def routed(self, first_event, stream, *, mode):
        assert mode == "hard_sparse"
        gate = torch.tensor([[1.0], [0.0]], device=first_event.device, dtype=torch.float32)
        logits = torch.tensor([[12.0], [-12.0]], device=first_event.device, dtype=torch.float32)
        return gate, logits

    router.forward = types.MethodType(routed, router)

    # Emulate a mixed-precision stage that promotes its executed branch to FP32.
    def promoted(self, events, state, *, hard, update_memory):
        assert hard
        y = (events.float() + 1.0)
        promoted_state = AERAState(
            stream=(state.stream.float() + 2.0),
            memory=FastMemoryState(state.memory.matrix.float() + 3.0),
        )
        dummy = {"start": {}, "end": {}}
        return y, promoted_state, dummy

    stage.forward_chunk = types.MethodType(promoted, stage)

    x = torch.randn(2, 8, model.cfg.d_model, dtype=torch.bfloat16)
    state = AERAState(
        stream=torch.randn(2, model.cfg.d_model, dtype=torch.bfloat16),
        memory=FastMemoryState(
            torch.randn(2, model.cfg.memory_dim, model.cfg.memory_dim, dtype=torch.bfloat16)
        ),
    )
    before_x = x.clone()
    before_stream = state.stream.clone()
    before_memory = state.memory.matrix.clone()

    y, new_state, info = model._route_one_stage(
        x,
        stage,
        state,
        router,
        route_mode="hard_sparse",
        update_memory=False,
    )

    assert y.dtype == x.dtype
    assert new_state.stream.dtype == state.stream.dtype
    assert new_state.memory.matrix.dtype == state.memory.matrix.dtype
    # Skipped example is bitwise untouched in residual and state.
    assert torch.equal(y[1], before_x[1])
    assert torch.equal(new_state.stream[1], before_stream[1])
    assert torch.equal(new_state.memory.matrix[1], before_memory[1])
    # Executed branch was merged after dtype restoration.
    assert torch.allclose(y[0].float(), (before_x[0].float() + 1.0), atol=0.02, rtol=0.0)
    assert info["executed_fraction"] == 0.5


def test_full_v9_hard_sparse_forward_is_finite():
    torch.manual_seed(92)
    model = HardwareAwareAERATextLMV9(cfg()).eval().to(dtype=torch.bfloat16)
    with torch.no_grad():
        # Force the only stage to run so the hard merge executes.
        model.stage_routers[0].proj.weight.zero_()
        model.stage_routers[0].proj.bias.fill_(12.0)
        tokens = torch.randint(0, model.cfg.vocab_size, (2, 16))
        out = model(tokens, hard=True, route_mode="hard_sparse", update_memory=False)
    logits = out["logits"]
    assert logits.shape == (2, 16, model.cfg.vocab_size)
    assert torch.isfinite(logits.float()).all()
