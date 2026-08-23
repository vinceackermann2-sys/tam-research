from __future__ import annotations

import types

import torch

from tam_research.aera_hardware_core import HardwareAERAConfig
from tam_research.aera_hardware_core_v8 import (
    HardwareAwareAERATextLMV8,
    MostlyHardRoutingSchedule,
)


def cfg(n_stages: int = 2) -> HardwareAERAConfig:
    return HardwareAERAConfig(
        vocab_size=47,
        d_model=32,
        n_stages=n_stages,
        n_heads=4,
        chunk_size=8,
        n_experts=4,
        max_active_experts=2,
        expert_mult=2,
        memory_dim=8,
        max_reason_steps=2,
        block_size=2,
    )


def _force_stage_gate(model: HardwareAwareAERATextLMV8, stage: int, run: bool) -> None:
    with torch.no_grad():
        router = model.stage_routers[stage]
        router.proj.weight.zero_()
        router.proj.bias.fill_(12.0 if run else -12.0)


def test_mostly_hard_schedule_is_actually_mostly_sparse():
    schedule = MostlyHardRoutingSchedule(calibration_every=8)
    modes = [schedule.mode_for_step(i) for i in range(16)]
    assert modes.count("hard_sparse") == 14
    assert modes.count("straight_through") == 2
    assert schedule.nominal_hard_sparse_fraction == 0.875


def test_hard_skip_does_not_execute_stage_and_preserves_state():
    torch.manual_seed(81)
    model = HardwareAwareAERATextLMV8(cfg(n_stages=1)).eval()
    _force_stage_gate(model, 0, False)
    tokens = torch.randint(0, model.cfg.vocab_size, (3, 8))
    initial = model.empty_state(tokens)
    before_stream = initial.stages[0].stream.clone()
    before_memory = initial.stages[0].memory.matrix.clone()

    calls = 0
    original = model.stages[0].forward_chunk

    def counted(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    model.stages[0].forward_chunk = types.MethodType(counted, model.stages[0])
    with torch.no_grad():
        out = model(tokens, state=initial, hard=True, route_mode="hard_sparse")

    assert calls == 0
    state = out["state"]
    assert torch.equal(state.stages[0].stream, before_stream)
    assert torch.equal(state.stages[0].memory.matrix, before_memory)
    assert model.last_stage_execution[0]["mean_executed_fraction"] == 0.0


def test_hard_run_executes_stage():
    torch.manual_seed(82)
    model = HardwareAwareAERATextLMV8(cfg(n_stages=1)).eval()
    _force_stage_gate(model, 0, True)
    tokens = torch.randint(0, model.cfg.vocab_size, (2, 8))
    calls = 0
    original = model.stages[0].forward_chunk

    def counted(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    model.stages[0].forward_chunk = types.MethodType(counted, model.stages[0])
    with torch.no_grad():
        model(tokens, hard=True, route_mode="hard_sparse")
    assert calls == 1
    assert model.last_stage_execution[0]["mean_executed_fraction"] == 1.0


def test_straight_through_task_loss_reaches_stage_router():
    torch.manual_seed(83)
    model = HardwareAwareAERATextLMV8(cfg(n_stages=1)).train()
    tokens = torch.randint(0, model.cfg.vocab_size, (4, 8))
    out = model(tokens, route_mode="straight_through", return_block_logits=True)
    terms = model.soft_objective(tokens, out, stage_compute_weight=0.001)
    terms["total"].backward()
    grad = model.stage_routers[0].proj.weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert float(grad.abs().sum()) > 0.0


def test_hard_sparse_task_loss_trains_executed_path():
    torch.manual_seed(84)
    model = HardwareAwareAERATextLMV8(cfg(n_stages=1)).train()
    _force_stage_gate(model, 0, True)
    tokens = torch.randint(0, model.cfg.vocab_size, (3, 8))
    out = model(
        tokens,
        hard=True,
        route_mode="hard_sparse",
        return_block_logits=True,
    )
    terms = model.hard_sparse_task_loss(tokens, out)
    terms["total"].backward()
    assert model.token_emb.weight.grad is not None
    assert torch.isfinite(model.token_emb.weight.grad).all()


def test_stage_routing_is_causal_within_chunk():
    torch.manual_seed(85)
    model = HardwareAwareAERATextLMV8(cfg(n_stages=2)).eval()
    for i in range(2):
        _force_stage_gate(model, i, True)

    a = torch.randint(0, model.cfg.vocab_size, (1, 8))
    b = a.clone()
    b[:, 6:] = (b[:, 6:] + 7) % model.cfg.vocab_size
    with torch.no_grad():
        ya = model(a, hard=True, route_mode="hard_sparse")["logits"]
        yb = model(b, hard=True, route_mode="hard_sparse")["logits"]
    # Changing positions 6-7 cannot influence logits at positions 0-5.
    assert torch.allclose(ya[:, :6], yb[:, :6], atol=2e-5, rtol=2e-5)


def test_soft_stage_compute_term_is_finite_and_bounded():
    torch.manual_seed(86)
    model = HardwareAwareAERATextLMV8(cfg(n_stages=2)).train()
    tokens = torch.randint(0, model.cfg.vocab_size, (3, 8))
    out = model(tokens, route_mode="soft")
    terms = model.soft_objective(tokens, out)
    assert torch.isfinite(terms["stage_compute"])
    assert 0.0 <= float(terms["stage_compute"]) <= 1.0
