from __future__ import annotations

import hashlib

import torch
import torch.nn.functional as F

from model import (
    ActionRequest,
    CortexSConfig,
    CortexSCore,
    FastAssociativeMemory,
    SafetyKernel,
    count_parameters,
)
from baseline import TransformerStateBaseline


def parameter_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for parameter in model.parameters():
        digest.update(parameter.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def test_forward_backward_is_finite():
    torch.manual_seed(1)
    model = CortexSCore()
    ops = torch.randint(0, 3, (8, 6))
    values = torch.randint(0, 16, (8, 6))
    targets = torch.randint(0, 16, (8, 6))
    logits, state, routes, halts = model(ops, values)
    loss = F.cross_entropy(logits.reshape(-1, 16), targets.reshape(-1))
    loss.backward()
    assert logits.shape == (8, 6, 16)
    assert state.hidden.shape == (8, 48)
    assert halts.shape == (8, 6)
    assert torch.isfinite(loss)
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    assert routes


def test_router_selects_exact_top_k():
    torch.manual_seed(2)
    cfg = CortexSConfig(num_experts=6, top_k=2)
    model = CortexSCore(cfg)
    ops = torch.randint(0, 3, (5, 3))
    values = torch.randint(0, 16, (5, 3))
    _, _, routes, _ = model(ops, values)
    assert routes
    assert all(route.shape == (5, 2) for route in routes)
    assert all(torch.all(route[:, 0] != route[:, 1]) for route in routes)


def test_persistent_state_changes_stream_result():
    torch.manual_seed(3)
    model = CortexSCore()
    first_ops = torch.tensor([[0, 0, 1]])
    first_values = torch.tensor([[5, 3, 7]])
    _, state, _, _ = model(first_ops, first_values)

    second_ops = torch.tensor([[0]])
    second_values = torch.tensor([[2]])
    continued, _, _, _ = model(second_ops, second_values, state=state)
    reset, _, _, _ = model(second_ops, second_values)
    assert not torch.allclose(continued, reset)


def test_fast_memory_learns_without_mutating_model_weights():
    torch.manual_seed(4)
    model = CortexSCore()
    before = parameter_digest(model)
    memory = FastAssociativeMemory(key_dim=4, value_dim=2)
    memory.write(torch.tensor([1.0, 0.0, 0.0, 0.0]), torch.tensor([0.25, 0.75]))
    memory.write(torch.tensor([0.0, 1.0, 0.0, 0.0]), torch.tensor([0.80, 0.20]))
    value, score = memory.retrieve(torch.tensor([0.99, 0.01, 0.0, 0.0]))
    after = parameter_digest(model)
    assert score > 0.99
    assert torch.allclose(value, torch.tensor([0.25, 0.75]))
    assert before == after


def test_safety_kernel_blocks_authority_escalation_by_request():
    kernel = SafetyKernel(
        {"email", "payment"},
        autonomous_amount_limit=100.0,
        allowed_targets={"approved@example.com"},
    )
    assert kernel.authorize(ActionRequest("email", target="approved@example.com")).allowed
    assert not kernel.authorize(ActionRequest("shell", target="approved@example.com")).allowed
    decision = kernel.authorize(ActionRequest("payment", amount=1000.0, target="approved@example.com"))
    assert not decision.allowed
    assert decision.reason == "human_approval_required"
    assert not kernel.authorize(ActionRequest("payment", amount=10.0, target="other@example.com")).allowed


def test_tiny_parameter_match_is_within_two_percent():
    cortex = CortexSCore()
    baseline = TransformerStateBaseline()
    gap = abs(count_parameters(cortex) - count_parameters(baseline)) / count_parameters(baseline)
    assert gap <= 0.02
