from __future__ import annotations

import torch
import torch.nn.functional as F

from language_model import (
    CortexSLM,
    CortexSLMConfig,
    TrulySparseMoE,
    affine_scan,
    parameter_count,
)


TRANSFORMER_25M_PARAMETERS = 24_940_288
CORTEX_S_25M_PARAMETERS = 25_022_776


def tiny_config() -> CortexSLMConfig:
    return CortexSLMConfig(
        vocab_size=64,
        d_model=32,
        n_layers=4,
        n_heads=4,
        max_seq_len=32,
        state_size=8,
        num_experts=4,
        top_k=2,
        expert_hidden=24,
        attention_every=2,
    )


def test_frozen_25m_parameter_match_is_within_half_percent():
    model = CortexSLM()
    assert parameter_count(model) == CORTEX_S_25M_PARAMETERS
    gap = abs(parameter_count(model) - TRANSFORMER_25M_PARAMETERS) / TRANSFORMER_25M_PARAMETERS
    assert gap <= 0.005


def test_affine_parallel_scan_matches_sequential_recurrence_with_initial_state():
    torch.manual_seed(101)
    a = torch.sigmoid(torch.randn(3, 17, 7))
    b = torch.randn(3, 17, 7)
    initial = torch.randn(3, 7)
    parallel = affine_scan(a, b, initial)

    state = initial
    reference = []
    for index in range(a.size(1)):
        state = a[:, index] * state + b[:, index]
        reference.append(state)
    sequential = torch.stack(reference, dim=1)
    assert torch.allclose(parallel, sequential, atol=2e-6, rtol=2e-6)


def test_language_model_is_causal():
    torch.manual_seed(102)
    model = CortexSLM(tiny_config()).eval()
    tokens = torch.randint(0, 64, (2, 12))
    changed = tokens.clone()
    changed[:, 7:] = torch.randint(0, 64, (2, 5))
    with torch.no_grad():
        original_logits = model(tokens)
        changed_logits = model(changed)
    # Sparse gather/scatter kernels are allowed last-bit floating-point variation
    # from work partitioning. Causality means no material dependence on future
    # tokens, not bit-identical reduction order.
    torch.testing.assert_close(
        original_logits[:, :7],
        changed_logits[:, :7],
        atol=2e-6,
        rtol=2e-6,
    )
    assert not torch.allclose(original_logits[:, 7:], changed_logits[:, 7:])


def test_persistent_state_changes_next_chunk_without_changing_tokens():
    torch.manual_seed(103)
    model = CortexSLM(tiny_config()).eval()
    tokens = torch.randint(0, 64, (2, 10))
    with torch.no_grad():
        _, state = model(tokens[:, :6], return_state=True)
        continued = model(tokens[:, 6:9], state=state)
        reset = model(tokens[:, 6:9])
    assert not torch.allclose(continued, reset)


def test_true_sparse_moe_executes_only_top_k_token_expert_assignments():
    torch.manual_seed(104)
    moe = TrulySparseMoE(d_model=24, num_experts=8, top_k=2, hidden=16)
    rows_seen = [0 for _ in moe.experts]
    hooks = []

    for expert_index, expert in enumerate(moe.experts):
        def record(module, args, index=expert_index):
            rows_seen[index] += int(args[0].shape[0])
        hooks.append(expert.register_forward_pre_hook(record))

    x = torch.randn(3, 11, 24, requires_grad=True)
    output = moe(x)
    loss = output.square().mean()
    loss.backward()
    for hook in hooks:
        hook.remove()

    token_count = 3 * 11
    assert sum(rows_seen) == token_count * 2
    assert sum(rows_seen) < token_count * 8
    assert moe.theoretical_executed_fraction == 0.25
    assert torch.isfinite(loss)
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_world_state_ablation_has_same_parameters_but_changes_computation():
    torch.manual_seed(105)
    model = CortexSLM(tiny_config()).eval()
    tokens = torch.randint(0, 64, (2, 8))
    before = parameter_count(model)
    with torch.no_grad():
        full = model(tokens)
        ablated = model(tokens, disable_world=True)
    after = parameter_count(model)
    assert before == after
    assert not torch.allclose(full, ablated)


def test_language_backward_is_finite():
    torch.manual_seed(106)
    model = CortexSLM(tiny_config())
    tokens = torch.randint(0, 64, (3, 9))
    targets = torch.randint(0, 64, (3, 9))
    logits = model(tokens)
    loss = F.cross_entropy(logits.reshape(-1, 64), targets.reshape(-1))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
