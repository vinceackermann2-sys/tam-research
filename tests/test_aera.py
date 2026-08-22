import torch

from tam_research.aera import AERAConfig, AERACore


def tiny_cfg() -> AERAConfig:
    return AERAConfig(
        d_model=32,
        n_experts=4,
        top_k_experts=1,
        expert_mult=2,
        memory_dim=8,
        max_reason_steps=3,
        halt_threshold=0.55,
    )


def test_sparse_expert_layer_executes_only_top_k_assignments():
    torch.manual_seed(1)
    cfg = tiny_cfg()
    model = AERACore(cfg).eval()
    x = torch.randn(2, 5, cfg.d_model)
    model(x, update_memory=False)
    stats = model.compute_stats()["experts"]
    assert stats is not None
    assert stats["assignments"] == 2 * 5 * cfg.top_k_experts
    assert stats["active_fraction_of_experts_per_token"] == 0.25


def test_fast_memory_updates_without_changing_model_parameters():
    torch.manual_seed(2)
    cfg = tiny_cfg()
    model = AERACore(cfg).eval()
    x = torch.randn(1, 6, cfg.d_model)
    before = [p.detach().clone() for p in model.parameters()]
    _, state0 = model(x, update_memory=False)
    _, state1 = model(x, state0, update_memory=True)
    after = [p.detach() for p in model.parameters()]

    for a, b in zip(before, after):
        torch.testing.assert_close(a, b)
    assert not torch.allclose(state0.memory.matrix, state1.memory.matrix)


def test_state_is_explicit_and_fresh_runs_do_not_share_memory():
    torch.manual_seed(3)
    cfg = tiny_cfg()
    model = AERACore(cfg).eval()
    history = torch.randn(1, 4, cfg.d_model)
    query = torch.randn(1, 3, cfg.d_model)

    _, learned_state = model(history, update_memory=True)
    with_history, _ = model(query, learned_state, update_memory=False)
    fresh_a, fresh_state_a = model(query, None, update_memory=False)
    fresh_b, fresh_state_b = model(query, None, update_memory=False)

    torch.testing.assert_close(fresh_a, fresh_b)
    torch.testing.assert_close(fresh_state_a.memory.matrix, fresh_state_b.memory.matrix)
    assert not torch.allclose(with_history, fresh_a)


def test_adaptive_reasoning_respects_hard_step_budget():
    torch.manual_seed(4)
    cfg = tiny_cfg()
    model = AERACore(cfg).eval()
    x = torch.randn(2, 7, cfg.d_model)
    model(x, update_memory=False)
    stats = model.compute_stats()["reasoning_steps"]
    assert stats is not None
    assert 1.0 <= stats["min"] <= cfg.max_reason_steps
    assert 1.0 <= stats["mean"] <= cfg.max_reason_steps
    assert 1.0 <= stats["max"] <= cfg.max_reason_steps
