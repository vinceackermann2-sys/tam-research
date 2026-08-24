import torch

from tam_research import aera_hardware_core_v22 as core
from tam_research import aera_real_language_v22 as v22
from tam_research import aera_real_language_v22_efficiency as eff
from tam_research.aera_hardware_core import HardwareAERAConfig
from tam_research.aera_hardware_core_v22 import HardwareAwareAERATextLMV22
from tam_research.aera_real_language import GRAD_ACCUM


def _tiny_model() -> HardwareAwareAERATextLMV22:
    cfg = HardwareAERAConfig(
        vocab_size=97,
        d_model=24,
        n_stages=4,
        n_heads=4,
        chunk_size=v22.CHUNK_SIZE,
        n_experts=4,
        max_active_experts=2,
        expert_mult=2,
        memory_dim=16,
        max_reason_steps=2,
        block_size=2,
    )
    return HardwareAwareAERATextLMV22(cfg)


def test_auxiliary_budget_is_frozen_per_optimizer_step_not_per_microbatch():
    assert GRAD_ACCUM == 4
    assert eff.MAX_MEMORY_AUX_EVENTS_PER_OPTIMIZER_STEP == 1024
    assert eff.MAX_MEMORY_AUX_EVENTS_PER_MICROBATCH == 256
    assert (
        eff.MAX_MEMORY_AUX_EVENTS_PER_MICROBATCH * GRAD_ACCUM
        == eff.MAX_MEMORY_AUX_EVENTS_PER_OPTIMIZER_STEP
    )
    protocol = eff.efficiency_protocol()
    assert protocol["architecture_changed"] is False
    assert protocol["memory_equations_changed"] is False
    assert protocol["objective_weights_changed"] is False
    assert protocol["memory_aux_events_per_optimizer_step"] == 1024
    assert protocol["gpu_training_authorized"] is False


def test_corrected_auxiliary_default_caps_each_microbatch_at_256_events():
    torch.manual_seed(8401)
    model = _tiny_model().train()
    # Two 256-token examples expose 510 adjacent pairs, so the 256 cap is active.
    tokens = torch.randint(0, model.cfg.vocab_size, (2, v22.CHUNK_SIZE))
    terms = eff.memory_auxiliary_terms(model, tokens, step=9)
    assert int(terms["memory_aux_events"].item()) == 256


def test_corrected_auxiliary_refuses_budget_above_per_microbatch_cap():
    torch.manual_seed(8402)
    model = _tiny_model().eval()
    tokens = torch.randint(0, model.cfg.vocab_size, (2, v22.CHUNK_SIZE))
    try:
        eff.memory_auxiliary_terms(
            model,
            tokens,
            step=0,
            max_events=eff.MAX_MEMORY_AUX_EVENTS_PER_MICROBATCH + 1,
        )
    except ValueError as exc:
        assert "per-microbatch budget" in str(exc)
    else:
        raise AssertionError("memory auxiliary accepted an over-budget microbatch")


def _fixture():
    torch.manual_seed(8403)
    batch, candidates, dim = 2, 5, 6
    matrix = torch.randn(batch, dim, dim, dtype=torch.float32) * 0.03
    inverse = torch.eye(dim, dtype=torch.float32).expand(batch, -1, -1).clone()
    keys = torch.nn.functional.normalize(
        torch.randn(batch, candidates, dim, dtype=torch.float32), dim=-1
    )
    targets = torch.tanh(torch.randn(batch, candidates, dim, dtype=torch.float32))
    strengths = torch.sigmoid(torch.randn(batch, candidates, 1, dtype=torch.float32))
    return matrix, inverse, keys, targets, strengths


def _run_with_grads(fn):
    inputs = [x.detach().clone().requires_grad_(True) for x in _fixture()]
    matrix, inverse = fn(*inputs)
    loss = matrix.square().mean() + 0.3 * inverse.square().mean()
    grads = torch.autograd.grad(loss, inputs)
    return matrix.detach(), inverse.detach(), [g.detach() for g in grads]


def test_torch_compile_exact_dual_delta_matches_eager_forward_and_gradients():
    eager_m, eager_p, eager_grads = _run_with_grads(
        core.interference_corrected_dual_delta_update
    )
    compiled = eff.make_compiled_dual_delta_update()
    got_m, got_p, got_grads = _run_with_grads(compiled)
    torch.testing.assert_close(got_m, eager_m, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(got_p, eager_p, atol=2e-5, rtol=2e-5)
    for got, expected in zip(got_grads, eager_grads):
        torch.testing.assert_close(got, expected, atol=5e-5, rtol=5e-5)


def test_efficiency_cpu_preflight_keeps_scientific_v22_unchanged():
    result = eff.cpu_preflight()
    assert result["scientific_architecture_changed"] is False
    assert result["scientific_objective_changed"] is False
    assert result["memory_off_logits_and_stream_bit_exact_v19"] is True
    assert result["memory_aux_events_per_microbatch"] == 256
    assert result["memory_aux_events_per_optimizer_step"] == 1024
    assert result["frozen_memory_aux_optimizer_step_cap"] == 1024
    assert result["seed8391_rerun_authorized"] is False
