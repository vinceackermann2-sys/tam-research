from __future__ import annotations

import copy
from pathlib import Path

import torch

from architectures.cortex_s import production_scan_integration_v1 as integration
from architectures.cortex_s.language_model import (
    CortexSLM,
    CortexSLMConfig,
    PersistentWorldState,
    affine_scan,
    parameter_count,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


def _tiny_config() -> CortexSLMConfig:
    return CortexSLMConfig(
        vocab_size=97,
        d_model=32,
        n_layers=2,
        n_heads=4,
        max_seq_len=16,
        state_size=8,
        num_experts=2,
        top_k=1,
        expert_hidden=16,
        attention_every=2,
    )


def test_cpu_dispatcher_is_existing_production_scan_with_matching_gradients() -> None:
    generator = torch.Generator().manual_seed(82001)
    a_left = torch.sigmoid(torch.randn(2, 13, 5, generator=generator)).requires_grad_()
    b_left = torch.randn(2, 13, 5, generator=generator).requires_grad_()
    i_left = torch.randn(2, 5, generator=generator).requires_grad_()
    a_right = a_left.detach().clone().requires_grad_()
    b_right = b_left.detach().clone().requires_grad_()
    i_right = i_left.detach().clone().requires_grad_()

    expected = affine_scan(a_left, b_left, i_left)
    actual = integration.production_affine_scan(a_right, b_right, i_right)
    assert torch.equal(actual, expected)

    grad_expected = torch.autograd.grad(expected.square().sum(), (a_left, b_left, i_left))
    grad_actual = torch.autograd.grad(actual.square().sum(), (a_right, b_right, i_right))
    for left, right in zip(grad_expected, grad_actual):
        assert torch.equal(left, right)


def test_cpu_fallback_never_uses_candidate_sequential_or_triton_path(monkeypatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("candidate path must not run for ordinary CPU fallback")

    monkeypatch.setattr(integration, "affine_scan_triton_candidate", forbidden)
    a = torch.full((2, 7, 3), 0.75)
    b = torch.randn(2, 7, 3, generator=torch.Generator().manual_seed(82002))
    expected = affine_scan(a, b)
    actual = integration.production_affine_scan(a, b)
    assert torch.equal(actual, expected)


def test_dispatch_seam_can_be_falsified_without_claiming_cuda_validation(monkeypatch) -> None:
    calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]] = []

    def fake_candidate(a, b, initial=None):
        calls.append((a, b, initial))
        return affine_scan(a, b, initial)

    monkeypatch.setattr(integration, "_should_use_triton", lambda tensor: True)
    monkeypatch.setattr(integration, "affine_scan_triton_candidate", fake_candidate)
    a = torch.full((1, 5, 2), 0.8)
    b = torch.ones_like(a)
    initial = torch.zeros(1, 2)
    result = integration.production_affine_scan(a, b, initial)
    assert len(calls) == 1
    assert torch.equal(result, affine_scan(a, b, initial))


def test_world_state_conversion_preserves_cpu_outputs_carry_and_parameter_objects() -> None:
    torch.manual_seed(82003)
    model = CortexSLM(_tiny_config())
    reference = copy.deepcopy(model)
    before_count = parameter_count(model)
    before_parameters = {name: parameter for name, parameter in model.named_parameters()}

    integration.convert_world_state_to_triton_scan(model)

    after_parameters = {name: parameter for name, parameter in model.named_parameters()}
    assert parameter_count(model) == before_count
    assert before_parameters.keys() == after_parameters.keys()
    assert all(before_parameters[name] is after_parameters[name] for name in before_parameters)
    assert all(isinstance(block.world, integration.ProductionScanPersistentWorldState) for block in model.blocks)

    x_left = torch.randn(2, 9, 32, generator=torch.Generator().manual_seed(82004)).requires_grad_()
    x_right = x_left.detach().clone().requires_grad_()
    state_left = torch.randn(2, 8, generator=torch.Generator().manual_seed(82005)).requires_grad_()
    state_right = state_left.detach().clone().requires_grad_()

    expected = reference.blocks[0].world(x_left, state_left)
    actual = model.blocks[0].world(x_right, state_right)
    for left, right in zip(expected, actual):
        assert torch.equal(left, right)

    grad_expected = torch.autograd.grad(
        expected[0].square().sum() + expected[1].square().sum(),
        (x_left, state_left),
    )
    grad_actual = torch.autograd.grad(
        actual[0].square().sum() + actual[1].square().sum(),
        (x_right, state_right),
    )
    for left, right in zip(grad_expected, grad_actual):
        assert torch.equal(left, right)


def test_conversion_is_explicitly_single_use() -> None:
    model = CortexSLM(_tiny_config())
    integration.convert_world_state_to_triton_scan(model)
    try:
        integration.convert_world_state_to_triton_scan(model)
    except TypeError as exc:
        assert "single-use" in str(exc)
    else:
        raise AssertionError("second scan conversion must fail closed")


def test_status_binds_consumed_817_and_authorizes_no_spend_or_training() -> None:
    status = integration.integration_status()
    assert status["classification"] == "ENGINEERING_PRODUCTION_INTEGRATION_ZERO_CREDIT_ONLY"
    assert status["consumed_predecessor_issue"] == 817
    assert status["consumed_predecessor_run"] == 34_450_068_844
    assert status["consumed_predecessor_job"] == 102_783_560_405
    assert status["consumed_predecessor_seed"] == 2_026_090_907
    assert status["isolated_scan_speedup"] == 3.055218648823612
    assert status["gpu_dispatch_authorized"] is False
    assert status["production_preflight_authorized"] is False
    assert status["full_training_authorized"] is False
    assert status["scientific_claim_authorized"] is False


def test_integration_source_adds_no_modal_gpu_or_full_training_entrypoint() -> None:
    path = REPO_ROOT / "architectures/cortex_s/production_scan_integration_v1.py"
    source = path.read_text(encoding="utf-8")
    assert "import modal" not in source
    assert "modal." not in source
    assert 'gpu="H100' not in source
    assert ".remote(" not in source
    assert "train_full" not in source
    assert "RESULT_ROOT" not in source
    assert "trigger_title" not in source
    assert "build_memory_lean_grouped_cortex_100m" in source
    assert "EXPECTED_CORTEX_PARAMS" in source


def test_historical_language_model_remains_unmodified_by_additive_candidate() -> None:
    source = (REPO_ROOT / "architectures/cortex_s/language_model.py").read_text(encoding="utf-8")
    assert "production_scan_integration_v1" not in source
    assert "affine_scan_triton_candidate" not in source
    assert "states = affine_scan(keep, (1.0 - keep) * candidate, state)" in source
