from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import torch
import torch.nn.functional as F

from architectures.cortex_s.autocast_ce_no_explicit_fp32_v1 import (
    _one_optimizer_step_autocast_ce,
    autocast_ce_no_explicit_fp32_training_builder,
    integration_status,
)
from architectures.cortex_s.language_model import CortexSLM, CortexSLMConfig


ROOT = Path(__file__).resolve().parents[3]
CANDIDATE = ROOT / "architectures/cortex_s/autocast_ce_no_explicit_fp32_v1.py"
TRAINER = ROOT / "architectures/cortex_s/experiments/scale100m_2b/train.py"
EXPECTED_TRAINER_SHA256 = "d0d91de09bba8b7631913b3e9c0407ab2bb0114f7bba133c123ee5598d490461"


def _tiny_model() -> CortexSLM:
    return CortexSLM(
        CortexSLMConfig(
            vocab_size=31,
            d_model=16,
            n_layers=2,
            n_heads=4,
            max_seq_len=16,
            state_size=8,
            num_experts=4,
            top_k=2,
            expert_hidden=12,
            attention_every=2,
        )
    )


def test_cpu_bf16_autocast_cross_entropy_returns_fp32_without_explicit_logits_cast() -> None:
    torch.manual_seed(20260912)
    logits = torch.randn(13, 31, dtype=torch.float32)
    targets = torch.randint(0, 31, (13,))
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        bf16_logits = logits.to(torch.bfloat16)
        loss = F.cross_entropy(bf16_logits, targets)
    assert bf16_logits.dtype == torch.bfloat16
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)


def test_candidate_loss_and_all_parameter_gradients_match_explicit_fp32_baseline() -> None:
    torch.manual_seed(20260913)
    reference = _tiny_model().train()
    candidate = _tiny_model().train()
    candidate.load_state_dict(reference.state_dict())
    tokens = torch.randint(0, reference.cfg.vocab_size, (2, 5))
    targets = torch.randint(0, reference.cfg.vocab_size, (2, 5))

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        ref_logits = reference(tokens)
        reference_loss = F.cross_entropy(
            ref_logits.float().reshape(-1, ref_logits.size(-1)),
            targets.reshape(-1),
        )
    reference_loss.backward()

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        candidate_logits = candidate(tokens)
        candidate_loss = F.cross_entropy(
            candidate_logits.reshape(-1, candidate_logits.size(-1)),
            targets.reshape(-1),
        )
    candidate_loss.backward()

    assert candidate_loss.dtype == torch.float32
    torch.testing.assert_close(candidate_loss, reference_loss, rtol=1e-5, atol=1e-5)
    reference_grads = {name: parameter.grad for name, parameter in reference.named_parameters()}
    candidate_grads = {name: parameter.grad for name, parameter in candidate.named_parameters()}
    assert reference_grads.keys() == candidate_grads.keys()
    for name in reference_grads:
        assert reference_grads[name] is not None, name
        assert candidate_grads[name] is not None, name
        torch.testing.assert_close(
            candidate_grads[name],
            reference_grads[name],
            rtol=2e-4,
            atol=2e-5,
            msg=lambda message, name=name: f"gradient mismatch for {name}: {message}",
        )


def test_candidate_changes_only_pre_ce_cast_and_preserves_training_contract() -> None:
    source = inspect.getsource(_one_optimizer_step_autocast_ce)
    assert "with training_module._autocast(device):" in source
    assert "logits = runner(x)" in source
    assert "F.cross_entropy(" in source
    assert "logits.reshape(-1, logits.size(-1))" in source
    assert "logits.float()" not in source
    assert "training_module.GRAD_ACCUM_STEPS" in source
    assert "torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)" in source
    assert "host_report = torch.stack(" in source
    assert source.count('.to(device="cpu")') == 1
    assert "running_value, finite_value = host_report.tolist()" in source
    assert source.index("host_report = torch.stack(") < source.index("optimizer.step()")
    assert "if not bool(finite_value):" in source
    assert "optimizer.step()" in source


def test_builder_is_scoped_and_restores_default_trainer_exactly() -> None:
    import architectures.cortex_s.autocast_ce_no_explicit_fp32_v1 as candidate_module
    import architectures.cortex_s.experiments.scale100m_2b.train as training_module

    original_step = training_module._one_optimizer_step
    original_compile = training_module._compile_model
    with autocast_ce_no_explicit_fp32_training_builder():
        assert training_module._one_optimizer_step is candidate_module._one_optimizer_step_autocast_ce
        assert training_module._compile_model is original_compile
    assert training_module._one_optimizer_step is original_step
    assert training_module._compile_model is original_compile


def test_default_production_trainer_bytes_remain_frozen() -> None:
    digest = hashlib.sha256(TRAINER.read_bytes()).hexdigest()
    assert digest == EXPECTED_TRAINER_SHA256


def test_zero_credit_candidate_contains_no_paid_or_scientific_authority() -> None:
    source = CANDIDATE.read_text(encoding="utf-8")
    lowered = source.lower()
    assert "import modal" not in lowered
    assert "modal.run" not in lowered
    assert "train_full_2b" not in source
    assert "8_100" not in source
    assert "48_131" not in source
    assert "48_132" not in source
    assert "48_133" not in source
    status = integration_status()
    assert status["classification"] == "ZERO_CREDIT_SYSTEMS_OPTIMIZATION_ONLY"
    assert status["systems_variant"] == "autocast_ce_no_explicit_fp32_v1"
    assert status["default_training_module_unchanged"] is True
    assert status["explicit_full_logits_fp32_cast"] is False
    assert status["cross_entropy_inside_autocast"] is True
    assert status["single_host_readback_preserved"] is True
    assert status["full_training_authorized"] is False
    assert status["scientific_claim_authorized"] is False
    assert status["gpu_dispatch_authorized"] is False
