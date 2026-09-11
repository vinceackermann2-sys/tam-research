from __future__ import annotations

import inspect
from pathlib import Path

import torch
import torch.nn.functional as F

from architectures.cortex_s.fused_linear_ce_v1 import (
    FusedLinearCETrainingRunner,
    LIGER_KERNEL_VERSION,
    LIGER_KERNEL_WHEEL_SHA256,
    _one_optimizer_step_fused_linear_ce,
    chunked_linear_cross_entropy_reference,
    forward_training_features,
    fused_linear_ce_training_builder,
    integration_status,
)
from architectures.cortex_s.language_model import CortexSLM, CortexSLMConfig


ROOT = Path(__file__).resolve().parents[3]
CANDIDATE = ROOT / "architectures/cortex_s/fused_linear_ce_v1.py"


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


def test_chunked_reference_matches_full_linear_ce_for_uneven_chunks_and_gradients() -> None:
    torch.manual_seed(20260911)
    hidden_base = torch.randn(11, 8)
    weight_base = torch.randn(17, 8)
    targets = torch.randint(0, 17, (11,))

    ref_hidden = hidden_base.clone().requires_grad_(True)
    ref_weight = weight_base.clone().requires_grad_(True)
    ref_loss = F.cross_entropy(F.linear(ref_hidden, ref_weight).float(), targets)
    ref_loss.backward()
    ref_hidden_grad = ref_hidden.grad.detach().clone()
    ref_weight_grad = ref_weight.grad.detach().clone()

    for chunk_rows in (1, 3, 7, 11, 32):
        hidden = hidden_base.clone().requires_grad_(True)
        weight = weight_base.clone().requires_grad_(True)
        loss = chunked_linear_cross_entropy_reference(
            hidden,
            weight,
            targets,
            chunk_rows=chunk_rows,
        )
        loss.backward()
        torch.testing.assert_close(loss, ref_loss, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(hidden.grad, ref_hidden_grad, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(weight.grad, ref_weight_grad, rtol=1e-5, atol=1e-6)


def test_training_features_reproduce_public_forward_logits_exactly() -> None:
    torch.manual_seed(101)
    model = _tiny_model().eval()
    tokens = torch.randint(0, model.cfg.vocab_size, (2, 5))
    with torch.no_grad():
        hidden = forward_training_features(model, tokens)
        reconstructed = F.linear(hidden, model.lm_head.weight)
        public_logits = model(tokens)
    assert torch.equal(reconstructed, public_logits)


def test_runner_without_targets_is_transparent_public_logits_path() -> None:
    torch.manual_seed(102)
    model = _tiny_model().eval()
    runner = FusedLinearCETrainingRunner(
        model,
        force_torch_reference=True,
        reference_chunk_rows=3,
    ).eval()
    tokens = torch.randint(0, model.cfg.vocab_size, (2, 5))
    with torch.no_grad():
        assert torch.equal(runner(tokens), model(tokens))


def test_runner_cpu_loss_and_all_parameter_gradients_match_reference() -> None:
    torch.manual_seed(103)
    reference = _tiny_model().train()
    candidate = _tiny_model().train()
    candidate.load_state_dict(reference.state_dict())
    tokens = torch.randint(0, reference.cfg.vocab_size, (2, 5))
    targets = torch.randint(0, reference.cfg.vocab_size, (2, 5))

    reference_loss = F.cross_entropy(
        reference(tokens).float().reshape(-1, reference.cfg.vocab_size),
        targets.reshape(-1),
    )
    reference_loss.backward()

    runner = FusedLinearCETrainingRunner(
        candidate,
        force_torch_reference=True,
        reference_chunk_rows=3,
    )
    candidate_loss = runner(tokens, targets)
    candidate_loss.backward()

    torch.testing.assert_close(candidate_loss, reference_loss, rtol=1e-6, atol=1e-6)
    reference_grads = {name: parameter.grad for name, parameter in reference.named_parameters()}
    candidate_grads = {name: parameter.grad for name, parameter in candidate.named_parameters()}
    assert reference_grads.keys() == candidate_grads.keys()
    for name in reference_grads:
        assert reference_grads[name] is not None, name
        assert candidate_grads[name] is not None, name
        torch.testing.assert_close(
            candidate_grads[name],
            reference_grads[name],
            rtol=2e-5,
            atol=2e-6,
            msg=lambda message, name=name: f"gradient mismatch for {name}: {message}",
        )


def test_tied_lm_head_identity_is_preserved_by_runner() -> None:
    model = _tiny_model()
    runner = FusedLinearCETrainingRunner(model, force_torch_reference=True)
    assert model.lm_head.weight is model.token_emb.weight
    assert runner.model.lm_head.weight is runner.model.token_emb.weight
    assert runner.model is model


def test_training_builder_is_scoped_and_restores_live_trainer() -> None:
    import architectures.cortex_s.experiments.scale100m_2b.train as training_module
    import architectures.cortex_s.fused_linear_ce_v1 as candidate_module

    original_compile = training_module._compile_model
    original_step = training_module._one_optimizer_step
    with fused_linear_ce_training_builder():
        assert training_module._compile_model is candidate_module._compile_fused_linear_ce_runner
        assert training_module._one_optimizer_step is candidate_module._one_optimizer_step_fused_linear_ce
    assert training_module._compile_model is original_compile
    assert training_module._one_optimizer_step is original_step


def test_fused_optimizer_step_preserves_v6_single_host_readback_contract() -> None:
    source = inspect.getsource(_one_optimizer_step_fused_linear_ce)
    assert "runner(x, y) / training_module.GRAD_ACCUM_STEPS" in source
    assert "torch.nn.utils.clip_grad_norm_" in source
    assert "host_report = torch.stack(" in source
    assert source.count('.to(device="cpu")') == 1
    assert "running_value, finite_value = host_report.tolist()" in source
    assert source.index("host_report = torch.stack(") < source.index("optimizer.step()")
    assert "float(loss.detach())" not in source
    assert "if not torch.isfinite(loss)" not in source


def test_cuda_candidate_is_pinned_fused_and_never_materializes_full_logits() -> None:
    source = CANDIDATE.read_text(encoding="utf-8")
    assert 'LIGER_KERNEL_VERSION = "0.8.2"' in source
    assert LIGER_KERNEL_VERSION == "0.8.2"
    assert LIGER_KERNEL_WHEEL_SHA256 == "84c0a7bc9bf4d4cf8ea5ba89ff84d28686afc94215b220851d9f57dc87852741"
    assert "LigerFusedLinearCrossEntropyLoss" in source
    assert 'installed = importlib.metadata.version("liger-kernel")' in source
    assert "self._liger_loss(weight, flat_hidden, flat_targets)" in source
    cuda_path = source[source.index("if self._liger_loss is not None:"):source.index("return chunked_linear_cross_entropy_reference(")]
    assert "F.linear" not in cuda_path
    assert ".float()" not in cuda_path
    assert "logits" not in cuda_path.replace("logits tensor", "")


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
    assert status["default_model_forward_unchanged"] is True
    assert status["default_training_module_unchanged"] is True
    assert status["cuda_materializes_full_logits"] is False
    assert status["full_training_authorized"] is False
    assert status["scientific_claim_authorized"] is False
    assert status["gpu_dispatch_authorized"] is False
