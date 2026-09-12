from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

import architectures.cortex_s.compiled_explicit_fp32_ce_v1 as candidate_module
from architectures.cortex_s.compiled_explicit_fp32_ce_v1 import (
    CLASSIFICATION,
    LOSS_COMPILE_MODE,
    LOSS_FULLGRAPH,
    SYSTEMS_VARIANT,
    _one_optimizer_step_compiled_explicit_fp32_ce,
    compiled_explicit_fp32_ce_training_builder,
    compiled_explicit_fp32_cross_entropy,
    explicit_fp32_cross_entropy,
    integration_status,
)
from architectures.cortex_s.language_model import CortexSLM, CortexSLMConfig


ROOT = Path(__file__).resolve().parents[3]
CANDIDATE = ROOT / "architectures/cortex_s/compiled_explicit_fp32_ce_v1.py"
TRAINER = ROOT / "architectures/cortex_s/experiments/scale100m_2b/train.py"
TRAINER_SHA256 = "d0d91de09bba8b7631913b3e9c0407ab2bb0114f7bba133c123ee5598d490461"


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


def test_explicit_fp32_loss_is_literal_production_reference() -> None:
    source = inspect.getsource(explicit_fp32_cross_entropy)
    trainer = TRAINER.read_text(encoding="utf-8")

    exact = "logits.float().reshape(-1, logits.size(-1))"
    assert exact in source
    assert exact in trainer
    assert "targets.reshape(-1)" in source
    assert "y.reshape(-1)" in trainer
    assert "F.cross_entropy(" in source
    assert "F.cross_entropy(" in trainer
    assert "torch.autocast" not in source


def test_compiled_loss_and_all_parameter_gradients_match_production_reference() -> None:
    """Exercise the actual torch.compile wrapper, not merely its eager helper."""

    torch.manual_seed(20260912)
    reference = _tiny_model().train()
    candidate = _tiny_model().train()
    candidate.load_state_dict(reference.state_dict())
    tokens = torch.randint(0, reference.cfg.vocab_size, (3, 7))
    targets = torch.randint(0, reference.cfg.vocab_size, (3, 7))

    reference_logits = reference(tokens)
    reference_loss = F.cross_entropy(
        reference_logits.float().reshape(-1, reference_logits.size(-1)),
        targets.reshape(-1),
    )
    reference_loss.backward()

    candidate_logits = candidate(tokens)
    candidate_loss = compiled_explicit_fp32_cross_entropy(candidate_logits, targets)
    candidate_loss.backward()

    torch.testing.assert_close(candidate_loss, reference_loss, rtol=1e-7, atol=1e-7)
    reference_grads = {name: parameter.grad for name, parameter in reference.named_parameters()}
    candidate_grads = {name: parameter.grad for name, parameter in candidate.named_parameters()}
    assert reference_grads.keys() == candidate_grads.keys()
    for name in reference_grads:
        assert reference_grads[name] is not None, name
        assert candidate_grads[name] is not None, name
        torch.testing.assert_close(
            candidate_grads[name],
            reference_grads[name],
            rtol=1e-6,
            atol=1e-7,
            msg=lambda message, name=name: f"compiled gradient mismatch for {name}: {message}",
        )


def test_loss_only_compiler_is_frozen_fullgraph_max_autotune() -> None:
    source = CANDIDATE.read_text(encoding="utf-8")
    lowered = source.lower()

    assert LOSS_COMPILE_MODE == "max-autotune-no-cudagraphs"
    assert LOSS_FULLGRAPH is True
    assert "compiled_explicit_fp32_cross_entropy = torch.compile(" in source
    assert "explicit_fp32_cross_entropy," in source
    assert "mode=LOSS_COMPILE_MODE" in source
    assert "fullgraph=LOSS_FULLGRAPH" in source
    assert "torch.compiler.disable" not in source
    assert "liger_kernel" not in lowered
    assert "ligerfused" not in lowered
    assert "from liger" not in lowered
    assert "import liger" not in lowered


def test_exact_production_optimizer_step_token_contract() -> None:
    import architectures.cortex_s.experiments.scale100m_2b.train as training_module

    assert training_module.GRAD_ACCUM_STEPS == 2
    assert training_module.MICRO_BATCH_SIZE == 64
    assert training_module.SEQ_LEN == 512
    assert (
        training_module.GRAD_ACCUM_STEPS
        * training_module.MICRO_BATCH_SIZE
        * training_module.SEQ_LEN
        == 65_536
    )


def test_builder_changes_only_optimizer_step_and_restores_exactly() -> None:
    import architectures.cortex_s.experiments.scale100m_2b.train as training_module

    original_compile = training_module._compile_model
    original_step = training_module._one_optimizer_step
    with compiled_explicit_fp32_ce_training_builder():
        assert training_module._compile_model is original_compile
        assert (
            training_module._one_optimizer_step
            is candidate_module._one_optimizer_step_compiled_explicit_fp32_ce
        )
    assert training_module._compile_model is original_compile
    assert training_module._one_optimizer_step is original_step


def test_optimizer_step_preserves_production_hot_path_contract() -> None:
    source = inspect.getsource(_one_optimizer_step_compiled_explicit_fp32_ce)

    assert "for _ in range(training_module.GRAD_ACCUM_STEPS):" in source
    assert "training_module.MICRO_BATCH_SIZE" in source
    assert "training_module.SEQ_LEN" in source
    assert "with training_module._autocast(device):" in source
    assert "logits = runner(x)" in source
    assert "compiled_explicit_fp32_cross_entropy(logits, y)" in source
    assert "/ training_module.GRAD_ACCUM_STEPS" in source
    assert "loss.backward()" in source
    assert "torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)" in source
    assert source.count('.to(device="cpu")') == 1
    assert "running_value, finite_value = host_report.tolist()" in source
    assert source.index("host_report = torch.stack(") < source.index("optimizer.step()")
    assert source.index("if not bool(finite_value):") < source.index("optimizer.step()")


def test_nonfinite_loss_aborts_before_optimizer_update(monkeypatch: pytest.MonkeyPatch) -> None:
    import architectures.cortex_s.experiments.scale100m_2b.train as training_module

    class TinyRunner(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([0.25, -0.25]))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            del x
            return self.weight.view(1, 1, 2)

    class FakeTrainData:
        def __init__(self) -> None:
            self.calls: list[tuple[int, int]] = []

        def batch(
            self,
            batch_size: int,
            seq_len: int,
            generator: torch.Generator,
            device: torch.device,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            del generator
            self.calls.append((batch_size, seq_len))
            x = torch.zeros((1, 1), dtype=torch.long, device=device)
            y = torch.zeros((1, 1), dtype=torch.long, device=device)
            return x, y

    class CountingSGD(torch.optim.SGD):
        def __init__(self, params) -> None:
            super().__init__(params, lr=0.1)
            self.step_calls = 0

        def step(self, closure=None):
            self.step_calls += 1
            return super().step(closure=closure)

    def nonfinite_compiled_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        del targets
        return logits.sum() * torch.tensor(float("inf"), device=logits.device)

    assert training_module.GRAD_ACCUM_STEPS == 2
    assert training_module.MICRO_BATCH_SIZE == 64
    assert training_module.SEQ_LEN == 512
    monkeypatch.setattr(
        candidate_module,
        "compiled_explicit_fp32_cross_entropy",
        nonfinite_compiled_loss,
    )

    model = TinyRunner()
    optimizer = CountingSGD(model.parameters())
    train_data = FakeTrainData()
    generator = torch.Generator(device="cpu").manual_seed(123)

    with pytest.raises(FloatingPointError):
        _one_optimizer_step_compiled_explicit_fp32_ce(
            model=model,
            runner=model,
            optimizer=optimizer,
            train_data=train_data,
            generator=generator,
            device=torch.device("cpu"),
            lr=0.1,
        )

    assert optimizer.step_calls == 0
    assert train_data.calls == [(64, 512), (64, 512)]


def test_production_trainer_is_unchanged() -> None:
    digest = hashlib.sha256(TRAINER.read_bytes()).hexdigest()
    assert digest == TRAINER_SHA256
    source = TRAINER.read_text(encoding="utf-8")
    assert "return torch.compile(model, mode=COMPILE_MODE, fullgraph=False)" in source
    assert "logits.float().reshape(-1, logits.size(-1))" in source


def test_candidate_has_no_paid_scientific_or_external_fused_loss_authority() -> None:
    source = CANDIDATE.read_text(encoding="utf-8")
    lowered = source.lower()

    assert "import modal" not in lowered
    assert "modal.run" not in lowered
    assert "h100" not in lowered
    assert "liger_kernel" not in lowered
    assert "ligerfused" not in lowered
    assert "from liger" not in lowered
    assert "import liger" not in lowered
    assert "train_full_2b" not in source
    assert "8_100" not in source
    assert "48_131" not in source
    assert "48_132" not in source
    assert "48_133" not in source
    assert "2_026_091_013" not in source

    status = integration_status()
    assert SYSTEMS_VARIANT == "compiled_explicit_fp32_ce_v1"
    assert CLASSIFICATION == "ZERO_CREDIT_SYSTEMS_OPTIMIZATION_ONLY"
    assert status["systems_variant"] == SYSTEMS_VARIANT
    assert status["classification"] == CLASSIFICATION
    assert status["loss_compile_mode"] == "max-autotune-no-cudagraphs"
    assert status["loss_fullgraph"] is True
    assert status["explicit_full_logits_fp32_cast_preserved"] is True
    assert status["cross_entropy_semantics_preserved"] is True
    assert status["production_model_compile_unchanged"] is True
    assert status["default_training_module_unchanged"] is True
    assert status["single_host_readback_preserved"] is True
    assert status["liger_used"] is False
    assert status["full_training_authorized"] is False
    assert status["scientific_claim_authorized"] is False
    assert status["gpu_dispatch_authorized"] is False
