from __future__ import annotations

import copy
import inspect

import pytest
import torch
import torch.nn.functional as F

import architectures.cortex_s.experiments.scale100m_2b.train as training


class _TinyLM(torch.nn.Module):
    def __init__(self, vocab_size: int = 17, width: int = 8) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, width)
        self.projection = torch.nn.Linear(width, vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.projection(self.embedding(tokens))


class _FixedBatches:
    def __init__(self, batches: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
        self._batches = [(x.clone(), y.clone()) for x, y in batches]
        self._index = 0

    def batch(
        self,
        _batch_size: int,
        _seq_len: int,
        _generator: torch.Generator,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = self._batches[self._index]
        self._index += 1
        return x.to(device), y.to(device)


def _batches(vocab_size: int = 17) -> list[tuple[torch.Tensor, torch.Tensor]]:
    generator = torch.Generator(device="cpu").manual_seed(2026091101)
    result: list[tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(training.GRAD_ACCUM_STEPS):
        x = torch.randint(0, vocab_size, (2, 5), generator=generator)
        y = torch.randint(0, vocab_size, (2, 5), generator=generator)
        result.append((x, y))
    return result


def _reference_pre_repair_step(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_data: _FixedBatches,
    generator: torch.Generator,
    device: torch.device,
    lr: float,
) -> float:
    """Exact CPU reference for the pre-#868 scalar semantics."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    running = 0.0
    for _ in range(training.GRAD_ACCUM_STEPS):
        x, y = train_data.batch(training.MICRO_BATCH_SIZE, training.SEQ_LEN, generator, device)
        logits = model(x)
        loss = F.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)),
            y.reshape(-1),
        ) / training.GRAD_ACCUM_STEPS
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite CORTEX-S training loss")
        loss.backward()
        running += float(loss.detach())
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if not torch.isfinite(grad_norm):
        raise FloatingPointError("non-finite CORTEX-S gradient norm")
    for group in optimizer.param_groups:
        group["lr"] = lr
    optimizer.step()
    return running


def _parameter_snapshot(model: torch.nn.Module) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in model.parameters()]


def test_hot_path_has_one_bundled_host_readback_and_preserves_float_api() -> None:
    source = inspect.getsource(training._one_optimizer_step)

    assert ") -> float:" in source
    assert "float(loss.detach())" not in source
    assert "if not torch.isfinite(loss)" not in source
    assert "if not torch.isfinite(grad_norm)" not in source
    assert "running_loss" in source
    assert "all_finite" in source
    assert "torch.logical_and(all_finite, torch.isfinite(grad_norm))" in source
    assert "host_report = torch.stack(" in source
    assert source.count('.to(device="cpu")') == 1
    assert "running_value, finite_value = host_report.tolist()" in source
    assert source.index("host_report = torch.stack(") < source.index("optimizer.step()")
    assert source.index("if not bool(finite_value):") < source.index("optimizer.step()")


def test_finite_step_matches_pre_repair_loss_and_parameter_update() -> None:
    torch.manual_seed(2026091102)
    reference_model = _TinyLM()
    candidate_model = copy.deepcopy(reference_model)
    batches = _batches()
    device = torch.device("cpu")
    lr = 0.03125

    reference_optimizer = torch.optim.SGD(reference_model.parameters(), lr=lr)
    candidate_optimizer = torch.optim.SGD(candidate_model.parameters(), lr=lr)
    reference_generator = torch.Generator(device="cpu").manual_seed(101)
    candidate_generator = torch.Generator(device="cpu").manual_seed(101)

    reference_loss = _reference_pre_repair_step(
        model=reference_model,
        optimizer=reference_optimizer,
        train_data=_FixedBatches(batches),
        generator=reference_generator,
        device=device,
        lr=lr,
    )
    candidate_loss = training._one_optimizer_step(
        model=candidate_model,  # type: ignore[arg-type]
        runner=candidate_model,
        optimizer=candidate_optimizer,
        train_data=_FixedBatches(batches),  # type: ignore[arg-type]
        generator=candidate_generator,
        device=device,
        lr=lr,
    )

    assert isinstance(candidate_loss, float)
    assert candidate_loss == pytest.approx(reference_loss, rel=1e-6, abs=1e-7)
    for reference_parameter, candidate_parameter in zip(
        reference_model.parameters(), candidate_model.parameters(), strict=True
    ):
        torch.testing.assert_close(candidate_parameter, reference_parameter, rtol=0.0, atol=0.0)


def test_nonfinite_loss_aborts_before_optimizer_update() -> None:
    torch.manual_seed(2026091103)
    model = _TinyLM()
    batches = _batches()
    device = torch.device("cpu")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    before = _parameter_snapshot(model)

    class _NaNRunner(torch.nn.Module):
        def __init__(self, wrapped: _TinyLM) -> None:
            super().__init__()
            self.wrapped = wrapped

        def forward(self, tokens: torch.Tensor) -> torch.Tensor:
            return self.wrapped(tokens) * torch.tensor(float("nan"), device=tokens.device)

    with pytest.raises(FloatingPointError, match="non-finite CORTEX-S training loss or gradient norm"):
        training._one_optimizer_step(
            model=model,  # type: ignore[arg-type]
            runner=_NaNRunner(model),
            optimizer=optimizer,
            train_data=_FixedBatches(batches),  # type: ignore[arg-type]
            generator=torch.Generator(device="cpu").manual_seed(102),
            device=device,
            lr=0.05,
        )

    for old, current in zip(before, model.parameters(), strict=True):
        torch.testing.assert_close(current, old, rtol=0.0, atol=0.0)


def test_nonfinite_gradient_aborts_before_optimizer_update() -> None:
    torch.manual_seed(2026091104)
    model = _TinyLM()
    batches = _batches()
    device = torch.device("cpu")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    before = _parameter_snapshot(model)
    handle = model.projection.weight.register_hook(
        lambda gradient: torch.full_like(gradient, float("inf"))
    )
    try:
        with pytest.raises(FloatingPointError, match="non-finite CORTEX-S training loss or gradient norm"):
            training._one_optimizer_step(
                model=model,  # type: ignore[arg-type]
                runner=model,
                optimizer=optimizer,
                train_data=_FixedBatches(batches),  # type: ignore[arg-type]
                generator=torch.Generator(device="cpu").manual_seed(103),
                device=device,
                lr=0.05,
            )
    finally:
        handle.remove()

    for old, current in zip(before, model.parameters(), strict=True):
        torch.testing.assert_close(current, old, rtol=0.0, atol=0.0)
