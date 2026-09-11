from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
import torch.nn.functional as F

from architectures.cortex_s.language_model import CortexSLM


SYSTEMS_VARIANT = "autocast_ce_no_explicit_fp32_v1"
CLASSIFICATION = "ZERO_CREDIT_SYSTEMS_OPTIMIZATION_ONLY"


def _one_optimizer_step_autocast_ce(
    *,
    model: CortexSLM,
    runner: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_data,
    generator: torch.Generator,
    device: torch.device,
    lr: float,
) -> float:
    """Production optimizer-step contract with only the explicit logits FP32 cast removed.

    Cross entropy remains inside the existing autocast region, so PyTorch owns the
    operation's precision policy. All batch, reduction, finite-check, gradient-clip,
    host-readback, optimizer, and learning-rate semantics are otherwise preserved.
    """

    import architectures.cortex_s.experiments.scale100m_2b.train as training_module

    model.train()
    runner.train()
    optimizer.zero_grad(set_to_none=True)
    running_loss: torch.Tensor | None = None
    all_finite: torch.Tensor | None = None
    for _ in range(training_module.GRAD_ACCUM_STEPS):
        x, y = train_data.batch(
            training_module.MICRO_BATCH_SIZE,
            training_module.SEQ_LEN,
            generator,
            device,
        )
        with training_module._autocast(device):
            logits = runner(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
            ) / training_module.GRAD_ACCUM_STEPS
        detached_loss = loss.detach()
        loss_finite = torch.isfinite(detached_loss)
        running_loss = detached_loss if running_loss is None else running_loss + detached_loss
        all_finite = loss_finite if all_finite is None else torch.logical_and(all_finite, loss_finite)
        loss.backward()

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if running_loss is None or all_finite is None:
        raise RuntimeError("gradient accumulation produced no CORTEX-S microbatches")
    all_finite = torch.logical_and(all_finite, torch.isfinite(grad_norm))

    # Preserve v6's single intentional device->CPU synchronization exactly.
    host_report = torch.stack(
        (running_loss, all_finite.to(dtype=running_loss.dtype))
    ).to(device="cpu")
    running_value, finite_value = host_report.tolist()
    if not bool(finite_value):
        raise FloatingPointError("non-finite CORTEX-S training loss or gradient norm")

    for group in optimizer.param_groups:
        group["lr"] = lr
    optimizer.step()
    return float(running_value)


@contextmanager
def autocast_ce_no_explicit_fp32_training_builder() -> Iterator[None]:
    """Temporarily select the v10 training-only CE systems candidate.

    The default production trainer is not edited. This context changes only the
    optimizer-step function for the duration of an explicitly selected systems run.
    It allocates no GPU, invokes no Modal job, and consumes no experiment seed.
    """

    import architectures.cortex_s.experiments.scale100m_2b.train as training_module

    original_step = training_module._one_optimizer_step
    training_module._one_optimizer_step = _one_optimizer_step_autocast_ce
    try:
        yield
    finally:
        training_module._one_optimizer_step = original_step


def integration_status() -> dict[str, object]:
    return {
        "systems_variant": SYSTEMS_VARIANT,
        "classification": CLASSIFICATION,
        "default_training_module_unchanged": True,
        "explicit_full_logits_fp32_cast": False,
        "cross_entropy_inside_autocast": True,
        "single_host_readback_preserved": True,
        "full_training_authorized": False,
        "scientific_claim_authorized": False,
        "gpu_dispatch_authorized": False,
    }
