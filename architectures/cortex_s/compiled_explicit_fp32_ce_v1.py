from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
import torch.nn.functional as F

from architectures.cortex_s.language_model import CortexSLM


SYSTEMS_VARIANT = "compiled_explicit_fp32_ce_v1"
CLASSIFICATION = "ZERO_CREDIT_SYSTEMS_OPTIMIZATION_ONLY"
LOSS_COMPILE_MODE = "max-autotune-no-cudagraphs"
LOSS_FULLGRAPH = True


def explicit_fp32_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Exact production CE semantics isolated behind a loss-only compile seam."""

    return F.cross_entropy(
        logits.float().reshape(-1, logits.size(-1)),
        targets.reshape(-1),
    )


compiled_explicit_fp32_cross_entropy = torch.compile(
    explicit_fp32_cross_entropy,
    mode=LOSS_COMPILE_MODE,
    fullgraph=LOSS_FULLGRAPH,
)


def _one_optimizer_step_compiled_explicit_fp32_ce(
    *,
    model: CortexSLM,
    runner: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_data,
    generator: torch.Generator,
    device: torch.device,
    lr: float,
) -> float:
    """Production optimizer step with only explicit-FP32 CE compiled separately."""

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
            loss = (
                compiled_explicit_fp32_cross_entropy(logits, y)
                / training_module.GRAD_ACCUM_STEPS
            )
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
def compiled_explicit_fp32_ce_training_builder() -> Iterator[None]:
    """Temporarily select loss-only compile without changing production defaults."""

    import architectures.cortex_s.experiments.scale100m_2b.train as training_module

    original_step = training_module._one_optimizer_step
    training_module._one_optimizer_step = _one_optimizer_step_compiled_explicit_fp32_ce
    try:
        yield
    finally:
        training_module._one_optimizer_step = original_step


def integration_status() -> dict[str, object]:
    return {
        "systems_variant": SYSTEMS_VARIANT,
        "classification": CLASSIFICATION,
        "loss_compile_mode": LOSS_COMPILE_MODE,
        "loss_fullgraph": LOSS_FULLGRAPH,
        "explicit_full_logits_fp32_cast_preserved": True,
        "cross_entropy_semantics_preserved": True,
        "production_model_compile_unchanged": True,
        "default_model_forward_unchanged": True,
        "default_training_module_unchanged": True,
        "single_host_readback_preserved": True,
        "liger_used": False,
        "full_training_authorized": False,
        "scientific_claim_authorized": False,
        "gpu_dispatch_authorized": False,
    }
