from __future__ import annotations

from contextlib import contextmanager
import importlib.metadata
from typing import Iterator

import torch
from torch import nn
import torch.nn.functional as F

from architectures.cortex_s.language_model import CortexSLM


SYSTEMS_VARIANT = "fused_linear_ce_liger_v1"
CLASSIFICATION = "ZERO_CREDIT_SYSTEMS_OPTIMIZATION_ONLY"
LIGER_KERNEL_VERSION = "0.8.2"
LIGER_KERNEL_WHEEL_SHA256 = "84c0a7bc9bf4d4cf8ea5ba89ff84d28686afc94215b220851d9f57dc87852741"
DEFAULT_REFERENCE_CHUNK_ROWS = 256


def _validate_training_pair(model: CortexSLM, tokens: torch.Tensor, targets: torch.Tensor) -> None:
    if tokens.ndim != 2 or targets.ndim != 2:
        raise ValueError("tokens and targets must both be [batch, sequence]")
    if tokens.shape != targets.shape:
        raise ValueError("tokens and targets must have identical shapes")
    if tokens.size(1) > model.cfg.max_seq_len:
        raise ValueError("sequence length exceeds model max_seq_len")
    if targets.numel() == 0:
        raise ValueError("targets must be non-empty")


def forward_training_features(model: CortexSLM, tokens: torch.Tensor) -> torch.Tensor:
    """Reproduce CortexSLM.forward exactly through the final normalization.

    This systems seam intentionally stops immediately before the tied lm_head. It
    does not alter the public model forward method and does not persist recurrent
    state across independent language-model training batches, matching the existing
    training path.
    """

    batch, length = tokens.shape
    if length > model.cfg.max_seq_len:
        raise ValueError(
            f"sequence length {length} exceeds max_seq_len={model.cfg.max_seq_len}"
        )
    state = model.initial_state(
        batch,
        device=tokens.device,
        dtype=model.token_emb.weight.dtype,
    )
    positions = torch.arange(length, device=tokens.device)
    x = model.token_emb(tokens) + model.pos_emb(positions)[None, :, :]
    for block, block_state in zip(model.blocks, state):
        x, _ = block(x, block_state, disable_world=False)
    return model.norm(x)


def chunked_linear_cross_entropy_reference(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    *,
    chunk_rows: int,
) -> torch.Tensor:
    """CPU/reference oracle for exact weighted row-chunk reduction semantics.

    The CUDA candidate uses Liger's fused operator instead. This oracle exists so
    ordinary CI can prove target alignment, uneven-chunk reduction, and gradients
    without importing Triton/Liger or allocating a GPU.
    """

    if hidden.ndim != 2 or weight.ndim != 2 or targets.ndim != 1:
        raise ValueError("expected hidden=[tokens, hidden], weight=[vocab, hidden], targets=[tokens]")
    if hidden.size(0) != targets.numel():
        raise ValueError("one target is required per hidden row")
    if hidden.size(1) != weight.size(1):
        raise ValueError("hidden width must match lm-head weight width")
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    if targets.numel() == 0:
        raise ValueError("targets must be non-empty")

    total = torch.zeros((), device=hidden.device, dtype=torch.float32)
    rows = hidden.size(0)
    for start in range(0, rows, chunk_rows):
        end = min(start + chunk_rows, rows)
        logits_chunk = F.linear(hidden[start:end], weight)
        total = total + F.cross_entropy(
            logits_chunk.float(),
            targets[start:end],
            reduction="sum",
        )
    return total / rows


class FusedLinearCETrainingRunner(nn.Module):
    """Training-only wrapper that fuses the tied output projection with CE on CUDA.

    `forward(tokens)` remains a transparent logits path for evaluation. Supplying
    `targets` selects the systems-only loss path. CPU deliberately uses the local
    chunked Torch oracle; CUDA requires the exactly pinned Liger release.
    """

    def __init__(
        self,
        model: CortexSLM,
        *,
        force_torch_reference: bool = False,
        reference_chunk_rows: int = DEFAULT_REFERENCE_CHUNK_ROWS,
    ) -> None:
        super().__init__()
        self.model = model
        self.reference_chunk_rows = int(reference_chunk_rows)
        if self.reference_chunk_rows <= 0:
            raise ValueError("reference_chunk_rows must be positive")

        parameter = next(model.parameters())
        self._use_liger = bool(parameter.is_cuda and not force_torch_reference)
        self._liger_loss: nn.Module | None = None
        if self._use_liger:
            installed = importlib.metadata.version("liger-kernel")
            if installed != LIGER_KERNEL_VERSION:
                raise RuntimeError(
                    f"liger-kernel version drift: {installed} != {LIGER_KERNEL_VERSION}"
                )
            from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

            self._liger_loss = LigerFusedLinearCrossEntropyLoss(
                reduction="mean",
                accum_dtype=torch.float32,
            )

    def _loss(self, hidden: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        flat_hidden = hidden.reshape(-1, hidden.size(-1))
        flat_targets = targets.reshape(-1)
        weight = self.model.lm_head.weight
        if self._liger_loss is not None:
            # The CUDA production candidate never creates a complete [tokens, vocab]
            # logits tensor. Liger performs linear + CE chunk-by-chunk and returns a
            # scalar mean loss while producing gradients for hidden + tied weight.
            return self._liger_loss(weight, flat_hidden, flat_targets)
        return chunked_linear_cross_entropy_reference(
            flat_hidden,
            weight,
            flat_targets,
            chunk_rows=self.reference_chunk_rows,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if targets is None:
            return self.model(tokens)
        _validate_training_pair(self.model, tokens, targets)
        hidden = forward_training_features(self.model, tokens)
        return self._loss(hidden, targets)


def _compile_fused_linear_ce_runner(model: CortexSLM) -> torch.nn.Module:
    import architectures.cortex_s.experiments.scale100m_2b.train as training_module

    training_module._enable_dynamic_compile_support()
    runner = FusedLinearCETrainingRunner(model)
    return torch.compile(runner, mode=training_module.COMPILE_MODE, fullgraph=False)


def _one_optimizer_step_fused_linear_ce(
    *,
    model: CortexSLM,
    runner: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_data,
    generator: torch.Generator,
    device: torch.device,
    lr: float,
) -> float:
    """Exact live optimizer-step contract with only logits+CE realization replaced."""

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
            loss = runner(x, y) / training_module.GRAD_ACCUM_STEPS
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
def fused_linear_ce_training_builder() -> Iterator[None]:
    """Temporarily select the v7 training-only fused-loss systems candidate.

    The default trainer remains untouched outside this context. The context itself
    allocates no GPU, invokes no Modal job, and consumes no experiment seed.
    """

    import architectures.cortex_s.experiments.scale100m_2b.train as training_module

    original_compile = training_module._compile_model
    original_step = training_module._one_optimizer_step
    training_module._compile_model = _compile_fused_linear_ce_runner
    training_module._one_optimizer_step = _one_optimizer_step_fused_linear_ce
    try:
        yield
    finally:
        training_module._compile_model = original_compile
        training_module._one_optimizer_step = original_step


def integration_status() -> dict[str, object]:
    return {
        "systems_variant": SYSTEMS_VARIANT,
        "classification": CLASSIFICATION,
        "liger_kernel_version": LIGER_KERNEL_VERSION,
        "liger_kernel_wheel_sha256": LIGER_KERNEL_WHEEL_SHA256,
        "default_model_forward_unchanged": True,
        "default_training_module_unchanged": True,
        "cuda_materializes_full_logits": False,
        "cpu_reference_is_chunked": True,
        "full_training_authorized": False,
        "scientific_claim_authorized": False,
        "gpu_dispatch_authorized": False,
    }
