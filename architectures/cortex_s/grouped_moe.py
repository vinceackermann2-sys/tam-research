from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Optional

import torch
from torch import nn
import torch.nn.functional as F

from architectures.cortex_s.language_model import CortexSLM, TrulySparseMoE, parameter_count
from architectures.cortex_s.systems_optimization import pack_topk_assignments


GROUPED_MM_ALIGNMENT_BYTES = 16
BF16_ALIGNMENT_ELEMENTS = GROUPED_MM_ALIGNMENT_BYTES // 2


def padded_hidden(hidden: int) -> int:
    """Round a logical BF16 expert width to the grouped-mm alignment boundary."""

    if hidden <= 0:
        raise ValueError("hidden must be positive")
    return ((hidden + BF16_ALIGNMENT_ELEMENTS - 1) // BF16_ALIGNMENT_ELEMENTS) * BF16_ALIGNMENT_ELEMENTS


def _require_grouped_layout(tensor: torch.Tensor, label: str) -> None:
    if tensor.ndim < 2:
        raise RuntimeError(f"{label} must be at least rank-2")
    if tensor.stride(-1) != 1:
        raise RuntimeError(f"{label} innermost stride must be 1, got {tensor.stride()}")
    row_stride_bytes = tensor.stride(-2) * tensor.element_size()
    if row_stride_bytes % GROUPED_MM_ALIGNMENT_BYTES:
        raise RuntimeError(
            f"{label} row stride must be a multiple of {GROUPED_MM_ALIGNMENT_BYTES} bytes, "
            f"got {row_stride_bytes} bytes from stride {tensor.stride()}"
        )


class PhysicalPaddedGroupedSparseMoE(nn.Module):
    """Production grouped-MoE backend validated by repair4.

    Logical trainable expert width remains unchanged. Runtime matrices are padded
    with zero channels to a real BF16-aligned physical width before grouped GEMM.
    Since the added W1 columns and W2 rows are zeros and GELU(0)=0, the padding
    carries no trainable parameters and does not alter the intended expert mapping.

    CPU/unsupported devices use a grouped reference implementation so repository
    correctness tests do not depend on CUDA grouped-mm availability.
    """

    def __init__(self, d_model: int, num_experts: int, top_k: int, hidden: int):
        super().__init__()
        if not 1 <= top_k <= num_experts:
            raise ValueError("top_k must be in [1, num_experts]")
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden = hidden
        self.physical_hidden = padded_hidden(hidden)
        self.router = nn.Linear(d_model, num_experts, bias=True)
        self.expert_w1 = nn.Parameter(torch.empty(num_experts, hidden, d_model))
        # Stored transposed relative to nn.Linear's second projection so runtime W2
        # is naturally [experts, hidden, d_model].
        self.expert_w2 = nn.Parameter(torch.empty(num_experts, hidden, d_model))
        self.last_counts: Optional[torch.Tensor] = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.router.reset_parameters()
        for expert in range(self.num_experts):
            nn.init.normal_(self.expert_w1[expert], mean=0.0, std=0.02)
            nn.init.normal_(self.expert_w2[expert], mean=0.0, std=0.02)

    @torch.no_grad()
    def copy_from_legacy(self, legacy: TrulySparseMoE) -> "PhysicalPaddedGroupedSparseMoE":
        if legacy.num_experts != self.num_experts or legacy.top_k != self.top_k:
            raise ValueError("legacy routing dimensions do not match grouped backend")
        self.router.weight.copy_(legacy.router.weight)
        self.router.bias.copy_(legacy.router.bias)
        for index, expert in enumerate(legacy.experts):
            first = expert[0]
            second = expert[2]
            if first.weight.shape != self.expert_w1[index].shape:
                raise ValueError("legacy first projection shape mismatch")
            if second.weight.T.shape != self.expert_w2[index].shape:
                raise ValueError("legacy second projection shape mismatch")
            self.expert_w1[index].copy_(first.weight)
            self.expert_w2[index].copy_(second.weight.T)
        return self

    @staticmethod
    def grouped_cuda_available(x: torch.Tensor) -> bool:
        return bool(
            hasattr(F, "grouped_mm")
            and x.is_cuda
            and torch.cuda.get_device_capability(x.device) >= (8, 0)
        )

    def _physical_matrices(self) -> tuple[torch.Tensor, torch.Tensor]:
        pad = self.physical_hidden - self.hidden
        w1 = self.expert_w1.transpose(-2, -1)
        w1_physical = F.pad(w1, (0, pad)).contiguous() if pad else w1.contiguous()
        w2_physical = (
            F.pad(self.expert_w2, (0, 0, 0, pad)).contiguous()
            if pad
            else self.expert_w2.contiguous()
        )
        return w1_physical, w2_physical

    @staticmethod
    def _reference_grouped_mm(
        packed: torch.Tensor,
        matrices: torch.Tensor,
        offsets: torch.Tensor,
    ) -> torch.Tensor:
        outputs: list[torch.Tensor] = []
        start = 0
        for expert_index, end_tensor in enumerate(offsets.cpu()):
            end = int(end_tensor)
            outputs.append(packed[start:end] @ matrices[expert_index])
            start = end
        return torch.cat(outputs, dim=0)

    def _grouped_mm(
        self,
        packed: torch.Tensor,
        matrices: torch.Tensor,
        offsets: torch.Tensor,
    ) -> torch.Tensor:
        if not self.grouped_cuda_available(packed):
            return self._reference_grouped_mm(packed, matrices, offsets)

        packed_bf16 = packed if packed.dtype == torch.bfloat16 else packed.to(dtype=torch.bfloat16)
        matrices_bf16 = matrices if matrices.dtype == torch.bfloat16 else matrices.to(dtype=torch.bfloat16)
        offsets_i32 = offsets if offsets.dtype == torch.int32 else offsets.to(dtype=torch.int32)
        _require_grouped_layout(packed_bf16, "production grouped_mm lhs")
        _require_grouped_layout(matrices_bf16, "production grouped_mm rhs")
        return F.grouped_mm(packed_bf16, matrices_bf16, offs=offsets_i32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        flat = x.reshape(-1, original_shape[-1])
        router_logits = self.router(flat)
        top_values, top_indices = router_logits.topk(self.top_k, dim=-1)
        assignment_weights = F.softmax(top_values.float(), dim=-1).to(flat.dtype)
        plan = pack_topk_assignments(top_indices, assignment_weights, num_experts=self.num_experts)
        packed = flat.index_select(0, plan.token_indices)

        w1_physical, w2_physical = self._physical_matrices()
        hidden_physical = self._grouped_mm(packed, w1_physical, plan.offsets)
        hidden_physical = F.gelu(hidden_physical)
        expert_output = self._grouped_mm(hidden_physical, w2_physical, plan.offsets)

        weighted = expert_output * plan.assignment_weights[:, None]
        output = torch.zeros_like(flat)
        output.index_add_(0, plan.token_indices, weighted)
        self.last_counts = plan.counts.detach()
        return output.view(original_shape)

    @property
    def theoretical_executed_fraction(self) -> float:
        return self.top_k / self.num_experts


def convert_to_production_grouped(model: CortexSLM) -> CortexSLM:
    """Replace every legacy MoE with the repair4-validated grouped backend."""

    before = parameter_count(model)
    for block in model.blocks:
        legacy = block.moe
        if not isinstance(legacy, TrulySparseMoE):
            raise TypeError("expected TrulySparseMoE before production grouped conversion")
        first = legacy.experts[0][0]
        device = legacy.router.weight.device
        dtype = legacy.router.weight.dtype
        cuda_devices: list[int] = []
        if device.type == "cuda":
            cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]
        # Conversion must not advance the scientific RNG stream merely because a
        # temporary candidate module allocates parameters before copying legacy ones.
        with torch.random.fork_rng(devices=cuda_devices):
            candidate = PhysicalPaddedGroupedSparseMoE(
                d_model=first.in_features,
                num_experts=legacy.num_experts,
                top_k=legacy.top_k,
                hidden=first.out_features,
            ).to(device=device, dtype=dtype)
        candidate.copy_from_legacy(legacy)
        block.moe = candidate
    after = parameter_count(model)
    if before != after:
        raise RuntimeError(f"production grouped conversion changed parameter count: {before} -> {after}")
    return model


def build_production_grouped_cortex_100m() -> CortexSLM:
    """Build the exact 100M model used by grouped preflight/full training v3."""

    from architectures.cortex_s.experiments.scale100m_2b.protocol import (
        CORTEX_100M_CONFIG,
        EXPECTED_CORTEX_PARAMS,
        validate_protocol,
    )

    model = CortexSLM(CORTEX_100M_CONFIG)
    model = convert_to_production_grouped(model)
    actual = parameter_count(model)
    if actual != EXPECTED_CORTEX_PARAMS:
        raise RuntimeError(f"grouped 100M parameter count drift: {actual} != {EXPECTED_CORTEX_PARAMS}")
    validate_protocol(actual_cortex_params=actual)
    return model


@contextmanager
def grouped_training_builder() -> Iterator[None]:
    """Install the grouped builder only for one explicit training/preflight call."""

    import architectures.cortex_s.experiments.scale100m_2b.train as training_module

    original = training_module.build_cortex_100m
    training_module.build_cortex_100m = build_production_grouped_cortex_100m
    try:
        yield
    finally:
        training_module.build_cortex_100m = original


def run_h100_grouped_calibration(**kwargs):
    import architectures.cortex_s.experiments.scale100m_2b.train as training_module

    with grouped_training_builder():
        result = training_module.run_h100_calibration(**kwargs)
    result = dict(result)
    result["production_moe_backend"] = "physical_padded_grouped_bf16"
    result["logical_expert_hidden"] = 338
    result["physical_expert_hidden"] = 344
    return result


def train_full_grouped_2b(**kwargs):
    import architectures.cortex_s.experiments.scale100m_2b.train as training_module

    with grouped_training_builder():
        result = training_module.train_full_2b(**kwargs)
    result = dict(result)
    result["production_moe_backend"] = "physical_padded_grouped_bf16"
    result["logical_expert_hidden"] = 338
    result["physical_expert_hidden"] = 344
    return result
