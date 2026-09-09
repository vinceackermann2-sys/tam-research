from __future__ import annotations

from contextlib import contextmanager
from types import MethodType
from typing import Iterator

import torch
import torch.nn.functional as F

from architectures.cortex_s.grouped_moe import (
    PhysicalPaddedGroupedSparseMoE,
    _json_safe_router_stats,
    padded_hidden,
)
from architectures.cortex_s.language_model import CortexSLM, TrulySparseMoE, parameter_count
from architectures.cortex_s.systems_optimization import RoutingPlan


SYSTEMS_VARIANT = "memory_lean_cast_before_gather_pad_v4"
CUDA_COMPUTE_DTYPE = torch.bfloat16


def pack_topk_assignments_lean(
    top_indices: torch.Tensor,
    weights: torch.Tensor,
    *,
    num_experts: int,
) -> RoutingPlan:
    """Semantics-identical stable expert packing with less temporary traffic.

    Flattened top-k positions already encode token id as position // top_k. The v3
    helper materialized an expanded token-id tensor and then gathered it by the
    expert sort order; it also gathered the sorted expert ids only to bincount them.
    v4 derives grouped token ids directly from the stable sort order and bincounts
    the original expert ids. Row order, selected experts and assignment weights are
    unchanged.
    """

    if top_indices.ndim != 2 or weights.shape != top_indices.shape:
        raise ValueError("top_indices and weights must have matching [tokens, top_k] shapes")
    if top_indices.dtype != torch.long:
        raise ValueError("top_indices must be torch.long")
    tokens, top_k = top_indices.shape
    if tokens <= 0 or top_k <= 0:
        raise ValueError("routing plan requires at least one token and one assignment")

    expert_ids = top_indices.reshape(-1)
    flat_weights = weights.reshape(-1)
    order = torch.argsort(expert_ids, stable=True)
    grouped_tokens = torch.div(order, top_k, rounding_mode="floor")
    grouped_weights = flat_weights.index_select(0, order)
    counts = torch.bincount(expert_ids, minlength=num_experts)
    offsets = counts.cumsum(dim=0).to(dtype=torch.int32)
    return RoutingPlan(
        token_indices=grouped_tokens,
        assignment_weights=grouped_weights,
        counts=counts,
        offsets=offsets,
    )


class MemoryLeanPhysicalPaddedGroupedSparseMoE(PhysicalPaddedGroupedSparseMoE):
    """v4 grouped backend that preserves v3 BF16 arithmetic while moving fewer bytes.

    The v3 H100 preflight proved the physical-padding grouped backend works but
    missed the frozen full-envelope gate. Two avoidable memory-traffic patterns are
    removed here without changing routing, trainable parameters, logical width or
    grouped BF16 values:

    1. cast the normalized token matrix to BF16 before top-2 row gathering, instead
       of gathering two FP32 rows per token and casting the doubled packed tensor;
    2. cast logical expert weights to BF16 before zero-padding to width 344, instead
       of padding FP32 matrices and then casting the larger physical matrices.

    CPU/reference execution remains FP32 and follows the inherited v3 path so the
    repository can test exact functional equivalence without CUDA.
    """

    def _physical_matrices_bf16(self) -> tuple[torch.Tensor, torch.Tensor]:
        pad = self.physical_hidden - self.hidden
        w1 = self.expert_w1.transpose(-2, -1).to(dtype=CUDA_COMPUTE_DTYPE)
        w2 = self.expert_w2.to(dtype=CUDA_COMPUTE_DTYPE)
        w1_physical = F.pad(w1, (0, pad)).contiguous() if pad else w1.contiguous()
        w2_physical = (
            F.pad(w2, (0, 0, 0, pad)).contiguous()
            if pad
            else w2.contiguous()
        )
        return w1_physical, w2_physical

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        flat = x.reshape(-1, original_shape[-1])
        router_logits = self.router(flat)
        top_values, top_indices = router_logits.topk(self.top_k, dim=-1)
        assignment_weights = F.softmax(top_values.float(), dim=-1).to(flat.dtype)
        plan = pack_topk_assignments_lean(
            top_indices,
            assignment_weights,
            num_experts=self.num_experts,
        )

        if self.grouped_cuda_available(flat):
            # v3 ultimately supplied BF16 to grouped_mm. Casting before the top-2
            # gather halves the gathered element width while preserving those exact
            # BF16 inputs.
            compute_flat = flat if flat.dtype == CUDA_COMPUTE_DTYPE else flat.to(CUDA_COMPUTE_DTYPE)
            packed = compute_flat.index_select(0, plan.token_indices)
            w1_physical, w2_physical = self._physical_matrices_bf16()
        else:
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


def convert_to_memory_lean_grouped(model: CortexSLM) -> CortexSLM:
    """Replace legacy MoE blocks with the v4 memory-lean grouped implementation."""

    before = parameter_count(model)
    for block in model.blocks:
        legacy = block.moe
        if not isinstance(legacy, TrulySparseMoE):
            raise TypeError("expected TrulySparseMoE before v4 grouped conversion")
        first = legacy.experts[0][0]
        device = legacy.router.weight.device
        dtype = legacy.router.weight.dtype
        cuda_devices: list[int] = []
        if device.type == "cuda":
            cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]
        with torch.random.fork_rng(devices=cuda_devices):
            candidate = MemoryLeanPhysicalPaddedGroupedSparseMoE(
                d_model=first.in_features,
                num_experts=legacy.num_experts,
                top_k=legacy.top_k,
                hidden=first.out_features,
            ).to(device=device, dtype=dtype)
        candidate.copy_from_legacy(legacy)
        block.moe = candidate
    after = parameter_count(model)
    if before != after:
        raise RuntimeError(f"v4 grouped conversion changed parameter count: {before} -> {after}")
    return model


def build_memory_lean_grouped_cortex_100m() -> CortexSLM:
    """Build the exact 100M model proposed for grouped preflight/full training v4."""

    from architectures.cortex_s.experiments.scale100m_2b.protocol import (
        CORTEX_100M_CONFIG,
        EXPECTED_CORTEX_PARAMS,
        validate_protocol,
    )

    model = CortexSLM(CORTEX_100M_CONFIG)
    model = convert_to_memory_lean_grouped(model)
    model.router_stats = MethodType(lambda self: _json_safe_router_stats(self), model)
    actual = parameter_count(model)
    if actual != EXPECTED_CORTEX_PARAMS:
        raise RuntimeError(f"v4 grouped 100M parameter count drift: {actual} != {EXPECTED_CORTEX_PARAMS}")
    validate_protocol(actual_cortex_params=actual)
    return model


@contextmanager
def memory_lean_grouped_training_builder() -> Iterator[None]:
    import architectures.cortex_s.experiments.scale100m_2b.train as training_module

    original = training_module.build_cortex_100m
    training_module.build_cortex_100m = build_memory_lean_grouped_cortex_100m
    try:
        yield
    finally:
        training_module.build_cortex_100m = original


def run_h100_memory_lean_grouped_calibration(**kwargs):
    import architectures.cortex_s.experiments.scale100m_2b.train as training_module

    with memory_lean_grouped_training_builder():
        result = training_module.run_h100_calibration(**kwargs)
    result = dict(result)
    result["production_moe_backend"] = "physical_padded_grouped_bf16"
    result["production_systems_variant"] = SYSTEMS_VARIANT
    result["logical_expert_hidden"] = 338
    result["physical_expert_hidden"] = 344
    return result


def train_full_memory_lean_grouped_2b(**kwargs):
    import architectures.cortex_s.experiments.scale100m_2b.train as training_module

    with memory_lean_grouped_training_builder():
        result = training_module.train_full_2b(**kwargs)
    result = dict(result)
    result["production_moe_backend"] = "physical_padded_grouped_bf16"
    result["production_systems_variant"] = SYSTEMS_VARIANT
    result["logical_expert_hidden"] = 338
    result["physical_expert_hidden"] = 344
    return result
