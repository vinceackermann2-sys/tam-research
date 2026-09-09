from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from .language_model import TrulySparseMoE


@dataclass(frozen=True)
class RoutingPlan:
    """Packed top-k assignments grouped by expert without changing routing semantics."""

    token_indices: torch.Tensor
    assignment_weights: torch.Tensor
    counts: torch.Tensor
    offsets: torch.Tensor


def pack_topk_assignments(
    top_indices: torch.Tensor,
    weights: torch.Tensor,
    *,
    num_experts: int,
) -> RoutingPlan:
    """Stable-pack token/expert assignments into expert-contiguous rows.

    ``top_indices`` and ``weights`` are the exact outputs of the existing router.
    Each token still contributes exactly ``top_k`` expert assignments.  The only
    change is physical row order so a grouped GEMM can replace many tiny GEMMs.
    """

    if top_indices.ndim != 2 or weights.shape != top_indices.shape:
        raise ValueError("top_indices and weights must have matching [tokens, top_k] shapes")
    if top_indices.dtype != torch.long:
        raise ValueError("top_indices must be torch.long")
    tokens, top_k = top_indices.shape
    if tokens <= 0 or top_k <= 0:
        raise ValueError("routing plan requires at least one token and one assignment")

    token_ids = (
        torch.arange(tokens, device=top_indices.device, dtype=torch.long)
        .unsqueeze(1)
        .expand(tokens, top_k)
        .reshape(-1)
    )
    expert_ids = top_indices.reshape(-1)
    flat_weights = weights.reshape(-1)

    # Stable ordering preserves the legacy row-major token/slot order within each
    # expert, reducing numerical drift in the expert GEMMs and scatter reduction.
    order = torch.argsort(expert_ids, stable=True)
    grouped_experts = expert_ids.index_select(0, order)
    grouped_tokens = token_ids.index_select(0, order)
    grouped_weights = flat_weights.index_select(0, order)
    counts = torch.bincount(grouped_experts, minlength=num_experts)
    offsets = counts.cumsum(dim=0).to(dtype=torch.int32)

    return RoutingPlan(
        token_indices=grouped_tokens,
        assignment_weights=grouped_weights,
        counts=counts,
        offsets=offsets,
    )


def affine_scan_slice_candidate(
    a: torch.Tensor,
    b: torch.Tensor,
    initial: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Systems-only affine-scan candidate with the same recurrence.

    The production scan constructs two padded tensors with ``torch.cat`` at every
    log2(sequence) stage.  This candidate keeps the same Hillis-Steele algebra but
    updates only the suffix that participates at each stage.  It is deliberately
    not wired into the production model until a separately authorized GPU systems
    benchmark demonstrates a real benefit.
    """

    if a.shape != b.shape or a.ndim != 3:
        raise ValueError("a and b must have matching [batch, sequence, state] shapes")
    length = a.size(1)
    pa, pb = a, b
    offset = 1
    while offset < length:
        old_a, old_b = pa, pb
        next_a = old_a.clone()
        next_b = old_b.clone()
        next_a[:, offset:] = old_a[:, offset:] * old_a[:, :-offset]
        next_b[:, offset:] = old_b[:, offset:] + old_a[:, offset:] * old_b[:, :-offset]
        pa, pb = next_a, next_b
        offset <<= 1
    if initial is not None:
        if initial.shape != (a.size(0), a.size(2)):
            raise ValueError("initial state has wrong shape")
        initial = initial.to(device=a.device, dtype=a.dtype)
        pb = pb + pa * initial[:, None, :]
    return pb


class GroupedSparseMoECandidate(nn.Module):
    """Semantics-preserving packed MoE candidate for systems validation.

    The legacy implementation launches each selected expert independently.  This
    candidate stores expert weights as stacked tensors and packs routed rows by
    expert so supported CUDA devices can execute two grouped GEMMs: one for the
    input projection and one for the output projection.  CPU and unsupported-device
    execution uses a transparent grouped-slice reference path for equivalence tests.

    This class is *not* the production CORTEX-S MoE yet.  It exists only to prove
    equivalence and establish a credible systems path before another paid benchmark.
    """

    def __init__(self, d_model: int, num_experts: int, top_k: int, hidden: int):
        super().__init__()
        if not 1 <= top_k <= num_experts:
            raise ValueError("top_k must be in [1, num_experts]")
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden = hidden
        self.router = nn.Linear(d_model, num_experts, bias=True)
        self.expert_w1 = nn.Parameter(torch.empty(num_experts, hidden, d_model))
        self.expert_w2 = nn.Parameter(torch.empty(num_experts, d_model, hidden))
        self.reset_parameters()
        self.last_counts: Optional[torch.Tensor] = None

    def reset_parameters(self) -> None:
        self.router.reset_parameters()
        # Match nn.Linear's per-expert default weight distribution.  Exact weights
        # for equivalence tests are copied from the legacy module below.
        for expert in range(self.num_experts):
            nn.init.kaiming_uniform_(self.expert_w1[expert], a=sqrt(5))
            nn.init.kaiming_uniform_(self.expert_w2[expert], a=sqrt(5))

    @torch.no_grad()
    def copy_from_legacy(self, legacy: TrulySparseMoE) -> "GroupedSparseMoECandidate":
        if legacy.num_experts != self.num_experts or legacy.top_k != self.top_k:
            raise ValueError("legacy routing dimensions do not match candidate")
        self.router.weight.copy_(legacy.router.weight)
        self.router.bias.copy_(legacy.router.bias)
        for expert_index, expert in enumerate(legacy.experts):
            first = expert[0]
            second = expert[2]
            if first.weight.shape != self.expert_w1[expert_index].shape:
                raise ValueError("legacy first expert projection has incompatible shape")
            if second.weight.shape != self.expert_w2[expert_index].shape:
                raise ValueError("legacy second expert projection has incompatible shape")
            self.expert_w1[expert_index].copy_(first.weight)
            self.expert_w2[expert_index].copy_(second.weight)
        return self

    @staticmethod
    def _grouped_mm_available(x: torch.Tensor) -> bool:
        if not hasattr(F, "grouped_mm") or not x.is_cuda:
            return False
        if torch.cuda.get_device_capability(x.device) < (8, 0):
            return False
        return x.dtype in {torch.bfloat16, torch.float16, torch.float32}

    def _grouped_mm_reference(
        self,
        packed: torch.Tensor,
        weights: torch.Tensor,
        offsets: torch.Tensor,
    ) -> torch.Tensor:
        outputs: list[torch.Tensor] = []
        start = 0
        # Reference-only fallback.  The CUDA candidate below contains no per-expert
        # GEMM loop and is the path that would need a fresh systems benchmark.
        for expert_index, end_tensor in enumerate(offsets.cpu()):
            end = int(end_tensor)
            outputs.append(packed[start:end] @ weights[expert_index])
            start = end
        return torch.cat(outputs, dim=0)

    def _grouped_mm(
        self,
        packed: torch.Tensor,
        weights: torch.Tensor,
        offsets: torch.Tensor,
    ) -> torch.Tensor:
        if self._grouped_mm_available(packed):
            # F.grouped_mm is not autocast-enabled.  Keep the operation in the
            # caller's current dtype; a future paid microbenchmark must compare
            # fp32 vs explicit bf16 casting before production integration.
            return F.grouped_mm(packed, weights, offs=offsets)
        return self._grouped_mm_reference(packed, weights, offsets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        flat = x.reshape(-1, original_shape[-1])
        router_logits = self.router(flat)
        top_values, top_indices = router_logits.topk(self.top_k, dim=-1)
        weights = F.softmax(top_values.float(), dim=-1).to(flat.dtype)
        plan = pack_topk_assignments(top_indices, weights, num_experts=self.num_experts)

        packed = flat.index_select(0, plan.token_indices)
        hidden = self._grouped_mm(
            packed,
            self.expert_w1.transpose(-2, -1),
            plan.offsets,
        )
        hidden = F.gelu(hidden)
        expert_output = self._grouped_mm(
            hidden,
            self.expert_w2.transpose(-2, -1),
            plan.offsets,
        )
        weighted = expert_output * plan.assignment_weights[:, None]

        output = torch.zeros_like(flat)
        output.index_add_(0, plan.token_indices, weighted)
        self.last_counts = plan.counts.detach()
        return output.view(original_shape)

    @property
    def theoretical_executed_fraction(self) -> float:
        return self.top_k / self.num_experts

    def logical_kernel_plan(self) -> dict[str, int]:
        """Return architecture-independent GEMM launch counts for review/tests."""

        return {
            "legacy_expert_gemms_per_layer": 2 * self.num_experts,
            "candidate_grouped_gemms_per_layer": 2,
            "selected_assignments_per_token": self.top_k,
            "possible_assignments_per_token": self.num_experts,
        }
