from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .aera_hardware_core import HardwareAERAConfig
from .aera_hardware_core_v5 import MixedPrecisionSafeAERAStage
from .aera_hardware_core_v6 import (
    BMMHardSparseExpertBank,
    HardwareAwareAERATextLMV6,
)


class NativeGroupedMMSparseExpertBank(BMMHardSparseExpertBank):
    """Production-shaped hard MoE execution using PyTorch native grouped GEMM.

    Training/soft routing deliberately keeps the established differentiable path.
    Hard CUDA inference packs only the selected top-1/top-2 chunk assignments,
    groups them by expert, explicitly casts *only the expert GEMM operands* to
    BF16, and executes each expert family in a single grouped-MM kernel call.

    The explicit expert-local cast is important: residual/LayerNorm plumbing may
    legitimately keep the surrounding activation in FP32 even under autocast.
    Requiring the entire residual stream to already be BF16 would silently disable
    the grouped kernel. The selected expert contribution is restored to the input
    residual dtype before accumulation.

    `torch.nn.functional.grouped_mm` is a hardware optimization, not an architecture
    semantic change. If the installed PyTorch/device does not expose/support it, the
    implementation falls back to the numerically equivalent v6 BMM path and records
    that fallback so no grouped-kernel efficiency claim can be made accidentally.
    """

    def __init__(self, cfg: HardwareAERAConfig):
        super().__init__(cfg)
        self.last_kernel: str = "uninitialized"
        self.last_input_dtype: str | None = None
        self.last_compute_dtype: str | None = None

    @staticmethod
    def native_grouped_mm_available(x: torch.Tensor) -> bool:
        op = getattr(F, "grouped_mm", None)
        if op is None or not x.is_cuda or not x.is_floating_point():
            return False
        major, _minor = torch.cuda.get_device_capability(x.device)
        return major >= 8

    def _native_grouped_two_layer(
        self,
        x: torch.Tensor,
        assignment_batch: torch.Tensor,
        assignment_expert: torch.Tensor,
        assignment_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Run selected chunk assignments grouped by expert.

        Each routing assignment owns an entire causal chunk [T,D]. Assignments are
        sorted by expert, their T rows are concatenated, and the two expert MLP
        projections are each issued as one grouped GEMM.

        PyTorch grouped_mm currently requires the final group offset to be strictly
        below the physical row count. We therefore append one physical sentinel row.
        The kernel returns an output row for that sentinel too, so *after every
        grouped GEMM* we slice back to the logical routed-row count before any
        activation or reshape. The sentinel must never become a real assignment.
        """

        _b, t, d = x.shape
        if assignment_batch.numel() == 0:
            return torch.zeros_like(x)

        residual_dtype = x.dtype
        compute_dtype = torch.bfloat16 if x.is_cuda else x.dtype
        self.last_input_dtype = str(residual_dtype)
        self.last_compute_dtype = str(compute_dtype)

        order = torch.argsort(assignment_expert, stable=True)
        batch_sorted = assignment_batch.index_select(0, order)
        expert_sorted = assignment_expert.index_select(0, order)
        weight_sorted = assignment_weight.index_select(0, order).to(compute_dtype)

        active_experts, assignment_counts = torch.unique_consecutive(
            expert_sorted, return_counts=True
        )
        packed = (
            x.index_select(0, batch_sorted)
            .to(dtype=compute_dtype)
            .contiguous()
        )  # [A,T,D]
        rows = packed.reshape(-1, d)
        logical_rows = int(rows.size(0))

        # grouped_mm requires at least one physical row after offs[-1].
        rows_with_sentinel = torch.cat((rows, torch.zeros_like(rows[:1])), dim=0)
        row_counts = assignment_counts.to(torch.int32) * int(t)
        offs = torch.cumsum(row_counts, dim=0, dtype=torch.int32)

        w1 = (
            self.w1.index_select(0, active_experts)
            .to(dtype=compute_dtype)
            .transpose(-2, -1)
            .contiguous()
        )
        hidden_physical = F.grouped_mm(rows_with_sentinel, w1, offs=offs)
        hidden = hidden_physical[:logical_rows]
        if hidden.size(0) != logical_rows:
            raise RuntimeError("grouped_mm first projection returned too few logical rows")
        hidden = F.gelu(hidden)

        hidden_with_sentinel = torch.cat((hidden, torch.zeros_like(hidden[:1])), dim=0)
        w2 = (
            self.w2.index_select(0, active_experts)
            .to(dtype=compute_dtype)
            .transpose(-2, -1)
            .contiguous()
        )
        y_physical = F.grouped_mm(hidden_with_sentinel, w2, offs=offs)
        y = y_physical[:logical_rows]
        if y.size(0) != logical_rows:
            raise RuntimeError("grouped_mm second projection returned too few logical rows")
        y = y.reshape(assignment_batch.numel(), t, d)
        y = y * weight_sorted[:, None, None]

        out = torch.zeros_like(x)
        out.index_add_(0, batch_sorted, y.to(dtype=residual_dtype))
        return out

    def forward(
        self,
        x: torch.Tensor,
        expert_logits: torch.Tensor,
        count_logits: torch.Tensor,
        *,
        hard: bool,
    ) -> torch.Tensor:
        if not hard or not self.native_grouped_mm_available(x):
            self.last_kernel = "soft_reference" if not hard else "bmm_fallback"
            self.last_input_dtype = str(x.dtype)
            self.last_compute_dtype = None
            return super().forward(x, expert_logits, count_logits, hard=hard)

        b, _t, _d = x.shape
        if expert_logits.shape != (b, self.n_experts):
            raise ValueError("expert_logits must be [batch,n_experts]")
        if count_logits.shape != (b, 2):
            raise ValueError("count_logits must be [batch,2]")

        route_probs = F.softmax(expert_logits.float(), dim=-1).to(x.dtype)
        selected_probs, idx = torch.topk(route_probs, self.max_active, dim=-1)
        chosen_count = count_logits.argmax(dim=-1) + 1

        batch_ids = torch.arange(b, device=x.device)
        assignment_batch = batch_ids
        assignment_expert = idx[:, 0]
        assignment_weight = torch.ones(b, device=x.device, dtype=x.dtype)

        if self.max_active >= 2:
            p1, p2 = selected_probs[:, 0], selected_probs[:, 1]
            denom = (p1 + p2).clamp_min(1e-6)
            use_second = chosen_count >= 2
            assignment_weight = torch.where(
                use_second, p1 / denom, torch.ones_like(p1)
            )
            second_batch = use_second.nonzero(as_tuple=False).squeeze(-1)
            if second_batch.numel() > 0:
                assignment_batch = torch.cat((assignment_batch, second_batch), dim=0)
                assignment_expert = torch.cat(
                    (assignment_expert, idx.index_select(0, second_batch)[:, 1]), dim=0
                )
                assignment_weight = torch.cat(
                    (
                        assignment_weight,
                        p2.index_select(0, second_batch)
                        / denom.index_select(0, second_batch),
                    ),
                    dim=0,
                )

        out = self._native_grouped_two_layer(
            x, assignment_batch, assignment_expert, assignment_weight
        )
        self.last_counts = chosen_count.detach().cpu()
        self.last_route_probs = route_probs.detach().float().mean(dim=0).cpu()
        self.last_kernel = "native_grouped_mm"
        return out

    def stats(self) -> dict[str, object] | None:
        base = super().stats()
        if base is None:
            return None
        return {
            **base,
            "hard_kernel": self.last_kernel,
            "hard_input_dtype": self.last_input_dtype,
            "hard_compute_dtype": self.last_compute_dtype,
        }


class NativeGroupedMMAERAStage(MixedPrecisionSafeAERAStage):
    def __init__(self, cfg: HardwareAERAConfig):
        super().__init__(cfg)
        self.experts = NativeGroupedMMSparseExpertBank(cfg)


class HardwareAwareAERATextLMV7(HardwareAwareAERATextLMV6):
    """Pre-100M AERA candidate with native grouped-MoE hard execution."""

    def __init__(
        self,
        cfg: HardwareAERAConfig = HardwareAERAConfig(),
        *,
        stream_forecast_tokens: int = 4,
    ):
        super().__init__(cfg, stream_forecast_tokens=stream_forecast_tokens)
        self.stages = nn.ModuleList(NativeGroupedMMAERAStage(cfg) for _ in range(cfg.n_stages))
