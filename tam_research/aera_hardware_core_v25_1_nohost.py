from __future__ import annotations

"""Execution-equivalent v25.1 adaptive telemetry without eager CUDA host copies.

Stage-router telemetry was already repaired in the merged v25.1 runtime. The active
expert and latent-reasoner lineage still copied diagnostic tensors to CPU inside the
forward path. This module preserves every expert/depth decision and computation while
keeping detached telemetry on the execution device until `stats()` is explicitly
requested.
"""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .aera_hardware_core import HardwareAERAConfig
from .aera_hardware_core_v5 import DtypeSafeChunkLatentReasoner
from .aera_hardware_core_v7 import NativeGroupedMMSparseExpertBank
from .aera_hardware_core_v25_1_compact import (
    HardwareAwareAERATextLMV251StableCompact,
    StableCompactExecutionEquivalentFICEMStage,
    stable_compaction_v25_1_protocol,
)


class ExecutionEquivalentNoHostDtypeSafeChunkLatentReasoner(
    DtypeSafeChunkLatentReasoner
):
    """Exact dtype-safe sparse reasoner with device-resident detached telemetry."""

    def __init__(self, source: DtypeSafeChunkLatentReasoner) -> None:
        nn.Module.__init__(self)
        self.max_steps = source.max_steps
        self.cell = source.cell
        self.last_steps = None
        self.last_expected = None

    def forward(
        self,
        summary: torch.Tensor,
        depth_logits: torch.Tensor,
        *,
        hard: bool,
    ) -> torch.Tensor:
        if depth_logits.shape != (summary.size(0), self.max_steps):
            raise ValueError("depth_logits shape mismatch")
        probs = F.softmax(depth_logits.float(), dim=-1).to(summary.dtype)
        values = torch.arange(
            1,
            self.max_steps + 1,
            device=summary.device,
            dtype=summary.dtype,
        )
        self.last_expected = (probs * values[None]).sum(dim=-1).detach()

        if not hard:
            current = summary
            states = []
            for _ in range(self.max_steps):
                updated = self.cell(current, current)
                current = updated.to(dtype=current.dtype)
                states.append(current)
            stacked = torch.stack(states, dim=1)
            self.last_steps = None
            return (stacked * probs[:, :, None]).sum(dim=1)

        chosen = depth_logits.argmax(dim=-1) + 1
        current = summary
        for step in range(1, self.max_steps + 1):
            idx = (chosen >= step).nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                break
            selected = current.index_select(0, idx)
            updated = self.cell(selected, selected).to(dtype=current.dtype)
            current = current.index_copy(0, idx, updated)
        self.last_steps = chosen.detach()
        return current


class ExecutionEquivalentNoHostNativeGroupedMMSparseExpertBank(
    NativeGroupedMMSparseExpertBank
):
    """Exact hard sparse expert execution with device-resident detached telemetry."""

    def __init__(self, source: NativeGroupedMMSparseExpertBank) -> None:
        nn.Module.__init__(self)
        self.n_experts = source.n_experts
        self.max_active = source.max_active
        self.w1 = source.w1
        self.w2 = source.w2
        self.last_counts = None
        self.last_route_probs = None
        self.last_second_batch_size = getattr(source, "last_second_batch_size", None)
        self.last_executed_expert_slots = getattr(
            source, "last_executed_expert_slots", None
        )
        self.last_kernel = getattr(source, "last_kernel", "uninitialized")
        self.last_input_dtype = getattr(source, "last_input_dtype", None)
        self.last_compute_dtype = getattr(source, "last_compute_dtype", None)

    def _bmm_hard_forward(
        self,
        x: torch.Tensor,
        expert_logits: torch.Tensor,
        count_logits: torch.Tensor,
    ) -> torch.Tensor:
        b, _, _ = x.shape
        if expert_logits.shape != (b, self.n_experts):
            raise ValueError("expert_logits must be [batch,n_experts]")
        if count_logits.shape != (b, 2):
            raise ValueError("count_logits must be [batch,2]")

        route_probs = F.softmax(expert_logits.float(), dim=-1).to(x.dtype)
        selected_probs, idx = torch.topk(route_probs, self.max_active, dim=-1)
        chosen_count = count_logits.argmax(dim=-1) + 1

        first_out = self._run_selected(x, idx[:, 0])
        if self.max_active == 1:
            out = first_out
        else:
            p1, p2 = selected_probs[:, 0], selected_probs[:, 1]
            denom = (p1 + p2).clamp_min(1e-6)
            use_second = chosen_count >= 2
            first_weight = torch.where(use_second, p1 / denom, torch.ones_like(p1))
            out = first_out * first_weight[:, None, None]

            second_batch = use_second.nonzero(as_tuple=False).squeeze(-1)
            if second_batch.numel() > 0:
                x2 = x.index_select(0, second_batch).contiguous()
                second_ids = idx.index_select(0, second_batch)[:, 1]
                second_out = self._run_selected(x2, second_ids)
                second_weight = (
                    p2.index_select(0, second_batch)
                    / denom.index_select(0, second_batch)
                )[:, None, None]
                out = out.index_add(
                    0,
                    second_batch,
                    (second_out * second_weight).to(out.dtype),
                )

        self.last_counts = chosen_count.detach()
        self.last_route_probs = route_probs.detach().float().mean(dim=0)
        self.last_kernel = "bmm_fallback"
        self.last_input_dtype = str(x.dtype)
        self.last_compute_dtype = None
        return out

    def _native_hard_forward(
        self,
        x: torch.Tensor,
        expert_logits: torch.Tensor,
        count_logits: torch.Tensor,
    ) -> torch.Tensor:
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
        self.last_counts = chosen_count.detach()
        self.last_route_probs = route_probs.detach().float().mean(dim=0)
        self.last_kernel = "native_grouped_mm"
        return out

    def forward(
        self,
        x: torch.Tensor,
        expert_logits: torch.Tensor,
        count_logits: torch.Tensor,
        *,
        hard: bool,
    ) -> torch.Tensor:
        if not hard:
            # Training/calibration stays on the exact inherited differentiable path.
            return super().forward(x, expert_logits, count_logits, hard=False)
        if self.native_grouped_mm_available(x):
            return self._native_hard_forward(x, expert_logits, count_logits)
        return self._bmm_hard_forward(x, expert_logits, count_logits)


class NoHostAdaptiveTelemetryStableCompactFICEMStage(
    StableCompactExecutionEquivalentFICEMStage
):
    """Merged stable-compaction stage with no eager host telemetry in hard inference."""

    def __init__(self, source: StableCompactExecutionEquivalentFICEMStage) -> None:
        super().__init__(source)
        if not isinstance(self.experts, NativeGroupedMMSparseExpertBank):
            raise TypeError("v25.1 no-host candidate requires native-grouped sparse experts")
        if not isinstance(self.reasoner, DtypeSafeChunkLatentReasoner):
            raise TypeError("v25.1 no-host candidate requires dtype-safe latent reasoner")
        self.experts = ExecutionEquivalentNoHostNativeGroupedMMSparseExpertBank(
            self.experts
        )
        self.reasoner = ExecutionEquivalentNoHostDtypeSafeChunkLatentReasoner(
            self.reasoner
        )


class HardwareAwareAERATextLMV251NoHostTelemetry(
    HardwareAwareAERATextLMV251StableCompact
):
    """CPU-gated final v25.1 candidate with compact writes and no adaptive host syncs."""

    def __init__(
        self,
        cfg: HardwareAERAConfig = HardwareAERAConfig(),
        *,
        stream_forecast_tokens: int = 4,
    ) -> None:
        super().__init__(cfg, stream_forecast_tokens=stream_forecast_tokens)
        self.stages = nn.ModuleList(
            NoHostAdaptiveTelemetryStableCompactFICEMStage(stage)
            for stage in self.stages
        )
        self.set_memory_pretraining_mode(False)


def no_host_adaptive_telemetry_v25_1_protocol() -> dict[str, Any]:
    protocol = dict(stable_compaction_v25_1_protocol())
    protocol.update(
        {
            "version": "aera-v25.1-execution-equivalent-no-host-adaptive-telemetry",
            "expert_hard_math_changed": False,
            "reasoner_math_changed": False,
            "expert_telemetry_forward_host_copy": False,
            "reasoner_telemetry_forward_host_copy": False,
            "expert_telemetry_storage": "detached on execution device; stats may materialize explicitly",
            "reasoner_telemetry_storage": "detached on execution device; stats may materialize explicitly",
            "physical_expert_sparsity_changed": False,
            "physical_reasoning_sparsity_changed": False,
            "learned_parameter_count_changed": False,
            "state_dict_schema_changed": False,
            "routing_policy_changed": False,
            "gpu_authorized": False,
            "scientific_training_authorized": False,
            "100m_authorized": False,
        }
    )
    return protocol
