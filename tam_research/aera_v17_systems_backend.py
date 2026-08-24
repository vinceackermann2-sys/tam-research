from __future__ import annotations

"""Systems-only execution substitutions for the frozen AERA-v17 architecture.

This module does not change routing, weights, training objectives, memory, attention,
or latent-depth decisions.  It only replaces two hard-inference implementations
whose overhead was isolated by the seed8331 systems probes:

* latent reasoning keeps exact selected-row GRU execution but avoids per-forward
  CPU diagnostics and the data-dependent Python early break;
* expert execution uses the established BMM path for small routed groups and only
  switches to native grouped GEMM at/above the empirically measured crossover.

The reference v17 model remains the source of truth.  ``install_v17_systems_backend``
copies weights into interface-compatible modules and preserves state-dict keys and
parameter count so checkpoint semantics remain unchanged.
"""

from typing import Any

import torch

from .aera_hardware_core_v5 import DtypeSafeChunkLatentReasoner
from .aera_hardware_core_v6 import BMMHardSparseExpertBank
from .aera_hardware_core_v7 import NativeGroupedMMSparseExpertBank

# Frozen from the L4 kernel-opportunity measurement: grouped_mm remained slower
# through selected=32 and became faster at selected=64 and 128.
GROUPED_MM_SELECTED_CROSSOVER = 64


class LowOverheadHardSparseReasoner(DtypeSafeChunkLatentReasoner):
    """Exact hard-depth recurrence without hot-path host diagnostics.

    In hard mode this executes the same rows for the same number of GRUCell steps as
    ``DtypeSafeChunkLatentReasoner``.  Unlike the reference diagnostic path it does
    not compute unused soft depth expectations, copy chosen depths to CPU, or branch
    out of the Python loop when later selections become empty.  Empty selections are
    valid and perform no useful recurrent assignments.

    Soft/calibration mode delegates unchanged to the reference implementation.
    ``last_steps`` is retained on-device and therefore only synchronizes if a caller
    explicitly requests stats later.
    """

    def forward(
        self,
        summary: torch.Tensor,
        depth_logits: torch.Tensor,
        *,
        hard: bool,
    ) -> torch.Tensor:
        if not hard:
            return super().forward(summary, depth_logits, hard=False)
        if depth_logits.shape != (summary.size(0), self.max_steps):
            raise ValueError("depth_logits shape mismatch")

        chosen = depth_logits.argmax(dim=-1) + 1
        current = summary
        for step in range(1, self.max_steps + 1):
            idx = torch.nonzero(chosen >= step, as_tuple=False).squeeze(-1)
            selected = current.index_select(0, idx)
            updated = self.cell(selected, selected).to(dtype=current.dtype)
            current = current.index_copy(0, idx, updated)

        self.last_expected = None
        self.last_steps = chosen.detach()
        return current


class HybridMeasuredSparseExpertBank(NativeGroupedMMSparseExpertBank):
    """Choose BMM vs grouped GEMM using the measured selected-group crossover.

    This is a systems policy only.  Expert IDs, top-1/top-2 decisions, mixture
    weights, and stored parameters are unchanged.  Soft mode always keeps the
    established differentiable BMM/reference path.
    """

    def __init__(self, cfg, *, grouped_crossover: int = GROUPED_MM_SELECTED_CROSSOVER):
        super().__init__(cfg)
        if grouped_crossover < 1:
            raise ValueError("grouped_crossover must be >=1")
        self.grouped_crossover = int(grouped_crossover)

    @staticmethod
    def choose_hard_kernel(
        selected_batch: int,
        *,
        grouped_available: bool,
        grouped_crossover: int = GROUPED_MM_SELECTED_CROSSOVER,
    ) -> str:
        if selected_batch < 0:
            raise ValueError("selected_batch must be nonnegative")
        if grouped_crossover < 1:
            raise ValueError("grouped_crossover must be >=1")
        return (
            "grouped_mm"
            if grouped_available and selected_batch >= grouped_crossover
            else "bmm"
        )

    def forward(
        self,
        x: torch.Tensor,
        expert_logits: torch.Tensor,
        count_logits: torch.Tensor,
        *,
        hard: bool,
    ) -> torch.Tensor:
        if not hard:
            self.last_kernel = "soft_reference"
            self.last_input_dtype = str(x.dtype)
            self.last_compute_dtype = None
            return BMMHardSparseExpertBank.forward(
                self, x, expert_logits, count_logits, hard=False
            )

        grouped_available = self.native_grouped_mm_available(x)
        choice = self.choose_hard_kernel(
            int(x.size(0)),
            grouped_available=grouped_available,
            grouped_crossover=self.grouped_crossover,
        )
        if choice == "grouped_mm":
            return NativeGroupedMMSparseExpertBank.forward(
                self, x, expert_logits, count_logits, hard=True
            )

        self.last_kernel = "bmm_measured_threshold"
        self.last_input_dtype = str(x.dtype)
        self.last_compute_dtype = None
        return BMMHardSparseExpertBank.forward(
            self, x, expert_logits, count_logits, hard=True
        )


def install_v17_systems_backend(
    model: Any,
    *,
    grouped_crossover: int = GROUPED_MM_SELECTED_CROSSOVER,
) -> Any:
    """Install execution-only substitutions while preserving learned parameters.

    The function intentionally mutates only ``stage.experts`` and ``stage.reasoner``.
    All other modules—including routing, controllers, attention, state, memory and
    output heads—remain the exact objects already present in the loaded checkpoint.
    """

    stages = getattr(model, "stages", None)
    if stages is None:
        raise TypeError("model must expose AERA stages")

    for stage in stages:
        old_experts = stage.experts
        new_experts = HybridMeasuredSparseExpertBank(
            stage.cfg,
            grouped_crossover=grouped_crossover,
        ).to(device=old_experts.w1.device, dtype=old_experts.w1.dtype)
        new_experts.load_state_dict(old_experts.state_dict(), strict=True)
        stage.experts = new_experts

        old_reasoner = stage.reasoner
        reason_device = old_reasoner.cell.weight_ih.device
        reason_dtype = old_reasoner.cell.weight_ih.dtype
        new_reasoner = LowOverheadHardSparseReasoner(
            stage.cfg.d_model,
            stage.cfg.max_reason_steps,
        ).to(device=reason_device, dtype=reason_dtype)
        new_reasoner.load_state_dict(old_reasoner.state_dict(), strict=True)
        stage.reasoner = new_reasoner

    return model


def systems_backend_protocol() -> dict[str, object]:
    return {
        "architecture_changed": False,
        "routing_changed": False,
        "training_objective_changed": False,
        "checkpoint_weights_changed": False,
        "hard_selected_depth_changed": False,
        "hard_expert_selection_changed": False,
        "grouped_mm_selected_crossover": GROUPED_MM_SELECTED_CROSSOVER,
        "reasoner_host_diagnostics_removed_from_hot_path": True,
    }
