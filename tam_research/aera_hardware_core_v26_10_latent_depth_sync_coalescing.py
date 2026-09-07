from __future__ import annotations

"""AERA-v26.10 latent-depth synchronization coalescing for issue #706.

Execution-only optimization. Learned equations, parameters, checkpoint/state-dict
schema, hard depth choices, and physical GRU-row sparsity are preserved.
"""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .aera_hardware_core_v25_1_nohost import (
    ExecutionEquivalentNoHostDtypeSafeChunkLatentReasoner,
)


class CoalescedDepthPrefixLatentReasonerV2610(
    ExecutionEquivalentNoHostDtypeSafeChunkLatentReasoner
):
    """Exact hard-depth GRU math with one host count materialization per invocation."""

    def __init__(
        self,
        source: ExecutionEquivalentNoHostDtypeSafeChunkLatentReasoner,
    ) -> None:
        # Do not call the source class constructor because it expects the older
        # source reasoner type. Re-register only the exact inherited trainable cell
        # under the same module path and keep telemetry non-persistent.
        nn.Module.__init__(self)
        self.max_steps = source.max_steps
        self.cell = source.cell
        self.last_steps = source.last_steps
        self.last_expected = source.last_expected
        self.last_active_prefix_counts: tuple[int, ...] | None = None
        self.last_dense_masked_execution = False

    def forward(
        self,
        summary: torch.Tensor,
        depth_logits: torch.Tensor,
        *,
        hard: bool,
    ) -> torch.Tensor:
        if not hard:
            self.last_active_prefix_counts = None
            self.last_dense_masked_execution = False
            return super().forward(summary, depth_logits, hard=False)

        if depth_logits.shape != (summary.size(0), self.max_steps):
            raise ValueError("depth_logits shape mismatch")

        # Preserve the inherited telemetry math byte-for-equation.
        probs = F.softmax(depth_logits.float(), dim=-1).to(summary.dtype)
        values = torch.arange(
            1,
            self.max_steps + 1,
            device=summary.device,
            dtype=summary.dtype,
        )
        self.last_expected = (probs * values[None]).sum(dim=-1).detach()

        # Preserve the exact hard choice.
        chosen = depth_logits.argmax(dim=-1) + 1

        # One stable descending permutation makes every later active population a
        # prefix. Stable ordering preserves original row order within equal depths.
        order = torch.argsort(chosen, dim=0, descending=True, stable=True)
        ordered = summary.index_select(0, order)

        # Compute all four physical prefix sizes on device, then intentionally
        # materialize this tiny vector once. This replaces up to max_steps
        # variable-output nonzero()/idx.numel() host synchronization points.
        counts_device = torch.stack(
            tuple((chosen >= step).sum() for step in range(1, self.max_steps + 1))
        )
        counts = tuple(int(v) for v in counts_device.to(device="cpu").tolist())
        self.last_active_prefix_counts = counts
        self.last_dense_masked_execution = False

        for count in counts:
            if count == 0:
                break
            selected = ordered[:count]
            updated = self.cell(selected, selected).to(dtype=ordered.dtype)
            if count == ordered.size(0):
                ordered = updated
            else:
                # Only the physically active prefix executes the GRUCell. Tail rows
                # are carried forward unchanged; no dense masking receives credit.
                ordered = torch.cat((updated, ordered[count:]), dim=0)

        # Restore the exact original row order once after all recurrent steps.
        current = summary.index_copy(0, order, ordered)
        self.last_steps = chosen.detach()
        return current


def install_latent_depth_sync_coalescing_v26_10(model: nn.Module) -> tuple[int, ...]:
    """Replace only existing no-host reasoners, preserving their exact cell objects."""

    stages = getattr(model, "stages", None)
    if stages is None:
        raise TypeError("v26.10 installer requires a staged AERA model")
    installed: list[int] = []
    for stage_index, stage in enumerate(stages):
        reasoner = getattr(stage, "reasoner", None)
        if not isinstance(reasoner, ExecutionEquivalentNoHostDtypeSafeChunkLatentReasoner):
            raise TypeError(
                "v26.10 installer requires frozen no-host dtype-safe latent reasoners"
            )
        if isinstance(reasoner, CoalescedDepthPrefixLatentReasonerV2610):
            raise RuntimeError("v26.10 latent-depth coalescing already installed")
        old_cell = reasoner.cell
        replacement = CoalescedDepthPrefixLatentReasonerV2610(reasoner)
        if replacement.cell is not old_cell:
            raise RuntimeError("v26.10 installer changed the inherited GRUCell object")
        stage.reasoner = replacement
        installed.append(stage_index)
    if not installed:
        raise RuntimeError("v26.10 installer found no stages")
    return tuple(installed)


def latent_depth_sync_coalescing_v26_10_protocol() -> dict[str, Any]:
    return {
        "version": "aera-v26.10-latent-depth-sync-coalescing",
        "research_issue": 706,
        "source_main": "4883abe053347379812901c5c2057da70c11e3e7",
        "source_tree": "27be25ba578aa530c1e6d832db1a38be4c8bc22a",
        "target": "launch_or_idle_reduction",
        "hard_only_execution_change": True,
        "hard_depth_choice_changed": False,
        "gru_cell_equation_changed": False,
        "gru_updates_per_example_changed": False,
        "inactive_rows_execute_gru": False,
        "dense_masked_sparse_credit": False,
        "stable_descending_depth_permutation": True,
        "equal_depth_original_order_preserved": True,
        "active_population_representation": "stable depth prefix",
        "host_count_materializations_per_reasoner_invocation_target": 1,
        "legacy_per_depth_nonzero_compactions_target": 0,
        "max_reason_steps": 4,
        "optional_stage_routing_changed": False,
        "expert_routing_changed": False,
        "ficem_changed": False,
        "memory_write_changed": False,
        "durable_state_changed": False,
        "learned_parameter_count_changed": False,
        "parameter_schema_changed": False,
        "state_dict_schema_changed": False,
        "checkpoint_bytes_changed": False,
        "soft_training_path_changed": False,
        "gpu_authorized": False,
        "scientific_training_authorized": False,
        "systems_pass_earned": False,
        "architecture_freeze_authorized": False,
        "s2_authorized": False,
        "fresh_scientific_seed_authorized": False,
        "independent_replication_credit": False,
        "100m_authorized": False,
        "breakthrough_proven": False,
    }
