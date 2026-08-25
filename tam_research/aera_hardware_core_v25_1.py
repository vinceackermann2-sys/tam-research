from __future__ import annotations

"""AERA-v25.1 semantic-preserving execution repair (#381).

V25.1 changes no learned parameter, checkpoint key, routing rule, memory address,
read/write budget, durable state schema, stream recurrence, expert, latent-depth,
or decoder semantics.  It only removes execution fragmentation observed after the
seed8471 checkpoint diagnostic:

* router diagnostics remain on-device until stats() is explicitly requested;
* the router probability computed for the decision is reused by dispatch;
* all-batch hard routes bypass select/copy/merge state plumbing;
* FICEM read gathers top-k values from a flattened slot buffer;
* a stage computes normalized causal identity/context once and reuses it for the
  semantically identical read and write paths.
"""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .aera import AERAState
from .aera_hardware_core import HardwareAERAConfig
from .aera_hardware_core_v8 import StageRouteGate
from .aera_hardware_core_v23 import select_budgeted_event_pairs, sparse_write_budget
from .aera_hardware_core_v24 import (
    ContextualEpisodicMemoryState,
    MIN_STRENGTH,
    READ_TEMPERATURE,
    READ_TOP_K,
    _blend_epi_state,
    _merge_epi_state,
    _restore_epi_dtype,
    _select_epi_state,
)
from .aera_hardware_core_v25 import (
    FactorizedIdentityContextEpisodicMemory,
    FactorizedIdentityContextEpisodicMemoryStage,
    HardwareAwareAERATextLMV25,
    causal_identity_context,
)


class LazyStatsStageRouteGate(StageRouteGate):
    """Exact v25 route gate without unconditional device-to-host diagnostics."""

    def __init__(self, source: StageRouteGate, *, always_run: bool = False) -> None:
        # Reuse the exact projection module so state-dict names/parameters are unchanged.
        nn.Module.__init__(self)
        if not hasattr(source, "proj"):
            raise TypeError("source route gate missing proj")
        self.proj = source.proj
        self.last_probability: torch.Tensor | None = None
        self.last_hard_gate: torch.Tensor | None = None
        self.always_run = bool(always_run)
        self.host_stat_copies = 0

    def route_with_probability(
        self,
        first_event: torch.Tensor,
        stream: torch.Tensor,
        *,
        mode: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if first_event.ndim != 2 or stream.shape != first_event.shape:
            raise ValueError("stage router expects [batch,d_model] event and stream")
        if mode not in {"soft", "straight_through", "hard_sparse"}:
            raise ValueError(f"unknown stage routing mode: {mode}")
        logits = self.proj(torch.cat((first_event, stream), dim=-1))
        prob = torch.sigmoid(logits)
        hard = (prob >= 0.5).to(prob.dtype)
        if mode == "soft":
            gate = prob
        elif mode == "straight_through":
            gate = hard.detach() - prob.detach() + prob
        else:
            gate = hard
        # Keep detached diagnostics on the producing device.  Explicit stats() is
        # the only place that is allowed to synchronize them to Python scalars.
        self.last_probability = prob.detach()
        self.last_hard_gate = hard.detach()
        return gate, logits, prob

    def forward(
        self,
        first_event: torch.Tensor,
        stream: torch.Tensor,
        *,
        mode: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gate, logits, _ = self.route_with_probability(first_event, stream, mode=mode)
        return gate, logits

    def stats(self) -> dict[str, float] | None:
        if self.last_probability is None or self.last_hard_gate is None:
            return None
        return {
            "mean_run_probability": float(self.last_probability.float().mean().item()),
            "hard_run_fraction": float(self.last_hard_gate.float().mean().item()),
        }


class PackedFactorizedIdentityContextEpisodicMemory(
    FactorizedIdentityContextEpisodicMemory
):
    """FICEM with an equivalent flat top-k value gather."""

    def __init__(self, source: FactorizedIdentityContextEpisodicMemory) -> None:
        nn.Module.__init__(self)
        required = (
            "memory_dim",
            "identity_dim",
            "context_dim",
            "capacity",
            "identity_proj",
            "context_proj",
            "v",
            "out",
            "differentiable_pretraining",
        )
        missing = [name for name in required if not hasattr(source, name)]
        if missing:
            raise TypeError(f"v25 memory source missing fields: {missing}")
        self.memory_dim = int(source.memory_dim)
        self.identity_dim = int(source.identity_dim)
        self.context_dim = int(source.context_dim)
        self.capacity = int(source.capacity)
        self.identity_proj = source.identity_proj
        self.context_proj = source.context_proj
        self.v = source.v
        self.out = source.out
        self.differentiable_pretraining = bool(source.differentiable_pretraining)
        self.last_flat_value_gather_calls = 0

    def read(
        self,
        identity_source: torch.Tensor,
        context_source: torch.Tensor,
        state: ContextualEpisodicMemoryState,
    ) -> torch.Tensor:
        if identity_source.ndim != 3 or context_source.ndim != 3:
            raise ValueError("read sources must be [batch,time,d_model]")
        if identity_source.shape != context_source.shape:
            raise ValueError("identity/context read sources must match")
        if state.keys.shape != state.values.shape:
            raise ValueError("episodic key/value state mismatch")
        if state.keys.shape[:2] != state.valid.shape:
            raise ValueError("episodic validity shape mismatch")
        if state.strengths.shape != state.valid.shape:
            raise ValueError("episodic strength shape mismatch")

        _, _, query = self.address_factors(identity_source, context_source)
        keys = F.normalize(state.keys, dim=-1)
        similarity = torch.einsum("btd,bsd->bts", query, keys)
        strength_bias = torch.log(
            state.strengths.clamp(MIN_STRENGTH, 1.0)
        )[:, None, :]
        logits = (similarity + strength_bias) / READ_TEMPERATURE
        masked = logits.masked_fill(~state.valid[:, None, :], -torch.inf)
        top_k = min(READ_TOP_K, self.capacity)
        top_logits, top_indices = torch.topk(masked, k=top_k, dim=-1)
        top_valid = state.valid[:, None, :].expand(
            -1, identity_source.size(1), -1
        ).gather(-1, top_indices)
        safe_logits = top_logits.masked_fill(~top_valid, -1e9)
        weights = torch.softmax(safe_logits.float(), dim=-1).to(identity_source.dtype)
        weights = weights * top_valid.to(weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        batch, time, selected = top_indices.shape
        slots = state.values.size(1)
        offsets = (
            torch.arange(batch, device=top_indices.device, dtype=top_indices.dtype)
            .view(batch, 1, 1)
            * slots
        )
        flat_indices = (top_indices + offsets).reshape(-1)
        gathered_values = state.values.reshape(batch * slots, self.memory_dim).index_select(
            0, flat_indices
        ).view(batch, time, selected, self.memory_dim)
        self.last_flat_value_gather_calls += 1
        recalled = (weights.unsqueeze(-1) * gathered_values).sum(dim=2)
        return self.out(recalled)


class PackedFactorizedIdentityContextEpisodicMemoryStage(
    FactorizedIdentityContextEpisodicMemoryStage
):
    """V25 stage with one normalized causal-context construction per chunk."""

    def __init__(self, source: FactorizedIdentityContextEpisodicMemoryStage) -> None:
        nn.Module.__init__(self)
        required = (
            "cfg",
            "norm",
            "controller",
            "state_to_chunk",
            "attn",
            "experts",
            "reasoner",
            "stream_cell",
            "stream_input_norm",
            "reason_to_chunk",
            "out_norm",
            "pair_write_gate",
            "memory",
        )
        missing = [name for name in required if not hasattr(source, name)]
        if missing:
            raise TypeError(f"v25 source stage missing modules: {missing}")
        self.cfg = source.cfg
        self.norm = source.norm
        self.controller = source.controller
        self.state_to_chunk = source.state_to_chunk
        self.attn = source.attn
        self.experts = source.experts
        self.reasoner = source.reasoner
        self.stream_cell = source.stream_cell
        self.stream_input_norm = source.stream_input_norm
        self.reason_to_chunk = source.reason_to_chunk
        self.out_norm = source.out_norm
        self.pair_write_gate = source.pair_write_gate
        if not isinstance(source.memory, FactorizedIdentityContextEpisodicMemory):
            raise TypeError("v25.1 stage requires v25 FICEM source")
        self.memory = PackedFactorizedIdentityContextEpisodicMemory(source.memory)
        self.last_start_controls = getattr(source, "last_start_controls", None)
        self.last_end_controls = getattr(source, "last_end_controls", None)
        self.last_selected_indices: torch.Tensor | None = None
        self.last_selected_count = 0
        self.last_candidate_count = 0
        self.last_pair_gate: torch.Tensor | None = None
        self.last_pair_strength: torch.Tensor | None = None
        self.last_vectorized_update_calls = 0
        self.last_causal_context_builds = 0
        self.last_duplicate_write_context_recomputations = 0

    def forward_chunk(
        self,
        events: torch.Tensor,
        state: AERAState | None,
        *,
        hard: bool,
        update_memory: bool,
    ) -> tuple[torch.Tensor, AERAState, dict[str, dict[str, torch.Tensor]]]:
        if events.ndim != 3 or events.size(1) < 1:
            raise ValueError("events must be nonempty [batch,time,d_model]")
        prior_state = state if state is not None else self.empty_state(events)
        if not isinstance(prior_state.memory, ContextualEpisodicMemoryState):
            raise TypeError("v25.1 stage received non-episodic memory state")

        # This is the inherited v19 stage equation in the same operation order,
        # with the v25 identity/context tensors retained for the later write.
        base_h = self.norm(events)
        start_control = self.controller(base_h[:, 0], prior_state.stream)
        self.last_start_controls = {
            key: value.detach() for key, value in start_control.items()
        }
        identity, causal_context, contextual = causal_identity_context(base_h)
        self.last_causal_context_builds = 1
        self.last_duplicate_write_context_recomputations = 0
        memory_read = self.memory.read(identity, causal_context, prior_state.memory)
        carried = self.state_to_chunk(prior_state.stream)
        context = (
            carried[:, None, :]
            + start_control["memory_read"][:, None, :] * memory_read
        )
        h = base_h + context
        h = h + self.attn(h)
        h = h + self.experts(
            h,
            start_control["expert_logits"],
            start_control["expert_count_logits"],
            hard=hard,
        )

        summary = h[:, -1]
        end_control = self.controller(summary, prior_state.stream)
        self.last_end_controls = {
            key: value.detach() for key, value in end_control.items()
        }
        reasoned = self.reasoner(
            summary,
            end_control["depth_logits"],
            hard=hard,
        )
        reason_chunk = h.new_zeros(h.shape)
        reason_chunk[:, -1] = self.reason_to_chunk(reasoned)
        h = self.out_norm(h + reason_chunk)
        stream_input = self.stream_input_norm(reasoned)
        stream = self.stream_cell(stream_input, prior_state.stream)
        next_state = AERAState(stream, prior_state.memory)
        controls = {"start": start_control, "end": end_control}

        candidate_count = max(int(events.size(1)) - 1, 0)
        self.last_candidate_count = candidate_count
        self.last_selected_count = 0
        self.last_selected_indices = None
        self.last_vectorized_update_calls = 0
        if not update_memory or candidate_count == 0:
            return h, next_state, controls

        contextual_address = contextual[:, :-1]
        payload_source = contextual[:, 1:]
        pair_features = torch.cat((contextual_address, payload_source), dim=-1)
        pair_logits = self.pair_write_gate(pair_features)
        pair_gate = torch.sigmoid(pair_logits)
        chunk_strength = (
            end_control["novelty"] * end_control["memory_write"]
        ).clamp(0.0, 1.0)
        write_strength = pair_gate * chunk_strength[:, None, :]
        selected = select_budgeted_event_pairs(
            contextual_address,
            payload_source,
            write_strength,
            pair_logits,
            differentiable_selector=self.memory.differentiable_pretraining,
        )
        if selected.hard_count != sparse_write_budget(candidate_count):
            raise RuntimeError("v25.1 sparse write budget mismatch")

        gather = selected.indices.unsqueeze(-1).expand(-1, -1, identity.size(-1))
        selected_identity = identity[:, :-1].gather(1, gather)
        selected_context = causal_context[:, :-1].gather(1, gather)
        self.last_pair_gate = pair_gate.detach()
        self.last_pair_strength = selected.strength.detach()
        self.last_selected_indices = selected.indices.detach()
        self.last_selected_count = selected.hard_count
        memory_state = self.memory.update_block(
            selected_identity,
            selected_context,
            selected.payload,
            selected.strength,
            prior_state.memory,
        )
        self.last_vectorized_update_calls = 1
        return h, AERAState(stream, memory_state), controls


class HardwareAwareAERATextLMV25_1(HardwareAwareAERATextLMV25):
    """Checkpoint-compatible v25 with packed/lazy execution only."""

    def __init__(
        self,
        cfg: HardwareAERAConfig = HardwareAERAConfig(),
        *,
        stream_forecast_tokens: int = 4,
    ) -> None:
        super().__init__(cfg, stream_forecast_tokens=stream_forecast_tokens)
        self.stages = nn.ModuleList(
            PackedFactorizedIdentityContextEpisodicMemoryStage(stage)
            for stage in self.stages
        )
        self.stage_routers = nn.ModuleList(
            LazyStatsStageRouteGate(router, always_run=(i == self.FOUNDATION_STAGE))
            for i, router in enumerate(self.stage_routers)
        )
        self.last_full_batch_route_fastpaths = 0
        self.last_partial_route_packs = 0
        self.last_empty_route_skips = 0
        self.last_host_stat_copies = 0
        self.set_memory_pretraining_mode(False)

    def _route_one_stage(
        self,
        x: torch.Tensor,
        stage: nn.Module,
        stage_state: AERAState,
        router: StageRouteGate,
        *,
        route_mode: str,
        update_memory: bool,
    ) -> tuple[torch.Tensor, AERAState, dict[str, object]]:
        if not isinstance(router, LazyStatsStageRouteGate):
            raise TypeError("v25.1 requires LazyStatsStageRouteGate")
        gate, logits, prob = router.route_with_probability(
            x[:, 0], stage_state.stream, mode=route_mode
        )

        if route_mode == "hard_sparse":
            # Stage0 is frozen to always run by v12.  Preserve its route output but
            # bypass any sparse selection machinery entirely.
            if router.always_run:
                processed, processed_state, controls = stage.forward_chunk(
                    x,
                    stage_state,
                    hard=True,
                    update_memory=update_memory,
                )
                processed = processed.to(dtype=x.dtype)
                processed_state = _restore_epi_dtype(stage_state, processed_state)
                self.last_full_batch_route_fastpaths += 1
                return processed, processed_state, {
                    "stage_route_probability": prob,
                    "stage_route_gate": gate,
                    "executed_fraction": 1.0,
                    "start": controls["start"],
                    "end": controls["end"],
                }

            run_idx = (gate[:, 0] >= 0.5).nonzero(as_tuple=False).squeeze(-1)
            if run_idx.numel() == 0:
                self.last_empty_route_skips += 1
                return x, stage_state, {
                    "stage_route_probability": prob,
                    "stage_route_gate": gate,
                    "executed_fraction": 0.0,
                    "start": None,
                    "end": None,
                }
            if run_idx.numel() == x.size(0):
                processed, processed_state, controls = stage.forward_chunk(
                    x,
                    stage_state,
                    hard=True,
                    update_memory=update_memory,
                )
                processed = processed.to(dtype=x.dtype)
                processed_state = _restore_epi_dtype(stage_state, processed_state)
                self.last_full_batch_route_fastpaths += 1
                return processed, processed_state, {
                    "stage_route_probability": prob,
                    "stage_route_gate": gate,
                    "executed_fraction": 1.0,
                    "start": controls["start"],
                    "end": controls["end"],
                }

            selected_x = x.index_select(0, run_idx)
            selected_state = _select_epi_state(stage_state, run_idx)
            selected_y, selected_new_state, selected_controls = stage.forward_chunk(
                selected_x,
                selected_state,
                hard=True,
                update_memory=update_memory,
            )
            selected_y = selected_y.to(dtype=x.dtype)
            selected_new_state = _restore_epi_dtype(selected_state, selected_new_state)
            self.last_partial_route_packs += 1
            return (
                x.index_copy(0, run_idx, selected_y),
                _merge_epi_state(stage_state, selected_new_state, run_idx),
                {
                    "stage_route_probability": prob,
                    "stage_route_gate": gate,
                    "executed_fraction": float(run_idx.numel() / x.size(0)),
                    "start": selected_controls["start"],
                    "end": selected_controls["end"],
                },
            )

        isolate = bool(
            route_mode == "straight_through"
            and getattr(self, "_isolate_router_task_gradient", False)
        )
        task_gate = gate.detach() if isolate else gate
        processed, processed_state, controls = stage.forward_chunk(
            x,
            stage_state,
            hard=False,
            update_memory=update_memory,
        )
        processed = processed.to(dtype=x.dtype)
        processed_state = _restore_epi_dtype(stage_state, processed_state)
        gate_for_residual = task_gate.to(dtype=x.dtype)
        y = x + gate_for_residual[:, None, :] * (processed - x)
        new_state = _blend_epi_state(
            stage_state,
            processed_state,
            task_gate,
            hard_validity=route_mode == "straight_through",
        )
        info: dict[str, object] = {
            "stage_route_probability": prob,
            "stage_route_gate": task_gate,
            "executed_fraction": 1.0,
            "start": controls["start"],
            "end": controls["end"],
        }
        if isolate:
            info["task_router_gradient_isolated"] = True
        return y, new_state, info

    def forward(self, *args: Any, **kwargs: Any) -> dict[str, object]:
        self.last_full_batch_route_fastpaths = 0
        self.last_partial_route_packs = 0
        self.last_empty_route_skips = 0
        result = super().forward(*args, **kwargs)
        self.last_host_stat_copies = sum(
            int(router.host_stat_copies)
            for router in self.stage_routers
            if isinstance(router, LazyStatsStageRouteGate)
        )
        return result

    def execution_stats(self) -> dict[str, int]:
        return {
            "full_batch_route_fastpaths": self.last_full_batch_route_fastpaths,
            "partial_route_packs": self.last_partial_route_packs,
            "empty_route_skips": self.last_empty_route_skips,
            "host_stat_copies": self.last_host_stat_copies,
            "causal_context_builds_last_stage_calls": sum(
                int(stage.last_causal_context_builds)
                for stage in self.stages
                if isinstance(stage, PackedFactorizedIdentityContextEpisodicMemoryStage)
            ),
            "duplicate_write_context_recomputations": sum(
                int(stage.last_duplicate_write_context_recomputations)
                for stage in self.stages
                if isinstance(stage, PackedFactorizedIdentityContextEpisodicMemoryStage)
            ),
        }


def packed_execution_protocol() -> dict[str, object]:
    return {
        "version": "aera-v25.1-semantic-preserving-packed-execution",
        "source": "aera-v25",
        "research_issue": 381,
        "learned_parameters_changed": False,
        "checkpoint_keys_changed": False,
        "routing_math_changed": False,
        "memory_address_math_changed": False,
        "memory_update_math_changed": False,
        "write_budget_changed": False,
        "read_top_k_changed": False,
        "read_temperature_changed": False,
        "persistent_state_schema_changed": False,
        "router_diagnostics_host_copy_on_forward": False,
        "router_probability_recomputed_in_dispatch": False,
        "all_batch_hard_route_select_merge_required": False,
        "duplicate_causal_context_for_write": False,
        "flat_topk_value_gather": True,
        "gpu_authorized": False,
        "100m_authorized": False,
    }
