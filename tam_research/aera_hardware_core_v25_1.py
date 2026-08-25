from __future__ import annotations

"""AERA-v25.1 execution-equivalent sparse/FICEM runtime repair.

Preregistered in issue #380 after the checkpoint-only #379 localization.
This revision preserves v25 learned semantics and state while removing only three
redundant execution costs:

1. direct-dispatch the architecturally mandatory foundation stage in hard-sparse
   mode instead of gathering/scattering the entire batch and episodic state;
2. reuse the normalized factorized causal representation already computed for the
   stage read when the same completed chunk is scored for sparse writes;
3. return exact zeros before FICEM top-k work when an entire selected batch has an
   empty episodic state.

No routing policy, memory equation, learned parameter, write budget, capacity,
state shape, objective, or scientific threshold is changed.
"""

from typing import Any

import torch
import torch.nn as nn

from .aera import AERAState
from .aera_hardware_core import HardwareAERAConfig
from .aera_hardware_core_v19 import TokenwiseFastMemoryStage
from .aera_hardware_core_v23 import select_budgeted_event_pairs, sparse_write_budget
from .aera_hardware_core_v24 import (
    ContextualEpisodicMemoryState,
    VectorizedContextualEpisodicMemoryStage,
    episodic_state_bytes_per_session,
    _restore_epi_dtype,
)
from .aera_hardware_core_v25 import (
    FactorizedIdentityContextEpisodicMemory,
    FactorizedIdentityContextEpisodicMemoryStage,
    HardwareAwareAERATextLMV25,
    causal_identity_context,
    factorized_identity_context_protocol,
)
from .aera_hardware_core_v8 import StageRouteGate


class ExecutionEquivalentFactorizedIdentityContextMemory(
    FactorizedIdentityContextEpisodicMemory
):
    """V25 FICEM with an exact all-empty read fast path only."""

    def __init__(self, source: FactorizedIdentityContextEpisodicMemory) -> None:
        # Do not call the v25 constructor: it expects the older q/k/v/out source
        # shape and would initialize new parameters. Reuse every v25 module object.
        nn.Module.__init__(self)
        self.memory_dim = source.memory_dim
        self.identity_dim = source.identity_dim
        self.context_dim = source.context_dim
        self.capacity = source.capacity
        self.identity_proj = source.identity_proj
        self.context_proj = source.context_proj
        self.v = source.v
        self.out = source.out
        self.differentiable_pretraining = source.differentiable_pretraining
        self.empty_read_fastpath_calls = 0

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

        # Under the frozen FICEM read, an all-invalid state masks every retrieval
        # logit and therefore produces an exact zero weighted value sum. `out` is
        # bias-free, so returning the final zero tensor is mathematically exact.
        # This scalar guard is batch-wide only; mixed-valid batches use v25 below.
        if not bool(state.valid.any()):
            self.empty_read_fastpath_calls += 1
            return torch.zeros(
                identity_source.size(0),
                identity_source.size(1),
                self.out.out_features,
                device=identity_source.device,
                dtype=identity_source.dtype,
            )
        return super().read(identity_source, context_source, state)


class ExecutionEquivalentFICEMStage(FactorizedIdentityContextEpisodicMemoryStage):
    """V25 stage with within-call factor reuse and no scientific semantic change."""

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
        if not isinstance(source.memory, FactorizedIdentityContextEpisodicMemory):
            raise TypeError("v25.1 requires a v25 factorized episodic memory source")

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
        self.memory = ExecutionEquivalentFactorizedIdentityContextMemory(source.memory)
        self.last_start_controls = getattr(source, "last_start_controls", None)
        self.last_end_controls = getattr(source, "last_end_controls", None)
        self.last_selected_indices: torch.Tensor | None = None
        self.last_selected_count = 0
        self.last_candidate_count = 0
        self.last_pair_gate: torch.Tensor | None = None
        self.last_pair_strength: torch.Tensor | None = None
        self.last_vectorized_update_calls = 0
        self._runtime_factor_cache: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor
        ] | None = None

    def _tokenwise_context(
        self,
        h: torch.Tensor,
        state: AERAState,
        start_control: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(state.memory, ContextualEpisodicMemoryState):
            raise TypeError("v25.1 requires contextual episodic state")
        identity, causal_context, contextual = causal_identity_context(h)
        # Ephemeral within-forward cache only. It is cleared in `finally` before
        # forward_chunk exits, never appears in state_dict, and is not session state.
        self._runtime_factor_cache = (identity, causal_context, contextual)
        memory_read = self.memory.read(identity, causal_context, state.memory)
        carried = self.state_to_chunk(state.stream)
        context = (
            carried[:, None, :]
            + start_control["memory_read"][:, None, :] * memory_read
        )
        return context, memory_read

    def forward_chunk(
        self,
        events: torch.Tensor,
        state: AERAState | None,
        *,
        hard: bool,
        update_memory: bool,
    ) -> tuple[torch.Tensor, AERAState, dict[str, dict[str, torch.Tensor]]]:
        prior_state = state if state is not None else self.empty_state(events)
        if not isinstance(prior_state.memory, ContextualEpisodicMemoryState):
            raise TypeError("v25.1 stage received non-episodic memory state")

        self._runtime_factor_cache = None
        try:
            # Execute the exact inherited tokenwise stage core while suppressing its
            # legacy write. Dynamic dispatch calls our `_tokenwise_context`, so the
            # exact normalized factor tensors used by the read are retained once.
            h_out, next_state, controls = TokenwiseFastMemoryStage.forward_chunk(
                self,
                events,
                prior_state,
                hard=hard,
                update_memory=False,
            )
            factors = self._runtime_factor_cache
            if factors is None:
                raise RuntimeError("v25.1 factor cache was not populated by stage read")

            candidate_count = max(int(events.size(1)) - 1, 0)
            self.last_candidate_count = candidate_count
            self.last_selected_count = 0
            self.last_selected_indices = None
            self.last_vectorized_update_calls = 0
            if not update_memory or candidate_count == 0:
                return h_out, next_state, controls

            identity, causal_context, contextual = factors
            contextual_address = contextual[:, :-1]
            payload_source = contextual[:, 1:]
            pair_features = torch.cat((contextual_address, payload_source), dim=-1)
            pair_logits = self.pair_write_gate(pair_features)
            pair_gate = torch.sigmoid(pair_logits)
            end_control = controls["end"]
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

            gather = selected.indices.unsqueeze(-1).expand(
                -1, -1, identity.size(-1)
            )
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
            return h_out, AERAState(next_state.stream, memory_state), controls
        finally:
            # This invariant is part of #380's CPU gate: never retain graph tensors
            # or hidden session information after the stage invocation returns/raises.
            self._runtime_factor_cache = None


class HardwareAwareAERATextLMV251(HardwareAwareAERATextLMV25):
    """State-dict-compatible v25 with execution-only runtime repairs."""

    def __init__(
        self,
        cfg: HardwareAERAConfig = HardwareAERAConfig(),
        *,
        stream_forecast_tokens: int = 4,
    ) -> None:
        super().__init__(cfg, stream_forecast_tokens=stream_forecast_tokens)
        self.stages = nn.ModuleList(
            ExecutionEquivalentFICEMStage(stage) for stage in self.stages
        )
        self.set_memory_pretraining_mode(False)
        self.foundation_direct_dispatch_calls = 0

    def _v24_stages_ready(self) -> bool:
        # This virtual hook is called repeatedly during inherited construction.
        # At the v24 boundary the stages are VectorizedContextualEpisodicMemoryStage;
        # v25 factorized stages and v25.1 stages are subclasses of that same contract.
        # Returning true for the whole v24+ family preserves the inherited setter's
        # intended construction-time dispatch without recognizing older v18-v23 stages.
        return bool(self.stages) and all(
            isinstance(stage, VectorizedContextualEpisodicMemoryStage)
            for stage in self.stages
        )

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
        is_foundation = router is self.stage_routers[self.FOUNDATION_STAGE]
        if route_mode != "hard_sparse" or not is_foundation:
            return super()._route_one_stage(
                x,
                stage,
                stage_state,
                router,
                route_mode=route_mode,
                update_memory=update_memory,
            )

        # Stage0's router parameters are frozen by the inherited architecture to a
        # constant RUN decision. Preserve the router call/metadata exactly, but do
        # not gather the entire batch/state only to scatter it back unchanged.
        gate, logits = router(x[:, 0], stage_state.stream, mode=route_mode)
        prob = torch.sigmoid(logits)
        if not bool((gate[:, 0] >= 0.5).all()):
            raise RuntimeError("v25.1 foundation-stage invariant violated")
        processed, processed_state, controls = stage.forward_chunk(
            x,
            stage_state,
            hard=True,
            update_memory=update_memory,
        )
        processed = processed.to(dtype=x.dtype)
        processed_state = _restore_epi_dtype(stage_state, processed_state)
        self.foundation_direct_dispatch_calls += 1
        return processed, processed_state, {
            "stage_route_probability": prob,
            "stage_route_gate": gate,
            "executed_fraction": 1.0,
            "start": controls["start"],
            "end": controls["end"],
        }


def execution_equivalent_v25_1_protocol() -> dict[str, Any]:
    source = dict(factorized_identity_context_protocol())
    source.update(
        {
            "version": "aera-v25.1-execution-equivalent-runtime",
            "research_issue": 380,
            "source_version": "aera-v25-factorized-identity-context-episodic-memory",
            "learned_equations_changed": False,
            "learned_parameter_count_changed": False,
            "state_dict_schema_changed": False,
            "routing_policy_changed": False,
            "foundation_stage_policy_changed": False,
            "foundation_stage_execution": "direct dispatch; no full-batch gather/scatter",
            "write_factor_representation_changed": False,
            "write_factor_execution": "reuse exact within-call read factors",
            "runtime_factor_cache_persistent": False,
            "all_empty_read_semantics": "exact zero early return",
            "mixed_valid_read_changed": False,
            "write_budget_changed": False,
            "real_language_selected_writes": sparse_write_budget(255),
            "vectorized_update_calls_per_completed_stage_chunk": 1,
            "state_bytes_real_language_four_stage_memory_dim50": episodic_state_bytes_per_session(
                n_stages=4, memory_dim=50
            ),
            "gpu_authorized": False,
            "scientific_training_authorized": False,
            "100m_authorized": False,
        }
    )
    return source
