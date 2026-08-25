from __future__ import annotations

"""AERA-v25.1 execution-equivalent sparse/FICEM runtime repair.

Preregistered in issue #380 after the checkpoint-only #379 localization and
continued under the broader execution-only systems boundary in issue #381.
This revision preserves v25 learned semantics and durable state while removing
only redundant execution costs:

1. direct-dispatch the architecturally mandatory foundation stage in hard-sparse
   mode instead of gathering/scattering the entire batch and episodic state;
2. reuse the causal representation already computed for the stage read when the
   same completed chunk is scored for sparse writes;
3. return exact zeros before FICEM top-k work when an entire selected batch is
   *known* empty, without forcing a CUDA tensor-to-host synchronization;
4. on a nonempty read, reuse that read's exact projected query and normalized
   prior-state keys in the same chunk's vectorized write update instead of
   projecting/normalizing the same tensors again;
5. retain router telemetry detached on the execution device during forward,
   deferring any host materialization until diagnostics explicitly request stats.

No routing policy, memory equation, learned parameter, write budget, capacity,
durable state shape, objective, or scientific threshold is changed.
"""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .aera import AERAState
from .aera_hardware_core import HardwareAERAConfig
from .aera_hardware_core_v19 import TokenwiseFastMemoryStage
from .aera_hardware_core_v23 import select_budgeted_event_pairs, sparse_write_budget
from .aera_hardware_core_v24 import (
    DUPLICATE_SIMILARITY,
    MIN_STRENGTH,
    READ_TEMPERATURE,
    READ_TOP_K,
    ContextualEpisodicMemoryState,
    VectorizedContextualEpisodicMemoryStage,
    _gather_slots,
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


_KNOWN_EMPTY_HINT = "_v25_1_known_empty"


def _set_known_empty_hint(state: ContextualEpisodicMemoryState, value: bool) -> None:
    # Python-only execution metadata. ContextualEpisodicMemoryState is deliberately
    # not slotted, so this does not change the dataclass/durable session schema.
    object.__setattr__(state, _KNOWN_EMPTY_HINT, bool(value))


def _known_empty_hint(state: ContextualEpisodicMemoryState) -> bool:
    return bool(getattr(state, _KNOWN_EMPTY_HINT, False))


class ExecutionEquivalentStageRouteGate(StageRouteGate):
    """Exact StageRouteGate math without eager CUDA-to-host telemetry copies."""

    def __init__(self, source: StageRouteGate) -> None:
        if not isinstance(source, StageRouteGate):
            raise TypeError("v25.1 router wrapper requires a StageRouteGate source")
        # Do not call StageRouteGate.__init__: that would allocate fresh parameters.
        nn.Module.__init__(self)
        self.proj = source.proj
        self.last_probability = source.last_probability
        self.last_hard_gate = source.last_hard_gate

    def forward(
        self,
        first_event: torch.Tensor,
        stream: torch.Tensor,
        *,
        mode: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if first_event.ndim != 2 or stream.shape != first_event.shape:
            raise ValueError("stage router expects [batch,d_model] event and stream")
        if mode not in {"soft", "straight_through", "hard_sparse"}:
            raise ValueError(f"unknown stage routing mode: {mode}")

        # Keep the exact inherited gate operation sequence. Only the detached
        # diagnostic storage stays on-device instead of synchronously copying to CPU.
        logits = self.proj(torch.cat((first_event, stream), dim=-1))
        prob = torch.sigmoid(logits)
        hard = (prob >= 0.5).to(prob.dtype)
        if mode == "soft":
            gate = prob
        elif mode == "straight_through":
            gate = hard.detach() - prob.detach() + prob
        else:
            gate = hard

        self.last_probability = prob.detach()
        self.last_hard_gate = hard.detach()
        return gate, logits


class ExecutionEquivalentFactorizedIdentityContextMemory(
    FactorizedIdentityContextEpisodicMemory
):
    """V25 FICEM with exact non-synchronizing and same-call reuse paths."""

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
        self.projected_update_reuse_calls = 0

    def empty_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> ContextualEpisodicMemoryState:
        state = super().empty_state(batch_size, device, dtype)
        _set_known_empty_hint(state, True)
        return state

    def update_block(
        self,
        identity_source: torch.Tensor,
        context_source: torch.Tensor,
        payload_source: torch.Tensor,
        write_strength: torch.Tensor,
        state: ContextualEpisodicMemoryState,
    ) -> ContextualEpisodicMemoryState:
        next_state = super().update_block(
            identity_source,
            context_source,
            payload_source,
            write_strength,
            state,
        )
        # Conservatively stop claiming emptiness after any write attempt. If every
        # strength happened to be zero, this is only a missed optimization: the
        # ordinary v25 read remains exact. No device scalar is inspected here.
        _set_known_empty_hint(next_state, False)
        return next_state

    def read_with_reuse(
        self,
        identity_source: torch.Tensor,
        context_source: torch.Tensor,
        state: ContextualEpisodicMemoryState,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Run exact v25 read math and expose safe same-call reusable tensors.

        The second/third results are the exact projected query and normalized prior
        keys used by this read. They are ephemeral graph tensors only; callers must
        never persist them in session state. Known-empty reads deliberately return
        ``None`` for both so the cheap empty path does not perform extra work.
        """
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

        known_empty = _known_empty_hint(state)
        cpu_verified_empty = (
            state.valid.device.type == "cpu" and not bool(state.valid.any())
        )
        if known_empty or cpu_verified_empty:
            self.empty_read_fastpath_calls += 1
            return (
                torch.zeros(
                    identity_source.size(0),
                    identity_source.size(1),
                    self.out.out_features,
                    device=identity_source.device,
                    dtype=identity_source.dtype,
                ),
                None,
                None,
            )

        # Keep the operation sequence identical to v25's nonempty read. The only
        # difference is retaining `query` and `keys` for this same stage invocation.
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

        expanded_values = state.values[:, None, :, :].expand(
            -1, identity_source.size(1), -1, -1
        )
        gathered_values = expanded_values.gather(
            2,
            top_indices.unsqueeze(-1).expand(-1, -1, -1, self.memory_dim),
        )
        recalled = (weights.unsqueeze(-1) * gathered_values).sum(dim=2)
        return self.out(recalled), query, keys

    def read(
        self,
        identity_source: torch.Tensor,
        context_source: torch.Tensor,
        state: ContextualEpisodicMemoryState,
    ) -> torch.Tensor:
        recalled, _, _ = self.read_with_reuse(identity_source, context_source, state)
        return recalled

    def _vectorized_update_from_projected(
        self,
        projected_new_keys: torch.Tensor,
        normalized_old_keys: torch.Tensor,
        payload_source: torch.Tensor,
        write_strength: torch.Tensor,
        state: ContextualEpisodicMemoryState,
        *,
        detach_inputs: bool,
    ) -> ContextualEpisodicMemoryState:
        """Exact v25 update after substituting already-computed equivalent tensors."""
        new_keys = projected_new_keys.detach() if detach_inputs else projected_new_keys
        normalized_old = (
            normalized_old_keys.detach() if detach_inputs else normalized_old_keys
        )
        payload = payload_source.detach() if detach_inputs else payload_source
        strength = write_strength.detach() if detach_inputs else write_strength
        old_keys = state.keys.detach() if detach_inputs else state.keys
        old_values = state.values.detach() if detach_inputs else state.values
        old_strengths = state.strengths.detach() if detach_inputs else state.strengths
        old_valid = state.valid.detach()

        new_values = torch.tanh(self.v(payload))
        new_strengths = strength[..., 0].clamp(0.0, 1.0)
        new_valid = new_strengths > 0.0

        incoming_similarity = torch.einsum("bkd,bjd->bkj", new_keys, new_keys)
        k_count = new_keys.size(1)
        position = torch.arange(k_count, device=new_keys.device)
        later = position[None, :, None] < position[None, None, :]
        shadowed_incoming = (
            incoming_similarity.ge(DUPLICATE_SIMILARITY)
            & new_valid[:, :, None]
            & new_valid[:, None, :]
            & later
        ).any(dim=2)
        new_valid = new_valid & ~shadowed_incoming

        similarity = torch.einsum("bkd,bsd->bks", new_keys, normalized_old)
        duplicate_old = (
            similarity.ge(DUPLICATE_SIMILARITY)
            & new_valid[:, :, None]
            & old_valid[:, None, :]
        ).any(dim=1)
        keep_old = old_valid & ~duplicate_old

        new_keys = new_keys.flip(1)
        new_values = new_values.flip(1)
        new_strengths = new_strengths.flip(1)
        new_valid = new_valid.flip(1)

        all_keys = torch.cat((new_keys, old_keys), dim=1)
        all_values = torch.cat((new_values, old_values), dim=1)
        all_strengths = torch.cat((new_strengths, old_strengths), dim=1)
        all_valid = torch.cat((new_valid, keep_old), dim=1)
        total = all_valid.size(1)
        slot_position = torch.arange(total, device=all_valid.device, dtype=torch.float32)
        priority = all_valid.float() * 2.0 - slot_position[None, :] * 1e-6
        keep_indices = torch.topk(
            priority,
            k=self.capacity,
            dim=1,
            largest=True,
            sorted=True,
        ).indices
        next_state = ContextualEpisodicMemoryState(
            keys=_gather_slots(all_keys, keep_indices),
            values=_gather_slots(all_values, keep_indices),
            strengths=_gather_slots(all_strengths, keep_indices),
            valid=_gather_slots(all_valid, keep_indices),
        )
        return next_state.detach() if detach_inputs else next_state

    def update_block_from_projected(
        self,
        projected_new_keys: torch.Tensor,
        normalized_old_keys: torch.Tensor,
        payload_source: torch.Tensor,
        write_strength: torch.Tensor,
        state: ContextualEpisodicMemoryState,
    ) -> ContextualEpisodicMemoryState:
        if projected_new_keys.ndim != 3:
            raise ValueError("projected write keys must be [batch,candidates,memory_dim]")
        if projected_new_keys.size(-1) != self.memory_dim:
            raise ValueError("projected write key dimension mismatch")
        if payload_source.ndim != 3 or payload_source.shape[:2] != projected_new_keys.shape[:2]:
            raise ValueError("projected keys and payload batch/candidate axes must match")
        if write_strength.shape != (*projected_new_keys.shape[:-1], 1):
            raise ValueError("write_strength must be [batch,candidates,1]")
        if normalized_old_keys.shape != state.keys.shape:
            raise ValueError("normalized prior-key shape mismatch")

        if self.differentiable_pretraining:
            next_state = self._vectorized_update_from_projected(
                projected_new_keys,
                normalized_old_keys,
                payload_source,
                write_strength,
                state,
                detach_inputs=False,
            )
        else:
            with torch.no_grad():
                next_state = self._vectorized_update_from_projected(
                    projected_new_keys,
                    normalized_old_keys,
                    payload_source,
                    write_strength,
                    state,
                    detach_inputs=True,
                )
        self.projected_update_reuse_calls += 1
        _set_known_empty_hint(next_state, False)
        return next_state


class ExecutionEquivalentFICEMStage(FactorizedIdentityContextEpisodicMemoryStage):
    """V25 stage with within-call tensor reuse and no scientific semantic change."""

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
        self.last_reused_read_key_update_calls = 0
        self._runtime_factor_cache: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor | None,
            torch.Tensor | None,
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
        memory_read, projected_query, normalized_old_keys = self.memory.read_with_reuse(
            identity,
            causal_context,
            state.memory,
        )
        # Ephemeral within-forward cache only. It is cleared in `finally` before
        # forward_chunk exits, never appears in state_dict, and is not session state.
        self._runtime_factor_cache = (
            identity,
            causal_context,
            contextual,
            projected_query,
            normalized_old_keys,
        )
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
            # legacy write. Dynamic dispatch calls our `_tokenwise_context`.
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
            self.last_reused_read_key_update_calls = 0
            if not update_memory or candidate_count == 0:
                return h_out, next_state, controls

            (
                identity,
                causal_context,
                contextual,
                projected_query,
                normalized_old_keys,
            ) = factors
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

            self.last_pair_gate = pair_gate.detach()
            self.last_pair_strength = selected.strength.detach()
            self.last_selected_indices = selected.indices.detach()
            self.last_selected_count = selected.hard_count

            if projected_query is not None and normalized_old_keys is not None:
                projected_gather = selected.indices.unsqueeze(-1).expand(
                    -1, -1, projected_query.size(-1)
                )
                selected_projected_keys = projected_query[:, :-1].gather(
                    1, projected_gather
                )
                memory_state = self.memory.update_block_from_projected(
                    selected_projected_keys,
                    normalized_old_keys,
                    selected.payload,
                    selected.strength,
                    prior_state.memory,
                )
                self.last_reused_read_key_update_calls = 1
            else:
                # The known-empty read intentionally did no projection/normalization;
                # retain the existing update path rather than defeating that fast path.
                source_gather = selected.indices.unsqueeze(-1).expand(
                    -1, -1, identity.size(-1)
                )
                selected_identity = identity[:, :-1].gather(1, source_gather)
                selected_context = causal_context[:, :-1].gather(1, source_gather)
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
            # Never retain graph tensors or hidden session information after the
            # stage invocation; same invariant as the earlier #380/#381 repair.
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
        self.stage_routers = nn.ModuleList(
            ExecutionEquivalentStageRouteGate(router) for router in self.stage_routers
        )
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
        # constant RUN decision. Preserve its call/metadata exactly. CPU CI checks
        # the invariant; CUDA does not synchronize a tensor solely to re-prove it.
        gate, logits = router(x[:, 0], stage_state.stream, mode=route_mode)
        prob = torch.sigmoid(logits)
        if gate.device.type == "cpu" and not bool((gate[:, 0] >= 0.5).all()):
            raise RuntimeError("v25.1 foundation-stage invariant violated")
        prior_known_empty = _known_empty_hint(stage_state.memory)
        processed, processed_state, controls = stage.forward_chunk(
            x,
            stage_state,
            hard=True,
            update_memory=update_memory,
        )
        processed = processed.to(dtype=x.dtype)
        processed_state = _restore_epi_dtype(stage_state, processed_state)
        # _restore_epi_dtype rebuilds the dataclass and intentionally drops dynamic
        # attributes. Preserve only the exact known-empty fact when no write was
        # requested; this lets consecutive empty chunks avoid CUDA scalar guards.
        if prior_known_empty and not update_memory:
            _set_known_empty_hint(processed_state.memory, True)
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
            "systems_authority_issue": 381,
            "source_version": "aera-v25-factorized-identity-context-episodic-memory",
            "learned_equations_changed": False,
            "learned_parameter_count_changed": False,
            "state_dict_schema_changed": False,
            "routing_policy_changed": False,
            "router_gate_math_changed": False,
            "router_telemetry_forward_host_copy": False,
            "router_telemetry_storage": "detached on execution device; stats materializes only when explicitly called",
            "router_state_dict_changed": False,
            "foundation_stage_policy_changed": False,
            "foundation_stage_execution": "direct dispatch; no full-batch gather/scatter",
            "write_factor_representation_changed": False,
            "write_factor_execution": "reuse exact within-call read factors and nonempty projected query/prior-key normalization",
            "projected_read_query_reused_for_write": True,
            "normalized_prior_keys_reused_for_write": True,
            "known_empty_write_reuse_fallback": "existing v25.1 update path",
            "runtime_factor_cache_persistent": False,
            "all_empty_read_semantics": "exact zero for known-empty state; exact v25 fallback otherwise",
            "known_empty_hint_persistent": False,
            "cuda_scalar_empty_read_sync": False,
            "cuda_scalar_foundation_invariant_sync": False,
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