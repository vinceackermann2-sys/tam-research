from __future__ import annotations

"""CPU-only #770 integration of #763 event addressing into the real #748 AERA path."""

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F

from tam_research import aera_issue748_memory_capability_seed1_harness_base as base
from tam_research.aera import AERAState, FastMemoryState
from tam_research.aera_hardware_core import HardwareAERAConfig, HardwareAERAState
from tam_research.aera_hardware_core_v18 import PretrainableDeltaFastMemory

RESEARCH_ISSUE = 770
PARENT_REPAIR_ISSUE = 763
FAILED_RESEARCH_ISSUE = 748
SOURCE_MAIN = "6cb1882aa5b02e4f8917a7cae44e6d1925b9e290"
SOURCE_TREE = "4892636545a54887ebe3079bc68597dba425a8e7"
CPU_TEST_SEED = 770_999
CONSUMED_SCIENTIFIC_SEED = 17_641
FROZEN_REPAIR_BLOB = "62ac12a9f7ecf5e50313853c79ca4c25bc8a3730"
FROZEN_REPAIR_TEST_BLOB = "8adf009295183590723c6630816706a8d9dc9796"
FROZEN_WRAPPER_BLOB = "afc939a69633f68ded05eb585c95a599a8f5c981"
FROZEN_BASE_BLOB = "40003b68987d026265b1d53fcb637f18277f9cac"
FROZEN_FIXTURE_BLOB = "981602432b684989f7a5011ff953e6965368c5e5"
FROZEN_V18_BLOB = "97861a2407876f62665b13140c2135b4a11d4597"
FROZEN_DELTA_BLOB = "ec0b5d29b3d4ac27bd60fd9c152480b9f177c3e9"
PAD, WRITE, UPDATE, QUERY, ANSWER, RESET, SEP, UNKNOWN = range(8)
GPU_AUTHORIZED = MODAL_AUTHORIZED = SCIENTIFIC_TRAINING_AUTHORIZED = False
SCIENTIFIC_EVALUATION_AUTHORIZED = SCIENTIFIC_SEED_AUTHORIZED = False
SEEDS_2_3_AUTHORIZED = SCIENTIFIC_RESULT_AUTHORIZED = False
SCIENTIFIC_CHECKPOINT_AUTHORIZED = SYSTEMS_OPTIMIZATION_AUTHORIZED = False
ARCHITECTURE_FREEZE_AUTHORIZED = SCALING_AUTHORIZED = BREAKTHROUGH_PROVEN = False


@dataclass(frozen=True)
class ParsedEvent:
    kind: str
    marker_index: int
    key_index: int | None = None
    value_index: int | None = None


def parse_events(tokens: Sequence[int]) -> tuple[ParsedEvent, ...]:
    out: list[ParsedEvent] = []
    i = 0
    while i < len(tokens):
        marker = int(tokens[i])
        if marker in {WRITE, UPDATE}:
            if i + 3 >= len(tokens) or int(tokens[i + 3]) != SEP:
                raise ValueError("malformed WRITE/UPDATE record")
            out.append(ParsedEvent("update" if marker == UPDATE else "write", i, i + 1, i + 2))
            i += 4
            continue
        if marker == QUERY:
            if i + 2 >= len(tokens) or int(tokens[i + 2]) != ANSWER:
                raise ValueError("malformed QUERY record")
            out.append(ParsedEvent("query", i, i + 1, None))
            i += 3
            continue
        if marker == RESET:
            out.append(ParsedEvent("reset", i))
            i += 1
            if i < len(tokens) and int(tokens[i]) == SEP:
                i += 1
            continue
        i += 1
    return tuple(out)


def _delta_impl(memory, key_x, value_x, strength, state):
    if key_x.ndim != 3 or key_x.size(1) != 1 or value_x.shape != key_x.shape:
        raise ValueError("event representations must be matching [batch,1,d_model]")
    if strength.shape != (*key_x.shape[:-1], 1):
        raise ValueError("strength must be [batch,1,1]")
    key = F.normalize(memory.k(key_x).squeeze(1), dim=-1)
    target = torch.tanh(memory.v(value_x).squeeze(1))
    matrix = memory.decay * state.matrix
    prediction = torch.einsum("bi,bij->bj", key, matrix)
    error = target - prediction
    eta = memory.lr * strength[:, 0].clamp(0.0, 1.0)
    matrix = matrix + torch.einsum("bi,bj->bij", key * eta, error)
    return FastMemoryState(matrix)


def event_aligned_delta_update(memory: PretrainableDeltaFastMemory, key_x, value_x, strength, state):
    """Production delta equation/parameters, but explicit causal key/value event inputs."""
    if memory.differentiable_pretraining:
        return _delta_impl(memory, key_x, value_x, strength, state)
    with torch.no_grad():
        updated = _delta_impl(
            memory,
            key_x.detach(),
            value_x.detach(),
            strength.detach(),
            FastMemoryState(state.matrix.detach()),
        )
    return FastMemoryState(updated.matrix.detach())


def _row_state(state: AERAState, row: int) -> AERAState:
    idx = torch.tensor([row], device=state.stream.device)
    return AERAState(
        stream=state.stream.index_select(0, idx),
        memory=FastMemoryState(state.memory.matrix.index_select(0, idx)),
    )


def _process_row(stage, source, raw_row, state, row: int, *, update_memory: bool):
    row_state = _row_state(state, row)
    memory_state = row_state.memory
    read = torch.zeros(1, source.size(-1), device=source.device, dtype=source.dtype)
    query_index = None
    trace: list[dict[str, object]] = []
    for event in parse_events(raw_row):
        if event.kind == "reset":
            memory_state = FastMemoryState(torch.zeros_like(memory_state.matrix))
            trace.append({"kind": "reset", "marker_index": event.marker_index})
            continue
        if event.kind in {"write", "update"}:
            assert event.key_index is not None and event.value_index is not None
            key_x = source[row:row + 1, event.key_index:event.key_index + 1]
            value_x = source[row:row + 1, event.value_index:event.value_index + 1]
            if update_memory:
                control = stage.controller(value_x.squeeze(1), row_state.stream)
                strength = (control["novelty"] * control["memory_write"]).clamp(0.0, 1.0)[:, None, :]
                memory_state = event_aligned_delta_update(stage.memory, key_x, value_x, strength, memory_state)
            trace.append({
                "kind": event.kind,
                "marker_index": event.marker_index,
                "key_index": event.key_index,
                "value_index": event.value_index,
                "key_input": key_x[0, 0].detach().clone(),
                "value_input": value_x[0, 0].detach().clone(),
            })
            continue
        if event.kind == "query":
            assert event.key_index is not None
            key_x = source[row:row + 1, event.key_index:event.key_index + 1]
            read = stage.memory.read(key_x, memory_state).squeeze(1)
            query_index = event.key_index
            trace.append({
                "kind": "query",
                "marker_index": event.marker_index,
                "key_index": event.key_index,
                "read_input": key_x[0, 0].detach().clone(),
                "recall": read[0].detach().clone(),
            })
    return memory_state, read, query_index, tuple(trace)


class IntegratedEventMemoryCapabilityLM(base.CapabilityAblationLM):
    """Actual #748/v18 model path with only memory event addressing repaired."""

    def _stage_forward_without_latent_reasoning(self, stage, events, state, *, intrinsic_hard, update_memory):
        h = stage.norm(events)
        if not hasattr(self, "_active_event_source") or not hasattr(self, "_active_tokens"):
            raise RuntimeError("event context is only valid inside forward_answer_tokens")
        source = stage.norm(self._active_event_source)
        raw = self._active_tokens
        if source.size(0) != events.size(0) or raw.size(0) != events.size(0):
            raise RuntimeError("event context batch mismatch")
        start_control = stage.controller(h[:, 0], state.stream)
        if self.memory_enabled:
            matrices, reads, qidxs, traces = [], [], [], []
            for row, row_tokens in enumerate(raw.detach().cpu().tolist()):
                row_memory, row_read, qidx, trace = _process_row(
                    stage, source, row_tokens, state, row, update_memory=update_memory
                )
                matrices.append(row_memory.matrix)
                reads.append(row_read)
                qidxs.append(qidx)
                traces.append(trace)
            memory_state = FastMemoryState(torch.cat(matrices, dim=0))
            memory_read = torch.cat(reads, dim=0)
        else:
            memory_state = FastMemoryState(torch.zeros_like(state.memory.matrix))
            memory_read = torch.zeros_like(state.stream)
            qidxs = [None] * events.size(0)
            traces = [tuple()] * events.size(0)
        carried = stage.state_to_chunk(state.stream) if self.recurrent_state_enabled else torch.zeros_like(state.stream)
        h = h + (start_control["state_read"] * carried)[:, None, :]
        if self.memory_enabled:
            term = start_control["memory_read"] * memory_read
            mask = torch.zeros(h.size(0), h.size(1), 1, device=h.device, dtype=h.dtype)
            for row, qidx in enumerate(qidxs):
                if qidx is not None and qidx + 1 < h.size(1):
                    mask[row, qidx + 1:, 0] = 1.0
            h = h + mask * term[:, None, :].to(h.dtype)
        h = h + stage.attn(h)
        h = h + stage.experts(h, start_control["expert_logits"], start_control["expert_count_logits"], hard=intrinsic_hard)
        end_summary = h[:, -1]
        end_control = stage.controller(end_summary, state.stream)
        h = stage.out_norm(h)
        final_stream = stage.stream_cell(end_summary, state.stream) if self.recurrent_state_enabled else torch.zeros_like(state.stream)
        self._trace_by_stage[id(stage)] = tuple(traces)
        self._qidx_by_stage[id(stage)] = tuple(qidxs)
        return h, AERAState(final_stream, memory_state), {"start": start_control, "end": end_control}

    def _run_stage(self, x, stage, stage_state, router, stage_index: int, *, route_mode: str, intrinsic_hard: bool, update_memory: bool):
        fixed = (not self.adaptive_routing_enabled) or stage_index == self.FOUNDATION_STAGE
        if fixed or route_mode != "hard_sparse":
            return super()._run_stage(
                x, stage, stage_state, router, stage_index,
                route_mode=route_mode, intrinsic_hard=intrinsic_hard, update_memory=update_memory,
            )
        gate_value, logits = router(x[:, 0], stage_state.stream, mode=route_mode)
        probability = torch.sigmoid(logits)
        run_idx = (gate_value[:, 0] >= 0.5).nonzero(as_tuple=False).squeeze(-1)
        if run_idx.numel() == 0:
            return x, stage_state, {
                "stage_route_probability": probability, "stage_route_gate": gate_value,
                "executed_fraction": 0.0, "start": None, "end": None,
            }
        selected_x = x.index_select(0, run_idx)
        selected_state = base._select_state(stage_state, run_idx)
        old_source, old_tokens = self._active_event_source, self._active_tokens
        self._active_event_source = old_source.index_select(0, run_idx)
        self._active_tokens = old_tokens.index_select(0, run_idx)
        try:
            selected_y, selected_new_state, controls = self._stage_forward_without_latent_reasoning(
                stage, selected_x, selected_state, intrinsic_hard=intrinsic_hard, update_memory=update_memory
            )
        finally:
            self._active_event_source, self._active_tokens = old_source, old_tokens
        y = x.index_copy(0, run_idx, selected_y.to(dtype=x.dtype))
        new_state = base._merge_state(stage_state, selected_new_state, run_idx)
        return y, new_state, {
            "stage_route_probability": probability, "stage_route_gate": gate_value,
            "executed_fraction": float(run_idx.numel() / x.size(0)),
            "start": controls["start"], "end": controls["end"],
        }

    def forward_answer_tokens(self, tokens, prediction_positions, reset_after_chunks, *, route_mode, intrinsic_hard, update_memory, state=None):
        if tokens.ndim != 2 or tokens.size(1) % self.cfg.chunk_size:
            raise ValueError("tokens must be [batch, whole_chunks]")
        batch, total_tokens = tokens.shape
        chunks = total_tokens // self.cfg.chunk_size
        if prediction_positions.shape != (batch,):
            raise ValueError("prediction_positions must be [batch]")
        if reset_after_chunks.shape != (batch, chunks):
            raise ValueError("reset_after_chunks shape mismatch")
        if state is None:
            state = self.empty_state(tokens)
        outputs, route_history = [], []
        execution = [0.0 for _ in self.stages]
        current_state = state
        self._trace_by_stage, self._qidx_by_stage = {}, {}
        for chunk_index in range(chunks):
            start = chunk_index * self.cfg.chunk_size
            chunk = tokens[:, start:start + self.cfg.chunk_size]
            pos = torch.arange(chunk.size(1), device=tokens.device)
            event_source = self.token_emb(chunk) + self.local_pos(pos)[None]
            x = event_source
            self._active_event_source, self._active_tokens = event_source, chunk
            new_states, stage_routes = [], []
            for stage_index, (stage, stage_state, router) in enumerate(zip(self.stages, current_state.stages, self.stage_routers)):
                x, new_state, info = self._run_stage(
                    x, stage, stage_state, router, stage_index,
                    route_mode=route_mode, intrinsic_hard=intrinsic_hard, update_memory=update_memory,
                )
                info["memory_events"] = self._trace_by_stage.get(id(stage), tuple())
                info["query_key_indices"] = self._qidx_by_stage.get(id(stage), tuple())
                new_states.append(new_state)
                stage_routes.append(info)
                execution[stage_index] += float(info["executed_fraction"])
            outputs.append(x)
            route_history.append(stage_routes)
            current_state = HardwareAERAState(new_states)
            current_state = base._reset_rows(current_state, reset_after_chunks[:, chunk_index])
        del self._active_event_source, self._active_tokens
        all_hidden = torch.cat(outputs, dim=1)
        if bool((prediction_positions < 0).any()) or bool((prediction_positions >= total_tokens).any()):
            raise ValueError("prediction position outside sequence")
        rows = torch.arange(batch, device=tokens.device)
        answer_hidden = self.norm(all_hidden[rows, prediction_positions])
        answer_logits = self.lm_head(answer_hidden)
        self.last_stage_execution = [
            {"stage": float(i), "mean_executed_fraction": execution[i] / chunks}
            for i in range(len(execution))
        ]
        self.last_memory_event_trace = route_history
        return {
            "answer_logits": answer_logits, "state": current_state,
            "stage_routes": route_history, "routing_mode": route_mode,
            "prediction_positions": prediction_positions,
        }


def cpu_probe_config() -> HardwareAERAConfig:
    return HardwareAERAConfig(
        vocab_size=8192, d_model=16, n_stages=4, n_heads=2, chunk_size=16,
        n_experts=2, max_active_experts=1, expert_mult=2, memory_dim=8,
        max_reason_steps=2, block_size=2, fast_memory_lr=0.2, fast_memory_decay=0.999,
    )


def build_cpu_model(variant_name: str, *, cfg: HardwareAERAConfig | None = None, seed: int = CPU_TEST_SEED):
    if int(seed) in base.SCIENTIFIC_MODEL_SEEDS:
        raise PermissionError("scientific model seeds are forbidden in issue770 CPU integration")
    torch.manual_seed(int(seed))
    return IntegratedEventMemoryCapabilityLM(variant_name, cpu_probe_config() if cfg is None else cfg).cpu()


def authority_snapshot() -> dict[str, object]:
    return {
        "research_issue": RESEARCH_ISSUE, "parent_repair_issue": PARENT_REPAIR_ISSUE,
        "source_main": SOURCE_MAIN, "source_tree": SOURCE_TREE, "cpu_test_seed": CPU_TEST_SEED,
        "consumed_scientific_seed": CONSUMED_SCIENTIFIC_SEED,
        "gpu_authorized": GPU_AUTHORIZED, "modal_authorized": MODAL_AUTHORIZED,
        "scientific_training_authorized": SCIENTIFIC_TRAINING_AUTHORIZED,
        "scientific_evaluation_authorized": SCIENTIFIC_EVALUATION_AUTHORIZED,
        "scientific_seed_authorized": SCIENTIFIC_SEED_AUTHORIZED,
        "seeds_2_3_authorized": SEEDS_2_3_AUTHORIZED,
        "scientific_result_authorized": SCIENTIFIC_RESULT_AUTHORIZED,
        "scientific_checkpoint_authorized": SCIENTIFIC_CHECKPOINT_AUTHORIZED,
        "systems_optimization_authorized": SYSTEMS_OPTIMIZATION_AUTHORIZED,
        "architecture_freeze_authorized": ARCHITECTURE_FREEZE_AUTHORIZED,
        "scaling_authorized": SCALING_AUTHORIZED, "breakthrough_proven": BREAKTHROUGH_PROVEN,
    }
