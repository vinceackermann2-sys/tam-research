from __future__ import annotations

"""Mechanistic CPU audit after the terminal AERA-v21 memory FAIL.

This is NOT a new architecture version and NOT independent replication evidence.
It deterministically reproduces the already-failed v21 synthetic training only so
we can instrument the learned memory pipeline and run counterfactual same-checkpoint
interventions that localize the broken link.
"""

from contextlib import ExitStack, contextmanager
import json
import math
from types import MethodType
from typing import Any, Iterator

import torch
import torch.nn.functional as F

from aera_v19_memory_necessity_cpu import (
    BATCH_SIZE,
    CHUNK_SIZE,
    EVAL_SEED,
    LEARNING_RATE,
    N_VALUES,
    SEED,
    TRAIN_STEPS,
    VALUE_START,
    WRITE,
    _evaluate,
    _query_logits,
    _loss_and_accuracy,
    make_batch,
)
from aera_v21_memory_necessity_cpu import build_model
from tam_research.aera_hardware_core_v21 import HardwareAwareAERATextLMV21

MATERIAL_ACCURACY_GAIN = 0.10
NEAR_ZERO_WRITE_STRENGTH = 0.01
NEAR_ZERO_MATRIX_NORM = 1e-4
GRAD_CHECKPOINTS = (0, TRAIN_STEPS // 4, TRAIN_STEPS // 2, (3 * TRAIN_STEPS) // 4, TRAIN_STEPS - 1)


def _tensor_grad_norm(parameter: torch.Tensor) -> float:
    grad = parameter.grad
    if grad is None:
        return 0.0
    return float(grad.detach().float().norm())


def _control_row_grad_norm(model: HardwareAwareAERATextLMV21, control_name: str) -> float:
    stage = model.stages[0]
    controller = stage.controller
    names = tuple(controller.CONTROL_NAMES)
    control_index = names.index(control_name)
    row = controller.n_experts + 2 + controller.max_reason_steps + control_index
    weight_grad = controller.proj.weight.grad
    bias_grad = controller.proj.bias.grad
    if weight_grad is None or bias_grad is None:
        return 0.0
    w = weight_grad[row].detach().float().norm()
    b = bias_grad[row].detach().float().abs()
    return float(torch.sqrt(w.square() + b.square()))


def _gradient_snapshot(model: HardwareAwareAERATextLMV21, step: int, loss: torch.Tensor) -> dict[str, float]:
    stage = model.stages[0]
    memory = stage.memory
    return {
        "step": float(step + 1),
        "loss": float(loss.detach()),
        "pair_gate": _tensor_grad_norm(stage.pair_write_gate.weight),
        "memory_q": _tensor_grad_norm(memory.q.weight),
        "memory_k": _tensor_grad_norm(memory.k.weight),
        "memory_v": _tensor_grad_norm(memory.v.weight),
        "memory_out": _tensor_grad_norm(memory.out.weight),
        "controller_memory_read": _control_row_grad_norm(model, "memory_read"),
        "controller_memory_write": _control_row_grad_norm(model, "memory_write"),
        "controller_novelty": _control_row_grad_norm(model, "novelty"),
        "lm_head": _tensor_grad_norm(model.lm_head.weight),
    }


def train_deterministic_reproduction(*, steps: int = TRAIN_STEPS) -> tuple[HardwareAwareAERATextLMV21, list[dict[str, float]]]:
    """Reproduce the failed v21 seed for instrumentation only.

    This uses the exact same seed, batches, optimizer and full-memory objective as
    #289.  It must never be counted as independent replication evidence.
    """
    model = build_model(SEED)
    model.train()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE,
        weight_decay=0.0,
    )
    snapshots: list[dict[str, float]] = []
    wanted = {x for x in GRAD_CHECKPOINTS if x < steps}
    wanted.add(steps - 1)
    for step in range(steps):
        batch = make_batch(BATCH_SIZE, SEED * 10000 + step)
        optimizer.zero_grad(set_to_none=True)
        loss, _ = _loss_and_accuracy(
            model,
            batch,
            update_memory=True,
            differentiable_memory=True,
        )
        loss.backward()
        if step in wanted:
            row = _gradient_snapshot(model, step, loss)
            snapshots.append(row)
            print("AERA_V21_MEMORY_PIPELINE_GRAD=" + json.dumps(row, sort_keys=True), flush=True)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return model, snapshots


@contextmanager
def _use_k_for_memory_read(model: HardwareAwareAERATextLMV21) -> Iterator[None]:
    originals: list[tuple[torch.nn.Module, object]] = []
    for stage in model.stages:
        memory = stage.memory
        original = memory.read
        originals.append((memory, original))

        def read_with_k(this: torch.nn.Module, x: torch.Tensor, state: Any) -> torch.Tensor:
            query = F.normalize(this.k(x), dim=-1)
            recalled = torch.einsum("btd,bdm->btm", query, state.matrix)
            return this.out(recalled)

        memory.read = MethodType(read_with_k, memory)
    try:
        yield
    finally:
        for memory, original in originals:
            memory.read = original  # type: ignore[method-assign]


@contextmanager
def _force_memory_read_gate_one(model: HardwareAwareAERATextLMV21) -> Iterator[None]:
    originals: list[tuple[torch.nn.Module, object]] = []
    for stage in model.stages:
        controller = stage.controller
        original = controller.forward
        originals.append((controller, original))

        def forward_with_read_one(this: torch.nn.Module, event: torch.Tensor, stream: torch.Tensor, _orig=original):
            result = _orig(event, stream)
            result = dict(result)
            result["memory_read"] = torch.ones_like(result["memory_read"])
            return result

        controller.forward = MethodType(forward_with_read_one, controller)
    try:
        yield
    finally:
        for controller, original in originals:
            controller.forward = original  # type: ignore[method-assign]


@contextmanager
def _oracle_pair_position_one(model: HardwareAwareAERATextLMV21) -> Iterator[None]:
    """Diagnostic-only pair selector: keep candidate index 1, suppress others."""
    originals: list[tuple[torch.nn.Module, object]] = []
    for stage in model.stages:
        gate = stage.pair_write_gate
        original = gate.forward
        originals.append((gate, original))

        def oracle_forward(this: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
            logits = torch.full((*x.shape[:-1], 1), -12.0, device=x.device, dtype=x.dtype)
            if x.size(1) > 1:
                logits[:, 1, :] = 12.0
            return logits

        gate.forward = MethodType(oracle_forward, gate)
    try:
        yield
    finally:
        for gate, original in originals:
            gate.forward = original  # type: ignore[method-assign]


def _evaluate_chunkwise(
    model: HardwareAwareAERATextLMV21,
    batch: Any,
    *,
    memory: bool = True,
    k_read: bool = False,
    read_gate_one: bool = False,
    oracle_pair: bool = False,
) -> dict[str, float]:
    """Evaluate with optional same-checkpoint counterfactual interventions."""
    model.eval()
    model.set_memory_pretraining_mode(False)
    state = None
    logits_parts: list[torch.Tensor] = []
    with ExitStack() as stack:
        if k_read:
            stack.enter_context(_use_k_for_memory_read(model))
        if read_gate_one:
            stack.enter_context(_force_memory_read_gate_one(model))
        if oracle_pair:
            stack.enter_context(_oracle_pair_position_one(model))

        for start in range(0, batch.tokens.size(1), CHUNK_SIZE):
            chunk = batch.tokens[:, start : start + CHUNK_SIZE]
            update = memory
            if oracle_pair:
                prefix = chunk[:, 0]
                if not bool(prefix.eq(prefix[0]).all()):
                    raise RuntimeError("oracle-pair audit expects synchronized chunk types")
                update = memory and int(prefix[0]) == WRITE
            out = model(
                chunk,
                state=state,
                hard=True,
                route_mode="hard_sparse",
                update_memory=update,
                return_block_logits=False,
            )
            logits = out["logits"]
            if not isinstance(logits, torch.Tensor):
                raise TypeError("expected tensor logits")
            logits_parts.append(logits)
            state = out["state"]

    logits = torch.cat(logits_parts, dim=1)
    selected = _query_logits(logits, batch.query_positions)
    target = batch.query_targets.to(selected.device)
    prediction = selected.argmax(dim=-1).cpu()
    return {
        "query_accuracy": float((prediction == batch.query_targets).float().mean()),
        "query_nll": float(F.cross_entropy(selected.float().reshape(-1, selected.size(-1)), target.reshape(-1))),
    }


@torch.no_grad()
def trace_activity(model: HardwareAwareAERATextLMV21, batch: Any) -> dict[str, float]:
    model.eval()
    model.set_memory_pretraining_mode(False)
    state = None
    read_gates: list[float] = []
    write_gates: list[float] = []
    novelty: list[float] = []
    pair_gates: list[float] = []
    pair_strengths: list[float] = []
    read_norms: list[float] = []
    token_norms: list[float] = []
    stream_norms: list[float] = []
    matrix_norms: list[float] = []
    after_writes = math.nan
    after_distractors = math.nan

    n_write_chunks = 12 + 4
    n_distractor_chunks = 2
    for chunk_index, start in enumerate(range(0, batch.tokens.size(1), CHUNK_SIZE)):
        chunk = batch.tokens[:, start : start + CHUNK_SIZE]
        if state is not None:
            stage0 = model.stages[0]
            pos = torch.arange(chunk.size(1), device=chunk.device)
            events = model.token_emb(chunk) + model.local_pos(pos)[None, :, :]
            h = stage0.norm(events)
            recalled = stage0.memory.read(h, state.stages[0].memory)
            carried = stage0.state_to_chunk(state.stages[0].stream)
            read_norms.append(float(recalled.float().norm(dim=-1).mean()))
            token_norms.append(float(h.float().norm(dim=-1).mean()))
            stream_norms.append(float(carried.float().norm(dim=-1).mean()))

        out = model(
            chunk,
            state=state,
            hard=True,
            route_mode="hard_sparse",
            update_memory=True,
            return_block_logits=False,
        )
        state = out["state"]
        stage0 = model.stages[0]
        stats = stage0.stats()
        start_controls = stats.get("start_controls") or {}
        end_controls = stats.get("end_controls") or {}
        read_gates.append(float(start_controls.get("memory_read", 0.0)))
        write_gates.append(float(end_controls.get("memory_write", 0.0)))
        novelty.append(float(end_controls.get("novelty", 0.0)))
        pair_gates.append(float(stats.get("event_pair_gate_mean", 0.0)))
        pair_strengths.append(float(stats.get("event_pair_write_strength_mean", 0.0)))
        matrix_norm = float(state.stages[0].memory.matrix.float().norm(dim=(1, 2)).mean())
        matrix_norms.append(matrix_norm)
        if chunk_index == n_write_chunks - 1:
            after_writes = matrix_norm
        if chunk_index == n_write_chunks + n_distractor_chunks - 1:
            after_distractors = matrix_norm

    def mean(xs: list[float]) -> float:
        return float(sum(xs) / max(len(xs), 1))

    return {
        "memory_read_gate_mean": mean(read_gates),
        "memory_write_gate_mean": mean(write_gates),
        "novelty_gate_mean": mean(novelty),
        "event_pair_gate_mean": mean(pair_gates),
        "event_pair_write_strength_mean": mean(pair_strengths),
        "memory_read_vector_norm_mean": mean(read_norms),
        "normalized_token_vector_norm_mean": mean(token_norms),
        "carried_stream_vector_norm_mean": mean(stream_norms),
        "memory_matrix_norm_mean": mean(matrix_norms),
        "memory_matrix_norm_after_writes": after_writes,
        "memory_matrix_norm_after_distractors": after_distractors,
        "memory_matrix_norm_final": matrix_norms[-1] if matrix_norms else 0.0,
    }


def _diagnose(evals: dict[str, dict[str, float]], activity: dict[str, float]) -> str:
    normal = evals["normal"]["query_accuracy"]
    gains = {
        "qk": evals["k_read"]["query_accuracy"] - normal,
        "read_gate": evals["read_gate_one"]["query_accuracy"] - normal,
        "oracle_pair": evals["oracle_pair"]["query_accuracy"] - normal,
        "qk_read": evals["k_read_plus_gate_one"]["query_accuracy"] - normal,
        "combined": evals["oracle_pair_plus_k_read_plus_gate_one"]["query_accuracy"] - normal,
    }
    if (
        activity["event_pair_write_strength_mean"] < NEAR_ZERO_WRITE_STRENGTH
        or activity["memory_matrix_norm_after_writes"] < NEAR_ZERO_MATRIX_NORM
    ):
        return "write_strength_or_matrix_activity_bottleneck"
    if gains["qk"] >= MATERIAL_ACCURACY_GAIN:
        return "q_k_alignment_bottleneck"
    if gains["read_gate"] >= MATERIAL_ACCURACY_GAIN:
        return "controller_memory_read_suppression_bottleneck"
    if gains["oracle_pair"] >= MATERIAL_ACCURACY_GAIN:
        return "write_selectivity_or_interference_bottleneck"
    if max(gains["qk_read"], gains["combined"]) >= MATERIAL_ACCURACY_GAIN:
        return "combined_learning_or_control_bottleneck"
    return "payload_decoding_injection_or_objective_bottleneck"


def run_audit(*, steps: int = TRAIN_STEPS) -> dict[str, Any]:
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    model, gradients = train_deterministic_reproduction(steps=steps)
    eval_batch = make_batch(24, EVAL_SEED)

    normal_reference = _evaluate(model, eval_batch, memory=True)
    normal_chunkwise = _evaluate_chunkwise(model, eval_batch)
    if abs(normal_reference["query_accuracy"] - normal_chunkwise["query_accuracy"]) > 1e-7:
        raise RuntimeError("chunkwise audit evaluator changed normal query accuracy")

    evals = {
        "normal": normal_chunkwise,
        "k_read": _evaluate_chunkwise(model, eval_batch, k_read=True),
        "read_gate_one": _evaluate_chunkwise(model, eval_batch, read_gate_one=True),
        "oracle_pair": _evaluate_chunkwise(model, eval_batch, oracle_pair=True),
        "k_read_plus_gate_one": _evaluate_chunkwise(model, eval_batch, k_read=True, read_gate_one=True),
        "oracle_pair_plus_k_read_plus_gate_one": _evaluate_chunkwise(
            model,
            eval_batch,
            k_read=True,
            read_gate_one=True,
            oracle_pair=True,
        ),
    }
    activity = trace_activity(model, eval_batch)
    diagnosis = _diagnose(evals, activity)
    normal = evals["normal"]["query_accuracy"]
    gains = {name: row["query_accuracy"] - normal for name, row in evals.items() if name != "normal"}
    return {
        "scope": "aera_v21_fast_memory_pipeline_audit_cpu",
        "diagnostic_reproduction_only": True,
        "independent_evidence": False,
        "steps": steps,
        "chance_accuracy": 1.0 / N_VALUES,
        "gradient_snapshots": gradients,
        "activity": activity,
        "counterfactuals": evals,
        "accuracy_gains_over_normal": gains,
        "diagnosis": diagnosis,
        "thresholds": {
            "material_accuracy_gain": MATERIAL_ACCURACY_GAIN,
            "near_zero_write_strength": NEAR_ZERO_WRITE_STRENGTH,
            "near_zero_matrix_norm": NEAR_ZERO_MATRIX_NORM,
        },
        "claims": {
            "production_change_authorized": False,
            "gpu_authorized": False,
            "v22_authorized": False,
            "architecture_freeze_authorized": False,
            "100m_authorized": False,
            "breakthrough_proven": False,
        },
    }


def main() -> None:
    result = run_audit()
    print("AERA_V21_MEMORY_PIPELINE_AUDIT_RESULT_JSON=" + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
