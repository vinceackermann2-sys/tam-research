from __future__ import annotations

"""CPU-only controlled gate for AERA-v24.1 causal end-to-end memory addressing.

Frozen by issue #356 after the first valid v24 result (#355) exposed an impossible
future-conditioned address auxiliary.  The v24 VCEM architecture is unchanged.
This candidate removes the q/k transition-identity contrastive loss and lets q/k
plus the sparse write selector learn only through causal query CE propagated
through differentiable VCEM reads/writes.  The decoder-aligned payload CE remains.
"""

import json
from typing import Any

import torch
import torch.nn.functional as F

from aera_v19_memory_necessity_cpu import (
    BATCH_SIZE,
    CHUNK_SIZE,
    FRESH_SESSION_CHANCE_TOLERANCE,
    FULL_ACCURACY_MIN,
    FULL_OVER_STREAM_MIN,
    LEARNING_RATE,
    N_KEYS,
    N_VALUES,
    OVERWRITE_ACCURACY_MIN,
    QUERY,
    SAME_CHECKPOINT_MEMORY_DROP_MIN,
    STALE_ERROR_MAX,
    TASK_VALIDITY_MIN,
    TRAIN_STEPS,
    VOCAB_SIZE,
    WRITE,
    _evaluate,
    _force_all_stages_run,
    _loss_and_accuracy,
    deployment_safety_and_isolation,
    diagnostic_config,
    make_batch,
    oracle_accuracy,
)
from aera_v21_memory_code_objective_cpu import _decode_with_frozen_model_head
from aera_v21_conflict_free_memory_objective_cpu import PAYLOAD_TOKEN_WEIGHT
from aera_v24_vectorized_contextual_episodic_memory_cpu import (
    AMBIGUOUS_CONTEXT_ERROR_MAX,
    AMBIGUOUS_CONTEXT_MIN,
    DIRECT_OVERWRITE_MIN,
    DIRECT_RECALL_MIN,
    DIRECT_STALE_MAX,
    MECHANISM_CURRENT_MIN,
    MECHANISM_OVERWRITE_MIN,
    MECHANISM_STALE_MAX,
    STATE_BYTES_MAX,
    TOKEN_ONLY_REFERENCE_MAX,
    context_disambiguation_probe,
    direct_episodic_read_evaluation,
    episodic_mechanism_probe,
)
from tam_research.aera_hardware_core_v24 import (
    HardwareAwareAERATextLMV24,
    VectorizedContextualEpisodicMemoryStage,
    causal_contextualize,
    episodic_state_bytes_per_session,
    vectorized_contextual_episodic_protocol,
)
from tam_research.aera_hardware_core_v23 import (
    select_budgeted_event_pairs,
    sparse_write_budget,
)

SEED = 8411
EVAL_SEED = 8412
ADDRESS_CONTRASTIVE_WEIGHT = 0.0


def build_model(seed: int) -> HardwareAwareAERATextLMV24:
    torch.manual_seed(seed)
    model = HardwareAwareAERATextLMV24(diagnostic_config())
    _force_all_stages_run(model)
    return model


def _contextual_events(model: HardwareAwareAERATextLMV24, tokens: torch.Tensor) -> torch.Tensor:
    if tokens.ndim != 2 or tokens.size(1) % CHUNK_SIZE:
        raise ValueError("tokens must be [batch,sequence] divisible by CHUNK_SIZE")
    chunks = tokens.view(tokens.size(0), -1, CHUNK_SIZE)
    pos = torch.arange(CHUNK_SIZE, device=tokens.device)
    events = model.token_emb(chunks) + model.local_pos(pos)[None, None, :, :]
    base_h = model.stages[0].norm(events).detach()
    return causal_contextualize(
        base_h.reshape(-1, CHUNK_SIZE, base_h.size(-1))
    ).reshape_as(base_h)


def payload_memory_terms(
    model: HardwareAwareAERATextLMV24,
    tokens: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Frozen v24 decoder-aligned payload objective, with no q/k address loss."""
    contextual = _contextual_events(model, tokens)
    chunks = tokens.view(tokens.size(0), -1, CHUNK_SIZE)
    payload_source = contextual[:, :, 1:]
    next_tokens = chunks[:, :, 1:]
    memory = model.stages[0].memory
    payload_code = memory.out(torch.tanh(memory.v(payload_source)))
    payload_logits = _decode_with_frozen_model_head(model, payload_code)
    payload_loss = F.cross_entropy(
        payload_logits.float().reshape(-1, payload_logits.size(-1)),
        next_tokens.reshape(-1),
    )
    payload_accuracy = (
        payload_logits.argmax(dim=-1) == next_tokens
    ).float().mean()
    return {
        "payload_token_loss": payload_loss,
        "payload_token_accuracy": payload_accuracy,
    }


@torch.no_grad()
def causal_objective_contradiction_probe(
    model: HardwareAwareAERATextLMV24,
) -> dict[str, bool | float]:
    """Prove the removed v24 address target contradicted causal address inputs."""
    key = 12
    value_a = 24
    value_b = 25
    filler = 4
    write_a = torch.tensor([[WRITE, key, value_a, filler, filler, filler]])
    write_b = torch.tensor([[WRITE, key, value_b, filler, filler, filler]])
    query = torch.tensor([[QUERY, key, value_a, filler, filler, filler]])

    def key_view(chunk: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(CHUNK_SIZE)
        events = model.token_emb(chunk) + model.local_pos(pos)[None, :, :]
        contextual = causal_contextualize(model.stages[0].norm(events))
        return contextual[:, 1]

    write_view_a = key_view(write_a)
    write_view_b = key_view(write_b)
    query_view = key_view(query)
    identity_a = key * VOCAB_SIZE + value_a
    identity_b = key * VOCAB_SIZE + value_b
    return {
        "same_write_address_bit_identical": bool(torch.equal(write_view_a, write_view_b)),
        "old_transition_identities_differ": bool(identity_a != identity_b),
        "write_and_query_views_differ": bool(not torch.equal(write_view_a, query_view)),
        "write_query_view_l2": float((write_view_a - query_view).pow(2).sum().sqrt()),
    }


def primary_query_gradient_probe() -> dict[str, float | bool]:
    """Show primary causal query CE alone reaches q/k/v/out and write selector."""
    model = build_model(9561)
    batch = make_batch(2, EVAL_SEED + 9561)
    model.train()
    model.zero_grad(set_to_none=True)
    query_loss, _ = _loss_and_accuracy(
        model,
        batch,
        update_memory=True,
        differentiable_memory=True,
    )
    query_loss.backward()
    stage = model.stages[0]
    grads = {
        "q": stage.memory.q.weight.grad,
        "k": stage.memory.k.weight.grad,
        "v": stage.memory.v.weight.grad,
        "out": stage.memory.out.weight.grad,
        "pair_write_gate": stage.pair_write_gate.weight.grad,
    }
    result: dict[str, float | bool] = {"loss_finite": bool(torch.isfinite(query_loss))}
    for name, grad in grads.items():
        finite = grad is not None and bool(torch.isfinite(grad).all())
        total = float(grad.abs().sum()) if grad is not None else 0.0
        result[f"{name}_grad_finite"] = finite
        result[f"{name}_grad_l1"] = total
        result[f"{name}_grad_nonzero"] = finite and total > 0.0
    return result


@torch.no_grad()
def heldout_address_and_selector_diagnostics(
    model: HardwareAwareAERATextLMV24,
    batch: Any,
) -> dict[str, float]:
    """Diagnostic-only query-view->latest-write q/k retrieval and selector coverage."""
    model.eval()
    stage = model.stages[0]
    if not isinstance(stage, VectorizedContextualEpisodicMemoryStage):
        raise TypeError("v24 stage0 type mismatch")
    chunks = batch.tokens.view(batch.tokens.size(0), -1, CHUNK_SIZE)
    pos = torch.arange(CHUNK_SIZE, device=batch.tokens.device)

    latest_write_view: list[dict[int, torch.Tensor]] = [dict() for _ in range(chunks.size(0))]
    query_rows: list[tuple[int, int, torch.Tensor]] = []
    selector_hits = 0
    selector_total = 0
    payload_correct = 0
    payload_total = 0

    for chunk_index in range(chunks.size(1)):
        chunk = chunks[:, chunk_index]
        events = model.token_emb(chunk) + model.local_pos(pos)[None, :, :]
        base_h = stage.norm(events)
        contextual = causal_contextualize(base_h)
        prefix = chunk[:, 0]

        write_mask = prefix.eq(WRITE)
        if bool(write_mask.any()):
            address_source = contextual[:, :-1]
            payload_source = contextual[:, 1:]
            pair_features = torch.cat((address_source, payload_source), dim=-1)
            pair_logits = stage.pair_write_gate(pair_features)
            pair_gate = torch.sigmoid(pair_logits)
            # Selector ranking is independent of the stage-level chunk scalar.
            selected = select_budgeted_event_pairs(
                address_source,
                payload_source,
                pair_gate,
                pair_logits,
                differentiable_selector=False,
            )
            for b in torch.nonzero(write_mask, as_tuple=False).flatten().tolist():
                key = int(chunk[b, 1])
                latest_write_view[b][key] = contextual[b, 1].detach().clone()
                selector_total += 1
                selector_hits += int(bool((selected.indices[b] == 1).any()))
                payload_code = stage.memory.out(
                    torch.tanh(stage.memory.v(contextual[b : b + 1, 2]))
                )
                payload_logits = _decode_with_frozen_model_head(model, payload_code)
                payload_correct += int(payload_logits.argmax(dim=-1).item() == int(chunk[b, 2]))
                payload_total += 1

        query_mask = prefix.eq(QUERY)
        if bool(query_mask.any()):
            for b in torch.nonzero(query_mask, as_tuple=False).flatten().tolist():
                query_rows.append((b, int(chunk[b, 1]), contextual[b, 1].detach().clone()))

    correct = 0
    total = 0
    margins: list[float] = []
    memory = stage.memory
    for b, key, query_view in query_rows:
        candidates = latest_write_view[b]
        if len(candidates) != N_KEYS or key not in candidates:
            continue
        ordered_keys = sorted(candidates)
        write_views = torch.stack([candidates[k] for k in ordered_keys])
        q = F.normalize(memory.q(query_view[None]), dim=-1)
        k = F.normalize(memory.k(write_views), dim=-1)
        similarities = (q @ k.transpose(0, 1))[0]
        predicted_key = ordered_keys[int(similarities.argmax())]
        correct += int(predicted_key == key)
        total += 1
        positive_index = ordered_keys.index(key)
        positive = similarities[positive_index]
        negative_mask = torch.ones_like(similarities, dtype=torch.bool)
        negative_mask[positive_index] = False
        margins.append(float(positive - similarities[negative_mask].max()))

    return {
        "query_to_latest_write_address_top1": correct / max(total, 1),
        "query_to_latest_write_address_margin": sum(margins) / max(len(margins), 1),
        "true_key_value_selector_coverage": selector_hits / max(selector_total, 1),
        "write_payload_token_accuracy": payload_correct / max(payload_total, 1),
    }


def train_pair_with_v24_1_objective(
    *,
    steps: int = TRAIN_STEPS,
) -> tuple[HardwareAwareAERATextLMV24, HardwareAwareAERATextLMV24, dict[str, Any]]:
    full = build_model(SEED)
    stream_only = build_model(SEED)
    for key, value in full.state_dict().items():
        torch.testing.assert_close(stream_only.state_dict()[key], value, atol=0.0, rtol=0.0)

    full.train()
    stream_only.train()
    full_opt = torch.optim.AdamW(
        [p for p in full.parameters() if p.requires_grad],
        lr=LEARNING_RATE,
        weight_decay=0.0,
    )
    stream_opt = torch.optim.AdamW(
        [p for p in stream_only.parameters() if p.requires_grad],
        lr=LEARNING_RATE,
        weight_decay=0.0,
    )
    history: list[dict[str, float]] = []
    for step in range(steps):
        batch = make_batch(BATCH_SIZE, SEED * 10000 + step)

        full_opt.zero_grad(set_to_none=True)
        query_loss, full_acc = _loss_and_accuracy(
            full,
            batch,
            update_memory=True,
            differentiable_memory=True,
        )
        payload = payload_memory_terms(full, batch.tokens)
        total = query_loss + PAYLOAD_TOKEN_WEIGHT * payload["payload_token_loss"]
        total.backward()
        torch.nn.utils.clip_grad_norm_(full.parameters(), 1.0)
        full_opt.step()

        stream_opt.zero_grad(set_to_none=True)
        stream_loss, stream_acc = _loss_and_accuracy(
            stream_only,
            batch,
            update_memory=False,
            differentiable_memory=False,
        )
        stream_loss.backward()
        torch.nn.utils.clip_grad_norm_(stream_only.parameters(), 1.0)
        stream_opt.step()

        if step in {0, steps // 4, steps // 2, (3 * steps) // 4, steps - 1}:
            row = {
                "step": float(step + 1),
                "total_loss": float(total.detach()),
                "query_loss": float(query_loss.detach()),
                "stream_only_loss": float(stream_loss.detach()),
                "full_query_accuracy": full_acc,
                "stream_only_query_accuracy": stream_acc,
                "payload_token_loss": float(payload["payload_token_loss"].detach()),
                "payload_token_accuracy": float(payload["payload_token_accuracy"].detach()),
            }
            history.append(row)
            print("AERA_V24_1_CAUSAL_TRAIN=" + json.dumps(row, sort_keys=True), flush=True)

    return full, stream_only, {"history": history}


def run_gate(*, steps: int = TRAIN_STEPS) -> dict[str, Any]:
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    mechanism = episodic_mechanism_probe()
    ambiguity = context_disambiguation_probe()
    state_bytes = episodic_state_bytes_per_session(n_stages=4, memory_dim=50)
    eval_batch = make_batch(24, EVAL_SEED)
    validity = oracle_accuracy(eval_batch)
    if validity < TASK_VALIDITY_MIN:
        raise RuntimeError(f"task validity failed before training: {validity}")

    full, stream_only, training = train_pair_with_v24_1_objective(steps=steps)
    full_eval = _evaluate(full, eval_batch, memory=True)
    memory_off = _evaluate(full, eval_batch, memory=False)
    stream_eval = _evaluate(stream_only, eval_batch, memory=False)
    direct = direct_episodic_read_evaluation(full, eval_batch)
    safety = deployment_safety_and_isolation(full, eval_batch)
    diagnostic = heldout_address_and_selector_diagnostics(full, eval_batch)
    contradiction = causal_objective_contradiction_probe(full)

    stage = full.stages[0]
    first_chunk = eval_batch.tokens[:, :CHUNK_SIZE]
    pos = torch.arange(CHUNK_SIZE)
    events = full.token_emb(first_chunk) + full.local_pos(pos)[None, :, :]
    _ = stage.forward_chunk(events, None, hard=True, update_memory=True)
    budget_ok = (
        stage.last_candidate_count == CHUNK_SIZE - 1
        and stage.last_selected_count == sparse_write_budget(CHUNK_SIZE - 1)
        and stage.last_selected_count == 2
        and stage.last_vectorized_update_calls == 1
    )

    chance = 1.0 / N_VALUES
    full_minus_stream = full_eval["query_accuracy"] - stream_eval["query_accuracy"]
    memory_drop = full_eval["query_accuracy"] - memory_off["query_accuracy"]
    checks = {
        "mechanism_current_ge_0_95": mechanism["current_accuracy"] >= MECHANISM_CURRENT_MIN,
        "mechanism_overwrite_ge_0_95": mechanism["overwrite_current_accuracy"] >= MECHANISM_OVERWRITE_MIN,
        "mechanism_stale_le_0_05": mechanism["stale_error"] <= MECHANISM_STALE_MAX,
        "context_disambiguation_ge_0_95": ambiguity["contextual_accuracy"] >= AMBIGUOUS_CONTEXT_MIN,
        "context_cross_error_le_0_05": ambiguity["contextual_cross_context_error"] <= AMBIGUOUS_CONTEXT_ERROR_MAX,
        "token_only_reference_le_0_75": ambiguity["token_only_reference_accuracy"] <= TOKEN_ONLY_REFERENCE_MAX,
        "task_validity_ge_0_95": validity >= TASK_VALIDITY_MIN,
        "direct_recall_ge_0_95": direct["overall_accuracy"] >= DIRECT_RECALL_MIN,
        "direct_overwrite_current_ge_0_80": direct["overwrite_current_value_accuracy"] >= DIRECT_OVERWRITE_MIN,
        "direct_stale_le_0_10": direct["overwrite_stale_value_error"] <= DIRECT_STALE_MAX,
        "full_query_accuracy_ge_0_80": full_eval["query_accuracy"] >= FULL_ACCURACY_MIN,
        "full_over_stream_only_ge_0_15": full_minus_stream >= FULL_OVER_STREAM_MIN,
        "same_checkpoint_memory_drop_ge_0_15": memory_drop >= SAME_CHECKPOINT_MEMORY_DROP_MIN,
        "full_overwrite_current_ge_0_80": full_eval["overwrite_current_value_accuracy"] >= OVERWRITE_ACCURACY_MIN,
        "full_stale_le_0_10": full_eval["stale_value_error_rate"] <= STALE_ERROR_MAX,
        "fresh_session_near_chance": safety["fresh_session_query_accuracy"] <= chance + FRESH_SESSION_CHANCE_TOLERANCE,
        "deployment_base_parameters_unchanged": bool(safety["deployment_base_parameters_unchanged"]),
        "session_isolation_exact": bool(safety["session_isolation_exact"]),
        "controlled_budget_2_of_5_one_vectorized_update": budget_ok,
        "state_bytes_le_80000": state_bytes <= STATE_BYTES_MAX,
    }
    passed = all(checks.values())
    if passed:
        interpretation = "v24_1_causal_end_to_end_addressing_passes_controlled_gate"
    elif diagnostic["query_to_latest_write_address_top1"] < 0.80:
        interpretation = "v24_1_end_to_end_only_address_learning_insufficient"
    elif diagnostic["true_key_value_selector_coverage"] < 0.80:
        interpretation = "v24_1_write_utility_selection_is_blocker"
    elif direct["overall_accuracy"] < DIRECT_RECALL_MIN:
        interpretation = "v24_1_payload_or_read_integration_is_blocker"
    elif full_eval["query_accuracy"] < FULL_ACCURACY_MIN:
        interpretation = "v24_1_direct_memory_not_integrated_into_end_to_end_prediction"
    else:
        interpretation = "v24_1_controlled_gate_partial_failure"

    return {
        "scope": "aera_v24_1_causal_end_to_end_addressing_cpu",
        "architecture_version": "v24",
        "objective_version": "v24.1-causal-end-to-end-addressing",
        "protocol": vectorized_contextual_episodic_protocol(),
        "gpu_used": False,
        "independent_evidence": False,
        "train_seed": SEED,
        "eval_seed": EVAL_SEED,
        "steps": steps,
        "address_contrastive_weight": ADDRESS_CONTRASTIVE_WEIGHT,
        "task": {"oracle_accuracy": validity, "chance_accuracy": chance},
        "state_bytes_real_language_four_stage_memory_dim50": state_bytes,
        "causal_objective_contradiction_probe": contradiction,
        "mechanism_probe": mechanism,
        "context_disambiguation_probe": ambiguity,
        "training": training,
        "heldout_address_and_selector_diagnostics": diagnostic,
        "direct_episodic_read": direct,
        "full_stream_plus_memory": {k: v for k, v in full_eval.items() if k != "predictions"},
        "same_checkpoint_memory_disabled": {k: v for k, v in memory_off.items() if k != "predictions"},
        "separately_trained_stream_only": {k: v for k, v in stream_eval.items() if k != "predictions"},
        "full_minus_stream_only_accuracy": full_minus_stream,
        "same_checkpoint_memory_contribution_accuracy": memory_drop,
        "deployment_safety": safety,
        "physical_budget_audit": {
            "candidates": stage.last_candidate_count,
            "selected": stage.last_selected_count,
            "vectorized_update_calls": stage.last_vectorized_update_calls,
        },
        "checks": checks,
        "pass": passed,
        "interpretation": interpretation,
        "claims": {
            "controlled_v24_1_memory_passed": passed,
            "l4_systems_benchmark_authorized": passed,
            "real_language_training_authorized": False,
            "architecture_frozen": False,
            "independent_replication_credit": False,
            "100m_authorized": False,
            "breakthrough_proven": False,
        },
    }


def main() -> None:
    result = run_gate()
    print("AERA_V24_1_CAUSAL_RESULT_JSON=" + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
