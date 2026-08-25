from __future__ import annotations

"""CPU-only controlled gate for AERA-v24 Vectorized Contextual Episodic Memory.

Frozen by issue #347. This script uses no GPU and no production corpus. It keeps
v23 routing/compute and the established delayed-associative protocol while
replacing the fast-memory state/read/write semantics exactly as preregistered.
"""

import json
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from aera_v19_memory_necessity_cpu import (
    BATCH_SIZE,
    CHUNK_SIZE,
    EVAL_SEED,
    FRESH_SESSION_CHANCE_TOLERANCE,
    FULL_ACCURACY_MIN,
    FULL_OVER_STREAM_MIN,
    LEARNING_RATE,
    N_VALUES,
    OVERWRITE_ACCURACY_MIN,
    QUERY,
    SAME_CHECKPOINT_MEMORY_DROP_MIN,
    SEED,
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
from aera_v21_collapse_resistant_memory_objective_cpu import (
    ADDRESS_TEMPERATURE,
    multi_positive_contrastive_loss,
)
from aera_v21_memory_code_objective_cpu import _decode_with_frozen_model_head
from aera_v21_conflict_free_memory_objective_cpu import (
    ADDRESS_CONTRASTIVE_WEIGHT,
    PAYLOAD_TOKEN_WEIGHT,
)
from tam_research.aera_hardware_core_v24 import (
    ContextualEpisodicMemoryState,
    HardwareAwareAERATextLMV24,
    VectorizedContextualEpisodicMemory,
    VectorizedContextualEpisodicMemoryStage,
    causal_contextualize,
    episodic_state_bytes_per_session,
    vectorized_contextual_episodic_protocol,
)

MECHANISM_CURRENT_MIN = 0.95
MECHANISM_OVERWRITE_MIN = 0.95
MECHANISM_STALE_MAX = 0.05
DIRECT_RECALL_MIN = 0.95
DIRECT_OVERWRITE_MIN = 0.80
DIRECT_STALE_MAX = 0.10
AMBIGUOUS_CONTEXT_MIN = 0.95
AMBIGUOUS_CONTEXT_ERROR_MAX = 0.05
TOKEN_ONLY_REFERENCE_MAX = 0.75
STATE_BYTES_MAX = 80_000


class _IdentityMemorySource(nn.Module):
    """Parameter-free probe source: shared identity q/k/v/out geometry."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.memory_dim = dim
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        eye = torch.eye(dim)
        with torch.no_grad():
            for layer in (self.q, self.k, self.v, self.out):
                layer.weight.copy_(eye)
        self.differentiable_pretraining = False


def build_model(seed: int) -> HardwareAwareAERATextLMV24:
    torch.manual_seed(seed)
    model = HardwareAwareAERATextLMV24(diagnostic_config())
    _force_all_stages_run(model)
    return model


def _contextual_pairs(
    model: HardwareAwareAERATextLMV24,
    tokens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Detached exact stage-0 contextual adjacent pairs plus observed transitions."""
    if tokens.ndim != 2 or tokens.size(1) % CHUNK_SIZE:
        raise ValueError("tokens must be [batch,sequence] divisible by CHUNK_SIZE")
    chunks = tokens.view(tokens.size(0), -1, CHUNK_SIZE)
    pos = torch.arange(CHUNK_SIZE, device=tokens.device)
    events = model.token_emb(chunks) + model.local_pos(pos)[None, None, :, :]
    base_h = model.stages[0].norm(events).detach()
    contextual = causal_contextualize(
        base_h.reshape(-1, CHUNK_SIZE, base_h.size(-1))
    ).reshape_as(base_h)
    current_tokens = chunks[:, :, :-1]
    next_tokens = chunks[:, :, 1:]
    # Frozen after #346 and before results: transition labels supervise q/k only;
    # next_tokens are never input to the query representation.
    transition_identity = current_tokens * VOCAB_SIZE + next_tokens
    return (
        contextual[:, :, :-1],
        contextual[:, :, 1:],
        next_tokens,
        transition_identity,
    )


def contextual_memory_terms(
    model: HardwareAwareAERATextLMV24,
    tokens: torch.Tensor,
) -> dict[str, torch.Tensor]:
    address_source, payload_source, next_tokens, identity = _contextual_pairs(
        model, tokens
    )
    memory = model.stages[0].memory
    q = memory.q(address_source).reshape(-1, memory.memory_dim)
    k = memory.k(address_source).reshape(-1, memory.memory_dim)
    address_loss = multi_positive_contrastive_loss(
        q,
        k,
        identity.reshape(-1),
        temperature=ADDRESS_TEMPERATURE,
    )
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
        "address_contrastive_loss": address_loss,
        "payload_token_loss": payload_loss,
        "payload_token_accuracy": payload_accuracy,
    }


@torch.no_grad()
def heldout_local_contextual_code(
    model: HardwareAwareAERATextLMV24,
    tokens: torch.Tensor,
) -> dict[str, float]:
    model.eval()
    address_source, payload_source, next_tokens, identity = _contextual_pairs(
        model, tokens
    )
    chunks = tokens.view(tokens.size(0), -1, CHUNK_SIZE)
    write_mask = chunks[:, :, 0].eq(WRITE)
    # Candidate index 1 is the controlled key->value transition.
    address = address_source[:, :, 1][write_mask]
    payload = payload_source[:, :, 1][write_mask]
    target = next_tokens[:, :, 1][write_mask]
    ids = identity[:, :, 1][write_mask]
    memory = model.stages[0].memory
    q = F.normalize(memory.q(address), dim=-1)
    k = F.normalize(memory.k(address), dim=-1)
    similarity = q @ k.transpose(0, 1)
    chosen = similarity.argmax(dim=1)
    address_identity_accuracy = float((ids[chosen] == ids).float().mean())
    positives = ids[:, None].eq(ids[None, :])
    best_positive = similarity.masked_fill(~positives, -torch.inf).max(dim=1).values
    best_negative = similarity.masked_fill(positives, -torch.inf).max(dim=1).values
    finite_negative = torch.isfinite(best_negative)
    margin = float(
        (best_positive[finite_negative] - best_negative[finite_negative]).mean()
    ) if bool(finite_negative.any()) else float("inf")
    payload_code = memory.out(torch.tanh(memory.v(payload)))
    payload_logits = _decode_with_frozen_model_head(model, payload_code)
    payload_accuracy = float((payload_logits.argmax(dim=-1) == target).float().mean())
    return {
        "write_transition_address_identity_top1": address_identity_accuracy,
        "write_transition_address_margin": margin,
        "write_payload_token_accuracy": payload_accuracy,
    }


@torch.no_grad()
def episodic_mechanism_probe() -> dict[str, float | bool]:
    """Capacity/overwrite probe independent of model training."""
    dim = 16
    memory = VectorizedContextualEpisodicMemory(_IdentityMemorySource(dim))
    state = memory.empty_state(1, torch.device("cpu"), torch.float32)
    keys = torch.eye(dim)[:12].unsqueeze(0)
    current = torch.arange(12)
    payload = F.one_hot(current, num_classes=dim).float().unsqueeze(0)
    strength = torch.ones(1, 12, 1)
    state = memory.update_block(keys, payload, strength, state)

    overwrite_keys = torch.tensor([1, 4, 7, 10])
    stale = current[overwrite_keys].clone()
    replacement = torch.tensor([12, 13, 14, 15])
    current = current.clone()
    current[overwrite_keys] = replacement
    overwrite_address = keys[:, overwrite_keys]
    overwrite_payload = F.one_hot(replacement, num_classes=dim).float().unsqueeze(0)
    state = memory.update_block(
        overwrite_address,
        overwrite_payload,
        torch.ones(1, len(overwrite_keys), 1),
        state,
    )
    prediction = memory.read(keys, state).argmax(dim=-1)[0]
    current_accuracy = float((prediction == current).float().mean())
    overwrite_prediction = prediction[overwrite_keys]
    overwrite_accuracy = float(
        (overwrite_prediction == current[overwrite_keys]).float().mean()
    )
    stale_error = float((overwrite_prediction == stale).float().mean())
    passed = (
        current_accuracy >= MECHANISM_CURRENT_MIN
        and overwrite_accuracy >= MECHANISM_OVERWRITE_MIN
        and stale_error <= MECHANISM_STALE_MAX
    )
    return {
        "current_accuracy": current_accuracy,
        "overwrite_current_accuracy": overwrite_accuracy,
        "stale_error": stale_error,
        "valid_slots": float(state.valid.sum()),
        "pass": passed,
    }


@torch.no_grad()
def context_disambiguation_probe() -> dict[str, float | bool]:
    """Same surface address, two causal contexts, two payloads."""
    dim = 8
    source = _IdentityMemorySource(dim)
    contextual_memory = VectorizedContextualEpisodicMemory(source)
    token_only_memory = VectorizedContextualEpisodicMemory(source)

    c1 = F.one_hot(torch.tensor(0), num_classes=dim).float()
    c2 = F.one_hot(torch.tensor(1), num_classes=dim).float()
    address = F.one_hot(torch.tensor(2), num_classes=dim).float()
    target1 = F.one_hot(torch.tensor(4), num_classes=dim).float()
    target2 = F.one_hot(torch.tensor(5), num_classes=dim).float()

    # Choose raw payload-stage events so contextualization makes the payload
    # exactly the target basis vector while addresses share the same surface event.
    raw_payload1 = target1 - 0.5 * (c1 + address)
    raw_payload2 = target2 - 0.5 * (c2 + address)
    seq1 = torch.stack((c1, address, raw_payload1)).unsqueeze(0)
    seq2 = torch.stack((c2, address, raw_payload2)).unsqueeze(0)
    ctx1 = causal_contextualize(seq1)[0]
    ctx2 = causal_contextualize(seq2)[0]
    contextual_address = torch.stack((ctx1[1], ctx2[1])).unsqueeze(0)
    contextual_payload = torch.stack((ctx1[2], ctx2[2])).unsqueeze(0)
    strengths = torch.ones(1, 2, 1)

    contextual_state = contextual_memory.empty_state(
        1, torch.device("cpu"), torch.float32
    )
    contextual_state = contextual_memory.update_block(
        contextual_address, contextual_payload, strengths, contextual_state
    )
    contextual_prediction = contextual_memory.read(
        contextual_address, contextual_state
    ).argmax(dim=-1)[0]
    target_ids = torch.tensor([4, 5])
    contextual_accuracy = float(
        (contextual_prediction == target_ids).float().mean()
    )
    contextual_error = 1.0 - contextual_accuracy

    token_address = torch.stack((address, address)).unsqueeze(0)
    token_state = token_only_memory.empty_state(1, torch.device("cpu"), torch.float32)
    token_state = token_only_memory.update_block(
        token_address, contextual_payload, strengths, token_state
    )
    token_prediction = token_only_memory.read(token_address, token_state).argmax(dim=-1)[0]
    token_accuracy = float((token_prediction == target_ids).float().mean())
    passed = (
        contextual_accuracy >= AMBIGUOUS_CONTEXT_MIN
        and contextual_error <= AMBIGUOUS_CONTEXT_ERROR_MAX
        and token_accuracy <= TOKEN_ONLY_REFERENCE_MAX
    )
    return {
        "contextual_accuracy": contextual_accuracy,
        "contextual_cross_context_error": contextual_error,
        "token_only_reference_accuracy": token_accuracy,
        "contextual_valid_slots": float(contextual_state.valid.sum()),
        "token_only_valid_slots": float(token_state.valid.sum()),
        "pass": passed,
    }


@torch.no_grad()
def direct_episodic_read_evaluation(
    model: HardwareAwareAERATextLMV24,
    batch: Any,
) -> dict[str, float]:
    """Read trained stage-0 episodic state directly before each query chunk update."""
    model.eval()
    model.set_memory_pretraining_mode(False)
    stage = model.stages[0]
    if not isinstance(stage, VectorizedContextualEpisodicMemoryStage):
        raise TypeError("v24 stage0 type mismatch")
    chunks = batch.tokens.view(batch.tokens.size(0), -1, CHUNK_SIZE)
    pos = torch.arange(CHUNK_SIZE, device=batch.tokens.device)
    state = None
    predictions: list[torch.Tensor] = []
    query_index = 0
    overwrite_predictions: list[torch.Tensor] = []
    overwrite_targets: list[torch.Tensor] = []
    overwrite_stale: list[torch.Tensor] = []

    for chunk_index in range(chunks.size(1)):
        chunk = chunks[:, chunk_index]
        events = model.token_emb(chunk) + model.local_pos(pos)[None, :, :]
        if state is None:
            state = stage.empty_state(events)
        base_h = stage.norm(events)
        contextual = causal_contextualize(base_h)
        recall = stage.memory.read(contextual, state.memory)
        if bool(chunk[:, 0].eq(QUERY).all()):
            decoded = _decode_with_frozen_model_head(model, recall[:, 1])
            prediction = decoded.argmax(dim=-1).cpu()
            predictions.append(prediction)
            overwrite_mask = batch.overwrite_mask[:, query_index]
            if bool(overwrite_mask.any()):
                overwrite_predictions.append(prediction[overwrite_mask])
                overwrite_targets.append(
                    batch.query_targets[:, query_index][overwrite_mask]
                )
                overwrite_stale.append(
                    batch.stale_targets[:, query_index][overwrite_mask]
                )
            query_index += 1
        _, state, _ = stage.forward_chunk(
            events,
            state,
            hard=True,
            update_memory=True,
        )

    if query_index != batch.query_targets.size(1):
        raise RuntimeError("direct episodic evaluator query count mismatch")
    prediction_matrix = torch.stack(predictions, dim=1)
    accuracy = float((prediction_matrix == batch.query_targets).float().mean())
    overwrite_prediction = torch.cat(overwrite_predictions)
    overwrite_target = torch.cat(overwrite_targets)
    overwrite_stale_target = torch.cat(overwrite_stale)
    overwrite_accuracy = float(
        (overwrite_prediction == overwrite_target).float().mean()
    )
    stale_error = float(
        (overwrite_prediction == overwrite_stale_target).float().mean()
    )
    return {
        "overall_accuracy": accuracy,
        "overwrite_current_value_accuracy": overwrite_accuracy,
        "overwrite_stale_value_error": stale_error,
    }


def train_pair_with_v24_objective(
    *,
    steps: int = TRAIN_STEPS,
) -> tuple[HardwareAwareAERATextLMV24, HardwareAwareAERATextLMV24, dict[str, Any]]:
    full = build_model(SEED)
    stream_only = build_model(SEED)
    for key, value in full.state_dict().items():
        torch.testing.assert_close(
            stream_only.state_dict()[key], value, atol=0.0, rtol=0.0
        )

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
        local = contextual_memory_terms(full, batch.tokens)
        total = (
            query_loss
            + ADDRESS_CONTRASTIVE_WEIGHT * local["address_contrastive_loss"]
            + PAYLOAD_TOKEN_WEIGHT * local["payload_token_loss"]
        )
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
                "address_contrastive_loss": float(
                    local["address_contrastive_loss"].detach()
                ),
                "payload_token_loss": float(local["payload_token_loss"].detach()),
                "payload_token_accuracy": float(
                    local["payload_token_accuracy"].detach()
                ),
            }
            history.append(row)
            print("AERA_V24_VCEM_TRAIN=" + json.dumps(row, sort_keys=True), flush=True)

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

    full, stream_only, training = train_pair_with_v24_objective(steps=steps)
    full_eval = _evaluate(full, eval_batch, memory=True)
    memory_off = _evaluate(full, eval_batch, memory=False)
    stream_eval = _evaluate(stream_only, eval_batch, memory=False)
    direct = direct_episodic_read_evaluation(full, eval_batch)
    safety = deployment_safety_and_isolation(full, eval_batch)
    local = heldout_local_contextual_code(full, eval_batch.tokens)

    # Exact physical budget/update-call audit on the trained stage.
    stage = full.stages[0]
    first_chunk = eval_batch.tokens[:, :CHUNK_SIZE]
    pos = torch.arange(CHUNK_SIZE)
    events = full.token_emb(first_chunk) + full.local_pos(pos)[None, :, :]
    _ = stage.forward_chunk(events, None, hard=True, update_memory=True)
    budget_ok = (
        stage.last_candidate_count == 5
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
        interpretation = "v24_contextual_episodic_memory_passes_controlled_gate"
    elif not ambiguity["pass"]:
        interpretation = "v24_context_representation_fails_ambiguity_gate"
    elif not mechanism["pass"] or direct["overall_accuracy"] < DIRECT_RECALL_MIN:
        interpretation = "v24_episodic_replacement_or_retrieval_insufficient"
    elif full_eval["query_accuracy"] < FULL_ACCURACY_MIN:
        interpretation = "v24_direct_memory_not_integrated_into_end_to_end_prediction"
    else:
        interpretation = "v24_controlled_gate_partial_failure"

    return {
        "scope": "aera_v24_vectorized_contextual_episodic_memory_cpu",
        "architecture_version": "v24",
        "protocol": vectorized_contextual_episodic_protocol(),
        "gpu_used": False,
        "independent_evidence": False,
        "steps": steps,
        "task": {
            "oracle_accuracy": validity,
            "chance_accuracy": chance,
        },
        "state_bytes_real_language_four_stage_memory_dim50": state_bytes,
        "mechanism_probe": mechanism,
        "context_disambiguation_probe": ambiguity,
        "training": training,
        "heldout_local_contextual_code": local,
        "direct_episodic_read": direct,
        "full_stream_plus_memory": {
            k: v for k, v in full_eval.items() if k != "predictions"
        },
        "same_checkpoint_memory_disabled": {
            k: v for k, v in memory_off.items() if k != "predictions"
        },
        "separately_trained_stream_only": {
            k: v for k, v in stream_eval.items() if k != "predictions"
        },
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
            "controlled_v24_memory_passed": passed,
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
    print("AERA_V24_VCEM_RESULT_JSON=" + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
