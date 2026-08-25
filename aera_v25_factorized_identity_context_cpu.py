from __future__ import annotations

"""CPU-only controlled gate for AERA-v25 Factorized Identity-Context Episodic Memory.

Preregistered in issue #359 after the valid v24.1 scientific failure #358.
V25 keeps the v24/v23 bounded sparse episodic machinery and v24.1 causal objective,
but factorizes each fixed-width address into a shared current-event identity half and
a shared strictly-prior causal-context half.
"""

import json
from typing import Any

import torch
import torch.nn as nn
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
)
from tam_research.aera_hardware_core_v23 import (
    select_budgeted_event_pairs,
    sparse_write_budget,
)
from tam_research.aera_hardware_core_v24 import episodic_state_bytes_per_session
from tam_research.aera_hardware_core_v25 import (
    FactorizedIdentityContextEpisodicMemory,
    FactorizedIdentityContextEpisodicMemoryStage,
    HardwareAwareAERATextLMV25,
    causal_identity_context,
    factorized_identity_context_protocol,
)

SEED = 8421
EVAL_SEED = 8422
ADDRESS_CONTRASTIVE_WEIGHT = 0.0


class _IdentityMemorySource(nn.Module):
    """Deterministic probe source with identity q/k/v/out geometry."""

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


def build_model(seed: int) -> HardwareAwareAERATextLMV25:
    torch.manual_seed(seed)
    model = HardwareAwareAERATextLMV25(diagnostic_config())
    _force_all_stages_run(model)
    return model


def _stage0_sources(
    model: HardwareAwareAERATextLMV25,
    tokens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if tokens.ndim != 2 or tokens.size(1) % CHUNK_SIZE:
        raise ValueError("tokens must be [batch,sequence] divisible by CHUNK_SIZE")
    chunks = tokens.view(tokens.size(0), -1, CHUNK_SIZE)
    pos = torch.arange(CHUNK_SIZE, device=tokens.device)
    events = model.token_emb(chunks) + model.local_pos(pos)[None, None, :, :]
    base_h = model.stages[0].norm(events).detach()
    flat = base_h.reshape(-1, CHUNK_SIZE, base_h.size(-1))
    identity, context, contextual = causal_identity_context(flat)
    shape = base_h.shape
    return (
        identity.reshape(shape),
        context.reshape(shape),
        contextual.reshape(shape),
        chunks,
    )


def payload_memory_terms(
    model: HardwareAwareAERATextLMV25,
    tokens: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Exact v24.1 decoder-aligned payload CE; no address auxiliary."""
    _, _, contextual, chunks = _stage0_sources(model, tokens)
    payload_source = contextual[:, :, 1:]
    next_tokens = chunks[:, :, 1:]
    memory = model.stages[0].memory
    payload_code = memory.out(torch.tanh(memory.v(payload_source)))
    payload_logits = _decode_with_frozen_model_head(model, payload_code)
    payload_loss = F.cross_entropy(
        payload_logits.float().reshape(-1, payload_logits.size(-1)),
        next_tokens.reshape(-1),
    )
    payload_accuracy = (payload_logits.argmax(dim=-1) == next_tokens).float().mean()
    return {
        "payload_token_loss": payload_loss,
        "payload_token_accuracy": payload_accuracy,
    }


@torch.no_grad()
def factorization_invariance_and_causality_probe(
    model: HardwareAwareAERATextLMV25,
) -> dict[str, bool | float]:
    """Prove shared identity, distinct causal context, and future-value invariance."""
    key = 12
    value_a = 24
    value_b = 25
    filler = 4
    write_a = torch.tensor([[WRITE, key, value_a, filler, filler, filler]])
    write_b = torch.tensor([[WRITE, key, value_b, filler, filler, filler]])
    query = torch.tensor([[QUERY, key, value_a, filler, filler, filler]])

    def factors(chunk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pos = torch.arange(CHUNK_SIZE)
        events = model.token_emb(chunk) + model.local_pos(pos)[None, :, :]
        h = model.stages[0].norm(events)
        identity, context, _ = causal_identity_context(h)
        memory = model.stages[0].memory
        i, c, combined = memory.address_factors(identity, context)
        return i[:, 1], c[:, 1], combined[:, 1]

    wi_a, wc_a, wa_a = factors(write_a)
    wi_b, wc_b, wa_b = factors(write_b)
    qi, qc, qa = factors(query)
    return {
        "write_query_identity_bit_identical": bool(torch.equal(wi_a, qi)),
        "write_query_context_differs": bool(not torch.equal(wc_a, qc)),
        "write_query_context_l2": float((wc_a - qc).pow(2).sum().sqrt()),
        "future_value_identity_bit_identical": bool(torch.equal(wi_a, wi_b)),
        "future_value_context_bit_identical": bool(torch.equal(wc_a, wc_b)),
        "future_value_combined_bit_identical": bool(torch.equal(wa_a, wa_b)),
        "write_query_combined_differs": bool(not torch.equal(wa_a, qa)),
    }


@torch.no_grad()
def factorized_mechanism_probe() -> dict[str, float | bool]:
    """Capacity/overwrite semantics with deterministic factorized identity addresses."""
    dim = 32
    memory = FactorizedIdentityContextEpisodicMemory(_IdentityMemorySource(dim))
    state = memory.empty_state(1, torch.device("cpu"), torch.float32)
    identities = torch.eye(dim)[:12].unsqueeze(0)
    contexts = torch.zeros_like(identities)
    current = torch.arange(12)
    payload = F.one_hot(current, num_classes=dim).float().unsqueeze(0)
    strength = torch.ones(1, 12, 1)
    state = memory.update_block(identities, contexts, payload, strength, state)

    overwrite_keys = torch.tensor([1, 4, 7, 10])
    stale = current[overwrite_keys].clone()
    replacement = torch.tensor([12, 13, 14, 15])
    current = current.clone()
    current[overwrite_keys] = replacement
    state = memory.update_block(
        identities[:, overwrite_keys],
        contexts[:, overwrite_keys],
        F.one_hot(replacement, num_classes=dim).float().unsqueeze(0),
        torch.ones(1, len(overwrite_keys), 1),
        state,
    )
    prediction = memory.read(identities, contexts, state).argmax(dim=-1)[0]
    current_accuracy = float((prediction == current).float().mean())
    overwrite_prediction = prediction[overwrite_keys]
    overwrite_accuracy = float((overwrite_prediction == current[overwrite_keys]).float().mean())
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
def factorized_context_disambiguation_probe() -> dict[str, float | bool]:
    """Same identity, two causal contexts, two payloads; identity-only is ambiguous."""
    dim = 8
    source = _IdentityMemorySource(dim)
    memory = FactorizedIdentityContextEpisodicMemory(source)
    identity_only = FactorizedIdentityContextEpisodicMemory(source)

    identity = F.one_hot(torch.tensor(2), num_classes=dim).float()
    c1 = F.one_hot(torch.tensor(0), num_classes=dim).float()
    c2 = F.one_hot(torch.tensor(1), num_classes=dim).float()
    target1 = F.one_hot(torch.tensor(4), num_classes=dim).float()
    target2 = F.one_hot(torch.tensor(5), num_classes=dim).float()
    identities = torch.stack((identity, identity)).unsqueeze(0)
    contexts = torch.stack((c1, c2)).unsqueeze(0)
    payloads = torch.stack((target1, target2)).unsqueeze(0)
    strengths = torch.ones(1, 2, 1)

    state = memory.empty_state(1, torch.device("cpu"), torch.float32)
    state = memory.update_block(identities, contexts, payloads, strengths, state)
    prediction = memory.read(identities, contexts, state).argmax(dim=-1)[0]
    targets = torch.tensor([4, 5])
    contextual_accuracy = float((prediction == targets).float().mean())
    contextual_error = 1.0 - contextual_accuracy

    zero_context = torch.zeros_like(contexts)
    token_state = identity_only.empty_state(1, torch.device("cpu"), torch.float32)
    token_state = identity_only.update_block(
        identities, zero_context, payloads, strengths, token_state
    )
    token_prediction = identity_only.read(
        identities, zero_context, token_state
    ).argmax(dim=-1)[0]
    token_accuracy = float((token_prediction == targets).float().mean())
    passed = (
        contextual_accuracy >= AMBIGUOUS_CONTEXT_MIN
        and contextual_error <= AMBIGUOUS_CONTEXT_ERROR_MAX
        and token_accuracy <= TOKEN_ONLY_REFERENCE_MAX
    )
    return {
        "contextual_accuracy": contextual_accuracy,
        "contextual_cross_context_error": contextual_error,
        "token_only_reference_accuracy": token_accuracy,
        "contextual_valid_slots": float(state.valid.sum()),
        "token_only_valid_slots": float(token_state.valid.sum()),
        "pass": passed,
    }


def primary_query_gradient_probe() -> dict[str, float | bool]:
    """Primary causal query CE alone must reach both factors, v/out, and selector."""
    model = build_model(9571)
    batch = make_batch(2, EVAL_SEED + 9571)
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
        "identity_proj": stage.memory.identity_proj.weight.grad,
        "context_proj": stage.memory.context_proj.weight.grad,
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
def heldout_factorized_address_and_selector_diagnostics(
    model: HardwareAwareAERATextLMV25,
    batch: Any,
) -> dict[str, float]:
    """Evaluation-only identity/combined latest-write geometry and selector coverage."""
    model.eval()
    stage = model.stages[0]
    if not isinstance(stage, FactorizedIdentityContextEpisodicMemoryStage):
        raise TypeError("v25 stage0 type mismatch")
    chunks = batch.tokens.view(batch.tokens.size(0), -1, CHUNK_SIZE)
    pos = torch.arange(CHUNK_SIZE, device=batch.tokens.device)

    latest: list[dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = [
        dict() for _ in range(chunks.size(0))
    ]
    queries: list[tuple[int, int, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    selector_hits = 0
    selector_total = 0
    payload_correct = 0
    payload_total = 0

    for chunk_index in range(chunks.size(1)):
        chunk = chunks[:, chunk_index]
        events = model.token_emb(chunk) + model.local_pos(pos)[None, :, :]
        base_h = stage.norm(events)
        identity_source, context_source, contextual = causal_identity_context(base_h)
        identity_factor, context_factor, combined = stage.memory.address_factors(
            identity_source, context_source
        )
        prefix = chunk[:, 0]

        write_mask = prefix.eq(WRITE)
        if bool(write_mask.any()):
            address_source = contextual[:, :-1]
            payload_source = contextual[:, 1:]
            pair_features = torch.cat((address_source, payload_source), dim=-1)
            pair_logits = stage.pair_write_gate(pair_features)
            pair_gate = torch.sigmoid(pair_logits)
            selected = select_budgeted_event_pairs(
                address_source,
                payload_source,
                pair_gate,
                pair_logits,
                differentiable_selector=False,
            )
            for b in torch.nonzero(write_mask, as_tuple=False).flatten().tolist():
                key = int(chunk[b, 1])
                latest[b][key] = (
                    identity_factor[b, 1].detach().clone(),
                    context_factor[b, 1].detach().clone(),
                    combined[b, 1].detach().clone(),
                )
                selector_total += 1
                selector_hits += int(bool((selected.indices[b] == 1).any()))
                payload_code = stage.memory.out(
                    torch.tanh(stage.memory.v(contextual[b : b + 1, 2]))
                )
                payload_logits = _decode_with_frozen_model_head(model, payload_code)
                payload_correct += int(
                    payload_logits.argmax(dim=-1).item() == int(chunk[b, 2])
                )
                payload_total += 1

        query_mask = prefix.eq(QUERY)
        if bool(query_mask.any()):
            for b in torch.nonzero(query_mask, as_tuple=False).flatten().tolist():
                queries.append(
                    (
                        b,
                        int(chunk[b, 1]),
                        identity_factor[b, 1].detach().clone(),
                        context_factor[b, 1].detach().clone(),
                        combined[b, 1].detach().clone(),
                    )
                )

    identity_correct = 0
    combined_correct = 0
    total = 0
    identity_margins: list[float] = []
    combined_margins: list[float] = []
    context_contributions: list[float] = []
    for b, key, q_identity, q_context, q_combined in queries:
        candidates = latest[b]
        if len(candidates) != N_KEYS or key not in candidates:
            continue
        ordered = sorted(candidates)
        write_identity = torch.stack([candidates[k][0] for k in ordered])
        write_context = torch.stack([candidates[k][1] for k in ordered])
        write_combined = torch.stack([candidates[k][2] for k in ordered])
        identity_sim = q_identity @ write_identity.transpose(0, 1)
        combined_sim = q_combined @ write_combined.transpose(0, 1)
        identity_correct += int(ordered[int(identity_sim.argmax())] == key)
        combined_correct += int(ordered[int(combined_sim.argmax())] == key)
        total += 1
        positive_index = ordered.index(key)
        negative_mask = torch.ones_like(combined_sim, dtype=torch.bool)
        negative_mask[positive_index] = False
        identity_margins.append(
            float(identity_sim[positive_index] - identity_sim[negative_mask].max())
        )
        negative_indices = torch.nonzero(negative_mask, as_tuple=False).flatten()
        best_negative_index = negative_indices[combined_sim[negative_mask].argmax()]
        combined_margins.append(
            float(combined_sim[positive_index] - combined_sim[best_negative_index])
        )
        context_contributions.append(
            0.5
            * float(
                q_context @ write_context[positive_index]
                - q_context @ write_context[best_negative_index]
            )
        )

    return {
        "identity_query_to_latest_write_top1": identity_correct / max(total, 1),
        "identity_query_to_latest_write_margin": sum(identity_margins) / max(len(identity_margins), 1),
        "combined_query_to_latest_write_top1": combined_correct / max(total, 1),
        "combined_query_to_latest_write_margin": sum(combined_margins) / max(len(combined_margins), 1),
        "context_margin_contribution": sum(context_contributions) / max(len(context_contributions), 1),
        "true_key_value_selector_coverage": selector_hits / max(selector_total, 1),
        "write_payload_token_accuracy": payload_correct / max(payload_total, 1),
    }


@torch.no_grad()
def direct_factorized_episodic_read_evaluation(
    model: HardwareAwareAERATextLMV25,
    batch: Any,
) -> dict[str, float]:
    """Read trained stage-0 v25 state directly before each query chunk update."""
    model.eval()
    model.set_memory_pretraining_mode(False)
    stage = model.stages[0]
    if not isinstance(stage, FactorizedIdentityContextEpisodicMemoryStage):
        raise TypeError("v25 stage0 type mismatch")
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
        identity, context, _ = causal_identity_context(base_h)
        recall = stage.memory.read(identity, context, state.memory)
        if bool(chunk[:, 0].eq(QUERY).all()):
            decoded = _decode_with_frozen_model_head(model, recall[:, 1])
            prediction = decoded.argmax(dim=-1).cpu()
            predictions.append(prediction)
            overwrite_mask = batch.overwrite_mask[:, query_index]
            if bool(overwrite_mask.any()):
                overwrite_predictions.append(prediction[overwrite_mask])
                overwrite_targets.append(batch.query_targets[:, query_index][overwrite_mask])
                overwrite_stale.append(batch.stale_targets[:, query_index][overwrite_mask])
            query_index += 1
        _, state, _ = stage.forward_chunk(
            events,
            state,
            hard=True,
            update_memory=True,
        )

    if query_index != batch.query_targets.size(1):
        raise RuntimeError("direct v25 evaluator query count mismatch")
    prediction_matrix = torch.stack(predictions, dim=1)
    accuracy = float((prediction_matrix == batch.query_targets).float().mean())
    overwrite_prediction = torch.cat(overwrite_predictions)
    overwrite_target = torch.cat(overwrite_targets)
    overwrite_stale_target = torch.cat(overwrite_stale)
    overwrite_accuracy = float((overwrite_prediction == overwrite_target).float().mean())
    stale_error = float((overwrite_prediction == overwrite_stale_target).float().mean())
    return {
        "overall_accuracy": accuracy,
        "overwrite_current_value_accuracy": overwrite_accuracy,
        "overwrite_stale_value_error": stale_error,
    }


def train_pair_with_v25_objective(
    *,
    steps: int = TRAIN_STEPS,
) -> tuple[HardwareAwareAERATextLMV25, HardwareAwareAERATextLMV25, dict[str, Any]]:
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
            print("AERA_V25_FICEM_TRAIN=" + json.dumps(row, sort_keys=True), flush=True)

    return full, stream_only, {"history": history}


def run_gate(*, steps: int = TRAIN_STEPS) -> dict[str, Any]:
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    mechanism = factorized_mechanism_probe()
    ambiguity = factorized_context_disambiguation_probe()
    state_bytes = episodic_state_bytes_per_session(n_stages=4, memory_dim=50)
    eval_batch = make_batch(24, EVAL_SEED)
    validity = oracle_accuracy(eval_batch)
    if validity < TASK_VALIDITY_MIN:
        raise RuntimeError(f"task validity failed before training: {validity}")

    full, stream_only, training = train_pair_with_v25_objective(steps=steps)
    full_eval = _evaluate(full, eval_batch, memory=True)
    memory_off = _evaluate(full, eval_batch, memory=False)
    stream_eval = _evaluate(stream_only, eval_batch, memory=False)
    direct = direct_factorized_episodic_read_evaluation(full, eval_batch)
    safety = deployment_safety_and_isolation(full, eval_batch)
    diagnostic = heldout_factorized_address_and_selector_diagnostics(full, eval_batch)
    invariance = factorization_invariance_and_causality_probe(full)

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
        "factorized_context_disambiguation_ge_0_95": ambiguity["contextual_accuracy"] >= AMBIGUOUS_CONTEXT_MIN,
        "factorized_context_cross_error_le_0_05": ambiguity["contextual_cross_context_error"] <= AMBIGUOUS_CONTEXT_ERROR_MAX,
        "identity_only_reference_le_0_75": ambiguity["token_only_reference_accuracy"] <= TOKEN_ONLY_REFERENCE_MAX,
        "cross_operation_identity_invariant": bool(invariance["write_query_identity_bit_identical"]),
        "context_factor_separates_operations": bool(invariance["write_query_context_differs"]),
        "future_value_causality_exact": bool(
            invariance["future_value_identity_bit_identical"]
            and invariance["future_value_context_bit_identical"]
            and invariance["future_value_combined_bit_identical"]
        ),
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
        "state_bytes_exact_77760": state_bytes == 77760,
        "state_bytes_le_80000": state_bytes <= STATE_BYTES_MAX,
    }
    passed = all(checks.values())
    identity_top1 = diagnostic["identity_query_to_latest_write_top1"]
    combined_top1 = diagnostic["combined_query_to_latest_write_top1"]
    if passed:
        interpretation = "v25_factorized_identity_context_memory_passes_controlled_gate"
    elif not mechanism["pass"] or not ambiguity["pass"]:
        interpretation = "v25_factorization_mechanism_or_ambiguity_invalid"
    elif identity_top1 >= 0.80 and combined_top1 < 0.80:
        interpretation = "v25_context_term_over_penalizes_cross_view_retrieval"
    elif combined_top1 >= 0.80 and direct["overall_accuracy"] < DIRECT_RECALL_MIN:
        interpretation = "v25_payload_or_read_pathway_is_blocker"
    elif direct["overall_accuracy"] >= DIRECT_RECALL_MIN and full_eval["query_accuracy"] < FULL_ACCURACY_MIN:
        interpretation = "v25_memory_integration_or_read_gating_is_blocker"
    elif combined_top1 < 0.80:
        interpretation = "v25_factorized_address_learning_insufficient"
    else:
        interpretation = "v25_controlled_gate_partial_failure"

    return {
        "scope": "aera_v25_factorized_identity_context_cpu",
        "architecture_version": "v25",
        "objective_version": "v24.1-causal-end-to-end-addressing",
        "protocol": factorized_identity_context_protocol(),
        "gpu_used": False,
        "independent_evidence": False,
        "train_seed": SEED,
        "eval_seed": EVAL_SEED,
        "steps": steps,
        "address_contrastive_weight": ADDRESS_CONTRASTIVE_WEIGHT,
        "task": {"oracle_accuracy": validity, "chance_accuracy": chance},
        "state_bytes_real_language_four_stage_memory_dim50": state_bytes,
        "factorization_invariance_and_causality_probe": invariance,
        "mechanism_probe": mechanism,
        "context_disambiguation_probe": ambiguity,
        "training": training,
        "heldout_factorized_address_and_selector_diagnostics": diagnostic,
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
            "controlled_v25_memory_passed": passed,
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
    print("AERA_V25_FICEM_RESULT_JSON=" + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
