from __future__ import annotations

"""CPU-only collapse-resistant memory-objective diagnostic for unchanged AERA-v21.

Issue #304 proved that the existing 16-d q/k and v/out projections have enough
capacity under explicit supervision.  This diagnostic therefore changes only the
training signal: negative-aware address matching plus latent/token payload utility.
It does not change AERA-v21 mechanics, parameters, memory size, or deployment.
"""

import json
from typing import Any

import torch
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
    SAME_CHECKPOINT_MEMORY_DROP_MIN,
    SEED,
    STALE_ERROR_MAX,
    TASK_VALIDITY_MIN,
    TRAIN_STEPS,
    VALUE_START,
    _evaluate,
    _loss_and_accuracy,
    deployment_safety_and_isolation,
    make_batch,
    oracle_accuracy,
)
from aera_v21_memory_code_objective_cpu import (
    _decode_with_frozen_model_head,
    _detached_stage0_event_pairs,
)
from aera_v21_memory_necessity_cpu import build_model
from aera_v21_memory_capacity_audit_cpu import _token_representation
from tam_research.aera_hardware_core_v21 import HardwareAwareAERATextLMV21

ADDRESS_TEMPERATURE = 0.10
ADDRESS_CONTRASTIVE_WEIGHT = 1.0
PAYLOAD_LATENT_WEIGHT = 1.0
PAYLOAD_TOKEN_WEIGHT = 1.0
ADDRESS_TOP1_MIN = 0.95
ADDRESS_MARGIN_MIN = 0.05
PAYLOAD_LEGAL_ACCURACY_MIN = 0.90


def _address_ids(tokens: torch.Tensor) -> torch.Tensor:
    if tokens.ndim != 2 or tokens.size(1) % CHUNK_SIZE:
        raise ValueError("tokens must be [batch,sequence] divisible by CHUNK_SIZE")
    chunks = tokens.view(tokens.size(0), -1, CHUNK_SIZE)
    address_tokens = chunks[:, :, :-1]
    positions = torch.arange(CHUNK_SIZE - 1, device=tokens.device)
    positions = positions.view(1, 1, -1).expand_as(address_tokens)
    # Text diagnostic identity is fully observed, not a synthetic task label.
    # Same token at the same local event position is a multi-positive match.
    return address_tokens * (CHUNK_SIZE - 1) + positions


def multi_positive_contrastive_loss(
    q: torch.Tensor,
    k: torch.Tensor,
    identity: torch.Tensor,
    *,
    temperature: float = ADDRESS_TEMPERATURE,
) -> torch.Tensor:
    if q.shape != k.shape or q.ndim != 2:
        raise ValueError("q and k must match [events,dim]")
    if identity.shape != (q.size(0),):
        raise ValueError("identity must be [events]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    q = F.normalize(q, dim=-1)
    k = F.normalize(k, dim=-1)
    logits = (q @ k.transpose(0, 1)) / temperature
    positives = identity[:, None].eq(identity[None, :])
    if not bool(positives.any(dim=1).all()):
        raise RuntimeError("every address must have at least one positive")
    neg_inf = torch.finfo(logits.dtype).min
    q_to_k = -(
        torch.logsumexp(logits.masked_fill(~positives, neg_inf), dim=1)
        - torch.logsumexp(logits, dim=1)
    ).mean()
    logits_t = logits.transpose(0, 1)
    k_to_q = -(
        torch.logsumexp(logits_t.masked_fill(~positives.transpose(0, 1), neg_inf), dim=1)
        - torch.logsumexp(logits_t, dim=1)
    ).mean()
    return 0.5 * (q_to_k + k_to_q)


def collapse_resistant_memory_terms(
    model: HardwareAwareAERATextLMV21,
    tokens: torch.Tensor,
) -> dict[str, torch.Tensor]:
    address_source, payload_source, next_tokens = _detached_stage0_event_pairs(model, tokens)
    memory = model.stages[0].memory

    q = memory.q(address_source).reshape(-1, memory.q.out_features)
    k = memory.k(address_source).reshape(-1, memory.k.out_features)
    ids = _address_ids(tokens).reshape(-1)
    address_loss = multi_positive_contrastive_loss(q, k, ids)

    payload_code = memory.out(torch.tanh(memory.v(payload_source)))
    payload_latent_cosine = F.cosine_similarity(
        payload_code,
        payload_source.detach(),
        dim=-1,
    )
    payload_latent_loss = (1.0 - payload_latent_cosine).mean()

    payload_logits = _decode_with_frozen_model_head(model, payload_code)
    payload_token_loss = F.cross_entropy(
        payload_logits.float().reshape(-1, payload_logits.size(-1)),
        next_tokens.reshape(-1),
    )
    payload_token_accuracy = (
        payload_logits.argmax(dim=-1) == next_tokens
    ).float().mean()

    return {
        "address_contrastive_loss": address_loss,
        "payload_latent_loss": payload_latent_loss,
        "payload_latent_cosine": payload_latent_cosine.mean(),
        "payload_token_loss": payload_token_loss,
        "payload_token_accuracy": payload_token_accuracy,
    }


@torch.no_grad()
def evaluate_local_memory_code(model: HardwareAwareAERATextLMV21) -> dict[str, float]:
    model.eval()
    device = model.token_emb.weight.device
    # The controlled task has twelve legal key tokens and sixteen legal values.
    from aera_v19_memory_necessity_cpu import KEY_START, N_KEYS

    key_tokens = torch.arange(KEY_START, KEY_START + N_KEYS, device=device)
    key_x = _token_representation(model, key_tokens, 1)
    memory = model.stages[0].memory
    q = F.normalize(memory.q(key_x), dim=-1)
    k = F.normalize(memory.k(key_x), dim=-1)
    similarity = q @ k.transpose(0, 1)
    target = torch.arange(N_KEYS, device=device)
    top1 = float((similarity.argmax(dim=-1) == target).float().mean())
    diag = similarity.diag()
    mask = ~torch.eye(N_KEYS, dtype=torch.bool, device=device)
    best_other = similarity.masked_fill(~mask, -torch.inf).max(dim=1).values
    margin = float((diag - best_other).mean())

    value_tokens = torch.arange(VALUE_START, VALUE_START + N_VALUES, device=device)
    value_x = _token_representation(model, value_tokens, 2)
    payload_code = memory.out(torch.tanh(memory.v(value_x)))
    logits = _decode_with_frozen_model_head(model, payload_code)
    legal = logits[:, VALUE_START : VALUE_START + N_VALUES]
    value_target = torch.arange(N_VALUES, device=device)
    payload_accuracy = float((legal.argmax(dim=-1) == value_target).float().mean())
    return {
        "address_top1_accuracy": top1,
        "address_mean_diag_minus_best_other_margin": margin,
        "payload_legal_value_accuracy": payload_accuracy,
    }


def train_pair_with_collapse_resistant_objective(
    *,
    steps: int = TRAIN_STEPS,
) -> tuple[HardwareAwareAERATextLMV21, HardwareAwareAERATextLMV21, dict[str, Any]]:
    full = build_model(SEED)
    stream_only = build_model(SEED)
    for key, value in full.state_dict().items():
        torch.testing.assert_close(stream_only.state_dict()[key], value, atol=0.0, rtol=0.0)

    initial_local = evaluate_local_memory_code(full)
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
        local = collapse_resistant_memory_terms(full, batch.tokens)
        total = (
            query_loss
            + ADDRESS_CONTRASTIVE_WEIGHT * local["address_contrastive_loss"]
            + PAYLOAD_LATENT_WEIGHT * local["payload_latent_loss"]
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
                "address_contrastive_loss": float(local["address_contrastive_loss"].detach()),
                "payload_latent_loss": float(local["payload_latent_loss"].detach()),
                "payload_latent_cosine": float(local["payload_latent_cosine"].detach()),
                "payload_token_loss": float(local["payload_token_loss"].detach()),
                "payload_token_accuracy": float(local["payload_token_accuracy"].detach()),
            }
            history.append(row)
            print("AERA_V21_COLLAPSE_RESISTANT_TRAIN=" + json.dumps(row, sort_keys=True), flush=True)

    return full, stream_only, {
        "history": history,
        "initial_local_code": initial_local,
        "final_local_code": evaluate_local_memory_code(full),
    }


def run_diagnostic(*, steps: int = TRAIN_STEPS) -> dict[str, Any]:
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    eval_batch = make_batch(24, EVAL_SEED)
    validity = oracle_accuracy(eval_batch)
    if validity < TASK_VALIDITY_MIN:
        raise RuntimeError(f"task validity failed before training: {validity}")

    full, stream_only, training = train_pair_with_collapse_resistant_objective(steps=steps)
    full_eval = _evaluate(full, eval_batch, memory=True)
    memory_off = _evaluate(full, eval_batch, memory=False)
    stream_eval = _evaluate(stream_only, eval_batch, memory=False)
    safety = deployment_safety_and_isolation(full, eval_batch)
    local = evaluate_local_memory_code(full)

    chance = 1.0 / N_VALUES
    full_minus_stream = full_eval["query_accuracy"] - stream_eval["query_accuracy"]
    memory_drop = full_eval["query_accuracy"] - memory_off["query_accuracy"]
    checks = {
        "task_validity_ge_0_95": validity >= TASK_VALIDITY_MIN,
        "address_top1_ge_0_95": local["address_top1_accuracy"] >= ADDRESS_TOP1_MIN,
        "address_margin_ge_0_05": local["address_mean_diag_minus_best_other_margin"] >= ADDRESS_MARGIN_MIN,
        "payload_legal_value_accuracy_ge_0_90": local["payload_legal_value_accuracy"] >= PAYLOAD_LEGAL_ACCURACY_MIN,
        "full_query_accuracy_ge_0_80": full_eval["query_accuracy"] >= FULL_ACCURACY_MIN,
        "full_over_stream_only_ge_0_15": full_minus_stream >= FULL_OVER_STREAM_MIN,
        "same_checkpoint_memory_drop_ge_0_15": memory_drop >= SAME_CHECKPOINT_MEMORY_DROP_MIN,
        "overwrite_current_value_accuracy_ge_0_80": full_eval["overwrite_current_value_accuracy"] >= OVERWRITE_ACCURACY_MIN,
        "stale_value_error_le_0_10": full_eval["stale_value_error_rate"] <= STALE_ERROR_MAX,
        "fresh_session_near_chance": safety["fresh_session_query_accuracy"] <= chance + FRESH_SESSION_CHANCE_TOLERANCE,
        "deployment_base_parameters_unchanged": bool(safety["deployment_base_parameters_unchanged"]),
        "session_isolation_exact": bool(safety["session_isolation_exact"]),
    }
    local_pass = all(
        checks[name]
        for name in (
            "address_top1_ge_0_95",
            "address_margin_ge_0_05",
            "payload_legal_value_accuracy_ge_0_90",
        )
    )
    e2e_pass = all(
        value for key, value in checks.items()
        if key not in {
            "address_top1_ge_0_95",
            "address_margin_ge_0_05",
            "payload_legal_value_accuracy_ge_0_90",
        }
    )
    if local_pass and e2e_pass:
        interpretation = "collapse_resistant_objective_solves_controlled_memory_bottleneck"
    elif local_pass:
        interpretation = "local_code_recoverable_but_end_to_end_memory_semantics_still_insufficient"
    else:
        interpretation = "collapse_resistant_objective_still_insufficient"

    return {
        "scope": "aera_v21_collapse_resistant_memory_objective_cpu",
        "architecture_changed": False,
        "independent_evidence": False,
        "steps": steps,
        "weights": {
            "address_contrastive": ADDRESS_CONTRASTIVE_WEIGHT,
            "payload_latent": PAYLOAD_LATENT_WEIGHT,
            "payload_token": PAYLOAD_TOKEN_WEIGHT,
            "temperature": ADDRESS_TEMPERATURE,
        },
        "task": {"oracle_accuracy": validity, "chance_accuracy": chance},
        "training": training,
        "heldout_local_code": local,
        "full_stream_plus_memory": {k: v for k, v in full_eval.items() if k != "predictions"},
        "same_checkpoint_memory_disabled": {k: v for k, v in memory_off.items() if k != "predictions"},
        "separately_trained_stream_only": {k: v for k, v in stream_eval.items() if k != "predictions"},
        "full_minus_stream_only_accuracy": full_minus_stream,
        "same_checkpoint_memory_contribution_accuracy": memory_drop,
        "deployment_safety": safety,
        "checks": checks,
        "local_pass": local_pass,
        "end_to_end_pass": e2e_pass,
        "pass": local_pass and e2e_pass,
        "interpretation": interpretation,
        "claims": {
            "production_v22_authorized": False,
            "gpu_authorized": False,
            "architecture_freeze_authorized": False,
            "100m_authorized": False,
            "breakthrough_proven": False,
        },
    }


def main() -> None:
    result = run_diagnostic()
    print("AERA_V21_COLLAPSE_RESISTANT_RESULT_JSON=" + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
