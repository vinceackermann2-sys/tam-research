from __future__ import annotations

"""CPU-only self-supervised memory-code diagnostic for unchanged AERA-v21.

Issue #297 showed that v21 writes/reads are physically active but the recalled
memory-space payload does not contain recoverable value information.  This
script does NOT change the architecture.  It asks whether the existing q/k/v/out
memory code can become useful when taught with general sequence-derived local
objectives instead of relying only on delayed query CE.

The auxiliary never receives synthetic key identities, delayed query labels, or
overwrite labels.  Event representations are detached so the auxiliary cannot
rewrite the backbone.  The decoder weights are detached so payload reconstruction
must improve the existing memory v/out code rather than create a decoder shortcut.
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
from aera_v21_memory_necessity_cpu import build_model
from tam_research.aera_hardware_core_v21 import HardwareAwareAERATextLMV21

QK_ALIGNMENT_WEIGHT = 1.0
PAYLOAD_RECONSTRUCTION_WEIGHT = 1.0
QK_IMPROVEMENT_MIN = 1e-3
PAYLOAD_LEGAL_ACCURACY_MIN = 0.90


def _detached_stage0_event_pairs(
    model: HardwareAwareAERATextLMV21,
    tokens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return exact stage-0 v21 event-pair sources, detached from the backbone."""
    if tokens.ndim != 2 or tokens.size(1) % CHUNK_SIZE:
        raise ValueError("tokens must be [batch,sequence] divisible by CHUNK_SIZE")
    chunks = tokens.view(tokens.size(0), -1, CHUNK_SIZE)
    pos = torch.arange(CHUNK_SIZE, device=tokens.device)
    events = model.token_emb(chunks) + model.local_pos(pos)[None, None, :, :]
    base_h = model.stages[0].norm(events).detach()
    return base_h[:, :, :-1], base_h[:, :, 1:], chunks[:, :, 1:]


def _decode_with_frozen_model_head(
    model: HardwareAwareAERATextLMV21,
    x: torch.Tensor,
) -> torch.Tensor:
    """Decode x while allowing gradients only through x, not norm/lm-head weights."""
    weight = model.norm.weight.detach() if model.norm.weight is not None else None
    bias = model.norm.bias.detach() if model.norm.bias is not None else None
    normalized = F.layer_norm(
        x,
        model.norm.normalized_shape,
        weight,
        bias,
        model.norm.eps,
    )
    head_weight = model.lm_head.weight.detach()
    head_bias = model.lm_head.bias.detach() if model.lm_head.bias is not None else None
    return F.linear(normalized, head_weight, head_bias)


def memory_code_terms(
    model: HardwareAwareAERATextLMV21,
    tokens: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """General local memory-code supervision derived only from observed sequence events."""
    address_source, payload_source, next_tokens = _detached_stage0_event_pairs(model, tokens)
    memory = model.stages[0].memory

    q = F.normalize(memory.q(address_source), dim=-1)
    k = F.normalize(memory.k(address_source), dim=-1)
    qk_cosine = (q * k).sum(dim=-1)
    qk_alignment_loss = (1.0 - qk_cosine).mean()

    payload_code = memory.out(torch.tanh(memory.v(payload_source)))
    payload_logits = _decode_with_frozen_model_head(model, payload_code)
    payload_reconstruction_loss = F.cross_entropy(
        payload_logits.float().reshape(-1, payload_logits.size(-1)),
        next_tokens.reshape(-1),
    )
    payload_accuracy = (
        payload_logits.argmax(dim=-1) == next_tokens
    ).float().mean()

    return {
        "qk_alignment_loss": qk_alignment_loss,
        "qk_cosine": qk_cosine.mean(),
        "payload_reconstruction_loss": payload_reconstruction_loss,
        "payload_reconstruction_accuracy": payload_accuracy,
    }


@torch.no_grad()
def evaluate_memory_code(
    model: HardwareAwareAERATextLMV21,
    batch: Any,
) -> dict[str, float]:
    """Held-out local diagnostics on actual WRITE key->value pairs."""
    model.eval()
    chunks = batch.tokens.view(batch.tokens.size(0), -1, CHUNK_SIZE)
    pos = torch.arange(CHUNK_SIZE, device=batch.tokens.device)
    events = model.token_emb(chunks) + model.local_pos(pos)[None, None, :, :]
    base_h = model.stages[0].norm(events)
    write_mask = chunks[:, :, 0].eq(1)
    write_h = base_h[write_mask]
    write_tokens = chunks[write_mask]
    if write_h.numel() == 0:
        raise RuntimeError("held-out diagnostic has no WRITE chunks")

    address = write_h[:, 1]
    payload = write_h[:, 2]
    value_target = write_tokens[:, 2]
    memory = model.stages[0].memory

    q = F.normalize(memory.q(address), dim=-1)
    k = F.normalize(memory.k(address), dim=-1)
    qk_cosine = float((q * k).sum(dim=-1).mean())

    payload_code = memory.out(torch.tanh(memory.v(payload)))
    logits = _decode_with_frozen_model_head(model, payload_code)
    legal = logits[:, VALUE_START : VALUE_START + N_VALUES]
    prediction = legal.argmax(dim=-1) + VALUE_START
    legal_accuracy = float((prediction == value_target).float().mean())
    full_accuracy = float((logits.argmax(dim=-1) == value_target).float().mean())
    return {
        "qk_cosine": qk_cosine,
        "payload_legal_value_accuracy": legal_accuracy,
        "payload_full_vocab_accuracy": full_accuracy,
    }


def train_pair_with_memory_code_objective(
    *, steps: int = TRAIN_STEPS,
) -> tuple[HardwareAwareAERATextLMV21, HardwareAwareAERATextLMV21, dict[str, Any]]:
    full = build_model(SEED)
    stream_only = build_model(SEED)
    for key, value in full.state_dict().items():
        torch.testing.assert_close(stream_only.state_dict()[key], value, atol=0.0, rtol=0.0)

    eval_probe = make_batch(24, EVAL_SEED + 301)
    initial_code = evaluate_memory_code(full, eval_probe)

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
        local = memory_code_terms(full, batch.tokens)
        total = (
            query_loss
            + QK_ALIGNMENT_WEIGHT * local["qk_alignment_loss"]
            + PAYLOAD_RECONSTRUCTION_WEIGHT * local["payload_reconstruction_loss"]
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
                "qk_alignment_loss": float(local["qk_alignment_loss"].detach()),
                "qk_cosine": float(local["qk_cosine"].detach()),
                "payload_reconstruction_loss": float(local["payload_reconstruction_loss"].detach()),
                "payload_reconstruction_accuracy": float(local["payload_reconstruction_accuracy"].detach()),
            }
            history.append(row)
            print("AERA_V21_MEMORY_CODE_TRAIN=" + json.dumps(row, sort_keys=True), flush=True)

    final_code = evaluate_memory_code(full, eval_probe)
    return full, stream_only, {
        "history": history,
        "initial_memory_code": initial_code,
        "final_memory_code": final_code,
    }


def run_diagnostic(*, steps: int = TRAIN_STEPS) -> dict[str, Any]:
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    eval_batch = make_batch(24, EVAL_SEED)
    validity = oracle_accuracy(eval_batch)
    if validity < TASK_VALIDITY_MIN:
        raise RuntimeError(f"task validity failed before training: {validity}")

    full, stream_only, training = train_pair_with_memory_code_objective(steps=steps)
    full_eval = _evaluate(full, eval_batch, memory=True)
    memory_off = _evaluate(full, eval_batch, memory=False)
    stream_eval = _evaluate(stream_only, eval_batch, memory=False)
    safety = deployment_safety_and_isolation(full, eval_batch)
    final_code = evaluate_memory_code(full, make_batch(24, EVAL_SEED + 302))

    chance = 1.0 / N_VALUES
    full_minus_stream = full_eval["query_accuracy"] - stream_eval["query_accuracy"]
    memory_drop = full_eval["query_accuracy"] - memory_off["query_accuracy"]
    initial_qk = training["initial_memory_code"]["qk_cosine"]
    qk_gain = final_code["qk_cosine"] - initial_qk

    checks = {
        "task_validity_ge_0_95": validity >= TASK_VALIDITY_MIN,
        "full_query_accuracy_ge_0_80": full_eval["query_accuracy"] >= FULL_ACCURACY_MIN,
        "full_over_stream_only_ge_0_15": full_minus_stream >= FULL_OVER_STREAM_MIN,
        "same_checkpoint_memory_drop_ge_0_15": memory_drop >= SAME_CHECKPOINT_MEMORY_DROP_MIN,
        "overwrite_current_value_accuracy_ge_0_80": full_eval["overwrite_current_value_accuracy"] >= OVERWRITE_ACCURACY_MIN,
        "stale_value_error_le_0_10": full_eval["stale_value_error_rate"] <= STALE_ERROR_MAX,
        "fresh_session_near_chance": safety["fresh_session_query_accuracy"] <= chance + FRESH_SESSION_CHANCE_TOLERANCE,
        "deployment_base_parameters_unchanged": bool(safety["deployment_base_parameters_unchanged"]),
        "session_isolation_exact": bool(safety["session_isolation_exact"]),
        "qk_alignment_improved": qk_gain >= QK_IMPROVEMENT_MIN,
        "payload_legal_value_accuracy_ge_0_90": final_code["payload_legal_value_accuracy"] >= PAYLOAD_LEGAL_ACCURACY_MIN,
    }
    local_code_pass = (
        checks["qk_alignment_improved"]
        and checks["payload_legal_value_accuracy_ge_0_90"]
    )
    end_to_end_pass = all(
        value for key, value in checks.items()
        if key not in {"qk_alignment_improved", "payload_legal_value_accuracy_ge_0_90"}
    )
    if local_code_pass and end_to_end_pass:
        interpretation = "training_objective_bottleneck"
    elif local_code_pass:
        interpretation = "local_code_learned_but_end_to_end_memory_still_insufficient"
    else:
        interpretation = "representation_projection_inadequate_even_with_direct_teaching"

    return {
        "scope": "aera_v21_self_supervised_memory_code_objective_cpu",
        "architecture_changed": False,
        "independent_evidence": False,
        "steps": steps,
        "weights": {
            "qk_alignment": QK_ALIGNMENT_WEIGHT,
            "payload_reconstruction": PAYLOAD_RECONSTRUCTION_WEIGHT,
        },
        "task": {"oracle_accuracy": validity, "chance_accuracy": chance},
        "training": training,
        "heldout_memory_code": final_code,
        "qk_cosine_gain": qk_gain,
        "full_stream_plus_memory": {k: v for k, v in full_eval.items() if k != "predictions"},
        "same_checkpoint_memory_disabled": {k: v for k, v in memory_off.items() if k != "predictions"},
        "separately_trained_stream_only": {k: v for k, v in stream_eval.items() if k != "predictions"},
        "full_minus_stream_only_accuracy": full_minus_stream,
        "same_checkpoint_memory_contribution_accuracy": memory_drop,
        "deployment_safety": safety,
        "checks": checks,
        "local_code_pass": local_code_pass,
        "end_to_end_pass": end_to_end_pass,
        "pass": local_code_pass and end_to_end_pass,
        "interpretation": interpretation,
        "claims": {
            "v22_authorized": False,
            "gpu_authorized": False,
            "architecture_freeze_authorized": False,
            "100m_authorized": False,
            "breakthrough_proven": False,
        },
    }


def main() -> None:
    result = run_diagnostic()
    print("AERA_V21_MEMORY_CODE_RESULT_JSON=" + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
