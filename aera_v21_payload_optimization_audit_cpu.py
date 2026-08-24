from __future__ import annotations

"""CPU-only payload optimization audit after AERA-v21 issue #307.

This does not change AERA.  It diagnoses why the unchanged v/out memory payload
projections fail under the full objective even though issue #304 proved 100%
upper-bound payload capacity.
"""

import copy
import json
import math
from typing import Any

import torch
import torch.nn.functional as F

from aera_v19_memory_necessity_cpu import (
    BATCH_SIZE,
    EVAL_SEED,
    LEARNING_RATE,
    N_VALUES,
    SEED,
    TRAIN_STEPS,
    VALUE_START,
    _loss_and_accuracy,
    make_batch,
)
from aera_v21_collapse_resistant_memory_objective_cpu import (
    collapse_resistant_memory_terms,
    evaluate_local_memory_code,
    train_pair_with_collapse_resistant_objective,
)
from aera_v21_memory_capacity_audit_cpu import _token_representation
from aera_v21_memory_necessity_cpu import build_model
from tam_research.aera_hardware_core_v21 import HardwareAwareAERATextLMV21

ISOLATED_STEPS = TRAIN_STEPS
ISOLATED_LR = LEARNING_RATE
LOCAL_CAPACITY_PASS = 0.95


def _payload_parameters(model: HardwareAwareAERATextLMV21) -> list[torch.nn.Parameter]:
    memory = model.stages[0].memory
    return [memory.v.weight, memory.out.weight]


def _flatten_payload_grad(model: HardwareAwareAERATextLMV21) -> torch.Tensor:
    rows = []
    for parameter in _payload_parameters(model):
        if parameter.grad is None:
            rows.append(torch.zeros_like(parameter).reshape(-1))
        else:
            rows.append(parameter.grad.detach().float().reshape(-1).clone())
    return torch.cat(rows)


def _grad_norm(parameters) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(parameter.grad.detach().float().pow(2).sum())
    return math.sqrt(total)


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    if float(a.norm()) == 0.0 or float(b.norm()) == 0.0:
        return 0.0
    return float(F.cosine_similarity(a[None, :], b[None, :]).item())


def gradient_snapshot(
    model: HardwareAwareAERATextLMV21,
    batch: Any,
) -> dict[str, Any]:
    """Measure v/out gradient contributions without taking an optimizer step."""
    vectors: dict[str, torch.Tensor] = {}
    norms: dict[str, float] = {}

    for name in ("query", "latent", "token"):
        model.zero_grad(set_to_none=True)
        if name == "query":
            loss, _ = _loss_and_accuracy(
                model,
                batch,
                update_memory=True,
                differentiable_memory=True,
            )
        else:
            local = collapse_resistant_memory_terms(model, batch.tokens)
            loss = local["payload_latent_loss"] if name == "latent" else local["payload_token_loss"]
        loss.backward()
        vectors[name] = _flatten_payload_grad(model)
        norms[name] = float(vectors[name].norm())

    model.zero_grad(set_to_none=True)
    query_loss, _ = _loss_and_accuracy(
        model,
        batch,
        update_memory=True,
        differentiable_memory=True,
    )
    local = collapse_resistant_memory_terms(model, batch.tokens)
    total = (
        query_loss
        + local["address_contrastive_loss"]
        + local["payload_latent_loss"]
        + local["payload_token_loss"]
    )
    total.backward()
    combined_payload = _flatten_payload_grad(model)
    full_norm = _grad_norm(model.parameters())
    payload_norm = float(combined_payload.norm())
    clip_multiplier = min(1.0, 1.0 / max(full_norm, 1e-12))

    return {
        "payload_gradient_norms": norms,
        "gradient_cosines": {
            "latent_vs_token": _cosine(vectors["latent"], vectors["token"]),
            "query_vs_token": _cosine(vectors["query"], vectors["token"]),
            "query_vs_latent": _cosine(vectors["query"], vectors["latent"]),
        },
        "full_objective_gradient_norm_before_clip": full_norm,
        "implied_global_clip_multiplier": clip_multiplier,
        "payload_gradient_norm_before_clip": payload_norm,
        "payload_gradient_norm_after_implied_clip": payload_norm * clip_multiplier,
    }


@torch.no_grad()
def payload_local_metrics(model: HardwareAwareAERATextLMV21) -> dict[str, float]:
    device = model.token_emb.weight.device
    values = torch.arange(VALUE_START, VALUE_START + N_VALUES, device=device)
    x = _token_representation(model, values, 2)
    memory = model.stages[0].memory
    code = memory.out(torch.tanh(memory.v(x)))
    latent_cosine = float(F.cosine_similarity(code, x, dim=-1).mean())
    local = evaluate_local_memory_code(model)
    return {
        "legal_value_accuracy": local["payload_legal_value_accuracy"],
        "latent_cosine": latent_cosine,
    }


def _freeze_except_payload(model: HardwareAwareAERATextLMV21) -> None:
    allowed = {id(parameter) for parameter in _payload_parameters(model)}
    for parameter in model.parameters():
        parameter.requires_grad_(id(parameter) in allowed)


def train_isolated_payload(
    source: HardwareAwareAERATextLMV21,
    objective: str,
    *,
    steps: int = ISOLATED_STEPS,
) -> dict[str, Any]:
    if objective not in {"token", "latent", "token_plus_latent"}:
        raise ValueError(objective)
    model = copy.deepcopy(source)
    _freeze_except_payload(model)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=ISOLATED_LR,
        weight_decay=0.0,
    )
    history: list[dict[str, float]] = []
    for step in range(steps):
        batch = make_batch(BATCH_SIZE, SEED * 20000 + step)
        optimizer.zero_grad(set_to_none=True)
        local = collapse_resistant_memory_terms(model, batch.tokens)
        if objective == "token":
            loss = local["payload_token_loss"]
        elif objective == "latent":
            loss = local["payload_latent_loss"]
        else:
            loss = local["payload_token_loss"] + local["payload_latent_loss"]
        loss.backward()
        optimizer.step()
        if step in {0, steps // 2, steps - 1}:
            history.append({
                "step": float(step + 1),
                "loss": float(loss.detach()),
                "token_accuracy": float(local["payload_token_accuracy"].detach()),
                "latent_cosine": float(local["payload_latent_cosine"].detach()),
            })
    return {
        "objective": objective,
        "history": history,
        "final": payload_local_metrics(model),
    }


def diagnose(result: dict[str, Any]) -> str:
    isolated = result["isolated_payload_training"]
    token = isolated["token"]["final"]["legal_value_accuracy"]
    both = isolated["token_plus_latent"]["final"]["legal_value_accuracy"]
    failed_full = result["reproduced_full_payload_accuracy"]
    conflict = result["post_reproduction_gradients"]["gradient_cosines"]["latent_vs_token"]
    if token >= LOCAL_CAPACITY_PASS and both < LOCAL_CAPACITY_PASS and conflict < 0.0:
        return "latent_payload_target_conflicts_with_decoder_aligned_token_target"
    if token < LOCAL_CAPACITY_PASS:
        return "production_budget_payload_optimization_bottleneck"
    if token >= LOCAL_CAPACITY_PASS and both >= LOCAL_CAPACITY_PASS and failed_full < LOCAL_CAPACITY_PASS:
        return "full_model_gradient_competition_or_clipping_bottleneck"
    return "mixed_payload_failure_requires_reaudit"


def run_audit(*, reproduction_steps: int = TRAIN_STEPS, isolated_steps: int = ISOLATED_STEPS) -> dict[str, Any]:
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    initial = build_model(SEED)
    initial_batch = make_batch(BATCH_SIZE, EVAL_SEED + 601)
    initial_gradients = gradient_snapshot(initial, initial_batch)

    reproduced, _stream, reproduction = train_pair_with_collapse_resistant_objective(
        steps=reproduction_steps
    )
    reproduced.eval()
    reproduced_payload = evaluate_local_memory_code(reproduced)["payload_legal_value_accuracy"]
    post_batch = make_batch(BATCH_SIZE, EVAL_SEED + 602)
    post_gradients = gradient_snapshot(reproduced, post_batch)

    # All isolated probes start from the same original initialization so the
    # only variable is the local payload objective.
    isolated_source = build_model(SEED)
    isolated = {
        name: train_isolated_payload(isolated_source, name, steps=isolated_steps)
        for name in ("token", "latent", "token_plus_latent")
    }
    result = {
        "scope": "aera_v21_payload_objective_conflict_gradient_scale_cpu",
        "architecture_changed": False,
        "independent_evidence": False,
        "learning_rate": ISOLATED_LR,
        "reproduction_steps": reproduction_steps,
        "isolated_steps": isolated_steps,
        "initial_gradients": initial_gradients,
        "post_reproduction_gradients": post_gradients,
        "reproduced_full_payload_accuracy": reproduced_payload,
        "reproduction_final_local_code": reproduction["final_local_code"],
        "isolated_payload_training": isolated,
        "prior_capacity_upper_bound": {
            "issue": 304,
            "lr": 1e-2,
            "steps": 1000,
            "payload_accuracy": 1.0,
        },
        "claims": {
            "gpu_authorized": False,
            "production_v22_authorized": False,
            "architecture_freeze_authorized": False,
            "100m_authorized": False,
            "breakthrough_proven": False,
        },
    }
    result["diagnosis"] = diagnose(result)
    return result


def main() -> None:
    result = run_audit()
    print("AERA_V21_PAYLOAD_OPTIMIZATION_AUDIT_RESULT_JSON=" + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
