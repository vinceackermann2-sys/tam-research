from __future__ import annotations

"""CPU-only capacity audit after AERA-v21 memory-code diagnostic #301.

This deterministically reproduces the already-failed #301 model only for
instrumentation, then asks whether the unchanged q/k and v/out projection spaces
have enough representational capacity under deliberately supervised upper-bound
probes.  The supervised probes are diagnostic-only and are never production
training objectives or independent evidence.
"""

import copy
import json
import math
from typing import Any

import torch
import torch.nn.functional as F

from aera_v19_memory_necessity_cpu import KEY_START, N_KEYS, N_VALUES, VALUE_START
from aera_v21_memory_code_objective_cpu import (
    _decode_with_frozen_model_head,
    train_pair_with_memory_code_objective,
)
from tam_research.aera_hardware_core_v21 import HardwareAwareAERATextLMV21

CAPACITY_STEPS = 1000
CAPACITY_LR = 1e-2
CAPACITY_TEMPERATURE = 0.10
CAPACITY_PASS = 0.95
CURRENT_SEPARABILITY_PASS = 0.95


def _token_representation(
    model: HardwareAwareAERATextLMV21,
    tokens: torch.Tensor,
    position: int,
) -> torch.Tensor:
    pos = model.local_pos(torch.tensor(position, device=tokens.device))
    x = model.token_emb(tokens) + pos[None, :]
    return model.stages[0].norm(x).detach()


@torch.no_grad()
def address_metrics(model: HardwareAwareAERATextLMV21) -> dict[str, Any]:
    keys = torch.arange(KEY_START, KEY_START + N_KEYS, device=model.token_emb.weight.device)
    x = _token_representation(model, keys, 1)
    memory = model.stages[0].memory
    q = F.normalize(memory.q(x), dim=-1)
    k = F.normalize(memory.k(x), dim=-1)
    similarity = q @ k.transpose(0, 1)
    target = torch.arange(N_KEYS, device=similarity.device)
    accuracy = float((similarity.argmax(dim=-1) == target).float().mean())
    diagonal = similarity.diag()
    mask = ~torch.eye(N_KEYS, dtype=torch.bool, device=similarity.device)
    off = similarity[mask]
    singular = torch.linalg.svdvals(q.float())
    if singular.numel() == 0 or float(singular[0]) == 0.0:
        effective_rank = 0
    else:
        effective_rank = int((singular >= singular[0] * 1e-2).sum())
    return {
        "q_to_k_top1_accuracy": accuracy,
        "diagonal_cosine_mean": float(diagonal.mean()),
        "off_diagonal_cosine_mean": float(off.mean()),
        "off_diagonal_cosine_max": float(off.max()),
        "q_effective_rank_1pct": effective_rank,
        "q_singular_values": [float(v) for v in singular],
    }


@torch.no_grad()
def payload_metrics(model: HardwareAwareAERATextLMV21) -> dict[str, float]:
    values = torch.arange(VALUE_START, VALUE_START + N_VALUES, device=model.token_emb.weight.device)
    x = _token_representation(model, values, 2)
    memory = model.stages[0].memory
    code = memory.out(torch.tanh(memory.v(x)))
    logits = _decode_with_frozen_model_head(model, code)
    legal = logits[:, VALUE_START : VALUE_START + N_VALUES]
    target = torch.arange(N_VALUES, device=legal.device)
    return {
        "legal_value_accuracy": float((legal.argmax(dim=-1) == target).float().mean()),
        "legal_value_nll": float(F.cross_entropy(legal.float(), target)),
    }


def _freeze_except(model: HardwareAwareAERATextLMV21, names: set[str]) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in names)


def supervised_address_capacity(
    source: HardwareAwareAERATextLMV21,
    *,
    steps: int = CAPACITY_STEPS,
) -> dict[str, Any]:
    model = copy.deepcopy(source)
    names = {"stages.0.memory.q.weight", "stages.0.memory.k.weight"}
    _freeze_except(model, names)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=CAPACITY_LR, weight_decay=0.0)
    keys = torch.arange(KEY_START, KEY_START + N_KEYS, device=model.token_emb.weight.device)
    x = _token_representation(model, keys, 1)
    target = torch.arange(N_KEYS, device=x.device)
    history: list[dict[str, float]] = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        memory = model.stages[0].memory
        q = F.normalize(memory.q(x), dim=-1)
        k = F.normalize(memory.k(x), dim=-1)
        logits = (q @ k.transpose(0, 1)) / CAPACITY_TEMPERATURE
        loss = F.cross_entropy(logits.float(), target)
        loss.backward()
        optimizer.step()
        if step in {0, steps // 2, steps - 1}:
            history.append({
                "step": float(step + 1),
                "loss": float(loss.detach()),
                "accuracy": float((logits.argmax(dim=-1) == target).float().mean().detach()),
            })
    metrics = address_metrics(model)
    return {"history": history, "final": metrics}


def supervised_payload_capacity(
    source: HardwareAwareAERATextLMV21,
    *,
    steps: int = CAPACITY_STEPS,
) -> dict[str, Any]:
    model = copy.deepcopy(source)
    names = {"stages.0.memory.v.weight", "stages.0.memory.out.weight"}
    _freeze_except(model, names)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=CAPACITY_LR, weight_decay=0.0)
    values = torch.arange(VALUE_START, VALUE_START + N_VALUES, device=model.token_emb.weight.device)
    x = _token_representation(model, values, 2)
    target = torch.arange(N_VALUES, device=x.device)
    history: list[dict[str, float]] = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        memory = model.stages[0].memory
        code = memory.out(torch.tanh(memory.v(x)))
        logits = _decode_with_frozen_model_head(model, code)
        legal = logits[:, VALUE_START : VALUE_START + N_VALUES]
        loss = F.cross_entropy(legal.float(), target)
        loss.backward()
        optimizer.step()
        if step in {0, steps // 2, steps - 1}:
            history.append({
                "step": float(step + 1),
                "loss": float(loss.detach()),
                "accuracy": float((legal.argmax(dim=-1) == target).float().mean().detach()),
            })
    metrics = payload_metrics(model)
    return {"history": history, "final": metrics}


def diagnose(
    current_address: dict[str, Any],
    current_payload: dict[str, float],
    address_capacity: dict[str, Any],
    payload_capacity: dict[str, Any],
) -> str:
    key_cap = address_capacity["final"]["q_to_k_top1_accuracy"] >= CAPACITY_PASS
    payload_cap = payload_capacity["final"]["legal_value_accuracy"] >= CAPACITY_PASS
    if not key_cap:
        return "q_k_projection_capacity_bottleneck"
    if not payload_cap:
        return "payload_projection_capacity_bottleneck"
    current_key_ok = current_address["q_to_k_top1_accuracy"] >= CURRENT_SEPARABILITY_PASS
    current_payload_ok = current_payload["legal_value_accuracy"] >= CAPACITY_PASS
    if not current_key_ok or not current_payload_ok:
        return "projection_capacity_sufficient_objective_design_bottleneck"
    return "capacity_and_current_code_high_reaudit_injection_task_semantics"


def run_audit() -> dict[str, Any]:
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    full, _stream, reproduction = train_pair_with_memory_code_objective()
    full.eval()
    current_address = address_metrics(full)
    current_payload = payload_metrics(full)
    address_capacity = supervised_address_capacity(full)
    payload_capacity = supervised_payload_capacity(full)
    diagnosis = diagnose(current_address, current_payload, address_capacity, payload_capacity)
    checks = {
        "supervised_key_capacity_ge_0_95": address_capacity["final"]["q_to_k_top1_accuracy"] >= CAPACITY_PASS,
        "supervised_payload_capacity_ge_0_95": payload_capacity["final"]["legal_value_accuracy"] >= CAPACITY_PASS,
    }
    return {
        "scope": "aera_v21_memory_address_collapse_payload_capacity_cpu",
        "diagnostic_reproduction_only": True,
        "independent_evidence": False,
        "reproduction_terminal_code": reproduction["final_memory_code"],
        "current_address": current_address,
        "current_payload": current_payload,
        "supervised_address_capacity": address_capacity,
        "supervised_payload_capacity": payload_capacity,
        "checks": checks,
        "diagnosis": diagnosis,
        "claims": {
            "gpu_authorized": False,
            "v22_authorized": False,
            "architecture_freeze_authorized": False,
            "100m_authorized": False,
            "breakthrough_proven": False,
        },
    }


def main() -> None:
    result = run_audit()
    print("AERA_V21_MEMORY_CAPACITY_AUDIT_RESULT_JSON=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
