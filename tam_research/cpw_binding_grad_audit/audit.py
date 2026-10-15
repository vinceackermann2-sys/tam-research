"""CPU-only RANDOM_INIT_GRADIENTS_ONLY audit for CPW #1335.

Never invoke an optimizer, Modal, CUDA, scientific seeds, checkpoints or metadata
as model inputs. This measures differentiable paths, NOT learned memory.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from tam_research.cpw_binding_science.harness import ARMS, build_arm, query_logits
from tam_research.cpw_binding_v2.task import (
    QUERY_POSITION,
    SCORED_DELAYS,
    VALUE_DELAYS,
    explicit_key_lookup,
    make_binding_batch,
)

ISSUE = 1335
AUDIT_SEED = 1_335_001
CLASSIFICATION = "RANDOM_INIT_GRADIENTS_ONLY"


def _norm(tensor: torch.Tensor | None) -> float:
    if tensor is None:
        return 0.0
    value = float(tensor.detach().float().norm().cpu())
    if not math.isfinite(value) or value < 0:
        raise AssertionError("nonfinite or negative gradient norm")
    return value


def one_arm(
    arm: str,
    *,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    delay: int,
    init_seed: int = AUDIT_SEED,
) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError(f"unknown frozen arm {arm!r}")
    if tokens.device.type != "cpu" or targets.device.type != "cpu":
        raise PermissionError("CPW #1335 diagnostic runs on CPU only")
    if tokens.ndim != 2 or targets.ndim != 1 or tokens.size(0) != targets.size(0):
        raise ValueError("invalid diagnostic input dimensions")
    if delay not in SCORED_DELAYS or tokens.size(1) != QUERY_POSITION + 1:
        raise ValueError("invalid frozen task shape or delay")

    torch.manual_seed(init_seed)
    model = build_arm(arm).cpu().train()
    activations: list[torch.Tensor] = []

    def _retain(_module, _inputs, output: torch.Tensor) -> None:
        output.retain_grad()
        activations.append(output)

    handle = model.token_emb.register_forward_hook(_retain)
    try:
        logits = query_logits(model, tokens)
        loss = F.cross_entropy(logits.float(), targets)
        if not bool(torch.isfinite(loss).item()):
            raise AssertionError("nonfinite loss")
        loss.backward()
    finally:
        handle.remove()

    if len(activations) != 1 or activations[0].grad is None:
        raise AssertionError("token embedding activation gradient unavailable")
    g = activations[0].grad.detach().float()
    assert g.shape[:2] == tokens.shape

    selected_value = QUERY_POSITION - delay
    selected_key = selected_value - 1
    nonselected_delay = next(d for d in VALUE_DELAYS if d != delay)
    distractor_value = QUERY_POSITION - nonselected_delay
    distractor_key = distractor_value - 1

    afm: dict[str, float] = {}
    if arm.startswith("afm_"):
        layer = model.afm_layers[0]
        module = model.blocks[layer].afm
        if module is None:
            raise AssertionError("AFM module missing")
        for field in ("key_proj", "value_proj", "out_proj", "gate_prev", "gate_cur"):
            afm[field + "_grad_l2"] = _norm(getattr(module, field).weight.grad)
        for name, value in model.memory_stats().items():
            afm["stat_" + name] = float(value)
    result = {
        "arm": arm,
        "classification": CLASSIFICATION,
        "training_performed": False,
        "gpu_requested": False,
        "issue": ISSUE,
        "delay": delay,
        "initial_query_nll": float(loss.detach().cpu()),
        "source_key_position": selected_key,
        "source_value_position": selected_value,
        "source_key_activation_grad_l2": _norm(g[:, selected_key]),
        "source_value_activation_grad_l2": _norm(g[:, selected_value]),
        "distractor_key_activation_grad_l2": _norm(g[:, distractor_key]),
        "distractor_value_activation_grad_l2": _norm(g[:, distractor_value]),
        "query_key_activation_grad_l2": _norm(g[:, QUERY_POSITION]),
        "far_prefix_activation_grad_l2": _norm(g[:, : QUERY_POSITION - 15]),
        "final_15_activation_grad_l2": _norm(g[:, QUERY_POSITION - 15 :]),
        "afm": afm,
    }
    if any(not math.isfinite(v) for v in [
        result[k] for k in result if k.endswith("_l2") or k == "initial_query_nll"
    ]):
        raise AssertionError("nonfinite diagnostic")
    return result


def audit(
    *,
    arms: Sequence[str] = ARMS,
    seed: int = AUDIT_SEED,
    delay: int = 256,
    batch_size: int = 1,
) -> dict[str, Any]:
    if batch_size < 1 or batch_size > 4:
        raise ValueError("CPU-only audit batch_size must be 1..4")
    if len(set(arms)) != len(arms) or not set(arms).issubset(ARMS):
        raise ValueError("arms must be unique frozen research arms")

    torch.set_num_threads(min(torch.get_num_threads(), 2))
    generator = torch.Generator(device="cpu").manual_seed(seed + 10_000)
    batch = make_binding_batch(
        batch_size=batch_size,
        generator=generator,
        delay=delay,
    )
    if not torch.equal(explicit_key_lookup(batch.tokens), batch.targets):
        raise AssertionError("corrected binding oracle mismatch")

    source_hash = hashlib.sha256(
        batch.tokens.numpy().tobytes() + batch.targets.numpy().tobytes()
    ).hexdigest()
    results = [
        one_arm(
            arm,
            tokens=batch.tokens,
            targets=batch.targets,
            delay=delay,
            init_seed=seed,
        )
        for arm in arms
    ]
    return {
        "classification": CLASSIFICATION,
        "issue": ISSUE,
        "audit_seed_not_scientific": seed,
        "source_sha256": source_hash,
        "distance": delay,
        "batch_size": batch_size,
        "frozen_arm_order": list(arms),
        "oracle_accuracy": 1.0,
        "training_performed": False,
        "gpu_requested": False,
        "scientific_seeds_consumed": False,
        "scientific_support_claimed": False,
        "results": results,
    }


if __name__ == "__main__":
    print("CPW_BINDING_GRADIENT_AUDIT=" + json.dumps(audit(), sort_keys=True))
