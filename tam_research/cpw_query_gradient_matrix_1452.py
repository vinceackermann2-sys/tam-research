"""CPW #1452: CPU-only four-query RANDOM-INIT gradient-routing matrices.

This is a source-bound engineering diagnostic, NOT trained recall, science,
a language-model comparison, a H100 attempt or a breakthrough claim.
"""
from __future__ import annotations

import hashlib
import json
import math
from statistics import median
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from tam_research.cpw_binding_science.harness import (
    ARMS, PARAMETERS, build_arm, query_logits,
)
from tam_research.cpw_binding_v2.task import (
    QUERY_POSITION, SCORED_DELAYS, VALUE_DELAYS,
    counterfactual_query, explicit_key_lookup, make_binding_batch,
)

CLASSIFICATION = "CPW_1452_RANDOM_INIT_QUERY_GRADIENT_ROUTING_ONLY"
ISSUE = 1452
ENGINEERING_SEED = 1_452_001
GPU_AUTHORIZED = False
NEW_SCIENTIFIC_ATTEMPT = False
FROZEN_DEPENDENCY_BLOBS = {
    "task": "79dc0ceda617985eba40f119343bdb2992aa395d",
    "harness": "63ab35fc82ea06d91ce2a9b5cbd1d2d9c56baf3f",
    "prior_single_query_audit": "49c6154e48fce7edace97ad630783153d39a6b0d",
}


def _counterfactual_source(seed: int) -> tuple[list[tuple[int, torch.Tensor, torch.Tensor]], str]:
    generator = torch.Generator(device="cpu").manual_seed(seed + 10_000)
    batch = make_binding_batch(batch_size=1, generator=generator, delay=32)
    rows: list[tuple[int, torch.Tensor, torch.Tensor]] = []
    prefix = batch.tokens[:, :QUERY_POSITION].clone()
    answers: set[int] = set()
    queries: set[int] = set()
    for delay in SCORED_DELAYS:
        tokens, targets = counterfactual_query(batch, new_delay=delay)
        if tokens.device.type != "cpu" or targets.device.type != "cpu":
            raise PermissionError("CPW #1452 CPU-only source policy")
        if not torch.equal(prefix, tokens[:, :QUERY_POSITION]):
            raise AssertionError("counterfactual source prefix changed")
        if not torch.equal(explicit_key_lookup(tokens), targets):
            raise AssertionError("explicit key lookup failed")
        queries.add(int(tokens[0, QUERY_POSITION]))
        answers.add(int(targets[0]))
        rows.append((delay, tokens, targets))
    if len(answers) != 4 or len(queries) != 4:
        raise AssertionError("four-query binding does not change all four targets")
    blind = int(rows[0][2][0])
    if sum(int(int(target[0]) == blind) for _, _, target in rows) != 1:
        raise AssertionError("query-blind fixed answer not exactly 1/4")
    digest = hashlib.sha256(prefix.numpy().tobytes()).hexdigest()
    return rows, digest


def _finite_norm(x: torch.Tensor) -> float:
    result = float(x.detach().float().norm().cpu())
    if not math.isfinite(result) or result < 0:
        raise AssertionError("invalid activation gradient norm")
    return result


def _row_metrics(g: torch.Tensor, query_delay: int) -> dict[str, Any]:
    if g.ndim != 3 or g.shape[0] != 1 or g.shape[1] != QUERY_POSITION + 1:
        raise ValueError("unexpected frozen token embedding activation shape")
    key_positions = [QUERY_POSITION - d - 1 for d in VALUE_DELAYS]
    value_positions = [QUERY_POSITION - d for d in VALUE_DELAYS]
    keys = [_finite_norm(g[:, i]) for i in key_positions]
    values = [_finite_norm(g[:, i]) for i in value_positions]
    selected = VALUE_DELAYS.index(query_delay)

    def metrics(magnitudes: list[float]) -> dict[str, Any]:
        total = sum(magnitudes)
        distractors = [x for idx, x in enumerate(magnitudes) if idx != selected]
        concentration = magnitudes[selected] / total if total > 0 else None
        return {
            "target_grad_l2": magnitudes[selected],
            "distractor_sum_grad_l2": sum(distractors),
            "distractor_median_grad_l2": median(distractors),
            "distractor_max_grad_l2": max(distractors),
            "target_fraction_of_all_eight": concentration,
            "normalized_eight_positions": (
                [x / total for x in magnitudes] if total > 0 else None
            ),
            "zero_norm": total == 0,
        }
    return {
        "query_delay": query_delay,
        "source_key_positions": key_positions,
        "source_value_positions": value_positions,
        "key_gradient_l2": keys,
        "value_gradient_l2": values,
        "selected_source_key_position": key_positions[selected],
        "selected_source_value_position": value_positions[selected],
        "key_concentration": metrics(keys),
        "value_concentration": metrics(values),
        "last_15_source_gradient_l2": _finite_norm(
            g[:, QUERY_POSITION - 15 : QUERY_POSITION]
        ),
        "query_token_gradient_l2": _finite_norm(g[:, QUERY_POSITION]),
    }


def _routing_change(rows: list[dict[str, Any]], field: str) -> dict[str, float | None]:
    pairs: list[float] = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            left = rows[i][field]["normalized_eight_positions"]
            right = rows[j][field]["normalized_eight_positions"]
            if left is not None and right is not None:
                pairs.append(sum(abs(a - b) for a, b in zip(left, right)) / 2.0)
    return {
        "mean_pairwise_normalized_total_variation": (
            sum(pairs) / len(pairs) if pairs else None
        ),
        "comparable_query_pairs": len(pairs),
    }


def one_arm(arm: str, *, seed: int = ENGINEERING_SEED) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError("only five frozen CPW research arms permitted")
    rows, fingerprint = _counterfactual_source(seed)
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    model = build_arm(arm).cpu().train()
    params = sum(p.numel() for p in model.parameters())
    if params != PARAMETERS[arm]:
        raise AssertionError("frozen arm parameter count changed")
    # Build ONCE and never modify parameter values between four queries.
    initial_param_sum = [p.detach().float().sum().item() for p in model.parameters()]
    gradients: list[torch.Tensor] = []

    def hook(_module, _inputs, output: torch.Tensor) -> None:
        output.retain_grad()
        gradients.append(output)

    handle = model.token_emb.register_forward_hook(hook)
    output_rows: list[dict[str, Any]] = []
    try:
        for delay, tokens, targets in rows:
            if tokens.device.type != "cpu" or targets.device.type != "cpu":
                raise PermissionError("CPW #1452 forbids GPU tensors")
            model.zero_grad(set_to_none=True)
            gradients.clear()
            logits = query_logits(model, tokens)
            loss = F.cross_entropy(logits.float(), targets)
            if not bool(torch.isfinite(loss).item()):
                raise AssertionError("nonfinite query-only CE")
            loss.backward()
            if len(gradients) != 1 or gradients[0].grad is None:
                raise AssertionError("embedding activation gradient missing")
            g = gradients[0].grad.detach().float()
            measurement = _row_metrics(g, delay)
            measurement["loss"] = float(loss.detach().cpu())
            measurement["query_token"] = int(tokens[0, QUERY_POSITION])
            measurement["target_token"] = int(targets[0])
            output_rows.append(measurement)
            gradients.clear()
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)

    for expected, parameter in zip(initial_param_sum, model.parameters()):
        if expected != parameter.detach().float().sum().item():
            raise AssertionError("parameters updated during random-initialization audit")
    if len(output_rows) != 4:
        raise AssertionError("missing counterfactual gradient row")
    if arm == "sequence_only":
        for row in output_rows:
            if any(abs(x) > 1e-12 for x in (
                *row["key_gradient_l2"], *row["value_gradient_l2"]
            )):
                raise AssertionError("local sequence model leaked far-source path")
    return {
        "arm": arm,
        "parameter_count": params,
        "source_prefix_sha256": fingerprint,
        "query_count": 4,
        "source_gradients_key_l2_4x8": [r["key_gradient_l2"] for r in output_rows],
        "source_gradients_value_l2_4x8": [r["value_gradient_l2"] for r in output_rows],
        "key_routing_change": _routing_change(output_rows, "key_concentration"),
        "value_routing_change": _routing_change(output_rows, "value_concentration"),
        "rows": output_rows,
        "training_performed": False,
        "gpu_requested": False,
        "science_seed_consumed": False,
    }


def audit(
    *, arms: Sequence[str] = ARMS, seed: int = ENGINEERING_SEED
) -> dict[str, Any]:
    if not arms or len(set(arms)) != len(arms) or not set(arms).issubset(ARMS):
        raise ValueError("arms must be nonempty unique frozen CPW arms")
    if type(seed) is not int or seed != ENGINEERING_SEED:
        raise ValueError("only preregistered CPW engineering seed allowed")
    results = [one_arm(arm, seed=seed) for arm in arms]
    if len({r["source_prefix_sha256"] for r in results}) != 1:
        raise AssertionError("five arms evaluated different source prefixes")
    return {
        "classification": CLASSIFICATION, "issue": ISSUE,
        "engineering_seed_not_scientific": seed,
        "source_dependency_blobs": FROZEN_DEPENDENCY_BLOBS,
        "four_query_delays": list(SCORED_DELAYS),
        "eight_source_pair_delays": list(VALUE_DELAYS),
        "source_prefix_sha256": results[0]["source_prefix_sha256"],
        "key_blind_accuracy": 0.25,
        "training_performed": False, "optimizer_steps": 0,
        "gpu_allocated": False, "scientific_attempt": False,
        "breakthrough_claim": False,
        "results": results,
        "interpretation": (
            "Four-query RANDOM-initialization source gradient ROUTING only; "
            "no learned binding, heldout score, Transformer win, or breakthrough."
        ),
    }


if __name__ == "__main__":
    print("CPW_1452_FOUR_QUERY_GRADIENT_MATRIX=" +
          json.dumps(audit(), sort_keys=True, allow_nan=False))
