"""CPW #1470 v2: CPU-only query/label factorial RANDOM-INIT gradient-routing matrices.

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

CLASSIFICATION = "CPW_1470_V2_RANDOM_INIT_FACTORIAL_QUERY_TARGET_ONLY"
ISSUE = 1470
ENGINEERING_SEED = 1_453_001
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
            raise PermissionError("CPW #1470 v2 CPU-only source policy")
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



def _state_sha(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if value.device.type != "cpu":
            raise PermissionError("CPU model state required")
        x = value.detach().contiguous()
        h.update(name.encode() + b"\0" + str(x.dtype).encode() + b"\0")
        h.update(str(tuple(x.shape)).encode() + b"\0" + x.numpy().tobytes())
    return h.hexdigest()


def _tv(a: list[float] | None, b: list[float] | None) -> float | None:
    if a is None or b is None:
        return None
    return sum(abs(x - y) for x, y in zip(a, b)) / 2


def _stats(grid: list[list[dict[str, Any]]], field: str) -> dict[str, Any]:
    q_delta, t_delta, masses, qdiag, qoff, tdiag, toff = ([] for _ in range(7))
    zero = 0
    def n(i: int, j: int) -> list[float] | None:
        return grid[i][j][field + "_concentration"]["normalized_eight_positions"]
    for i in range(4):
        for j in range(4):
            g = grid[i][j][field + "_gradient_l2"]
            masses.append(sum(g))
            zero += n(i, j) is None
            if n(i, j) is not None:
                q_share = n(i, j)[VALUE_DELAYS.index(SCORED_DELAYS[i])]
                t_share = n(i, j)[VALUE_DELAYS.index(SCORED_DELAYS[j])]
                (qdiag if i == j else qoff).append(q_share)
                (tdiag if i == j else toff).append(t_share)
    for i in range(4):
        for k in range(i + 1, 4):
            for j in range(4):
                v = _tv(n(i, j), n(k, j))
                if v is not None: q_delta.append(v)
    for i in range(4):
        for j in range(4):
            for k in range(j + 1, 4):
                v = _tv(n(i, j), n(i, k))
                if v is not None: t_delta.append(v)
    def avg(v: list[float]) -> float | None:
        return sum(v) / len(v) if v else None
    return {
        "query_fixed_label_mean_tv": avg(q_delta), "query_pair_count": len(q_delta),
        "label_fixed_query_mean_tv": avg(t_delta), "label_pair_count": len(t_delta),
        "query_slot_diag": avg(qdiag), "query_slot_offdiag": avg(qoff),
        "label_slot_diag": avg(tdiag), "label_slot_offdiag": avg(toff),
        "mean_unnormalized_mass": avg(masses), "zero_norm_cells": zero,
    }


def _signed_contrasts(signed: list[list[torch.Tensor]]) -> dict[str, Any]:
    query, label, interaction, split = [], [], [], []
    for i in range(4):
        for j in range(i + 1, 4):
            diag = signed[i][i] - signed[j][j]
            pieces = (signed[i][i] - signed[j][i]) + (signed[j][i] - signed[j][j])
            split.append(_finite_norm(diag - pieces))
            for t in range(4):
                query.append(_finite_norm(signed[i][t] - signed[j][t]))
                label.append(_finite_norm(signed[t][i] - signed[t][j]))
            for a in range(4):
                for b in range(a + 1, 4):
                    interaction.append(_finite_norm(
                        signed[i][a] - signed[j][a] - signed[i][b] + signed[j][b]))
    return {
        "query_fixed_label_signed_mean_l2": sum(query) / len(query),
        "label_fixed_query_signed_mean_l2": sum(label) / len(label),
        "query_label_interaction_signed_mean_l2": sum(interaction) / len(interaction),
        "interaction_count": len(interaction),
        "max_diagonal_telescoping_residual": max(split),
    }


def one_arm(arm: str, *, seed: int = ENGINEERING_SEED) -> dict[str, Any]:
    if arm not in ARMS or type(seed) is not int or seed != ENGINEERING_SEED:
        raise ValueError("only preregistered CPW arm and seed allowed")
    torch.set_num_threads(1)
    sources, digest = _counterfactual_source(seed)
    if [r[0] for r in sources] != list(SCORED_DELAYS):
        raise AssertionError("scored query delays changed")
    queries = [int(tokens[0, QUERY_POSITION]) for _, tokens, _ in sources]
    targets = [int(target[0]) for _, _, target in sources]
    if len(set(queries)) != 4 or len(set(targets)) != 4:
        raise AssertionError("nonunique factorial query/label IDs")
    if any(tokens.device.type != "cpu" for _, tokens, _ in sources):
        raise PermissionError("no GPU")
    torch.manual_seed(seed)
    model = build_arm(arm).cpu().eval()
    params = sum(p.numel() for p in model.parameters())
    if params != PARAMETERS[arm]:
        raise AssertionError("frozen arm parameter drift")
    state_sha = _state_sha(model)
    activations: list[torch.Tensor] = []
    def capture(_module, _inputs, output: torch.Tensor) -> None:
        activations.append(output)
    hook = model.token_emb.register_forward_hook(capture)
    key_pos = [QUERY_POSITION - d - 1 for d in VALUE_DELAYS]
    value_pos = [QUERY_POSITION - d for d in VALUE_DELAYS]
    observed: list[list[dict[str, Any]]] = []
    signed_keys: list[list[torch.Tensor]] = []
    signed_values: list[list[torch.Tensor]] = []
    derivative_residuals: list[float] = []
    try:
        for i, (delay, tokens, _) in enumerate(sources):
            activations.clear()
            logits = query_logits(model, tokens)
            if len(activations) != 1 or not torch.isfinite(logits).all():
                raise AssertionError("invalid differentiable query forward")
            act = activations[0]
            if act.device.type != "cpu" or not act.requires_grad:
                raise AssertionError("CPU differentiable embedding required")
            row, vec = [], []
            for j, label_id in enumerate(targets):
                y = torch.tensor([label_id], dtype=torch.long)
                loss = F.cross_entropy(logits.float(), y)
                if not torch.isfinite(loss).item():
                    raise AssertionError("nonfinite CE")
                g, = torch.autograd.grad(loss, act, retain_graph=True)
                g = g.detach().float()
                m = _row_metrics(g, delay)
                m.update({
                    "query_index": i, "label_index": j,
                    "query_token": queries[i], "loss_label": label_id,
                    "correct_binding_label": i == j,
                    "loss": float(loss.detach().item()),
                })
                row.append(m)
                vec.append(g[0, key_pos + value_pos].clone())
            # Exact CE-label effect equals a logit-difference derivative at fixed q.
            oracle_grad, = torch.autograd.grad(
                logits[0, targets[1]] - logits[0, targets[0]], act)
            expected = vec[0] - vec[1]
            actual = oracle_grad.detach().float()[0, key_pos + value_pos]
            err = _finite_norm(expected - actual)
            if err > 1e-4 + 1e-3 * max(_finite_norm(expected), _finite_norm(actual)):
                raise AssertionError("fixed-query label-contrast gradient identity failed")
            derivative_residuals.append(err)
            observed.append(row)
            signed_keys.append([v[:len(key_pos)].clone() for v in vec])
            signed_values.append([v[len(key_pos):].clone() for v in vec])
            activations.clear()
    finally:
        hook.remove()
        model.zero_grad(set_to_none=True)
    if _state_sha(model) != state_sha:
        raise AssertionError("model tensors changed during measurement")
    if arm == "sequence_only":
        if any(x != 0.0 for row in observed for cell in row for k in (
            "key_gradient_l2", "value_gradient_l2") for x in cell[k]):
            raise AssertionError("sequence-only far source leaked gradients")
    return {
        "arm": arm, "parameter_count": params, "source_prefix_sha256": digest,
        "model_initial_and_final_state_sha256": state_sha,
        "query_tokens": queries, "target_tokens": targets,
        "grid_4x4": observed,
        "key_metrics": _stats(observed, "key"),
        "value_metrics": _stats(observed, "value"),
        "signed_key_differences": _signed_contrasts(signed_keys),
        "signed_value_differences": _signed_contrasts(signed_values),
        "max_fixed_query_logit_identity_residual": max(derivative_residuals),
        "training_performed": False, "optimizer_steps": 0,
        "gpu_requested": False, "scientific_seed_consumed": False,
    }


def audit(*, arms: Sequence[str] = ARMS, seed: int = ENGINEERING_SEED) -> dict[str, Any]:
    if type(seed) is not int or seed != ENGINEERING_SEED:
        raise ValueError("only preregistered engineering seed allowed")
    if not arms or len(set(arms)) != len(arms) or not set(arms).issubset(ARMS):
        raise ValueError("only nonempty unique frozen arms allowed")
    results = [one_arm(a, seed=seed) for a in arms]
    if len({r["source_prefix_sha256"] for r in results}) != 1:
        raise AssertionError("source prefix changed between arms")
    return {
        "classification": CLASSIFICATION, "issue": ISSUE,
        "engineering_seed_not_scientific": seed,
        "frozen_dependency_blobs": FROZEN_DEPENDENCY_BLOBS,
        "query_delays": list(SCORED_DELAYS), "source_delays": list(VALUE_DELAYS),
        "source_prefix_sha256": results[0]["source_prefix_sha256"],
        "key_blind_accuracy": .25, "training_performed": False,
        "optimizer_steps": 0, "gpu_allocated": False,
        "scientific_attempt": False, "breakthrough_claim": False,
        "results": results,
        "interpretation": "random-init differential gradients; NOT learned recall or model quality",
    }


if __name__ == "__main__":
    print("CPW_1470_V2_FACTORIAL=" + json.dumps(audit(), sort_keys=True, allow_nan=False))
