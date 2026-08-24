from __future__ import annotations

"""CPU-only controlled gate for AERA-v22 interference-corrected fast memory."""

import json
from typing import Any

import torch
import torch.nn.functional as F

from aera_v19_memory_necessity_cpu import (
    BATCH_SIZE,
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
    _evaluate,
    _force_all_stages_run,
    _loss_and_accuracy,
    deployment_safety_and_isolation,
    diagnostic_config,
    make_batch,
    oracle_accuracy,
)
from aera_v21_collapse_resistant_memory_objective_cpu import (
    ADDRESS_MARGIN_MIN,
    ADDRESS_TOP1_MIN,
    PAYLOAD_LEGAL_ACCURACY_MIN,
    evaluate_local_memory_code,
)
from aera_v21_conflict_free_memory_objective_cpu import (
    ADDRESS_CONTRASTIVE_WEIGHT,
    PAYLOAD_TOKEN_WEIGHT,
    conflict_free_memory_terms,
)
from aera_v21_write_kinetics_audit_cpu import _evaluate_mode
from tam_research.aera_hardware_core_v22 import (
    HardwareAwareAERATextLMV22,
    interference_corrected_dual_delta_update,
)

RAW_RECALL_MIN = 0.95
MECHANISM_CURRENT_MIN = 0.95
MECHANISM_STALE_MAX = 0.05


def build_model(seed: int) -> HardwareAwareAERATextLMV22:
    torch.manual_seed(seed)
    model = HardwareAwareAERATextLMV22(diagnostic_config())
    _force_all_stages_run(model)
    return model


@torch.no_grad()
def dual_delta_mechanism_probe() -> dict[str, float | bool]:
    """Nonorthogonal code-level write/overwrite gate for the new update itself."""
    d = 16
    n_keys = 12
    g = torch.Generator().manual_seed(9220)
    base = torch.eye(n_keys, d)
    keys = F.normalize(base + 0.08 * torch.randn(n_keys, d, generator=g), dim=-1)
    matrix = torch.zeros(1, d, d)
    inverse = torch.eye(d).unsqueeze(0)

    current = torch.arange(n_keys)
    stale = torch.full((n_keys,), -1, dtype=torch.long)
    overwrite_keys = torch.tensor([1, 4, 7, 10])
    overwrite_values = torch.tensor([12, 13, 14, 15])

    def write(key_index: int, value_index: int) -> None:
        nonlocal matrix, inverse
        key = keys[key_index].view(1, 1, d)
        target = F.one_hot(torch.tensor([[value_index]]), num_classes=d).float()
        strength = torch.ones(1, 1, 1)
        matrix, inverse = interference_corrected_dual_delta_update(
            matrix,
            inverse,
            key,
            target,
            strength,
        )

    for key_index in range(n_keys):
        write(key_index, int(current[key_index]))
    for key_index, value_index in zip(overwrite_keys.tolist(), overwrite_values.tolist()):
        stale[key_index] = current[key_index]
        current[key_index] = value_index
        write(key_index, value_index)

    prediction = (keys @ matrix[0]).argmax(dim=-1)
    current_accuracy = float((prediction == current).float().mean())
    overwrite_prediction = prediction[overwrite_keys]
    overwrite_current = float(
        (overwrite_prediction == current[overwrite_keys]).float().mean()
    )
    stale_error = float(
        (overwrite_prediction == stale[overwrite_keys]).float().mean()
    )
    kk = keys @ keys.transpose(0, 1)
    mask = ~torch.eye(n_keys, dtype=torch.bool)
    offdiag = kk[mask].abs()
    passed = (
        current_accuracy >= MECHANISM_CURRENT_MIN
        and overwrite_current >= MECHANISM_CURRENT_MIN
        and stale_error <= MECHANISM_STALE_MAX
    )
    return {
        "current_accuracy": current_accuracy,
        "overwrite_current_accuracy": overwrite_current,
        "stale_error": stale_error,
        "key_abs_offdiag_mean": float(offdiag.mean()),
        "key_abs_offdiag_max": float(offdiag.max()),
        "pass": passed,
    }


def train_pair_with_v22_objective(
    *,
    steps: int = TRAIN_STEPS,
) -> tuple[HardwareAwareAERATextLMV22, HardwareAwareAERATextLMV22, dict[str, Any]]:
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
        local = conflict_free_memory_terms(full, batch.tokens)
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
                "address_contrastive_loss": float(local["address_contrastive_loss"].detach()),
                "payload_token_loss": float(local["payload_token_loss"].detach()),
                "payload_token_accuracy": float(local["payload_token_accuracy"].detach()),
            }
            history.append(row)
            print("AERA_V22_DUAL_DELTA_TRAIN=" + json.dumps(row, sort_keys=True), flush=True)

    return full, stream_only, {
        "history": history,
        "initial_local_code": initial_local,
        "final_local_code": evaluate_local_memory_code(full),
    }


def run_gate(*, steps: int = TRAIN_STEPS) -> dict[str, Any]:
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    mechanism = dual_delta_mechanism_probe()
    eval_batch = make_batch(24, EVAL_SEED)
    validity = oracle_accuracy(eval_batch)
    if validity < TASK_VALIDITY_MIN:
        raise RuntimeError(f"task validity failed before training: {validity}")

    full, stream_only, training = train_pair_with_v22_objective(steps=steps)
    full_eval = _evaluate(full, eval_batch, memory=True)
    memory_off = _evaluate(full, eval_batch, memory=False)
    stream_eval = _evaluate(stream_only, eval_batch, memory=False)
    safety = deployment_safety_and_isolation(full, eval_batch)
    local = evaluate_local_memory_code(full)
    chunkwise = _evaluate_mode(full, eval_batch)
    if abs(chunkwise["final"]["overall_accuracy"] - full_eval["query_accuracy"]) > 1e-7:
        raise RuntimeError("chunkwise raw evaluator changed final v22 query accuracy")
    raw = chunkwise["raw_memory_decode"]

    chance = 1.0 / N_VALUES
    full_minus_stream = full_eval["query_accuracy"] - stream_eval["query_accuracy"]
    memory_drop = full_eval["query_accuracy"] - memory_off["query_accuracy"]
    checks = {
        "mechanism_probe_pass": bool(mechanism["pass"]),
        "task_validity_ge_0_95": validity >= TASK_VALIDITY_MIN,
        "address_top1_ge_0_95": local["address_top1_accuracy"] >= ADDRESS_TOP1_MIN,
        "address_margin_ge_0_05": local["address_mean_diag_minus_best_other_margin"] >= ADDRESS_MARGIN_MIN,
        "payload_legal_value_accuracy_ge_0_90": local["payload_legal_value_accuracy"] >= PAYLOAD_LEGAL_ACCURACY_MIN,
        "raw_recall_ge_0_95": raw["overall_accuracy"] >= RAW_RECALL_MIN,
        "raw_overwrite_current_ge_0_80": raw["overwrite_current_value_accuracy"] >= OVERWRITE_ACCURACY_MIN,
        "raw_stale_le_0_10": raw["overwrite_stale_value_error"] <= STALE_ERROR_MAX,
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
    raw_pass = all(
        checks[name]
        for name in (
            "raw_recall_ge_0_95",
            "raw_overwrite_current_ge_0_80",
            "raw_stale_le_0_10",
        )
    )
    end_to_end_names = (
        "task_validity_ge_0_95",
        "full_query_accuracy_ge_0_80",
        "full_over_stream_only_ge_0_15",
        "same_checkpoint_memory_drop_ge_0_15",
        "overwrite_current_value_accuracy_ge_0_80",
        "stale_value_error_le_0_10",
        "fresh_session_near_chance",
        "deployment_base_parameters_unchanged",
        "session_isolation_exact",
    )
    end_to_end_pass = all(checks[name] for name in end_to_end_names)
    passed = bool(mechanism["pass"]) and local_pass and raw_pass and end_to_end_pass
    if passed:
        interpretation = "v22_dual_delta_solves_controlled_integrated_fast_memory_blocker"
    elif local_pass and raw_pass:
        interpretation = "raw_memory_passes_but_downstream_integration_remains"
    elif local_pass:
        interpretation = "dual_delta_write_dynamics_still_insufficient"
    else:
        interpretation = "v22_training_interaction_changed_local_memory_code"

    return {
        "scope": "aera_v22_interference_corrected_dual_delta_cpu",
        "architecture_version": "v22",
        "gpu_used": False,
        "independent_evidence": False,
        "mechanism_probe": mechanism,
        "training": training,
        "heldout_local_code": local,
        "raw_memory_decode": raw,
        "full_stream_plus_memory": {k: v for k, v in full_eval.items() if k != "predictions"},
        "same_checkpoint_memory_disabled": {k: v for k, v in memory_off.items() if k != "predictions"},
        "separately_trained_stream_only": {k: v for k, v in stream_eval.items() if k != "predictions"},
        "full_minus_stream_only_accuracy": full_minus_stream,
        "same_checkpoint_memory_contribution_accuracy": memory_drop,
        "deployment_safety": safety,
        "checks": checks,
        "local_pass": local_pass,
        "raw_pass": raw_pass,
        "end_to_end_pass": end_to_end_pass,
        "pass": passed,
        "interpretation": interpretation,
        "claims": {
            "controlled_memory_blocker_solved": passed,
            "real_language_memory_advantage_proven": False,
            "architecture_frozen": False,
            "100m_authorized": False,
            "breakthrough_proven": False,
        },
    }


def main() -> None:
    result = run_gate()
    print("AERA_V22_DUAL_DELTA_RESULT_JSON=" + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
