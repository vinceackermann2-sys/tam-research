from __future__ import annotations

"""CPU-only #278-equivalent delayed-associative diagnostic for AERA-v21."""

import json
from typing import Any

import torch

from aera_v19_memory_necessity_cpu import (
    BATCH_SIZE, EVAL_SEED, FRESH_SESSION_CHANCE_TOLERANCE, FULL_ACCURACY_MIN,
    FULL_OVER_STREAM_MIN, LEARNING_RATE, N_VALUES, OVERWRITE_ACCURACY_MIN,
    SAME_CHECKPOINT_MEMORY_DROP_MIN, SEED, STALE_ERROR_MAX, TASK_VALIDITY_MIN,
    TRAIN_STEPS, _evaluate, _force_all_stages_run, _loss_and_accuracy,
    deployment_safety_and_isolation, diagnostic_config, make_batch, oracle_accuracy,
)
from tam_research.aera_hardware_core_v21 import HardwareAwareAERATextLMV21


def build_model(seed: int) -> HardwareAwareAERATextLMV21:
    torch.manual_seed(seed)
    model = HardwareAwareAERATextLMV21(diagnostic_config())
    _force_all_stages_run(model)
    return model


def train_pair(*, steps: int = TRAIN_STEPS):
    full = build_model(SEED)
    stream_only = build_model(SEED)
    for key, value in full.state_dict().items():
        torch.testing.assert_close(stream_only.state_dict()[key], value, atol=0.0, rtol=0.0)

    full.train(); stream_only.train()
    full_opt = torch.optim.AdamW([p for p in full.parameters() if p.requires_grad], lr=LEARNING_RATE, weight_decay=0.0)
    stream_opt = torch.optim.AdamW([p for p in stream_only.parameters() if p.requires_grad], lr=LEARNING_RATE, weight_decay=0.0)
    history: list[dict[str, float]] = []
    for step in range(steps):
        batch = make_batch(BATCH_SIZE, SEED * 10000 + step)

        full_opt.zero_grad(set_to_none=True)
        full_loss, full_acc = _loss_and_accuracy(full, batch, update_memory=True, differentiable_memory=True)
        full_loss.backward(); torch.nn.utils.clip_grad_norm_(full.parameters(), 1.0); full_opt.step()

        stream_opt.zero_grad(set_to_none=True)
        stream_loss, stream_acc = _loss_and_accuracy(stream_only, batch, update_memory=False, differentiable_memory=False)
        stream_loss.backward(); torch.nn.utils.clip_grad_norm_(stream_only.parameters(), 1.0); stream_opt.step()

        if step in {0, steps // 4, steps // 2, (3 * steps) // 4, steps - 1}:
            row = {"step": float(step + 1), "full_loss": float(full_loss.detach()), "stream_only_loss": float(stream_loss.detach()), "full_accuracy": full_acc, "stream_only_accuracy": stream_acc}
            history.append(row)
            print("AERA_V21_MEMORY_NECESSITY_TRAIN=" + json.dumps(row, sort_keys=True), flush=True)
    return full, stream_only, {"history": history}


def run_diagnostic(*, steps: int = TRAIN_STEPS) -> dict[str, Any]:
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    eval_batch = make_batch(24, EVAL_SEED)
    validity = oracle_accuracy(eval_batch)
    if validity < TASK_VALIDITY_MIN:
        raise RuntimeError(f"task validity failed before training: {validity}")

    full, stream_only, training = train_pair(steps=steps)
    full_eval = _evaluate(full, eval_batch, memory=True)
    full_memory_off = _evaluate(full, eval_batch, memory=False)
    stream_eval = _evaluate(stream_only, eval_batch, memory=False)
    safety = deployment_safety_and_isolation(full, eval_batch)
    chance = 1.0 / N_VALUES
    full_minus_stream = full_eval["query_accuracy"] - stream_eval["query_accuracy"]
    same_ckpt_gain = full_eval["query_accuracy"] - full_memory_off["query_accuracy"]
    checks = {
        "task_validity_ge_0_95": validity >= TASK_VALIDITY_MIN,
        "full_query_accuracy_ge_0_80": full_eval["query_accuracy"] >= FULL_ACCURACY_MIN,
        "full_over_stream_only_ge_0_15": full_minus_stream >= FULL_OVER_STREAM_MIN,
        "same_checkpoint_memory_drop_ge_0_15": same_ckpt_gain >= SAME_CHECKPOINT_MEMORY_DROP_MIN,
        "overwrite_current_value_accuracy_ge_0_80": full_eval["overwrite_current_value_accuracy"] >= OVERWRITE_ACCURACY_MIN,
        "stale_value_error_le_0_10": full_eval["stale_value_error_rate"] <= STALE_ERROR_MAX,
        "fresh_session_near_chance": safety["fresh_session_query_accuracy"] <= chance + FRESH_SESSION_CHANCE_TOLERANCE,
        "deployment_base_parameters_unchanged": safety["deployment_base_parameters_unchanged"],
        "session_isolation_exact": safety["session_isolation_exact"],
    }
    return {
        "version": "aera-v21-event-pair-memory-necessity-cpu",
        "task": {"oracle_accuracy": validity, "chance_accuracy": chance, "frozen_source": "#278"},
        "full_stream_plus_memory": {k: v for k, v in full_eval.items() if k != "predictions"},
        "same_checkpoint_memory_disabled": {k: v for k, v in full_memory_off.items() if k != "predictions"},
        "separately_trained_stream_only": {k: v for k, v in stream_eval.items() if k != "predictions"},
        "full_minus_stream_only_accuracy": full_minus_stream,
        "same_checkpoint_memory_contribution_accuracy": same_ckpt_gain,
        "deployment_safety": safety,
        "training": training,
        "checks": checks,
        "pass": all(checks.values()),
    }


def main() -> None:
    print("AERA_V21_MEMORY_NECESSITY_RESULT_JSON=" + json.dumps(run_diagnostic(), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
