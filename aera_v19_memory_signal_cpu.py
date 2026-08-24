from __future__ import annotations

"""CPU-only root-cause diagnostic for AERA-v19 integrated fast memory.

Issue #278 proved that ordinary delayed query CE does not teach the integrated
v19 memory path, even when associative memory is necessary.  This diagnostic
keeps the production architecture/equation unchanged and asks whether the
existing *single end-of-chunk summary* has enough representational capacity if
we directly supervise its memory address and payload during training.

The auxiliary labels are synthetic-task-only diagnostic supervision.  They are
not a production language-training objective and do not authorize any GPU run.
"""

from contextlib import contextmanager
import json
from typing import Any, Iterator

import torch
import torch.nn.functional as F

from aera_v19_memory_necessity_cpu import (
    BATCH_SIZE,
    CHUNK_SIZE,
    EVAL_SEED,
    FRESH_SESSION_CHANCE_TOLERANCE,
    KEY_START,
    LEARNING_RATE,
    N_KEYS,
    N_VALUES,
    OVERWRITE_ACCURACY_MIN,
    SEED,
    STALE_ERROR_MAX,
    TASK_VALIDITY_MIN,
    VALUE_START,
    WRITE,
    _evaluate,
    build_model,
    deployment_safety_and_isolation,
    make_batch,
    oracle_accuracy,
)
from tam_research.aera_hardware_core_v19 import HardwareAwareAERATextLMV19

TRAIN_STEPS = 500
AUX_ADDRESS_WEIGHT = 1.0
AUX_VALUE_WEIGHT = 1.0
ADDRESS_TEMPERATURE = 0.10

FULL_ACCURACY_MIN = 0.80
SAME_CHECKPOINT_MEMORY_DROP_MIN = 0.15
ADDRESS_ACCURACY_MIN = 0.90
VALUE_ACCURACY_MIN = 0.90


@contextmanager
def _capture_stage0_reasoned(model: HardwareAwareAERATextLMV19) -> Iterator[list[torch.Tensor]]:
    """Capture the existing stage-0 causal end-of-chunk latent summaries."""
    captured: list[torch.Tensor] = []

    def hook(_module: torch.nn.Module, _inputs: tuple[object, ...], output: torch.Tensor) -> None:
        captured.append(output)

    handle = model.stages[0].reasoner.register_forward_hook(hook)
    try:
        yield captured
    finally:
        handle.remove()


def _candidate_key_queries(model: HardwareAwareAERATextLMV19) -> torch.Tensor:
    """Represent the exact key-token position used by v19 token-wise reads."""
    device = model.token_emb.weight.device
    keys = torch.arange(KEY_START, KEY_START + N_KEYS, device=device)
    pos = model.local_pos(torch.tensor(1, device=device))
    x = model.token_emb(keys) + pos[None, :]
    stage = model.stages[0]
    x = stage.norm(x)
    return F.normalize(stage.memory.q(x), dim=-1)


def _memory_utility_terms(
    model: HardwareAwareAERATextLMV19,
    tokens: torch.Tensor,
    captured: list[torch.Tensor],
) -> dict[str, torch.Tensor]:
    if tokens.size(1) % CHUNK_SIZE:
        raise ValueError("diagnostic sequence must divide exactly into chunks")
    n_chunks = tokens.size(1) // CHUNK_SIZE
    if len(captured) != n_chunks:
        raise RuntimeError(f"expected {n_chunks} stage-0 summaries, got {len(captured)}")

    summaries = torch.stack(captured, dim=1)  # [batch,chunks,d_model]
    chunks = tokens.view(tokens.size(0), n_chunks, CHUNK_SIZE)
    write_mask = chunks[:, :, 0].eq(WRITE)
    if not bool(write_mask.any()):
        raise RuntimeError("diagnostic batch has no writes")

    write_summary = summaries[write_mask]
    key_token = chunks[:, :, 1][write_mask]
    value_token = chunks[:, :, 2][write_mask]

    stage = model.stages[0]
    memory = stage.memory

    write_address = F.normalize(memory.k(write_summary), dim=-1)
    key_queries = _candidate_key_queries(model)
    address_logits = write_address @ key_queries.transpose(0, 1)
    address_logits = address_logits / ADDRESS_TEMPERATURE
    address_target = key_token - KEY_START
    address_loss = F.cross_entropy(address_logits.float(), address_target)
    address_accuracy = (address_logits.argmax(dim=-1) == address_target).float().mean()

    # Use the exact existing v/out projections.  This does not add a value head.
    payload = memory.out(torch.tanh(memory.v(write_summary)))
    payload_logits = model.lm_head(model.norm(payload))
    value_loss = F.cross_entropy(payload_logits.float(), value_token)
    value_accuracy = (payload_logits.argmax(dim=-1) == value_token).float().mean()

    return {
        "address_loss": address_loss,
        "address_accuracy": address_accuracy,
        "value_loss": value_loss,
        "value_accuracy": value_accuracy,
    }


def _query_loss_and_accuracy(logits: torch.Tensor, batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
    rows = torch.arange(logits.size(0), device=logits.device)[:, None]
    selected = logits[rows, batch.query_positions.to(logits.device)]
    target = batch.query_targets.to(logits.device)
    loss = F.cross_entropy(selected.float().reshape(-1, logits.size(-1)), target.reshape(-1))
    accuracy = (selected.argmax(dim=-1) == target).float().mean()
    return loss, accuracy


def train_with_memory_signal(*, steps: int = TRAIN_STEPS) -> tuple[HardwareAwareAERATextLMV19, dict[str, Any]]:
    model = build_model(SEED)
    model.train()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE,
        weight_decay=0.0,
    )
    history: list[dict[str, float]] = []

    for step in range(steps):
        batch = make_batch(BATCH_SIZE, SEED * 20000 + step)
        optimizer.zero_grad(set_to_none=True)
        model.set_memory_pretraining_mode(True)
        try:
            with _capture_stage0_reasoned(model) as captured:
                out = model(
                    batch.tokens,
                    hard=True,
                    route_mode="hard_sparse",
                    update_memory=True,
                    return_block_logits=False,
                )
            logits = out["logits"]
            assert isinstance(logits, torch.Tensor)
            query_loss, query_accuracy = _query_loss_and_accuracy(logits, batch)
            aux = _memory_utility_terms(model, batch.tokens, captured)
            total = (
                query_loss
                + AUX_ADDRESS_WEIGHT * aux["address_loss"]
                + AUX_VALUE_WEIGHT * aux["value_loss"]
            )
            total.backward()
        finally:
            model.set_memory_pretraining_mode(False)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step in {0, steps // 4, steps // 2, (3 * steps) // 4, steps - 1}:
            row = {
                "step": float(step + 1),
                "total_loss": float(total.detach()),
                "query_loss": float(query_loss.detach()),
                "query_accuracy": float(query_accuracy.detach()),
                "address_accuracy": float(aux["address_accuracy"].detach()),
                "value_accuracy": float(aux["value_accuracy"].detach()),
            }
            history.append(row)
            print("AERA_V19_MEMORY_SIGNAL_TRAIN=" + json.dumps(row, sort_keys=True), flush=True)

    return model, {"history": history}


@torch.no_grad()
def evaluate_auxiliary(model: HardwareAwareAERATextLMV19, batch: Any) -> dict[str, float]:
    model.eval()
    model.set_memory_pretraining_mode(False)
    with _capture_stage0_reasoned(model) as captured:
        _ = model(
            batch.tokens,
            hard=True,
            route_mode="hard_sparse",
            update_memory=True,
            return_block_logits=False,
        )
    aux = _memory_utility_terms(model, batch.tokens, captured)
    return {
        "address_loss": float(aux["address_loss"]),
        "address_accuracy": float(aux["address_accuracy"]),
        "value_loss": float(aux["value_loss"]),
        "value_accuracy": float(aux["value_accuracy"]),
    }


def run_diagnostic(*, steps: int = TRAIN_STEPS) -> dict[str, Any]:
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    eval_batch = make_batch(24, EVAL_SEED + 100)
    validity = oracle_accuracy(eval_batch)
    if validity < TASK_VALIDITY_MIN:
        raise RuntimeError(f"task validity failed before training: {validity}")

    model, training = train_with_memory_signal(steps=steps)
    full = _evaluate(model, eval_batch, memory=True)
    memory_off = _evaluate(model, eval_batch, memory=False)
    auxiliary = evaluate_auxiliary(model, eval_batch)
    safety = deployment_safety_and_isolation(model, eval_batch)

    chance = 1.0 / N_VALUES
    memory_drop = full["query_accuracy"] - memory_off["query_accuracy"]
    checks = {
        "task_validity_ge_0_95": validity >= TASK_VALIDITY_MIN,
        "full_query_accuracy_ge_0_80": full["query_accuracy"] >= FULL_ACCURACY_MIN,
        "same_checkpoint_memory_drop_ge_0_15": memory_drop >= SAME_CHECKPOINT_MEMORY_DROP_MIN,
        "overwrite_current_value_accuracy_ge_0_80": full["overwrite_current_value_accuracy"] >= OVERWRITE_ACCURACY_MIN,
        "stale_value_error_le_0_10": full["stale_value_error_rate"] <= STALE_ERROR_MAX,
        "aux_address_accuracy_ge_0_90": auxiliary["address_accuracy"] >= ADDRESS_ACCURACY_MIN,
        "aux_value_accuracy_ge_0_90": auxiliary["value_accuracy"] >= VALUE_ACCURACY_MIN,
        "fresh_session_near_chance": safety["fresh_session_query_accuracy"] <= chance + FRESH_SESSION_CHANCE_TOLERANCE,
        "deployment_base_parameters_unchanged": bool(safety["deployment_base_parameters_unchanged"]),
        "session_isolation_exact": bool(safety["session_isolation_exact"]),
    }
    passed = all(checks.values())
    interpretation = (
        "training_signal_bottleneck"
        if passed
        else "single_summary_representation_or_injection_inadequate"
    )
    result = {
        "scope": "aera_v19_memory_signal_vs_representation_cpu",
        "steps": steps,
        "task": {
            "oracle_accuracy": validity,
            "chance_accuracy": chance,
            "keys": N_KEYS,
            "values": N_VALUES,
        },
        "training": training,
        "full_stream_plus_memory": {k: v for k, v in full.items() if k != "predictions"},
        "same_checkpoint_memory_disabled": {k: v for k, v in memory_off.items() if k != "predictions"},
        "same_checkpoint_memory_contribution_accuracy": memory_drop,
        "auxiliary_memory_signal": auxiliary,
        "deployment_safety": safety,
        "checks": checks,
        "pass": passed,
        "interpretation": interpretation,
        "claims": {
            "production_memory_objective_authorized": False,
            "gpu_authorized": False,
            "architecture_freeze_authorized": False,
            "100m_authorized": False,
            "breakthrough_proven": False,
        },
    }
    return result


def main() -> None:
    result = run_diagnostic()
    print("AERA_V19_MEMORY_SIGNAL_RESULT_JSON=" + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
