from __future__ import annotations

"""CPU-only diagnostic for integrated AERA-v19 fast-memory utility.

This is not a new architecture version.  It asks a narrower falsifiable question:
can the already-merged v19 memory pathway learn to provide information that the
recurrent stream alone does not on a task where associative recall is necessary?

No production corpus, real-language checkpoint, routing policy, or GPU is touched.
"""

from dataclasses import dataclass
import json
import math
from typing import Any

import torch
import torch.nn.functional as F

from tam_research.aera_hardware_core import HardwareAERAConfig
from tam_research.aera_hardware_core_v19 import HardwareAwareAERATextLMV19

SEED = 8401
EVAL_SEED = 8402
VOCAB_SIZE = 64
CHUNK_SIZE = 6
N_KEYS = 12
N_VALUES = 16
N_OVERWRITES = 4
N_DISTRACTOR_CHUNKS = 2
N_QUERY_KEYS = N_KEYS
BATCH_SIZE = 6
TRAIN_STEPS = 500
LEARNING_RATE = 4e-3

WRITE = 1
QUERY = 2
DISTRACT = 3
FILLER_START = 4
KEY_START = 12
VALUE_START = KEY_START + N_KEYS
FILLER_END = KEY_START

TASK_VALIDITY_MIN = 0.95
FULL_ACCURACY_MIN = 0.80
FULL_OVER_STREAM_MIN = 0.15
SAME_CHECKPOINT_MEMORY_DROP_MIN = 0.15
OVERWRITE_ACCURACY_MIN = 0.80
STALE_ERROR_MAX = 0.10
FRESH_SESSION_CHANCE_TOLERANCE = 0.10


@dataclass
class AssociativeBatch:
    tokens: torch.Tensor
    query_positions: torch.Tensor
    query_targets: torch.Tensor
    query_keys: torch.Tensor
    overwrite_mask: torch.Tensor
    stale_targets: torch.Tensor


def diagnostic_config() -> HardwareAERAConfig:
    return HardwareAERAConfig(
        vocab_size=VOCAB_SIZE,
        d_model=24,
        n_stages=4,
        n_heads=4,
        chunk_size=CHUNK_SIZE,
        n_experts=2,
        max_active_experts=1,
        expert_mult=2,
        memory_dim=16,
        max_reason_steps=2,
        block_size=2,
    )


def _force_all_stages_run(model: HardwareAwareAERATextLMV19) -> None:
    """Hold whole-stage routing constant so this experiment isolates memory."""
    with torch.no_grad():
        for router in model.stage_routers:
            router.proj.weight.zero_()
            router.proj.bias.fill_(12.0)
    for router in model.stage_routers:
        for parameter in router.parameters():
            parameter.requires_grad_(False)


def build_model(seed: int) -> HardwareAwareAERATextLMV19:
    torch.manual_seed(seed)
    model = HardwareAwareAERATextLMV19(diagnostic_config())
    _force_all_stages_run(model)
    return model


def _filler(g: torch.Generator, count: int) -> list[int]:
    return torch.randint(FILLER_START, FILLER_END, (count,), generator=g).tolist()


def _chunk(prefix: int, key: int | None, value: int | None, g: torch.Generator) -> list[int]:
    row = [prefix]
    if key is not None:
        row.append(key)
    if value is not None:
        row.append(value)
    row.extend(_filler(g, CHUNK_SIZE - len(row)))
    if len(row) != CHUNK_SIZE:
        raise AssertionError("chunk construction mismatch")
    return row


def make_batch(batch_size: int, seed: int) -> AssociativeBatch:
    """Build randomized write/overwrite/distractor/query sessions.

    Every example writes all keys with random values, overwrites a random subset,
    waits through distractor chunks, then queries every key in random order.  The
    value token follows the key token in each query chunk, so the logit at the key
    position must predict a value that is not yet causally visible in that chunk.
    """
    g = torch.Generator().manual_seed(seed)
    rows: list[list[int]] = []
    all_positions: list[list[int]] = []
    all_targets: list[list[int]] = []
    all_keys: list[list[int]] = []
    all_overwrite: list[list[bool]] = []
    all_stale: list[list[int]] = []

    key_tokens = list(range(KEY_START, KEY_START + N_KEYS))
    value_tokens = list(range(VALUE_START, VALUE_START + N_VALUES))

    for _ in range(batch_size):
        current: dict[int, int] = {}
        stale: dict[int, int] = {}
        tokens: list[int] = []

        initial_values = torch.randint(0, N_VALUES, (N_KEYS,), generator=g).tolist()
        for key, value_index in zip(key_tokens, initial_values):
            value = value_tokens[value_index]
            current[key] = value
            tokens.extend(_chunk(WRITE, key, value, g))

        overwrite_order = torch.randperm(N_KEYS, generator=g)[:N_OVERWRITES].tolist()
        overwritten = {key_tokens[i] for i in overwrite_order}
        for i in overwrite_order:
            key = key_tokens[i]
            old = current[key]
            new_index = int(torch.randint(0, N_VALUES - 1, (1,), generator=g))
            new = value_tokens[new_index]
            if new >= old:
                new += 1
            if new not in value_tokens:
                new = value_tokens[0]
            if new == old:
                raise AssertionError("overwrite must change the value")
            stale[key] = old
            current[key] = new
            tokens.extend(_chunk(WRITE, key, new, g))

        for _d in range(N_DISTRACTOR_CHUNKS):
            tokens.extend(_chunk(DISTRACT, None, None, g))

        query_order = torch.randperm(N_KEYS, generator=g).tolist()
        positions: list[int] = []
        targets: list[int] = []
        qkeys: list[int] = []
        overwrite_flags: list[bool] = []
        stale_values: list[int] = []
        for i in query_order:
            key = key_tokens[i]
            value = current[key]
            start = len(tokens)
            query_chunk = _chunk(QUERY, key, value, g)
            tokens.extend(query_chunk)
            positions.append(start + 1)  # key token predicts the following value
            targets.append(value)
            qkeys.append(key)
            overwrite_flags.append(key in overwritten)
            stale_values.append(stale.get(key, -1))

        rows.append(tokens)
        all_positions.append(positions)
        all_targets.append(targets)
        all_keys.append(qkeys)
        all_overwrite.append(overwrite_flags)
        all_stale.append(stale_values)

    lengths = {len(row) for row in rows}
    if len(lengths) != 1:
        raise AssertionError(f"batch sequence lengths disagree: {lengths}")
    return AssociativeBatch(
        tokens=torch.tensor(rows, dtype=torch.long),
        query_positions=torch.tensor(all_positions, dtype=torch.long),
        query_targets=torch.tensor(all_targets, dtype=torch.long),
        query_keys=torch.tensor(all_keys, dtype=torch.long),
        overwrite_mask=torch.tensor(all_overwrite, dtype=torch.bool),
        stale_targets=torch.tensor(all_stale, dtype=torch.long),
    )


def oracle_accuracy(batch: AssociativeBatch) -> float:
    correct = 0
    total = 0
    for row in batch.tokens.tolist():
        current: dict[int, int] = {}
        for start in range(0, len(row), CHUNK_SIZE):
            chunk = row[start : start + CHUNK_SIZE]
            if chunk[0] == WRITE:
                current[chunk[1]] = chunk[2]
            elif chunk[0] == QUERY:
                total += 1
                correct += int(current.get(chunk[1]) == chunk[2])
    return correct / max(total, 1)


def _query_logits(logits: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    batch = torch.arange(logits.size(0), device=logits.device)[:, None]
    return logits[batch, positions.to(logits.device)]


def _loss_and_accuracy(
    model: HardwareAwareAERATextLMV19,
    batch: AssociativeBatch,
    *,
    update_memory: bool,
    differentiable_memory: bool,
) -> tuple[torch.Tensor, float]:
    model.set_memory_pretraining_mode(differentiable_memory)
    try:
        out = model(
            batch.tokens,
            hard=True,
            route_mode="hard_sparse",
            update_memory=update_memory,
            return_block_logits=False,
        )
    finally:
        model.set_memory_pretraining_mode(False)
    logits = out["logits"]
    assert isinstance(logits, torch.Tensor)
    selected = _query_logits(logits, batch.query_positions)
    target = batch.query_targets.to(selected.device)
    loss = F.cross_entropy(selected.float().reshape(-1, VOCAB_SIZE), target.reshape(-1))
    accuracy = float((selected.argmax(dim=-1) == target).float().mean().detach())
    return loss, accuracy


def train_pair(*, steps: int = TRAIN_STEPS) -> tuple[HardwareAwareAERATextLMV19, HardwareAwareAERATextLMV19, dict[str, Any]]:
    full = build_model(SEED)
    stream_only = build_model(SEED)
    for key, value in full.state_dict().items():
        torch.testing.assert_close(stream_only.state_dict()[key], value, atol=0.0, rtol=0.0)

    full.train()
    stream_only.train()
    full_opt = torch.optim.AdamW([p for p in full.parameters() if p.requires_grad], lr=LEARNING_RATE, weight_decay=0.0)
    stream_opt = torch.optim.AdamW([p for p in stream_only.parameters() if p.requires_grad], lr=LEARNING_RATE, weight_decay=0.0)

    history: list[dict[str, float]] = []
    for step in range(steps):
        batch = make_batch(BATCH_SIZE, SEED * 10000 + step)

        full_opt.zero_grad(set_to_none=True)
        full_loss, full_acc = _loss_and_accuracy(
            full, batch, update_memory=True, differentiable_memory=True
        )
        full_loss.backward()
        torch.nn.utils.clip_grad_norm_(full.parameters(), 1.0)
        full_opt.step()

        stream_opt.zero_grad(set_to_none=True)
        stream_loss, stream_acc = _loss_and_accuracy(
            stream_only, batch, update_memory=False, differentiable_memory=False
        )
        stream_loss.backward()
        torch.nn.utils.clip_grad_norm_(stream_only.parameters(), 1.0)
        stream_opt.step()

        if step in {0, steps // 4, steps // 2, (3 * steps) // 4, steps - 1}:
            row = {
                "step": float(step + 1),
                "full_loss": float(full_loss.detach()),
                "stream_only_loss": float(stream_loss.detach()),
                "full_accuracy": full_acc,
                "stream_only_accuracy": stream_acc,
            }
            history.append(row)
            print("AERA_V19_MEMORY_NECESSITY_TRAIN=" + json.dumps(row, sort_keys=True), flush=True)
    return full, stream_only, {"history": history}


@torch.no_grad()
def _evaluate(model: HardwareAwareAERATextLMV19, batch: AssociativeBatch, *, memory: bool) -> dict[str, Any]:
    model.eval()
    model.set_memory_pretraining_mode(False)
    out = model(
        batch.tokens,
        hard=True,
        route_mode="hard_sparse",
        update_memory=memory,
        return_block_logits=False,
    )
    logits = out["logits"]
    assert isinstance(logits, torch.Tensor)
    selected = _query_logits(logits, batch.query_positions)
    target = batch.query_targets
    prediction = selected.argmax(dim=-1).cpu()
    accuracy = float((prediction == target).float().mean())
    nll = float(F.cross_entropy(selected.float().reshape(-1, VOCAB_SIZE), target.reshape(-1)))

    overwrite = batch.overwrite_mask
    overwrite_accuracy = float((prediction[overwrite] == target[overwrite]).float().mean())
    stale = batch.stale_targets
    stale_error = float((prediction[overwrite] == stale[overwrite]).float().mean())
    return {
        "query_accuracy": accuracy,
        "query_nll": nll,
        "overwrite_current_value_accuracy": overwrite_accuracy,
        "stale_value_error_rate": stale_error,
        "predictions": prediction,
    }


def _query_only_batch(source: AssociativeBatch) -> AssociativeBatch:
    rows: list[list[int]] = []
    positions: list[list[int]] = []
    for b in range(source.tokens.size(0)):
        row: list[int] = []
        pos: list[int] = []
        # Two distractor chunks ensure the test still crosses state boundaries, but
        # no key/value write for the queried bindings is provided.
        g = torch.Generator().manual_seed(EVAL_SEED + 5000 + b)
        for _ in range(2):
            row.extend(_chunk(DISTRACT, None, None, g))
        for q in range(source.query_keys.size(1)):
            key = int(source.query_keys[b, q])
            value = int(source.query_targets[b, q])
            start = len(row)
            row.extend(_chunk(QUERY, key, value, g))
            pos.append(start + 1)
        rows.append(row)
        positions.append(pos)
    return AssociativeBatch(
        tokens=torch.tensor(rows, dtype=torch.long),
        query_positions=torch.tensor(positions, dtype=torch.long),
        query_targets=source.query_targets.clone(),
        query_keys=source.query_keys.clone(),
        overwrite_mask=source.overwrite_mask.clone(),
        stale_targets=source.stale_targets.clone(),
    )


@torch.no_grad()
def deployment_safety_and_isolation(model: HardwareAwareAERATextLMV19, eval_batch: AssociativeBatch) -> dict[str, Any]:
    model.eval()
    model.set_memory_pretraining_mode(False)
    versions_before = [p._version for p in model.parameters()]
    _ = model(
        eval_batch.tokens,
        hard=True,
        route_mode="hard_sparse",
        update_memory=True,
        return_block_logits=False,
    )
    base_parameters_unchanged = versions_before == [p._version for p in model.parameters()]

    fresh = _query_only_batch(eval_batch)
    first = model(
        fresh.tokens,
        state=None,
        hard=True,
        route_mode="hard_sparse",
        update_memory=False,
        return_block_logits=False,
    )["logits"]
    # Exercise a separate session and discard its returned state.
    other = make_batch(eval_batch.tokens.size(0), EVAL_SEED + 999)
    _ = model(
        other.tokens,
        state=None,
        hard=True,
        route_mode="hard_sparse",
        update_memory=True,
        return_block_logits=False,
    )
    second = model(
        fresh.tokens,
        state=None,
        hard=True,
        route_mode="hard_sparse",
        update_memory=False,
        return_block_logits=False,
    )["logits"]
    assert isinstance(first, torch.Tensor) and isinstance(second, torch.Tensor)
    session_isolation_exact = bool(torch.equal(first, second))
    selected = _query_logits(first, fresh.query_positions)
    fresh_accuracy = float((selected.argmax(dim=-1).cpu() == fresh.query_targets).float().mean())
    return {
        "deployment_base_parameters_unchanged": base_parameters_unchanged,
        "session_isolation_exact": session_isolation_exact,
        "fresh_session_query_accuracy": fresh_accuracy,
    }


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
    checks = {
        "task_validity_ge_0_95": validity >= TASK_VALIDITY_MIN,
        "full_query_accuracy_ge_0_80": full_eval["query_accuracy"] >= FULL_ACCURACY_MIN,
        "full_over_stream_only_ge_0_15": (full_eval["query_accuracy"] - stream_eval["query_accuracy"]) >= FULL_OVER_STREAM_MIN,
        "same_checkpoint_memory_drop_ge_0_15": (full_eval["query_accuracy"] - full_memory_off["query_accuracy"]) >= SAME_CHECKPOINT_MEMORY_DROP_MIN,
        "overwrite_current_value_accuracy_ge_0_80": full_eval["overwrite_current_value_accuracy"] >= OVERWRITE_ACCURACY_MIN,
        "stale_value_error_le_0_10": full_eval["stale_value_error_rate"] <= STALE_ERROR_MAX,
        "fresh_session_near_chance": safety["fresh_session_query_accuracy"] <= chance + FRESH_SESSION_CHANCE_TOLERANCE,
        "deployment_base_parameters_unchanged": bool(safety["deployment_base_parameters_unchanged"]),
        "session_isolation_exact": bool(safety["session_isolation_exact"]),
    }
    result = {
        "scope": "aera-v19-integrated-fast-memory-necessity-cpu-diagnostic-v1",
        "seed": SEED,
        "eval_seed": EVAL_SEED,
        "gpu_used": False,
        "production_data_changed": False,
        "architecture_version_changed": False,
        "routing_held_constant_all_stages_run": True,
        "task": {
            "chunk_size": CHUNK_SIZE,
            "keys": N_KEYS,
            "values": N_VALUES,
            "overwrites_per_session": N_OVERWRITES,
            "distractor_chunks": N_DISTRACTOR_CHUNKS,
            "query_keys": N_QUERY_KEYS,
            "chance_accuracy": chance,
            "oracle_accuracy": validity,
        },
        "model": {
            "stored_parameters": sum(p.numel() for p in full.parameters()),
            "stream_state_scalars_per_stage": diagnostic_config().d_model,
            "fast_memory_scalars_per_stage": diagnostic_config().memory_dim ** 2,
            "stages": diagnostic_config().n_stages,
        },
        "training": training,
        "full_stream_plus_memory": {k: v for k, v in full_eval.items() if k != "predictions"},
        "same_checkpoint_memory_disabled": {k: v for k, v in full_memory_off.items() if k != "predictions"},
        "separately_trained_stream_only": {k: v for k, v in stream_eval.items() if k != "predictions"},
        "full_minus_stream_only_accuracy": full_eval["query_accuracy"] - stream_eval["query_accuracy"],
        "same_checkpoint_memory_contribution_accuracy": full_eval["query_accuracy"] - full_memory_off["query_accuracy"],
        "deployment_safety": safety,
        "checks": checks,
        "pass": all(checks.values()),
        "claims": {
            "integrated_memory_necessity_proven_on_controlled_task": all(checks.values()),
            "real_language_memory_advantage_proven": False,
            "architecture_frozen": False,
            "100m_authorized": False,
            "breakthrough_proven": False,
        },
    }
    return result


def protocol_summary() -> dict[str, Any]:
    return {
        "seed": SEED,
        "eval_seed": EVAL_SEED,
        "cpu_only": True,
        "train_steps": TRAIN_STEPS,
        "batch_size": BATCH_SIZE,
        "thresholds": {
            "task_validity_min": TASK_VALIDITY_MIN,
            "full_accuracy_min": FULL_ACCURACY_MIN,
            "full_over_stream_min": FULL_OVER_STREAM_MIN,
            "same_checkpoint_memory_drop_min": SAME_CHECKPOINT_MEMORY_DROP_MIN,
            "overwrite_accuracy_min": OVERWRITE_ACCURACY_MIN,
            "stale_error_max": STALE_ERROR_MAX,
            "fresh_session_chance_tolerance": FRESH_SESSION_CHANCE_TOLERANCE,
        },
        "gpu_authorized": False,
    }


def main() -> None:
    result = run_diagnostic()
    print("AERA_V19_MEMORY_NECESSITY_RESULT_JSON=" + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
