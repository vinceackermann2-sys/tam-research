from __future__ import annotations

"""CPU-only post-#316 residual raw-recall localization for unchanged AERA-v21.

The #316 diagnostic showed that a unit delta step makes integrated memory
behaviorally useful on the controlled task, but its strictest stage-0 raw recall
control remained below the frozen 95% threshold.  This audit keeps all learned
parameters frozen after an exact #313 diagnostic reproduction and distinguishes
q/k read mismatch, decay attenuation, and sequential matrix superposition.
"""

from contextlib import ExitStack, contextmanager
import json
from typing import Any, Iterator

import torch
import torch.nn.functional as F

from aera_v19_memory_necessity_cpu import (
    CHUNK_SIZE,
    EVAL_SEED,
    KEY_START,
    N_KEYS,
    N_VALUES,
    OVERWRITE_ACCURACY_MIN,
    STALE_ERROR_MAX,
    VALUE_START,
    WRITE,
    make_batch,
)
from aera_v21_conflict_free_memory_objective_cpu import (
    train_pair_with_conflict_free_objective,
)
from aera_v21_memory_capacity_audit_cpu import _token_representation
from aera_v21_memory_code_objective_cpu import _decode_with_frozen_model_head
from aera_v21_write_kinetics_audit_cpu import (
    _evaluate_mode,
    _keep_candidate_one_preserve_gate,
    _temporary_memory_lr,
)
from tam_research.aera_hardware_core_v21 import HardwareAwareAERATextLMV21

RAW_RECALL_MIN = 0.95
DIAGNOSTIC_MEMORY_LR = 1.0
PRODUCTION_DIAGNOSTIC_DECAY = 0.999
NO_DECAY = 1.0


@contextmanager
def _temporary_memory_decay(
    model: HardwareAwareAERATextLMV21,
    decay: float,
) -> Iterator[None]:
    originals: list[tuple[torch.nn.Module, float]] = []
    for stage in model.stages:
        memory = stage.memory
        originals.append((memory, float(memory.decay)))
        memory.decay = float(decay)
    try:
        yield
    finally:
        for memory, original in originals:
            memory.decay = original


def _prefix(chunk: torch.Tensor) -> int:
    prefix = chunk[:, 0]
    if not bool(prefix.eq(prefix[0]).all()):
        raise RuntimeError("audit expects synchronized chunk types across batch")
    return int(prefix[0])


def _group_raw_metrics(
    prediction: torch.Tensor,
    batch: Any,
    target_minus_stale_margin: torch.Tensor,
) -> dict[str, Any]:
    target = batch.query_targets
    overwrite = batch.overwrite_mask
    stale = batch.stale_targets
    non_overwrite = ~overwrite

    def accuracy(lhs: torch.Tensor, rhs: torch.Tensor, mask: torch.Tensor) -> float:
        if not bool(mask.any()):
            return float("nan")
        return float((lhs[mask] == rhs[mask]).float().mean())

    return {
        "overall_accuracy": float((prediction == target).float().mean()),
        "non_overwrite_accuracy": accuracy(prediction, target, non_overwrite),
        "overwrite_current_value_accuracy": accuracy(prediction, target, overwrite),
        "overwrite_stale_value_error": accuracy(prediction, stale, overwrite),
        "overwrite_target_minus_stale_logit_margin": (
            float(target_minus_stale_margin[overwrite].float().mean())
            if bool(overwrite.any())
            else float("nan")
        ),
        "query_ordinal_accuracy": [
            float((prediction[:, i] == target[:, i]).float().mean())
            for i in range(prediction.size(1))
        ],
    }


def _decode_recall(
    model: HardwareAwareAERATextLMV21,
    recalled: torch.Tensor,
) -> torch.Tensor:
    stage0 = model.stages[0]
    payload_code = stage0.memory.out(recalled)
    return _decode_with_frozen_model_head(model, payload_code)


def _query_vector(
    model: HardwareAwareAERATextLMV21,
    chunk: torch.Tensor,
    *,
    read: str,
) -> torch.Tensor:
    stage0 = model.stages[0]
    pos = torch.arange(chunk.size(1), device=chunk.device)
    events = model.token_emb(chunk) + model.local_pos(pos)[None, :, :]
    h = stage0.norm(events)[:, 1]
    if read == "q":
        return F.normalize(stage0.memory.q(h), dim=-1)
    if read == "k":
        return F.normalize(stage0.memory.k(h), dim=-1)
    raise ValueError(read)


@torch.no_grad()
def sequential_raw_recall(
    model: HardwareAwareAERATextLMV21,
    batch: Any,
    *,
    read: str,
    decay: float,
) -> dict[str, Any]:
    """Strict #316 WRITE-only/candidate-1/lr=1 state with q or k raw reads."""
    model.eval()
    model.set_memory_pretraining_mode(False)
    state = None
    predictions: list[torch.Tensor] = []
    margins: list[torch.Tensor] = []
    query_index = 0

    with ExitStack() as stack:
        stack.enter_context(_keep_candidate_one_preserve_gate(model))
        stack.enter_context(_temporary_memory_lr(model, DIAGNOSTIC_MEMORY_LR))
        stack.enter_context(_temporary_memory_decay(model, decay))

        for start in range(0, batch.tokens.size(1), CHUNK_SIZE):
            chunk = batch.tokens[:, start : start + CHUNK_SIZE]
            prefix = _prefix(chunk)
            if prefix != WRITE:
                if state is not None and query_index < batch.query_targets.size(1):
                    # Only actual QUERY chunks have position-1 targets.  In this
                    # frozen task, all queries occur after writes+distractors.
                    from aera_v19_memory_necessity_cpu import QUERY
                    if prefix == QUERY:
                        query = _query_vector(model, chunk, read=read)
                        matrix = state.stages[0].memory.matrix
                        recalled = torch.einsum("bi,bij->bj", query, matrix)
                        logits = _decode_recall(model, recalled)
                        legal = logits[:, VALUE_START : VALUE_START + N_VALUES]
                        predictions.append((legal.argmax(dim=-1) + VALUE_START).cpu())

                        target = batch.query_targets[:, query_index].to(logits.device)
                        stale = batch.stale_targets[:, query_index].to(logits.device)
                        overwrite = batch.overwrite_mask[:, query_index].to(logits.device)
                        safe_stale = torch.where(overwrite, stale, target)
                        row = torch.arange(logits.size(0), device=logits.device)
                        margins.append((logits[row, target] - logits[row, safe_stale]).cpu())
                        query_index += 1

            out = model(
                chunk,
                state=state,
                hard=True,
                route_mode="hard_sparse",
                update_memory=(prefix == WRITE),
                return_block_logits=False,
            )
            state = out["state"]

    if query_index != batch.query_targets.size(1):
        raise RuntimeError("query count mismatch")
    prediction = torch.stack(predictions, dim=1)
    margin = torch.stack(margins, dim=1)
    return _group_raw_metrics(prediction, batch, margin)


def _least_squares_matrix(k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Minimum-norm M satisfying K M ~= V; diagnostic upper bound only."""
    if k.ndim != 2 or v.ndim != 2 or k.size(0) != v.size(0):
        raise ValueError("K and V must be [bindings,dim] with matching rows")
    return torch.linalg.lstsq(k.float(), v.float()).solution.to(v.dtype)


@torch.no_grad()
def least_squares_raw_recall(
    model: HardwareAwareAERATextLMV21,
    batch: Any,
    *,
    read: str,
) -> dict[str, Any]:
    """Oracle matrix-capacity upper bound from latest observed WRITE bindings."""
    model.eval()
    model.set_memory_pretraining_mode(False)
    stage0 = model.stages[0]
    batch_size = batch.tokens.size(0)
    latest: list[dict[int, tuple[torch.Tensor, torch.Tensor]]] = [
        {} for _ in range(batch_size)
    ]
    predictions: list[torch.Tensor] = []
    margins: list[torch.Tensor] = []
    query_index = 0

    for start in range(0, batch.tokens.size(1), CHUNK_SIZE):
        chunk = batch.tokens[:, start : start + CHUNK_SIZE]
        prefix = _prefix(chunk)
        pos = torch.arange(chunk.size(1), device=chunk.device)
        events = model.token_emb(chunk) + model.local_pos(pos)[None, :, :]
        base_h = stage0.norm(events)

        if prefix == WRITE:
            address = base_h[:, 1]
            payload = base_h[:, 2]
            for b in range(batch_size):
                latest[b][int(chunk[b, 1])] = (address[b].detach(), payload[b].detach())
            continue

        from aera_v19_memory_necessity_cpu import QUERY
        if prefix != QUERY:
            continue

        query = _query_vector(model, chunk, read=read)
        recalled_rows: list[torch.Tensor] = []
        for b in range(batch_size):
            if len(latest[b]) != N_KEYS:
                raise RuntimeError(f"expected {N_KEYS} latest bindings, got {len(latest[b])}")
            ordered = [latest[b][key] for key in sorted(latest[b])]
            addresses = torch.stack([row[0] for row in ordered], dim=0)
            payloads = torch.stack([row[1] for row in ordered], dim=0)
            k = F.normalize(stage0.memory.k(addresses), dim=-1)
            v = torch.tanh(stage0.memory.v(payloads))
            matrix = _least_squares_matrix(k, v)
            recalled_rows.append(query[b] @ matrix)
        recalled = torch.stack(recalled_rows, dim=0)
        logits = _decode_recall(model, recalled)
        legal = logits[:, VALUE_START : VALUE_START + N_VALUES]
        predictions.append((legal.argmax(dim=-1) + VALUE_START).cpu())

        target = batch.query_targets[:, query_index].to(logits.device)
        stale = batch.stale_targets[:, query_index].to(logits.device)
        overwrite = batch.overwrite_mask[:, query_index].to(logits.device)
        safe_stale = torch.where(overwrite, stale, target)
        row = torch.arange(logits.size(0), device=logits.device)
        margins.append((logits[row, target] - logits[row, safe_stale]).cpu())
        query_index += 1

    if query_index != batch.query_targets.size(1):
        raise RuntimeError("least-squares query count mismatch")
    return _group_raw_metrics(
        torch.stack(predictions, dim=1),
        batch,
        torch.stack(margins, dim=1),
    )


def _effective_rank(singular_values: torch.Tensor) -> int:
    if singular_values.numel() == 0:
        return 0
    threshold = float(singular_values.max()) * 1e-4
    return int((singular_values > threshold).sum())


def _cosine_offdiag_stats(x: torch.Tensor) -> tuple[float, float]:
    x = F.normalize(x.float(), dim=-1)
    gram = x @ x.transpose(0, 1)
    mask = ~torch.eye(gram.size(0), dtype=torch.bool, device=gram.device)
    off = gram[mask].abs()
    return float(off.mean()), float(off.max())


@torch.no_grad()
def learned_geometry(model: HardwareAwareAERATextLMV21) -> dict[str, Any]:
    model.eval()
    device = model.token_emb.weight.device
    stage0 = model.stages[0]
    memory = stage0.memory

    key_tokens = torch.arange(KEY_START, KEY_START + N_KEYS, device=device)
    key_x = _token_representation(model, key_tokens, 1)
    q = F.normalize(memory.q(key_x), dim=-1)
    k = F.normalize(memory.k(key_x), dim=-1)
    qk = q @ k.transpose(0, 1)
    target = torch.arange(N_KEYS, device=device)
    diag = qk.diag()
    mask = ~torch.eye(N_KEYS, dtype=torch.bool, device=device)
    best_other = qk.masked_fill(~mask, -torch.inf).max(dim=1).values
    k_s = torch.linalg.svdvals(k.float())
    nonzero = k_s[k_s > float(k_s.max()) * 1e-4]
    k_condition = float(k_s.max() / nonzero.min()) if nonzero.numel() else float("inf")
    kk_mean, kk_max = _cosine_offdiag_stats(k)

    value_tokens = torch.arange(VALUE_START, VALUE_START + N_VALUES, device=device)
    value_x = _token_representation(model, value_tokens, 2)
    latent_v = torch.tanh(memory.v(value_x))
    payload_code = memory.out(latent_v)
    latent_s = torch.linalg.svdvals(latent_v.float())
    payload_s = torch.linalg.svdvals(payload_code.float())
    latent_mean, latent_max = _cosine_offdiag_stats(latent_v)
    payload_mean, payload_max = _cosine_offdiag_stats(payload_code)

    return {
        "q_to_k_top1_accuracy": float((qk.argmax(dim=-1) == target).float().mean()),
        "qk_diagonal_cosine_mean": float(diag.mean()),
        "qk_best_other_cosine_mean": float(best_other.mean()),
        "qk_diag_minus_best_other_margin_mean": float((diag - best_other).mean()),
        "k_k_abs_offdiag_cosine_mean": kk_mean,
        "k_k_abs_offdiag_cosine_max": kk_max,
        "k_singular_values": [float(x) for x in k_s],
        "k_effective_rank": _effective_rank(k_s),
        "k_condition_number_effective": k_condition,
        "latent_payload_abs_offdiag_cosine_mean": latent_mean,
        "latent_payload_abs_offdiag_cosine_max": latent_max,
        "latent_payload_singular_values": [float(x) for x in latent_s],
        "latent_payload_effective_rank": _effective_rank(latent_s),
        "decoded_payload_abs_offdiag_cosine_mean": payload_mean,
        "decoded_payload_abs_offdiag_cosine_max": payload_max,
        "decoded_payload_singular_values": [float(x) for x in payload_s],
        "decoded_payload_effective_rank": _effective_rank(payload_s),
    }


def _raw_pass(row: dict[str, Any]) -> bool:
    return bool(
        row["overall_accuracy"] >= RAW_RECALL_MIN
        and row["overwrite_current_value_accuracy"] >= OVERWRITE_ACCURACY_MIN
        and row["overwrite_stale_value_error"] <= STALE_ERROR_MAX
    )


def run_audit() -> dict[str, Any]:
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    full, _, training = train_pair_with_conflict_free_objective()
    eval_batch = make_batch(24, EVAL_SEED)

    strict_reference = _evaluate_mode(
        full,
        eval_batch,
        write_chunks_only=True,
        keep_candidate_one=True,
        memory_lr=DIAGNOSTIC_MEMORY_LR,
    )
    sequential = {
        "q_decay_0_999": sequential_raw_recall(
            full, eval_batch, read="q", decay=PRODUCTION_DIAGNOSTIC_DECAY
        ),
        "k_decay_0_999": sequential_raw_recall(
            full, eval_batch, read="k", decay=PRODUCTION_DIAGNOSTIC_DECAY
        ),
        "q_decay_1_0": sequential_raw_recall(
            full, eval_batch, read="q", decay=NO_DECAY
        ),
        "k_decay_1_0": sequential_raw_recall(
            full, eval_batch, read="k", decay=NO_DECAY
        ),
    }
    baseline = sequential["q_decay_0_999"]
    prior = strict_reference["raw_memory_decode"]
    if abs(baseline["overall_accuracy"] - prior["overall_accuracy"]) > 1e-7:
        raise RuntimeError("new sequential q-read baseline changed #316 strict raw recall")
    if abs(
        baseline["overwrite_current_value_accuracy"]
        - prior["overwrite_current_value_accuracy"]
    ) > 1e-7:
        raise RuntimeError("new sequential baseline changed #316 overwrite raw recall")

    least_squares = {
        "q_read": least_squares_raw_recall(full, eval_batch, read="q"),
        "k_read": least_squares_raw_recall(full, eval_batch, read="k"),
    }
    geometry = learned_geometry(full)

    findings = {
        "decay_attenuation_sufficient": (
            not _raw_pass(sequential["q_decay_0_999"])
            and _raw_pass(sequential["q_decay_1_0"])
        ),
        "qk_read_mismatch_sufficient_at_decay_0_999": (
            not _raw_pass(sequential["q_decay_0_999"])
            and _raw_pass(sequential["k_decay_0_999"])
        ),
        "qk_read_mismatch_sufficient_at_decay_1_0": (
            not _raw_pass(sequential["q_decay_1_0"])
            and _raw_pass(sequential["k_decay_1_0"])
        ),
        "least_squares_q_read_passes": _raw_pass(least_squares["q_read"]),
        "sequential_superposition_or_update_dynamics_remain": (
            _raw_pass(least_squares["q_read"])
            and not _raw_pass(sequential["q_decay_0_999"])
            and not _raw_pass(sequential["q_decay_1_0"])
        ),
        "least_squares_localizes_qk_geometry": (
            not _raw_pass(least_squares["q_read"])
            and _raw_pass(least_squares["k_read"])
        ),
        "even_k_read_least_squares_below_raw_gate": not _raw_pass(
            least_squares["k_read"]
        ),
    }

    return {
        "scope": "aera_v21_post_316_residual_raw_recall_localization_cpu",
        "diagnostic_reproduction_only": True,
        "independent_evidence": False,
        "training_protocol": {
            "source": "exact_313_conflict_free_training",
            "steps": 500,
            "lr": 4e-3,
            "final_local_code": training["final_local_code"],
        },
        "strict_316_reconfirmation": {
            "final": strict_reference["final"],
            "raw_memory_decode": strict_reference["raw_memory_decode"],
        },
        "geometry": geometry,
        "sequential_counterfactuals": sequential,
        "least_squares_upper_bound": least_squares,
        "thresholds": {
            "raw_recall_min": RAW_RECALL_MIN,
            "overwrite_accuracy_min": OVERWRITE_ACCURACY_MIN,
            "stale_error_max": STALE_ERROR_MAX,
            "diagnostic_memory_lr": DIAGNOSTIC_MEMORY_LR,
            "production_diagnostic_decay": PRODUCTION_DIAGNOSTIC_DECAY,
            "no_decay": NO_DECAY,
        },
        "findings": findings,
        "claims": {
            "production_change_authorized": False,
            "gpu_authorized": False,
            "real_language_run_authorized": False,
            "v22_authorized": False,
            "architecture_freeze_authorized": False,
            "100m_authorized": False,
            "breakthrough_proven": False,
        },
    }


def main() -> None:
    result = run_audit()
    print(
        "AERA_V21_RESIDUAL_RAW_RECALL_AUDIT_RESULT_JSON="
        + json.dumps(result, sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
