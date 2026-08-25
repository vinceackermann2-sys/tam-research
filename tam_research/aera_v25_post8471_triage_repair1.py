from __future__ import annotations

"""Bounded-memory implementation repair for frozen issue #369 diagnostic.

Preregistered by #372 after #371 failed before any metric due CUDA OOM.
This module changes evaluation plumbing only: models are constructed/loaded as
ordinary tensors, all diagnostic measurements run under inference mode, and
full-vocabulary CE reductions are accumulated over fixed 32-token time slices.
The v25 architecture, source checkpoints, data, samples, alphas, masks,
statistics, thresholds, and interpretations remain frozen by #369.
"""

from contextlib import nullcontext
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from . import aera_v25_post8471_triage as base

REPAIR_ISSUE = 372
SOURCE_FAILED_TRIGGER = 371
SOURCE_FAILED_ACTIONS_RUN = 32856397733
LOSS_TIME_SLICE = 32


def repair_protocol() -> dict[str, Any]:
    protocol = dict(base.frozen_protocol())
    protocol.update(
        {
            "repair_issue": REPAIR_ISSUE,
            "source_failed_trigger": SOURCE_FAILED_TRIGGER,
            "source_failed_actions_run": SOURCE_FAILED_ACTIONS_RUN,
            "semantic_change": "inference_only_plus_fixed_32_token_ce_slices",
            "loss_time_slice_tokens": LOSS_TIME_SLICE,
            "source_checkpoint_seed": base.SOURCE_SEED,
            "training_performed": False,
            "checkpoint_mutation_authorized": False,
            "scientific_protocol_changed": False,
        }
    )
    return protocol


def sliced_token_nll(
    logits: torch.Tensor,
    y: torch.Tensor,
    *,
    start: int = 0,
    end: int | None = None,
    slice_tokens: int = LOSS_TIME_SLICE,
) -> torch.Tensor:
    """Exact token NLL in bounded time slices, returned on CPU in original order."""
    if logits.ndim != 3 or y.ndim != 2 or logits.shape[:2] != y.shape:
        raise ValueError("logits/y must be [batch,time,vocab] and [batch,time]")
    if slice_tokens != LOSS_TIME_SLICE:
        raise ValueError(f"repair1 slice size is frozen to {LOSS_TIME_SLICE}")
    stop = y.size(1) if end is None else int(end)
    if not (0 <= start < stop <= y.size(1)):
        raise ValueError("invalid token-loss range")

    rows: list[torch.Tensor] = []
    for left in range(int(start), stop, LOSS_TIME_SLICE):
        right = min(left + LOSS_TIME_SLICE, stop)
        local_logits = logits[:, left:right].float()
        local_y = y[:, left:right]
        local = F.cross_entropy(
            local_logits.reshape(-1, local_logits.size(-1)),
            local_y.reshape(-1),
            reduction="none",
        ).reshape(y.size(0), right - left)
        rows.append(local.detach().cpu())
        del local, local_logits, local_y
    return torch.cat(rows, dim=1)


def sliced_per_example_nll(
    logits: torch.Tensor,
    y: torch.Tensor,
    *,
    start: int = 0,
    end: int | None = None,
) -> torch.Tensor:
    return sliced_token_nll(logits, y, start=start, end=end).mean(dim=1)


def sliced_chunk_mean_nll(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Matched-Transformer per-chunk difficulty, exact original ordering."""
    if y.size(1) != base.SEQ_LEN or base.SEQ_LEN % base.CHUNK_SIZE:
        raise ValueError("repair1 requires exact two-chunk real-language geometry")
    chunks = []
    for start in range(0, base.SEQ_LEN, base.CHUNK_SIZE):
        chunks.append(
            sliced_per_example_nll(
                logits,
                y,
                start=start,
                end=start + base.CHUNK_SIZE,
            )
        )
    return torch.stack(chunks, dim=1)


def bounded_condition_losses(
    model,
    batches: Sequence[tuple[torch.Tensor, torch.Tensor]],
    *,
    update_memory: bool,
    read_scales: float | Sequence[float] | None,
) -> dict[str, torch.Tensor]:
    overall: list[torch.Tensor] = []
    second: list[torch.Tensor] = []
    context = (
        base.scaled_ficem_reads(model, read_scales)
        if read_scales is not None
        else nullcontext()
    )
    with context:
        for x, y in batches:
            output = base._model_forward(model, x, update_memory=update_memory)
            logits = output.get("logits")
            if not isinstance(logits, torch.Tensor):
                raise RuntimeError("AERA output missing logits")
            overall.append(sliced_per_example_nll(logits, y))
            second.append(
                sliced_per_example_nll(logits, y, start=base.CHUNK_SIZE)
            )
            del logits, output
    return {
        "overall": torch.cat(overall),
        "second_chunk": torch.cat(second),
    }


def bounded_adaptivity_diagnostic(
    model,
    transformer,
    *,
    data_dir: str,
    device: torch.device,
) -> dict[str, Any]:
    versions_before = base.parameter_versions(model)
    batches = base._fixed_batches(
        data_dir=data_dir,
        batches=base.ADAPTIVITY_BATCHES,
        batch_size=base.ADAPTIVITY_BATCH_SIZE,
        seed=base.DIAGNOSTIC_SEED + 1,
        device=device,
    )
    difficulty_rows: list[torch.Tensor] = []
    position_rows: list[torch.Tensor] = []
    enabled_compute: list[torch.Tensor] = []
    enabled_stages: list[torch.Tensor] = []
    disabled_compute: list[torch.Tensor] = []
    disabled_stages: list[torch.Tensor] = []

    for x, y in batches:
        with base._autocast(device):
            transformer_logits = transformer(x)
        chunk_difficulty = sliced_chunk_mean_nll(transformer_logits, y)
        del transformer_logits

        enabled = base._model_forward(model, x, update_memory=True)
        disabled = base._model_forward(model, x, update_memory=False)
        enabled_counts, enabled_matrix = base._route_arrays(
            enabled, base.ADAPTIVITY_BATCH_SIZE
        )
        disabled_counts, disabled_matrix = base._route_arrays(
            disabled, base.ADAPTIVITY_BATCH_SIZE
        )
        del enabled, disabled

        for chunk_index in range(base.SEQ_LEN // base.CHUNK_SIZE):
            difficulty_rows.append(chunk_difficulty[:, chunk_index])
            position_rows.append(
                torch.full(
                    (base.ADAPTIVITY_BATCH_SIZE,),
                    chunk_index,
                    dtype=torch.long,
                )
            )
            enabled_compute.append(enabled_counts[chunk_index].detach().cpu())
            enabled_stages.append(enabled_matrix[chunk_index].detach().cpu())
            disabled_compute.append(disabled_counts[chunk_index].detach().cpu())
            disabled_stages.append(disabled_matrix[chunk_index].detach().cpu())
        del chunk_difficulty

    difficulty = torch.cat(difficulty_rows)
    positions = torch.cat(position_rows)
    if difficulty.numel() != base.EXPECTED_CHUNKS:
        raise RuntimeError(
            f"unexpected adaptivity sample count {difficulty.numel()} != {base.EXPECTED_CHUNKS}"
        )
    enabled_result = base.adaptivity_summary(
        difficulty,
        torch.cat(enabled_compute),
        positions,
        torch.cat(enabled_stages),
        bootstrap_seed=base.DIAGNOSTIC_SEED + 101,
    )
    disabled_result = base.adaptivity_summary(
        difficulty,
        torch.cat(disabled_compute),
        positions,
        torch.cat(disabled_stages),
        bootstrap_seed=base.DIAGNOSTIC_SEED + 102,
    )
    versions_after = base.parameter_versions(model)
    return {
        "difficulty_definition": "matched Transformer per-chunk held-out next-token NLL",
        "memory_enabled_hard_sparse": enabled_result,
        "memory_disabled_hard_sparse": disabled_result,
        "parameter_versions_unchanged": versions_before == versions_after,
    }


def run_checkpoint_triage_repair1(*, data_dir: str, run_dir: str) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("issue372 repair1 checkpoint triage requires one CUDA L4")
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    base.validate_production_data(data_dir)

    # Construct + strict-load OUTSIDE inference mode so parameters remain ordinary
    # tensors and the existing parameter-version nonmutation audit stays valid.
    aera, transformer = base._load_models(run_dir=run_dir, device=device)
    versions_initial = base.parameter_versions(aera)

    original_condition_losses = base._condition_losses
    try:
        base._condition_losses = bounded_condition_losses
        with torch.inference_mode():
            memory = base.memory_diagnostic(aera, data_dir=data_dir, device=device)
            if not memory["parameter_versions_unchanged"]:
                raise RuntimeError("memory diagnostic changed parameter versions")
            torch.cuda.empty_cache()

            adaptivity = bounded_adaptivity_diagnostic(
                aera,
                transformer,
                data_dir=data_dir,
                device=device,
            )
            if not adaptivity["parameter_versions_unchanged"]:
                raise RuntimeError("adaptivity diagnostic changed parameter versions")
            torch.cuda.empty_cache()

            systems = base.systems_diagnostic(aera, transformer, device=device)
            if not systems["parameter_versions_unchanged"]:
                raise RuntimeError("systems diagnostic changed parameter versions")
    finally:
        base._condition_losses = original_condition_losses

    versions_final = base.parameter_versions(aera)
    recommendations = base._recommendations(memory, adaptivity, systems)
    return {
        "scope": "aera_v25_post_seed8471_checkpoint_only_triage_issue369_repair1",
        "protocol": repair_protocol(),
        "gpu": torch.cuda.get_device_name(device),
        "source_checkpoint": {
            "seed": base.SOURCE_SEED,
            "run_dir": run_dir,
            "strict_load": True,
            "parameter_versions_unchanged": versions_initial == versions_final,
        },
        "memory_diagnostic": memory,
        "adaptivity_diagnostic": adaptivity,
        "systems_diagnostic": systems,
        "recommendations": recommendations,
        "repair1": {
            "repair_issue": REPAIR_ISSUE,
            "source_failed_trigger": SOURCE_FAILED_TRIGGER,
            "source_failed_actions_run": SOURCE_FAILED_ACTIONS_RUN,
            "loss_time_slice_tokens": LOSS_TIME_SLICE,
            "models_constructed_outside_inference_mode": True,
            "diagnostic_families_run_under_inference_mode": True,
        },
        "training_performed": False,
        "checkpoint_mutated": False,
        "optimizer_created": False,
        "backward_performed": False,
        "new_model_checkpoint_written": False,
        "primary_issue368_result_changed": False,
        "claims": {
            "v25_pass": False,
            "architecture_freeze_authorized": False,
            "s2_authorized": False,
            "independent_replication_credit": False,
            "100m_authorized": False,
            "breakthrough_proven": False,
        },
    }


def source_result_path(run_dir: str) -> Path:
    return Path(run_dir) / "result.json"
