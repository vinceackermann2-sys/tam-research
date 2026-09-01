from __future__ import annotations

"""Localization-only diagnostic for the exhausted #418 BF16 correctness FAIL.

This module reconstructs the exact fifth ordinary #411 fixture
(`bfloat16_batch8_mixed`) from DESIGN_SEED=408411 and reports correctness
subgates.  It contains no timing, profiling, training, checkpoint, corpus, or
scientific-seed path and does not modify the v26.3 candidate backend.
"""

from typing import Any

import torch

from .aera_hardware_core_v26 import TorchFICEMReferenceBackend
from .aera_hardware_core_v26_3_ficem_read_triton import TritonFICEMReadBackend
from . import aera_v26_3_ficem_read_probe as probe

RESEARCH_ISSUE = 423
SOURCE_MAIN = "7a10d91258f7daa0247369554533e3b2f6445bad"
SOURCE_FAILED_ISSUE = 418
SOURCE_FAILED_TRIGGER = 422
SOURCE_FAILED_ACTIONS_RUN = 33499743719
TARGET_ROW = "bfloat16_batch8_mixed"
TARGET_DTYPE = "bfloat16"
TARGET_BATCH = 8
TARGET_VALIDITY = "mixed"
TARGET_ORDINAL = 5
MAX_COORDINATES = 16


def cpu_contract_preflight() -> dict[str, Any]:
    if probe.DESIGN_SEED != 408_411:
        raise RuntimeError("#423 requires the original #411 design seed")
    if (probe.BF16_ATOL, probe.BF16_RTOL) != (1e-2, 1e-2):
        raise RuntimeError("#423 BF16 tolerance drifted")
    if probe.DTYPE_NAMES != ("float32", "bfloat16"):
        raise RuntimeError("#423 dtype ordering drifted")
    if probe.BATCH_SIZES != (8, 64):
        raise RuntimeError("#423 batch ordering drifted")
    if probe.VALIDITY_KINDS != ("mixed", "full"):
        raise RuntimeError("#423 validity ordering drifted")
    ordinary_rows = [
        probe._row_key(dtype_name, batch_size, validity_kind)
        for dtype_name in probe.DTYPE_NAMES
        for batch_size in probe.BATCH_SIZES
        for validity_kind in probe.VALIDITY_KINDS
    ]
    if ordinary_rows[TARGET_ORDINAL - 1] != TARGET_ROW:
        raise RuntimeError("#423 target ordinal no longer reproduces failed row")
    return {
        "research_issue": RESEARCH_ISSUE,
        "source_main": SOURCE_MAIN,
        "source_failed_issue": SOURCE_FAILED_ISSUE,
        "source_failed_trigger": SOURCE_FAILED_TRIGGER,
        "source_failed_actions_run": SOURCE_FAILED_ACTIONS_RUN,
        "target_row": TARGET_ROW,
        "target_ordinal": TARGET_ORDINAL,
        "design_seed": probe.DESIGN_SEED,
        "design_seed_is_scientific_seed": False,
        "bf16_atol": probe.BF16_ATOL,
        "bf16_rtol": probe.BF16_RTOL,
        "original_global_case_order_preserved": True,
        "resampling": False,
        "rejection_sampling": False,
        "fixture_nudging": False,
        "timing_authorized": False,
        "profiling_authorized": False,
        "performance_decision_authorized": False,
        "model_loaded": False,
        "checkpoint_loaded": False,
        "corpus_accessed": False,
        "training_performed": False,
        "optimizer_created": False,
        "backward_performed": False,
        "scientific_seed_consumed": False,
        "architecture_freeze_authorized": False,
        "s2_authorized": False,
        "100m_authorized": False,
        "breakthrough_proven": False,
    }


def reconstruct_failed_case(memory, device: torch.device):
    """Replay the exact #411 ordinary-case generator through ordinal five."""
    generator = torch.Generator().manual_seed(probe.DESIGN_SEED)
    ordinal = 0
    for dtype_name in probe.DTYPE_NAMES:
        for batch_size in probe.BATCH_SIZES:
            for validity_kind in probe.VALIDITY_KINDS:
                ordinal += 1
                case = probe.make_case(
                    dtype_name=dtype_name,
                    batch_size=batch_size,
                    validity_kind=validity_kind,
                    generator=generator,
                    device=device,
                )
                key = probe._row_key(dtype_name, batch_size, validity_kind)
                if ordinal == TARGET_ORDINAL:
                    if key != TARGET_ROW:
                        raise RuntimeError("#423 replay reached wrong target row")
                    return case
    raise RuntimeError("#423 failed to reconstruct target row")


def _error_stats(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    ref = reference.float()
    cand = candidate.float()
    diff = (cand - ref).abs()
    allowed = atol + rtol * ref.abs()
    failing = diff > allowed
    nonzero = ref.abs() > 0
    if bool(nonzero.any()):
        max_relative = float((diff[nonzero] / ref.abs()[nonzero]).max())
    else:
        max_relative = 0.0
    fail_count = int(failing.sum())
    total = int(failing.numel())
    return {
        "allclose": bool(torch.allclose(reference, candidate, atol=atol, rtol=rtol)),
        "atol": atol,
        "rtol": rtol,
        "max_abs_error": float(diff.max()),
        "mean_abs_error": float(diff.mean()),
        "failing_element_count": fail_count,
        "failing_element_fraction": fail_count / total if total else 0.0,
        "max_relative_error_nonzero_reference": max_relative,
    }


def _selection_masks(
    masked_logits: torch.Tensor,
    valid: torch.Tensor,
    reference_indices: torch.Tensor,
    candidate_indices: torch.Tensor,
) -> dict[str, torch.Tensor]:
    capacity = int(masked_logits.size(-1))
    candidate_long = candidate_indices.to(torch.long)
    reference_long = reference_indices.to(torch.long)
    candidate_in_range = (candidate_long >= 0) & (candidate_long < capacity)
    safe_candidate = candidate_long.clamp(0, capacity - 1)
    raw_set_equal = torch.all(
        torch.sort(reference_long, dim=-1).values
        == torch.sort(safe_candidate, dim=-1).values,
        dim=-1,
    ) & torch.all(candidate_in_range, dim=-1)

    boundary_values = torch.topk(
        masked_logits, k=probe.READ_TOP_K + 1, dim=-1
    ).values
    cutoff = boundary_values[..., probe.READ_TOP_K - 1]
    fifth = boundary_values[..., probe.READ_TOP_K]
    distinct = cutoff != fifth
    tied = ~distinct

    sorted_candidate = torch.sort(safe_candidate, dim=-1).values
    unique = torch.all(
        sorted_candidate[..., 1:] != sorted_candidate[..., :-1], dim=-1
    )
    expanded_valid = valid[:, None, :].expand(-1, masked_logits.size(1), -1)
    selected_valid = torch.all(expanded_valid.gather(-1, safe_candidate), dim=-1)
    selected_logits = masked_logits.gather(-1, safe_candidate)
    selected_meet_cutoff = torch.all(
        selected_logits >= cutoff.unsqueeze(-1), dim=-1
    )
    slot_ids = torch.arange(capacity, device=masked_logits.device, dtype=torch.long)
    membership = torch.any(
        safe_candidate.unsqueeze(-1) == slot_ids.view(1, 1, 1, capacity), dim=-2
    )
    strictly_above = masked_logits > cutoff.unsqueeze(-1)
    strict_above_included = torch.all(~strictly_above | membership, dim=-1)
    tied_semantically_valid = (
        torch.all(candidate_in_range, dim=-1)
        & unique
        & selected_valid
        & selected_meet_cutoff
        & strict_above_included
    )
    return {
        "safe_candidate": safe_candidate,
        "reference_long": reference_long,
        "raw_set_equal": raw_set_equal,
        "cutoff": cutoff,
        "fifth": fifth,
        "distinct": distinct,
        "tied": tied,
        "unique": unique,
        "selected_valid": selected_valid,
        "selected_meet_cutoff": selected_meet_cutoff,
        "strict_above_included": strict_above_included,
        "tied_semantically_valid": tied_semantically_valid,
    }


def _coordinate_records(
    mask: torch.Tensor,
    masks: dict[str, torch.Tensor],
    *,
    tied: bool,
) -> list[dict[str, Any]]:
    coordinates = torch.nonzero(mask, as_tuple=False)[:MAX_COORDINATES]
    records: list[dict[str, Any]] = []
    for coordinate in coordinates:
        batch_index = int(coordinate[0])
        time_index = int(coordinate[1])
        record: dict[str, Any] = {
            "batch": batch_index,
            "time": time_index,
            "reference_top4": [
                int(x)
                for x in masks["reference_long"][batch_index, time_index].tolist()
            ],
            "candidate_top4": [
                int(x)
                for x in masks["safe_candidate"][batch_index, time_index].tolist()
            ],
            "fourth_cutoff": float(masks["cutoff"][batch_index, time_index]),
            "fifth_value": float(masks["fifth"][batch_index, time_index]),
            "fourth_minus_fifth_gap": float(
                masks["cutoff"][batch_index, time_index]
                - masks["fifth"][batch_index, time_index]
            ),
            "raw_selected_set_equal": bool(
                masks["raw_set_equal"][batch_index, time_index]
            ),
        }
        if tied:
            record.update(
                {
                    "candidate_unique": bool(
                        masks["unique"][batch_index, time_index]
                    ),
                    "candidate_selected_valid": bool(
                        masks["selected_valid"][batch_index, time_index]
                    ),
                    "candidate_selected_meet_cutoff": bool(
                        masks["selected_meet_cutoff"][batch_index, time_index]
                    ),
                    "strict_above_included": bool(
                        masks["strict_above_included"][batch_index, time_index]
                    ),
                    "tied_selection_semantically_valid": bool(
                        masks["tied_semantically_valid"][batch_index, time_index]
                    ),
                }
            )
        records.append(record)
    return records


def run_localization() -> dict[str, Any]:
    contract = cpu_contract_preflight()
    if not torch.cuda.is_available():
        raise RuntimeError("#423 localization requires CUDA")
    device = torch.device("cuda")
    device_name = torch.cuda.get_device_name(0)
    if "L4" not in device_name.upper():
        raise RuntimeError(f"#423 requires NVIDIA L4, found {device_name}")

    memory = probe.build_memory(device)
    reference = TorchFICEMReferenceBackend()
    candidate = TritonFICEMReadBackend()
    case = reconstruct_failed_case(memory, device)

    identity_before = case.identity.clone()
    context_before = case.context.clone()
    state_before = probe._clone_state(case.state)

    reference_result = probe._full_read(reference, memory, case)
    candidate_result = probe._full_read(candidate, memory, case)
    torch.cuda.synchronize()

    query, keys, similarity = probe._diagnostic_tail_inputs(memory, case)
    with torch.no_grad():
        reference_recalled, reference_indices = probe._reference_tail(
            similarity, case.state
        )
        candidate_recalled, candidate_indices = probe.fused_ficem_read_tail(
            similarity,
            case.state.strengths,
            case.state.valid,
            case.state.values,
            return_top_indices=True,
        )
    torch.cuda.synchronize()
    if candidate_indices is None:
        raise RuntimeError("#423 candidate diagnostic indices missing")

    masked_logits = probe._reference_masked_logits(similarity, case.state)
    selection = probe._tie_aware_top4_equivalence(
        masked_logits,
        case.state.valid,
        reference_indices,
        candidate_indices,
    )
    masks = _selection_masks(
        masked_logits, case.state.valid, reference_indices, candidate_indices
    )

    distinct_mismatch = masks["distinct"] & ~masks["raw_set_equal"]
    tied_mask = masks["tied"]
    distinct_mismatch_count = int(distinct_mismatch.sum())
    tied_count = int(tied_mask.sum())

    atol, rtol = probe.BF16_ATOL, probe.BF16_RTOL
    pre_out = _error_stats(
        reference_recalled, candidate_recalled, atol=atol, rtol=rtol
    )
    final_out = _error_stats(
        reference_result.recalled,
        candidate_result.recalled,
        atol=atol,
        rtol=rtol,
    )
    reuse_exact = {
        "projected_query": bool(
            reference_result.projected_query is not None
            and candidate_result.projected_query is not None
            and torch.equal(
                reference_result.projected_query, candidate_result.projected_query
            )
            and torch.equal(reference_result.projected_query, query)
        ),
        "normalized_old_keys": bool(
            reference_result.normalized_old_keys is not None
            and candidate_result.normalized_old_keys is not None
            and torch.equal(
                reference_result.normalized_old_keys,
                candidate_result.normalized_old_keys,
            )
            and torch.equal(reference_result.normalized_old_keys, keys)
        ),
    }
    source_unchanged = bool(
        torch.equal(case.identity, identity_before)
        and torch.equal(case.context, context_before)
        and probe._state_equal(case.state, state_before)
    )
    finite = bool(
        torch.isfinite(reference_recalled).all()
        and torch.isfinite(candidate_recalled).all()
        and torch.isfinite(reference_result.recalled).all()
        and torch.isfinite(candidate_result.recalled).all()
    )
    dtype_device_shape_exact = bool(
        reference_recalled.shape == candidate_recalled.shape
        and reference_recalled.dtype == candidate_recalled.dtype
        and reference_recalled.device == candidate_recalled.device
        and reference_result.recalled.shape == candidate_result.recalled.shape
        and reference_result.recalled.dtype == candidate_result.recalled.dtype
        and reference_result.recalled.device == candidate_result.recalled.device
        and candidate_result.recalled.device.type == "cuda"
    )

    subgates = {
        "selection_semantically_equivalent": selection[
            "selection_semantically_equivalent"
        ],
        "pre_out_recalled_allclose": pre_out["allclose"],
        "final_out_allclose": final_out["allclose"],
        "projected_query_exact": reuse_exact["projected_query"],
        "normalized_old_keys_exact": reuse_exact["normalized_old_keys"],
        "source_unchanged": source_unchanged,
        "finite": finite,
        "dtype_device_shape_exact": dtype_device_shape_exact,
    }
    failed_subgates = [name for name, passed in subgates.items() if not passed]

    boundary_values = torch.topk(
        masked_logits, k=probe.READ_TOP_K + 1, dim=-1
    ).values
    gaps = (
        boundary_values[..., probe.READ_TOP_K - 1].float()
        - boundary_values[..., probe.READ_TOP_K].float()
    )

    return {
        "contract": contract,
        "device": device_name,
        "target_row": TARGET_ROW,
        "target_ordinal": TARGET_ORDINAL,
        "aggregate_correctness_pass": len(failed_subgates) == 0,
        "failed_subgates": failed_subgates,
        "subgates": subgates,
        "selection": selection,
        "distinct_query_mismatch_count": distinct_mismatch_count,
        "distinct_query_mismatch_examples": _coordinate_records(
            distinct_mismatch, masks, tied=False
        ),
        "tied_query_count": tied_count,
        "tied_query_examples": _coordinate_records(tied_mask, masks, tied=True),
        "pre_out_recalled_error": pre_out,
        "final_out_error": final_out,
        "reference_boundary_diagnostics": {
            "minimum_fourth_minus_fifth_gap": float(gaps.min()),
            "maximum_fourth_minus_fifth_gap": float(gaps.max()),
            "mean_fourth_minus_fifth_gap": float(gaps.mean()),
            "tie_query_count": tied_count,
            "total_query_count": int(gaps.numel()),
        },
        "localization_only": True,
        "timing_performed": False,
        "profiling_performed": False,
        "performance_decision": None,
        "candidate_backend_modified": False,
        "model_loaded": False,
        "checkpoint_loaded": False,
        "corpus_accessed": False,
        "training_performed": False,
        "optimizer_created": False,
        "backward_performed": False,
        "scientific_seed_consumed": False,
        "architecture_freeze_authorized": False,
        "s2_authorized": False,
        "100m_authorized": False,
        "breakthrough_proven": False,
    }
