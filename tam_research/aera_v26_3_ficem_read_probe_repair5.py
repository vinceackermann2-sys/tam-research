from __future__ import annotations

"""Repair5 diagnostic successor for the frozen v26.3 FICEM read probe.

The historical issue411/#418 probe remains byte-for-byte frozen.  This module
reuses its fixtures, constants, timing/profiling helpers, and acceptance equations,
but evaluates the three standalone diagnostic reference operations under the same
precision context as the real full read.  This is required for repair5 because the
actual inherited BF16 reference executes inside CUDA BF16 autocast and returns an
FP32 recalled pre-out, as localized by #456/#460.

Issue #477 is CPU-first and authorizes no GPU execution by itself.
"""

import math
from typing import Any

import torch

from . import aera_v26_3_ficem_read_probe as frozen


RESEARCH_ISSUE = 477
HISTORICAL_PROBE_GIT_BLOB = "16aa99b9f6f0a1d11bd7bf5f36b2f0b1fb97047b"
REPAIR5_BACKEND_GIT_BLOB = "263f68eb1186a8ac14a08fc4b4df1fc5b292c711"
SOURCE_MAIN = "2debbdb93e97ef8cc646f9730b83d61d3dcbda1a"
SOURCE_FAILED_GATE = 474
SOURCE_FAILED_TRIGGER = 476
SOURCE_FAILED_RUN = 33608906596
SOURCE_FAILED_JOB = 100179200965

# Freeze the original public synthetic contract by direct alias, not copied values.
DESIGN_SEED = frozen.DESIGN_SEED
D_MODEL = frozen.D_MODEL
MEMORY_DIM = frozen.MEMORY_DIM
CAPACITY = frozen.CAPACITY
TIME = frozen.TIME
BATCH_SIZES = frozen.BATCH_SIZES
DTYPE_NAMES = frozen.DTYPE_NAMES
VALIDITY_KINDS = frozen.VALIDITY_KINDS
WARMUP_CALLS = frozen.WARMUP_CALLS
TIMED_ROUNDS = frozen.TIMED_ROUNDS
CALLS_PER_ROUND = frozen.CALLS_PER_ROUND
FP32_ATOL = frozen.FP32_ATOL
FP32_RTOL = frozen.FP32_RTOL
BF16_ATOL = frozen.BF16_ATOL
BF16_RTOL = frozen.BF16_RTOL
MAX_GEOMEAN_LATENCY_RATIO = frozen.MAX_GEOMEAN_LATENCY_RATIO
MAX_ROW_LATENCY_RATIO = frozen.MAX_ROW_LATENCY_RATIO
MAX_FULL_EVENT_RATIO = frozen.MAX_FULL_EVENT_RATIO


def issue477_protocol() -> dict[str, Any]:
    protocol = dict(frozen.issue411_protocol())
    protocol.update(
        {
            "probe_successor_issue": RESEARCH_ISSUE,
            "source_main_issue477": SOURCE_MAIN,
            "historical_probe_git_blob": HISTORICAL_PROBE_GIT_BLOB,
            "repair5_backend_git_blob": REPAIR5_BACKEND_GIT_BLOB,
            "source_failed_gate": SOURCE_FAILED_GATE,
            "source_failed_trigger": SOURCE_FAILED_TRIGGER,
            "source_failed_run": SOURCE_FAILED_RUN,
            "source_failed_job": SOURCE_FAILED_JOB,
            "diagnostic_reference_precision_context_corrected": True,
            "historical_probe_modified": False,
            "candidate_path_changed_by_probe_successor": False,
            "fixtures_changed_by_probe_successor": False,
            "thresholds_changed_by_probe_successor": False,
            "timing_changed_by_probe_successor": False,
            "gpu_authorized_by_issue477": False,
            "scientific_seed_consumed": False,
            "end_to_end_systems_authorized": False,
            "architecture_freeze_authorized": False,
            "s2_authorized": False,
            "fresh_scientific_seed_authorized": False,
            "100m_authorized": False,
            "breakthrough_proven": False,
        }
    )
    return protocol


def cpu_contract_preflight() -> dict[str, Any]:
    historical = frozen.cpu_contract_preflight()
    if DESIGN_SEED != 408_411:
        raise RuntimeError("issue477 design seed drifted")
    if (D_MODEL, MEMORY_DIM, CAPACITY, TIME) != (200, 50, 48, 256):
        raise RuntimeError("issue477 geometry drifted")
    if BATCH_SIZES != (8, 64):
        raise RuntimeError("issue477 batch sizes drifted")
    if DTYPE_NAMES != ("float32", "bfloat16"):
        raise RuntimeError("issue477 dtype order drifted")
    if VALIDITY_KINDS != ("mixed", "full"):
        raise RuntimeError("issue477 validity order drifted")
    if (WARMUP_CALLS, TIMED_ROUNDS, CALLS_PER_ROUND) != (10, 5, 100):
        raise RuntimeError("issue477 timing protocol drifted")
    if (FP32_ATOL, FP32_RTOL, BF16_ATOL, BF16_RTOL) != (
        1e-5,
        1e-5,
        1e-2,
        1e-2,
    ):
        raise RuntimeError("issue477 tolerances drifted")
    if (
        MAX_GEOMEAN_LATENCY_RATIO != 0.90
        or MAX_ROW_LATENCY_RATIO != 1.05
        or MAX_FULL_EVENT_RATIO != 0.75
    ):
        raise RuntimeError("issue477 PASS thresholds drifted")
    return {
        "historical_contract": historical,
        "protocol": issue477_protocol(),
        "gpu_authorized_by_cpu_preflight": False,
        "synthetic_only": True,
        "scientific_seed_consumed": False,
    }


def _reference_tail_in_full_read_context(
    dtype_name: str,
    similarity: torch.Tensor,
    state: frozen.ContextualEpisodicMemoryState,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the historical diagnostic tail in the real full-read context."""
    with torch.no_grad(), frozen._precision_context(dtype_name):
        return frozen._reference_tail(similarity, state)


def _reference_masked_logits_in_full_read_context(
    dtype_name: str,
    similarity: torch.Tensor,
    state: frozen.ContextualEpisodicMemoryState,
) -> torch.Tensor:
    with torch.no_grad(), frozen._precision_context(dtype_name):
        return frozen._reference_masked_logits(similarity, state)


def correctness_row_repair5(
    memory: frozen.CoalescedFICEMMemory,
    case: frozen.ReadCase,
    reference: frozen.TorchFICEMReferenceBackend,
    candidate: frozen.TritonFICEMReadBackend,
) -> dict[str, Any]:
    """Frozen correctness equation with only diagnostic reference context fixed."""
    identity_before = case.identity.clone()
    context_before = case.context.clone()
    state_before = frozen._clone_state(case.state)

    # Production full-read comparison remains exactly the historical path.
    reference_result = frozen._full_read(reference, memory, case)
    candidate_result = frozen._full_read(candidate, memory, case)
    torch.cuda.synchronize()

    query, keys, similarity = frozen._diagnostic_tail_inputs(memory, case)
    reference_recalled, reference_indices = _reference_tail_in_full_read_context(
        case.dtype_name, similarity, case.state
    )
    with torch.no_grad():
        candidate_recalled, candidate_indices = frozen.fused_ficem_read_tail(
            similarity,
            case.state.strengths,
            case.state.valid,
            case.state.values,
            return_top_indices=True,
        )
    torch.cuda.synchronize()
    if candidate_indices is None:
        raise RuntimeError("issue477 candidate did not return diagnostic top indices")

    masked_logits = _reference_masked_logits_in_full_read_context(
        case.dtype_name, similarity, case.state
    )
    selection = frozen._tie_aware_top4_equivalence(
        masked_logits,
        case.state.valid,
        reference_indices,
        candidate_indices,
    )
    atol, rtol = frozen._tolerances(case.dtype_name)
    recalled_close = torch.allclose(
        reference_recalled, candidate_recalled, atol=atol, rtol=rtol
    )
    final_close = torch.allclose(
        reference_result.recalled,
        candidate_result.recalled,
        atol=atol,
        rtol=rtol,
    )
    reuse_exact = (
        reference_result.projected_query is not None
        and candidate_result.projected_query is not None
        and reference_result.normalized_old_keys is not None
        and candidate_result.normalized_old_keys is not None
        and torch.equal(reference_result.projected_query, candidate_result.projected_query)
        and torch.equal(
            reference_result.normalized_old_keys,
            candidate_result.normalized_old_keys,
        )
        and torch.equal(reference_result.projected_query, query)
        and torch.equal(reference_result.normalized_old_keys, keys)
    )
    source_unchanged = (
        torch.equal(case.identity, identity_before)
        and torch.equal(case.context, context_before)
        and frozen._state_equal(case.state, state_before)
    )
    finite = all(
        bool(torch.isfinite(t).all())
        for t in (
            reference_recalled,
            candidate_recalled,
            reference_result.recalled,
            candidate_result.recalled,
        )
    )
    meta_exact = (
        reference_recalled.shape == candidate_recalled.shape
        and reference_recalled.dtype == candidate_recalled.dtype
        and reference_recalled.device == candidate_recalled.device
        and reference_result.recalled.shape == candidate_result.recalled.shape
        and reference_result.recalled.dtype == candidate_result.recalled.dtype
        and reference_result.recalled.device == candidate_result.recalled.device
        and candidate_result.recalled.device.type == "cuda"
    )
    passed = bool(
        selection["selection_semantically_equivalent"]
        and recalled_close
        and final_close
        and reuse_exact
        and source_unchanged
        and finite
        and meta_exact
    )
    return {
        "pass": passed,
        "selected_top4_set_exact": selection["raw_selected_set_equal_all_queries"],
        "selection_semantically_equivalent": selection[
            "selection_semantically_equivalent"
        ],
        "distinct_selected_set_exact": selection["distinct_selected_set_exact"],
        "tied_selection_semantically_valid": selection[
            "tied_selection_semantically_valid"
        ],
        "tie_query_count": selection["tie_query_count"],
        "total_query_count": selection["total_query_count"],
        "tie_query_fraction": selection["tie_query_fraction"],
        "tied_raw_selected_set_match_count": selection[
            "tied_raw_selected_set_match_count"
        ],
        "tied_raw_selected_set_match_fraction": selection[
            "tied_raw_selected_set_match_fraction"
        ],
        "pre_out_recalled_close": recalled_close,
        "final_out_close": final_close,
        "query_and_normalized_keys_bit_exact": reuse_exact,
        "source_unchanged": source_unchanged,
        "finite": finite,
        "dtype_device_shape_exact": meta_exact,
        "atol": atol,
        "rtol": rtol,
        "pre_out_max_abs_diff": float(
            (reference_recalled.float() - candidate_recalled.float()).abs().max()
        ),
        "final_out_max_abs_diff": float(
            (reference_result.recalled.float() - candidate_result.recalled.float()).abs().max()
        ),
    }


def near_tie_correctness_repair5(
    dtype_name: str, device: torch.device
) -> dict[str, Any]:
    """Exact historical near-tie fixture with actual reference precision context."""
    dtype = frozen._dtype_from_name(dtype_name)
    similarity = torch.full(
        (1, 1, CAPACITY), -1.0, dtype=dtype, device=device
    )
    if dtype_name == "float32":
        values = [1.0, 0.999, 0.998, 0.997, 0.996]
    else:
        values = [1.0, 0.98, 0.96, 0.94, 0.92]
    similarity[0, 0, :5] = torch.tensor(values, dtype=dtype, device=device)
    strengths = torch.ones((1, CAPACITY), dtype=dtype, device=device)
    valid = torch.ones((1, CAPACITY), dtype=torch.bool, device=device)
    generator = torch.Generator().manual_seed(
        DESIGN_SEED + (3 if dtype_name == "float32" else 4)
    )
    payload = torch.randn(1, CAPACITY, MEMORY_DIM, generator=generator).to(
        dtype=dtype, device=device
    )
    state = frozen.ContextualEpisodicMemoryState(
        keys=torch.zeros_like(payload),
        values=payload,
        strengths=strengths,
        valid=valid,
    )
    reference_recalled, reference_indices = _reference_tail_in_full_read_context(
        dtype_name, similarity, state
    )
    # Candidate invocation remains the exact fused primitive; no cast/coercion.
    candidate_recalled, candidate_indices = frozen.fused_ficem_read_tail(
        similarity, strengths, valid, payload, return_top_indices=True
    )
    torch.cuda.synchronize()
    if candidate_indices is None:
        raise RuntimeError("issue477 near-tie diagnostic missing indices")
    atol, rtol = frozen._tolerances(dtype_name)
    selected_exact = torch.equal(
        torch.sort(reference_indices, dim=-1).values,
        torch.sort(candidate_indices.to(torch.long), dim=-1).values,
    )
    recalled_close = torch.allclose(
        reference_recalled, candidate_recalled, atol=atol, rtol=rtol
    )
    return {
        "pass": bool(selected_exact and recalled_close),
        "selected_top4_set_exact": bool(selected_exact),
        "recalled_close": bool(recalled_close),
        "reference_dtype": str(reference_recalled.dtype),
        "candidate_dtype": str(candidate_recalled.dtype),
        "dtype_exact": reference_recalled.dtype == candidate_recalled.dtype,
        "max_abs_diff": float(
            (reference_recalled.float() - candidate_recalled.float()).abs().max()
        ),
    }


def run_ficem_read_probe_repair5() -> dict[str, Any]:
    """Run the frozen gate with only the preregistered diagnostic context repair."""
    cpu_contract_preflight()
    if not torch.cuda.is_available():
        raise RuntimeError("issue477 requires CUDA")
    device = torch.device("cuda")
    device_name = torch.cuda.get_device_name(0)
    if "L4" not in device_name.upper():
        raise RuntimeError(f"issue477 requires NVIDIA L4, found {device_name}")

    memory = frozen.build_memory(device)
    reference = frozen.TorchFICEMReferenceBackend()
    candidate = frozen.TritonFICEMReadBackend()
    generator = torch.Generator().manual_seed(DESIGN_SEED)

    near_tie = {
        dtype_name: near_tie_correctness_repair5(dtype_name, device)
        for dtype_name in DTYPE_NAMES
    }
    if not all(row["pass"] for row in near_tie.values()):
        raise RuntimeError("issue477 near-tie correctness failed")

    known_empty: dict[str, dict[str, Any]] = {}
    for dtype_name in DTYPE_NAMES:
        for batch_size in BATCH_SIZES:
            key = f"{dtype_name}_batch{batch_size}_empty"
            known_empty[key] = frozen.known_empty_case(
                memory, dtype_name, batch_size, device, reference, candidate
            )
            if not known_empty[key]["pass"]:
                raise RuntimeError(
                    f"issue477 known-empty correctness failed for {key}"
                )

    rows: dict[str, dict[str, Any]] = {}
    for dtype_name in DTYPE_NAMES:
        for batch_size in BATCH_SIZES:
            for validity_kind in VALIDITY_KINDS:
                case = frozen.make_case(
                    dtype_name=dtype_name,
                    batch_size=batch_size,
                    validity_kind=validity_kind,
                    generator=generator,
                    device=device,
                )
                key = frozen._row_key(dtype_name, batch_size, validity_kind)
                correctness = correctness_row_repair5(
                    memory, case, reference, candidate
                )
                if not correctness["pass"]:
                    raise RuntimeError(f"issue477 correctness failed for {key}")

                calls = {
                    "reference": lambda case=case: frozen._full_read(
                        reference, memory, case
                    ),
                    "candidate": lambda case=case: frozen._full_read(
                        candidate, memory, case
                    ),
                }
                timing = frozen._timed_summaries(calls)
                reference_us = timing["reference"]["median_us_per_call"]
                candidate_us = timing["candidate"]["median_us_per_call"]
                latency_ratio = candidate_us / reference_us
                profiles = {
                    name: frozen._cuda_profile(call) for name, call in calls.items()
                }
                reference_events = profiles["reference"]["cuda_device_events"]
                candidate_events = profiles["candidate"]["cuda_device_events"]
                if reference_events <= 0 or candidate_events <= 0:
                    raise RuntimeError(
                        f"issue477 profiler found no CUDA events for {key}"
                    )
                event_ratio = candidate_events / reference_events

                _, _, similarity = frozen._diagnostic_tail_inputs(memory, case)
                tail_call = (
                    lambda similarity=similarity, case=case: frozen.fused_ficem_read_tail(
                        similarity,
                        case.state.strengths,
                        case.state.valid,
                        case.state.values,
                        return_top_indices=False,
                    )
                )
                tail_profile = frozen._cuda_profile(tail_call)

                rows[key] = {
                    "dtype": dtype_name,
                    "batch_size": batch_size,
                    "validity_kind": validity_kind,
                    "correctness": correctness,
                    "timing": timing,
                    "latency_ratio_candidate_over_reference": float(latency_ratio),
                    "profiles": profiles,
                    "full_cuda_event_ratio_candidate_over_reference": float(
                        event_ratio
                    ),
                    "candidate_tail_profile": tail_profile,
                    "vram": {
                        name: frozen._peak_vram(call) for name, call in calls.items()
                    },
                    "row_latency_pass": bool(
                        latency_ratio <= MAX_ROW_LATENCY_RATIO
                    ),
                    "full_event_ratio_pass": bool(
                        event_ratio <= MAX_FULL_EVENT_RATIO
                    ),
                    "single_tail_kernel_pass": bool(
                        tail_profile["cuda_device_events"] == 1
                        and tail_profile["triton_read_tail_events"] == 1
                    ),
                    "candidate_no_reference_tail_ops_pass": bool(
                        profiles["candidate"]["relevant_operator_calls"]["topk"]
                        == 0
                        and profiles["candidate"]["relevant_operator_calls"][
                            "softmax"
                        ]
                        == 0
                        and profiles["candidate"]["relevant_operator_calls"][
                            "gather"
                        ]
                        == 0
                    ),
                }

    geomeans: dict[str, float] = {}
    geomean_pass: dict[str, bool] = {}
    for dtype_name in DTYPE_NAMES:
        ratios = [
            row["latency_ratio_candidate_over_reference"]
            for row in rows.values()
            if row["dtype"] == dtype_name
        ]
        geomean = math.exp(sum(math.log(value) for value in ratios) / len(ratios))
        geomeans[dtype_name] = float(geomean)
        geomean_pass[dtype_name] = bool(
            geomean <= MAX_GEOMEAN_LATENCY_RATIO
        )

    correctness_pass = all(row["correctness"]["pass"] for row in rows.values())
    known_empty_pass = all(row["pass"] for row in known_empty.values())
    near_tie_pass = all(row["pass"] for row in near_tie.values())
    row_latency_pass = all(row["row_latency_pass"] for row in rows.values())
    event_ratio_pass = all(row["full_event_ratio_pass"] for row in rows.values())
    single_tail_kernel_pass = all(
        row["single_tail_kernel_pass"] for row in rows.values()
    )
    no_reference_tail_ops_pass = all(
        row["candidate_no_reference_tail_ops_pass"] for row in rows.values()
    )
    overall_pass = bool(
        correctness_pass
        and known_empty_pass
        and near_tie_pass
        and row_latency_pass
        and event_ratio_pass
        and single_tail_kernel_pass
        and no_reference_tail_ops_pass
        and all(geomean_pass.values())
    )

    return {
        "protocol": issue477_protocol(),
        "device": device_name,
        "near_tie_correctness": near_tie,
        "known_empty_correctness": known_empty,
        "rows": rows,
        "geomean_latency_ratio_by_dtype": geomeans,
        "geomean_latency_pass_by_dtype": geomean_pass,
        "correctness_pass": correctness_pass,
        "known_empty_pass": known_empty_pass,
        "near_tie_pass": near_tie_pass,
        "row_latency_pass": row_latency_pass,
        "full_event_ratio_pass": event_ratio_pass,
        "single_tail_kernel_pass": single_tail_kernel_pass,
        "candidate_no_reference_tail_ops_pass": no_reference_tail_ops_pass,
        "overall_pass": overall_pass,
        "decision": "PASS" if overall_pass else "FAIL",
        "synthetic_only": True,
        "memory_module_random_weights_only": True,
        "model_loaded": False,
        "checkpoint_loaded": False,
        "corpus_accessed": False,
        "training_performed": False,
        "optimizer_created": False,
        "backward_performed": False,
        "scientific_seed_consumed": False,
        "end_to_end_systems_authorized": False,
        "architecture_freeze_authorized": False,
        "s2_authorized": False,
        "fresh_scientific_seed_authorized": False,
        "100m_authorized": False,
        "breakthrough_proven": False,
    }
