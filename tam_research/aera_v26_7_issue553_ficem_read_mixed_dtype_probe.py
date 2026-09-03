from __future__ import annotations

"""Issue #553 one-shot synthetic gate for AERA-v26.7 mixed-dtype FICEM READ.

The historical #479 repair5 surface stays decision-bearing and is rerun through
the v26.7 backend/tail without changing its fixtures, order, tolerances, timing,
or profiler thresholds. The only extension is the two preregistered
compute-versus-durable FP32/BF16 layouts from #551/#553.
"""

from dataclasses import dataclass
import math
from typing import Any, Callable

import torch
import torch.nn.functional as F

from . import aera_v26_3_ficem_read_probe as frozen
from . import aera_v26_3_ficem_read_probe_repair5 as repair5
from .aera_hardware_core_v24 import ContextualEpisodicMemoryState
from .aera_hardware_core_v25_1 import _set_known_empty_hint
from .aera_hardware_core_v26 import TorchFICEMReferenceBackend
from .aera_hardware_core_v26_7_ficem_read_mixed_dtype import (
    MixedDtypeTritonFICEMReadWriteBackend,
    fused_ficem_read_tail_mixed_dtype,
    mixed_dtype_ficem_read_write_v26_7_protocol,
)

RESEARCH_ISSUE = 553
SOURCE_MAIN = "89ef42e447fd797146a45cf4ea869e3012542761"
V26_7_BACKEND_BLOB = "d8133c6b204b1ee5f23955255fb2fb09d09bd723"
REPAIR5_BACKEND_BLOB = "263f68eb1186a8ac14a08fc4b4df1fc5b292c711"
V26_6_WRITE_BACKEND_BLOB = "d45c262314a0b4691f26812a279937a225043ad9"
HISTORICAL_PROBE_BLOB = "16aa99b9f6f0a1d11bd7bf5f36b2f0b1fb97047b"
REPAIR5_PROBE_BLOB = "6fd6518e10ed1ef4115863f98ac591ffd77ce903"
ISSUE530_SYSTEMS_BLOB = "9d5a3c31f4a3862f96b957540baa2e0ec6a84c6b"
V26_INTERFACE_BLOB = "268644ac4edee15a4cc4e29d3fed7f61eeb3caa7"
STABLE_REFERENCE_BLOB = "4e336b6e1a6238dac782fa320751d68281493ee1"

ISSUE479_TRIGGER = 484
ISSUE479_RUN = 33618950619
ISSUE479_JOB = 100211244996
ISSUE545_TRIGGER = 550
ISSUE545_RUN = 33686037672
ISSUE545_JOB = 100433658768
ISSUE545_FAILURE = "FICEM read-tail floating dtypes must match"
ISSUE552_HEAD = "c7b74cac061f5d0233df38261c60d12d18126eed"
ISSUE552_CPU_RUN = 33722918693
ISSUE552_CPU_JOB = 100545652710

DESIGN_SEED = repair5.DESIGN_SEED
D_MODEL = repair5.D_MODEL
MEMORY_DIM = repair5.MEMORY_DIM
CAPACITY = repair5.CAPACITY
TIME = repair5.TIME
BATCH_SIZES = repair5.BATCH_SIZES
DTYPE_NAMES = repair5.DTYPE_NAMES
VALIDITY_KINDS = repair5.VALIDITY_KINDS
WARMUP_CALLS = repair5.WARMUP_CALLS
TIMED_ROUNDS = repair5.TIMED_ROUNDS
CALLS_PER_ROUND = repair5.CALLS_PER_ROUND
FP32_ATOL = repair5.FP32_ATOL
FP32_RTOL = repair5.FP32_RTOL
BF16_ATOL = repair5.BF16_ATOL
BF16_RTOL = repair5.BF16_RTOL
MAX_GEOMEAN_LATENCY_RATIO = repair5.MAX_GEOMEAN_LATENCY_RATIO
MAX_ROW_LATENCY_RATIO = repair5.MAX_ROW_LATENCY_RATIO
MAX_FULL_EVENT_RATIO = repair5.MAX_FULL_EVENT_RATIO

MIXED_LAYOUTS: tuple[tuple[str, str], ...] = (
    ("bfloat16", "float32"),
    ("float32", "bfloat16"),
)


@dataclass(frozen=True)
class MixedReadCase:
    compute_dtype_name: str
    durable_dtype_name: str
    batch_size: int
    validity_kind: str
    identity: torch.Tensor
    context: torch.Tensor
    state: ContextualEpisodicMemoryState


def issue553_protocol() -> dict[str, Any]:
    protocol = dict(mixed_dtype_ficem_read_write_v26_7_protocol())
    protocol.update(
        {
            "probe_version": "aera-v26.7-issue553-ficem-read-mixed-dtype-l4",
            "research_issue": RESEARCH_ISSUE,
            "source_main_issue553": SOURCE_MAIN,
            "v26_7_backend_blob": V26_7_BACKEND_BLOB,
            "repair5_backend_blob": REPAIR5_BACKEND_BLOB,
            "v26_6_write_backend_blob": V26_6_WRITE_BACKEND_BLOB,
            "historical_probe_blob": HISTORICAL_PROBE_BLOB,
            "repair5_probe_blob": REPAIR5_PROBE_BLOB,
            "issue530_systems_blob": ISSUE530_SYSTEMS_BLOB,
            "v26_interface_blob": V26_INTERFACE_BLOB,
            "stable_reference_blob": STABLE_REFERENCE_BLOB,
            "issue479_trigger": ISSUE479_TRIGGER,
            "issue479_run": ISSUE479_RUN,
            "issue479_job": ISSUE479_JOB,
            "issue545_trigger": ISSUE545_TRIGGER,
            "issue545_run": ISSUE545_RUN,
            "issue545_job": ISSUE545_JOB,
            "issue545_failure": ISSUE545_FAILURE,
            "issue552_head": ISSUE552_HEAD,
            "issue552_cpu_run": ISSUE552_CPU_RUN,
            "issue552_cpu_job": ISSUE552_CPU_JOB,
            "historical_surface_preserved": True,
            "historical_surface_candidate_is_v26_7": True,
            "mixed_layouts": [list(item) for item in MIXED_LAYOUTS],
            "mixed_regular_generator_continues_historical_stream": True,
            "mixed_timing_decision_bearing": False,
            "historical_timing_decision_bearing": True,
            "complementary_fp32_compute_bf16_durable_full_backend_required": False,
            "integration_bf16_compute_fp32_durable_full_backend_required": True,
            "gpu_authorized_by_probe_module": False,
            "end_to_end_systems_authorized": False,
            "architecture_freeze_authorized": False,
            "s2_authorized": False,
            "fresh_scientific_seed_authorized": False,
            "independent_replication_credit": False,
            "100m_authorized": False,
            "breakthrough_proven": False,
        }
    )
    return protocol


def cpu_contract_preflight_issue553() -> dict[str, Any]:
    inherited = repair5.cpu_contract_preflight()
    if DESIGN_SEED != 408_411:
        raise RuntimeError("issue553 design seed drifted")
    if (D_MODEL, MEMORY_DIM, CAPACITY, TIME) != (200, 50, 48, 256):
        raise RuntimeError("issue553 geometry drifted")
    if BATCH_SIZES != (8, 64):
        raise RuntimeError("issue553 batch order drifted")
    if DTYPE_NAMES != ("float32", "bfloat16"):
        raise RuntimeError("issue553 historical dtype order drifted")
    if VALIDITY_KINDS != ("mixed", "full"):
        raise RuntimeError("issue553 validity order drifted")
    if MIXED_LAYOUTS != (("bfloat16", "float32"), ("float32", "bfloat16")):
        raise RuntimeError("issue553 mixed layout order drifted")
    if (WARMUP_CALLS, TIMED_ROUNDS, CALLS_PER_ROUND) != (10, 5, 100):
        raise RuntimeError("issue553 historical timing drifted")
    if (FP32_ATOL, FP32_RTOL, BF16_ATOL, BF16_RTOL) != (
        1e-5,
        1e-5,
        1e-2,
        1e-2,
    ):
        raise RuntimeError("issue553 tolerance drifted")
    if (
        MAX_GEOMEAN_LATENCY_RATIO != 0.90
        or MAX_ROW_LATENCY_RATIO != 1.05
        or MAX_FULL_EVENT_RATIO != 0.75
    ):
        raise RuntimeError("issue553 historical thresholds drifted")
    protocol = issue553_protocol()
    forbidden = (
        "end_to_end_systems_authorized",
        "architecture_freeze_authorized",
        "s2_authorized",
        "fresh_scientific_seed_authorized",
        "100m_authorized",
        "breakthrough_proven",
    )
    if any(protocol[key] for key in forbidden):
        raise RuntimeError("issue553 CPU contract unexpectedly authorizes higher work")
    return {
        "inherited_repair5_contract": inherited,
        "protocol": protocol,
        "gpu_authorized_by_cpu_preflight": False,
        "synthetic_only": True,
        "scientific_seed_consumed": False,
    }


def _dtype(name: str) -> torch.dtype:
    return frozen._dtype_from_name(name)


def _tolerances(compute_dtype_name: str) -> tuple[float, float]:
    return (
        (FP32_ATOL, FP32_RTOL)
        if compute_dtype_name == "float32"
        else (BF16_ATOL, BF16_RTOL)
    )


def _clone_mixed_case(case: MixedReadCase) -> tuple[torch.Tensor, torch.Tensor, ContextualEpisodicMemoryState]:
    return case.identity.clone(), case.context.clone(), frozen._clone_state(case.state)


def _historical_near_tie_v26_7(dtype_name: str, device: torch.device) -> dict[str, Any]:
    dtype = _dtype(dtype_name)
    similarity = torch.full((1, 1, CAPACITY), -1.0, dtype=dtype, device=device)
    values = (
        [1.0, 0.999, 0.998, 0.997, 0.996]
        if dtype_name == "float32"
        else [1.0, 0.98, 0.96, 0.94, 0.92]
    )
    similarity[0, 0, :5] = torch.tensor(values, dtype=dtype, device=device)
    strengths = torch.ones((1, CAPACITY), dtype=dtype, device=device)
    valid = torch.ones((1, CAPACITY), dtype=torch.bool, device=device)
    generator = torch.Generator().manual_seed(
        DESIGN_SEED + (3 if dtype_name == "float32" else 4)
    )
    payload = torch.randn(1, CAPACITY, MEMORY_DIM, generator=generator).to(
        dtype=dtype, device=device
    )
    state = ContextualEpisodicMemoryState(
        keys=torch.zeros_like(payload),
        values=payload,
        strengths=strengths,
        valid=valid,
    )
    reference_recalled, reference_indices = repair5._reference_tail_in_full_read_context(
        dtype_name, similarity, state
    )
    candidate_recalled, candidate_indices = fused_ficem_read_tail_mixed_dtype(
        similarity,
        strengths,
        valid,
        payload,
        return_top_indices=True,
    )
    torch.cuda.synchronize()
    if candidate_indices is None:
        raise RuntimeError("issue553 historical near-tie missing candidate indices")
    atol, rtol = _tolerances(dtype_name)
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


def _historical_correctness_v26_7(
    memory: frozen.CoalescedFICEMMemory,
    case: frozen.ReadCase,
    reference: TorchFICEMReferenceBackend,
    candidate: MixedDtypeTritonFICEMReadWriteBackend,
) -> dict[str, Any]:
    identity_before = case.identity.clone()
    context_before = case.context.clone()
    state_before = frozen._clone_state(case.state)

    reference_result = frozen._full_read(reference, memory, case)
    candidate_result = frozen._full_read(candidate, memory, case)
    torch.cuda.synchronize()

    query, keys, similarity = frozen._diagnostic_tail_inputs(memory, case)
    reference_recalled, reference_indices = repair5._reference_tail_in_full_read_context(
        case.dtype_name, similarity, case.state
    )
    with torch.no_grad():
        candidate_recalled, candidate_indices = fused_ficem_read_tail_mixed_dtype(
            similarity,
            case.state.strengths,
            case.state.valid,
            case.state.values,
            return_top_indices=True,
        )
    torch.cuda.synchronize()
    if candidate_indices is None:
        raise RuntimeError("issue553 historical row missing candidate indices")

    masked_logits = repair5._reference_masked_logits_in_full_read_context(
        case.dtype_name, similarity, case.state
    )
    selection = frozen._tie_aware_top4_equivalence(
        masked_logits,
        case.state.valid,
        reference_indices,
        candidate_indices,
    )
    atol, rtol = _tolerances(case.dtype_name)
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
        "selection_semantically_equivalent": selection["selection_semantically_equivalent"],
        "distinct_selected_set_exact": selection["distinct_selected_set_exact"],
        "tied_selection_semantically_valid": selection["tied_selection_semantically_valid"],
        "selected_top4_set_exact": selection["raw_selected_set_equal_all_queries"],
        "tie_query_count": selection["tie_query_count"],
        "total_query_count": selection["total_query_count"],
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


def _tail_profile_with_cast_accounting(call: Callable[[], Any]) -> dict[str, Any]:
    call()
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
        profile_memory=False,
    ) as profile:
        output = call()
    torch.cuda.synchronize()
    del output

    cuda_events = 0
    triton_events = 0
    for event in profile.events():
        device_type = getattr(event, "device_type", None)
        if device_type == torch.autograd.DeviceType.CUDA or str(device_type).endswith("CUDA"):
            cuda_events += 1
            if "ficem_read_tail_kernel" in str(getattr(event, "name", "")):
                triton_events += 1

    operators = {
        "topk": 0,
        "softmax": 0,
        "gather": 0,
        "_to_copy": 0,
        "copy_": 0,
    }
    for item in profile.key_averages():
        key = str(item.key).lower()
        for token in operators:
            if token in key:
                operators[token] += int(item.count)
    return {
        "cuda_device_events": int(cuda_events),
        "triton_read_tail_events": int(triton_events),
        "relevant_operator_calls": operators,
    }


def _make_mixed_case(
    *,
    compute_dtype_name: str,
    durable_dtype_name: str,
    batch_size: int,
    validity_kind: str,
    generator: torch.Generator,
    device: torch.device,
) -> MixedReadCase:
    compute_dtype = _dtype(compute_dtype_name)
    durable_dtype = _dtype(durable_dtype_name)
    identity = frozen._cpu_randn(
        (batch_size, TIME, D_MODEL), generator=generator, dtype=compute_dtype
    ).to(device)
    context = frozen._cpu_randn(
        (batch_size, TIME, D_MODEL), generator=generator, dtype=compute_dtype
    ).to(device)
    keys = frozen._cpu_randn(
        (batch_size, CAPACITY, MEMORY_DIM), generator=generator, dtype=durable_dtype
    ).to(device)
    values = frozen._cpu_randn(
        (batch_size, CAPACITY, MEMORY_DIM), generator=generator, dtype=durable_dtype
    ).to(device)
    strengths = (
        torch.rand(batch_size, CAPACITY, generator=generator).mul_(0.95).add_(0.05)
    ).to(dtype=durable_dtype, device=device)
    if validity_kind == "full":
        valid = torch.ones(batch_size, CAPACITY, dtype=torch.bool, device=device)
    else:
        valid_cpu = torch.rand(batch_size, CAPACITY, generator=generator) > 0.35
        valid_cpu[:, :8] = True
        valid = valid_cpu.to(device)
    state = ContextualEpisodicMemoryState(
        keys=keys,
        values=values,
        strengths=strengths,
        valid=valid,
    )
    _set_known_empty_hint(state, False)
    return MixedReadCase(
        compute_dtype_name,
        durable_dtype_name,
        batch_size,
        validity_kind,
        identity,
        context,
        state,
    )


def _direct_similarity_from_compute_sources(case: MixedReadCase) -> torch.Tensor:
    left = F.normalize(case.identity[..., :MEMORY_DIM], dim=-1)
    slots = F.normalize(case.context[:, :CAPACITY, :MEMORY_DIM], dim=-1)
    with torch.no_grad(), frozen._precision_context(case.compute_dtype_name):
        similarity = torch.einsum("btd,bsd->bts", left, slots)
    return similarity.contiguous()


def _mixed_similarity(
    memory: frozen.CoalescedFICEMMemory,
    case: MixedReadCase,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    if (case.compute_dtype_name, case.durable_dtype_name) == ("bfloat16", "float32"):
        read_case = frozen.ReadCase(
            case.compute_dtype_name,
            case.batch_size,
            case.validity_kind,
            case.identity,
            case.context,
            case.state,
        )
        query, keys, similarity = frozen._diagnostic_tail_inputs(memory, read_case)
        return similarity, query, keys
    return _direct_similarity_from_compute_sources(case), None, None


def _learned_out_pair(
    memory: frozen.CoalescedFICEMMemory,
    compute_dtype_name: str,
    reference_recalled: torch.Tensor,
    candidate_recalled: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad(), frozen._precision_context(compute_dtype_name):
        return memory.out(reference_recalled), memory.out(candidate_recalled)


def _mixed_correctness(
    memory: frozen.CoalescedFICEMMemory,
    case: MixedReadCase,
    reference: TorchFICEMReferenceBackend,
    candidate: MixedDtypeTritonFICEMReadWriteBackend,
) -> dict[str, Any]:
    identity_before, context_before, state_before = _clone_mixed_case(case)
    similarity, diagnostic_query, diagnostic_keys = _mixed_similarity(memory, case)
    expected_similarity_dtype = _dtype(case.compute_dtype_name)
    if similarity.dtype != expected_similarity_dtype:
        raise RuntimeError(
            f"issue553 mixed similarity dtype drift: got={similarity.dtype} "
            f"expected={expected_similarity_dtype}"
        )

    reference_recalled, reference_indices = repair5._reference_tail_in_full_read_context(
        case.compute_dtype_name, similarity, case.state
    )
    candidate_recalled, candidate_indices = fused_ficem_read_tail_mixed_dtype(
        similarity,
        case.state.strengths,
        case.state.valid,
        case.state.values,
        return_top_indices=True,
    )
    torch.cuda.synchronize()
    if candidate_indices is None:
        raise RuntimeError("issue553 mixed row missing candidate indices")

    masked_logits = repair5._reference_masked_logits_in_full_read_context(
        case.compute_dtype_name, similarity, case.state
    )
    selection = frozen._tie_aware_top4_equivalence(
        masked_logits,
        case.state.valid,
        reference_indices,
        candidate_indices,
    )
    atol, rtol = _tolerances(case.compute_dtype_name)
    recalled_close = torch.allclose(
        reference_recalled, candidate_recalled, atol=atol, rtol=rtol
    )
    reference_final, candidate_final = _learned_out_pair(
        memory,
        case.compute_dtype_name,
        reference_recalled,
        candidate_recalled,
    )
    final_close = torch.allclose(
        reference_final, candidate_final, atol=atol, rtol=rtol
    )

    expected_pre_out_dtype = torch.float32
    meta_exact = (
        candidate_recalled.shape == (case.batch_size, TIME, MEMORY_DIM)
        and candidate_recalled.dtype == expected_pre_out_dtype
        and candidate_recalled.device.type == "cuda"
        and reference_recalled.shape == candidate_recalled.shape
        and reference_recalled.dtype == candidate_recalled.dtype
        and reference_recalled.device == candidate_recalled.device
        and reference_final.shape == candidate_final.shape
        and reference_final.dtype == candidate_final.dtype
        and reference_final.device == candidate_final.device
    )
    finite = all(
        bool(torch.isfinite(t).all())
        for t in (
            reference_recalled,
            candidate_recalled,
            reference_final,
            candidate_final,
        )
    )

    full_backend = {
        "required": (case.compute_dtype_name, case.durable_dtype_name)
        == ("bfloat16", "float32"),
        "exercised": False,
        "pass": True,
        "final_out_close": True,
        "query_and_normalized_keys_bit_exact": True,
        "dtype_device_shape_exact": True,
        "no_reference_tail_ops": True,
    }
    if full_backend["required"]:
        read_case = frozen.ReadCase(
            case.compute_dtype_name,
            case.batch_size,
            case.validity_kind,
            case.identity,
            case.context,
            case.state,
        )
        reference_result = frozen._full_read(reference, memory, read_case)
        candidate_result = frozen._full_read(candidate, memory, read_case)
        torch.cuda.synchronize()
        full_close = torch.allclose(
            reference_result.recalled,
            candidate_result.recalled,
            atol=atol,
            rtol=rtol,
        )
        reuse_exact = (
            diagnostic_query is not None
            and diagnostic_keys is not None
            and reference_result.projected_query is not None
            and candidate_result.projected_query is not None
            and reference_result.normalized_old_keys is not None
            and candidate_result.normalized_old_keys is not None
            and torch.equal(reference_result.projected_query, candidate_result.projected_query)
            and torch.equal(
                reference_result.normalized_old_keys,
                candidate_result.normalized_old_keys,
            )
            and torch.equal(reference_result.projected_query, diagnostic_query)
            and torch.equal(reference_result.normalized_old_keys, diagnostic_keys)
        )
        full_meta = (
            reference_result.recalled.shape == candidate_result.recalled.shape
            and reference_result.recalled.dtype == candidate_result.recalled.dtype
            and reference_result.recalled.device == candidate_result.recalled.device
            and candidate_result.recalled.device.type == "cuda"
        )
        full_profile = frozen._cuda_profile(
            lambda: frozen._full_read(candidate, memory, read_case)
        )
        no_tail_ops = (
            full_profile["relevant_operator_calls"]["topk"] == 0
            and full_profile["relevant_operator_calls"]["softmax"] == 0
            and full_profile["relevant_operator_calls"]["gather"] == 0
        )
        full_backend = {
            "required": True,
            "exercised": True,
            "pass": bool(full_close and reuse_exact and full_meta and no_tail_ops),
            "final_out_close": bool(full_close),
            "query_and_normalized_keys_bit_exact": bool(reuse_exact),
            "dtype_device_shape_exact": bool(full_meta),
            "no_reference_tail_ops": bool(no_tail_ops),
            "candidate_profile": full_profile,
        }

    source_unchanged = (
        torch.equal(case.identity, identity_before)
        and torch.equal(case.context, context_before)
        and frozen._state_equal(case.state, state_before)
    )

    tail_call = lambda: fused_ficem_read_tail_mixed_dtype(
        similarity,
        case.state.strengths,
        case.state.valid,
        case.state.values,
        return_top_indices=False,
    )
    tail_profile = _tail_profile_with_cast_accounting(tail_call)
    topology_pass = bool(
        tail_profile["cuda_device_events"] == 1
        and tail_profile["triton_read_tail_events"] == 1
        and tail_profile["relevant_operator_calls"]["topk"] == 0
        and tail_profile["relevant_operator_calls"]["softmax"] == 0
        and tail_profile["relevant_operator_calls"]["gather"] == 0
        and tail_profile["relevant_operator_calls"]["_to_copy"] == 0
        and tail_profile["relevant_operator_calls"]["copy_"] == 0
    )

    timing = frozen._timed_summaries(
        {
            "reference_tail": lambda: repair5._reference_tail_in_full_read_context(
                case.compute_dtype_name, similarity, case.state
            ),
            "candidate_tail": tail_call,
        }
    )
    diagnostic_latency_ratio = (
        timing["candidate_tail"]["median_us_per_call"]
        / timing["reference_tail"]["median_us_per_call"]
    )

    passed = bool(
        selection["selection_semantically_equivalent"]
        and recalled_close
        and final_close
        and source_unchanged
        and finite
        and meta_exact
        and topology_pass
        and full_backend["pass"]
    )
    return {
        "pass": passed,
        "compute_dtype": case.compute_dtype_name,
        "durable_dtype": case.durable_dtype_name,
        "batch_size": case.batch_size,
        "validity_kind": case.validity_kind,
        "similarity_dtype": str(similarity.dtype),
        "strengths_dtype": str(case.state.strengths.dtype),
        "values_dtype": str(case.state.values.dtype),
        "selection_semantically_equivalent": selection["selection_semantically_equivalent"],
        "distinct_selected_set_exact": selection["distinct_selected_set_exact"],
        "tied_selection_semantically_valid": selection["tied_selection_semantically_valid"],
        "pre_out_recalled_close": bool(recalled_close),
        "final_out_close": bool(final_close),
        "source_unchanged": bool(source_unchanged),
        "finite": bool(finite),
        "dtype_device_shape_exact": bool(meta_exact),
        "topology_pass": topology_pass,
        "tail_profile": tail_profile,
        "full_backend": full_backend,
        "timing_diagnostic_only": timing,
        "diagnostic_latency_ratio_candidate_over_reference": float(
            diagnostic_latency_ratio
        ),
        "latency_decision_bearing": False,
        "atol": atol,
        "rtol": rtol,
        "pre_out_max_abs_diff": float(
            (reference_recalled.float() - candidate_recalled.float()).abs().max()
        ),
        "final_out_max_abs_diff": float(
            (reference_final.float() - candidate_final.float()).abs().max()
        ),
    }


def _mixed_near_tie(
    memory: frozen.CoalescedFICEMMemory,
    compute_dtype_name: str,
    durable_dtype_name: str,
    device: torch.device,
) -> dict[str, Any]:
    compute_dtype = _dtype(compute_dtype_name)
    durable_dtype = _dtype(durable_dtype_name)
    similarity = torch.full(
        (1, 1, CAPACITY), -1.0, dtype=compute_dtype, device=device
    )
    values = (
        [1.0, 0.999, 0.998, 0.997, 0.996]
        if compute_dtype_name == "float32"
        else [1.0, 0.98, 0.96, 0.94, 0.92]
    )
    similarity[0, 0, :5] = torch.tensor(values, dtype=compute_dtype, device=device)
    strengths = torch.ones((1, CAPACITY), dtype=durable_dtype, device=device)
    valid = torch.ones((1, CAPACITY), dtype=torch.bool, device=device)
    generator = torch.Generator().manual_seed(
        DESIGN_SEED + (3 if compute_dtype_name == "float32" else 4)
    )
    payload = torch.randn(1, CAPACITY, MEMORY_DIM, generator=generator).to(
        dtype=durable_dtype, device=device
    )
    state = ContextualEpisodicMemoryState(
        keys=torch.zeros_like(payload),
        values=payload,
        strengths=strengths,
        valid=valid,
    )
    similarity_before = similarity.clone()
    strengths_before = strengths.clone()
    valid_before = valid.clone()
    payload_before = payload.clone()

    reference_recalled, reference_indices = repair5._reference_tail_in_full_read_context(
        compute_dtype_name, similarity, state
    )
    candidate_recalled, candidate_indices = fused_ficem_read_tail_mixed_dtype(
        similarity,
        strengths,
        valid,
        payload,
        return_top_indices=True,
    )
    torch.cuda.synchronize()
    if candidate_indices is None:
        raise RuntimeError("issue553 mixed near-tie missing candidate indices")
    atol, rtol = _tolerances(compute_dtype_name)
    selected_exact = torch.equal(
        torch.sort(reference_indices, dim=-1).values,
        torch.sort(candidate_indices.to(torch.long), dim=-1).values,
    )
    recalled_close = torch.allclose(
        reference_recalled, candidate_recalled, atol=atol, rtol=rtol
    )
    reference_final, candidate_final = _learned_out_pair(
        memory,
        compute_dtype_name,
        reference_recalled,
        candidate_recalled,
    )
    final_close = torch.allclose(
        reference_final, candidate_final, atol=atol, rtol=rtol
    )
    source_unchanged = (
        torch.equal(similarity, similarity_before)
        and torch.equal(strengths, strengths_before)
        and torch.equal(valid, valid_before)
        and torch.equal(payload, payload_before)
    )
    tail_profile = _tail_profile_with_cast_accounting(
        lambda: fused_ficem_read_tail_mixed_dtype(
            similarity,
            strengths,
            valid,
            payload,
            return_top_indices=False,
        )
    )
    topology_pass = bool(
        tail_profile["cuda_device_events"] == 1
        and tail_profile["triton_read_tail_events"] == 1
        and tail_profile["relevant_operator_calls"]["_to_copy"] == 0
        and tail_profile["relevant_operator_calls"]["copy_"] == 0
    )
    dtype_exact = (
        reference_recalled.dtype == candidate_recalled.dtype == torch.float32
        and reference_final.dtype == candidate_final.dtype
    )
    return {
        "pass": bool(
            selected_exact
            and recalled_close
            and final_close
            and source_unchanged
            and topology_pass
            and dtype_exact
        ),
        "compute_dtype": compute_dtype_name,
        "durable_dtype": durable_dtype_name,
        "selected_top4_set_exact": bool(selected_exact),
        "pre_out_recalled_close": bool(recalled_close),
        "final_out_close": bool(final_close),
        "source_unchanged": bool(source_unchanged),
        "topology_pass": topology_pass,
        "dtype_exact": bool(dtype_exact),
        "tail_profile": tail_profile,
        "max_abs_diff": float(
            (reference_recalled.float() - candidate_recalled.float()).abs().max()
        ),
    }


def _mixed_known_empty(
    memory: frozen.CoalescedFICEMMemory,
    reference: TorchFICEMReferenceBackend,
    candidate: MixedDtypeTritonFICEMReadWriteBackend,
    compute_dtype_name: str,
    durable_dtype_name: str,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    compute_dtype = _dtype(compute_dtype_name)
    durable_dtype = _dtype(durable_dtype_name)
    generator = torch.Generator().manual_seed(DESIGN_SEED + 1000 + batch_size)
    identity = torch.randn(batch_size, TIME, D_MODEL, generator=generator).to(
        dtype=compute_dtype, device=device
    )
    context = torch.randn(batch_size, TIME, D_MODEL, generator=generator).to(
        dtype=compute_dtype, device=device
    )
    state = ContextualEpisodicMemoryState(
        keys=torch.zeros(
            batch_size, CAPACITY, MEMORY_DIM, dtype=durable_dtype, device=device
        ),
        values=torch.zeros(
            batch_size, CAPACITY, MEMORY_DIM, dtype=durable_dtype, device=device
        ),
        strengths=torch.zeros(
            batch_size, CAPACITY, dtype=durable_dtype, device=device
        ),
        valid=torch.zeros(batch_size, CAPACITY, dtype=torch.bool, device=device),
    )
    _set_known_empty_hint(state, True)
    read_case = frozen.ReadCase(
        compute_dtype_name,
        batch_size,
        "empty",
        identity,
        context,
        state,
    )
    reference_result = frozen._full_read(reference, memory, read_case)
    candidate_result = frozen._full_read(candidate, memory, read_case)
    profile = frozen._cuda_profile(
        lambda: frozen._full_read(candidate, memory, read_case)
    )
    expected = torch.zeros_like(candidate_result.recalled)
    return {
        "pass": bool(
            torch.equal(reference_result.recalled, candidate_result.recalled)
            and torch.equal(candidate_result.recalled, expected)
            and candidate_result.projected_query is None
            and candidate_result.normalized_old_keys is None
            and profile["triton_read_tail_events"] == 0
        ),
        "compute_dtype": compute_dtype_name,
        "durable_dtype": durable_dtype_name,
        "batch_size": batch_size,
        "zero_output_exact": bool(
            torch.equal(reference_result.recalled, candidate_result.recalled)
            and torch.equal(candidate_result.recalled, expected)
        ),
        "reuse_is_none": bool(
            candidate_result.projected_query is None
            and candidate_result.normalized_old_keys is None
        ),
        "triton_read_tail_events": profile["triton_read_tail_events"],
        "candidate_profile": profile,
    }


def _historical_surface(
    memory: frozen.CoalescedFICEMMemory,
    reference: TorchFICEMReferenceBackend,
    candidate: MixedDtypeTritonFICEMReadWriteBackend,
    generator: torch.Generator,
    device: torch.device,
) -> dict[str, Any]:
    near_tie = {
        dtype_name: _historical_near_tie_v26_7(dtype_name, device)
        for dtype_name in DTYPE_NAMES
    }

    known_empty: dict[str, dict[str, Any]] = {}
    for dtype_name in DTYPE_NAMES:
        for batch_size in BATCH_SIZES:
            key = f"{dtype_name}_batch{batch_size}_empty"
            known_empty[key] = frozen.known_empty_case(
                memory,
                dtype_name,
                batch_size,
                device,
                reference,
                candidate,
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
                correctness = _historical_correctness_v26_7(
                    memory, case, reference, candidate
                )
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
                        f"issue553 historical profiler found no CUDA events for {key}"
                    )
                event_ratio = candidate_events / reference_events
                _, _, similarity = frozen._diagnostic_tail_inputs(memory, case)
                tail_profile = frozen._cuda_profile(
                    lambda similarity=similarity, case=case: fused_ficem_read_tail_mixed_dtype(
                        similarity,
                        case.state.strengths,
                        case.state.valid,
                        case.state.values,
                        return_top_indices=False,
                    )
                )
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
                        profiles["candidate"]["relevant_operator_calls"]["topk"] == 0
                        and profiles["candidate"]["relevant_operator_calls"]["softmax"] == 0
                        and profiles["candidate"]["relevant_operator_calls"]["gather"] == 0
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
        "authoritative_predecessor_issue": 479,
        "candidate_backend": candidate.name,
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
    }


def run_ficem_read_mixed_dtype_probe_v26_7() -> dict[str, Any]:
    cpu_contract_preflight_issue553()
    if not torch.cuda.is_available():
        raise RuntimeError("issue553 requires CUDA")
    device = torch.device("cuda")
    device_name = torch.cuda.get_device_name(0)
    if "L4" not in device_name.upper():
        raise RuntimeError(f"issue553 requires NVIDIA L4, found {device_name}")

    memory = frozen.build_memory(device)
    reference = TorchFICEMReferenceBackend()
    candidate = MixedDtypeTritonFICEMReadWriteBackend()
    generator = torch.Generator().manual_seed(DESIGN_SEED)

    historical = _historical_surface(
        memory,
        reference,
        candidate,
        generator,
        device,
    )

    mixed_rows: dict[str, dict[str, Any]] = {}
    for compute_dtype_name, durable_dtype_name in MIXED_LAYOUTS:
        for batch_size in BATCH_SIZES:
            for validity_kind in VALIDITY_KINDS:
                case = _make_mixed_case(
                    compute_dtype_name=compute_dtype_name,
                    durable_dtype_name=durable_dtype_name,
                    batch_size=batch_size,
                    validity_kind=validity_kind,
                    generator=generator,
                    device=device,
                )
                key = (
                    f"compute_{compute_dtype_name}_durable_{durable_dtype_name}_"
                    f"batch{batch_size}_{validity_kind}"
                )
                mixed_rows[key] = _mixed_correctness(
                    memory,
                    case,
                    reference,
                    candidate,
                )

    mixed_near_tie = {
        f"compute_{compute}_durable_{durable}": _mixed_near_tie(
            memory, compute, durable, device
        )
        for compute, durable in MIXED_LAYOUTS
    }

    mixed_known_empty: dict[str, dict[str, Any]] = {}
    for compute, durable in MIXED_LAYOUTS:
        for batch_size in BATCH_SIZES:
            key = f"compute_{compute}_durable_{durable}_batch{batch_size}_empty"
            mixed_known_empty[key] = _mixed_known_empty(
                memory,
                reference,
                candidate,
                compute,
                durable,
                batch_size,
                device,
            )

    mixed_rows_pass = all(row["pass"] for row in mixed_rows.values())
    mixed_near_tie_pass = all(row["pass"] for row in mixed_near_tie.values())
    mixed_known_empty_pass = all(
        row["pass"] for row in mixed_known_empty.values()
    )
    mixed_overall_pass = bool(
        mixed_rows_pass and mixed_near_tie_pass and mixed_known_empty_pass
    )
    overall_pass = bool(historical["overall_pass"] and mixed_overall_pass)

    return {
        "protocol": issue553_protocol(),
        "device": device_name,
        "historical": historical,
        "mixed": {
            "rows": mixed_rows,
            "near_tie": mixed_near_tie,
            "known_empty": mixed_known_empty,
            "rows_pass": mixed_rows_pass,
            "near_tie_pass": mixed_near_tie_pass,
            "known_empty_pass": mixed_known_empty_pass,
            "overall_pass": mixed_overall_pass,
            "timing_decision_bearing": False,
        },
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
        "independent_replication_credit": False,
        "100m_authorized": False,
        "breakthrough_proven": False,
    }
