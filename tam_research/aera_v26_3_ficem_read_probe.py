from __future__ import annotations

"""Issue #411 production-shaped synthetic gate for the fused FICEM read tail.

Issue #419 repairs only ordinary performance-fixture eligibility: each frozen row
uses a deterministic row-local generator and accepts the first full synthetic case
whose exact reference fourth/fifth boundary is distinct. Candidate output, latency,
profiling and PASS state never participate in fixture selection.
"""

from contextlib import nullcontext
from dataclasses import dataclass
import math
import statistics
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .aera_hardware_core_v24 import (
    MIN_STRENGTH,
    READ_TEMPERATURE,
    READ_TOP_K,
    ContextualEpisodicMemoryState,
)
from .aera_hardware_core_v25 import FactorizedIdentityContextEpisodicMemory
from .aera_hardware_core_v25_1 import (
    ExecutionEquivalentFactorizedIdentityContextMemory,
    _set_known_empty_hint,
)
from .aera_hardware_core_v25_1_compact import (
    StableCompactExecutionEquivalentFactorizedIdentityContextMemory,
)
from .aera_hardware_core_v26 import CoalescedFICEMMemory, TorchFICEMReferenceBackend
from .aera_hardware_core_v26_3_ficem_read_triton import (
    TritonFICEMReadBackend,
    fused_ficem_read_tail,
    fused_ficem_read_v26_3_protocol,
)

RESEARCH_ISSUE = 411
DESIGN_SEED = 408_411
D_MODEL = 200
MEMORY_DIM = 50
CAPACITY = 48
TIME = 256
BATCH_SIZES: tuple[int, ...] = (8, 64)
DTYPE_NAMES: tuple[str, ...] = ("float32", "bfloat16")
VALIDITY_KINDS: tuple[str, ...] = ("mixed", "full")
MAX_FIXTURE_CANDIDATES = 32

WARMUP_CALLS = 10
TIMED_ROUNDS = 5
CALLS_PER_ROUND = 100
FP32_ATOL = 1e-5
FP32_RTOL = 1e-5
BF16_ATOL = 1e-2
BF16_RTOL = 1e-2
MAX_GEOMEAN_LATENCY_RATIO = 0.90
MAX_ROW_LATENCY_RATIO = 1.05
MAX_FULL_EVENT_RATIO = 0.75


class _RawFICEMSource(nn.Module):
    """Only the source geometry needed to construct the real FICEM memory module."""

    def __init__(self) -> None:
        super().__init__()
        self.memory_dim = MEMORY_DIM
        self.q = nn.Linear(D_MODEL, MEMORY_DIM, bias=False)
        self.k = nn.Linear(D_MODEL, MEMORY_DIM, bias=False)
        self.v = nn.Linear(D_MODEL, MEMORY_DIM, bias=False)
        self.out = nn.Linear(MEMORY_DIM, D_MODEL, bias=False)
        self.differentiable_pretraining = False


@dataclass(frozen=True)
class ReadCase:
    dtype_name: str
    batch_size: int
    validity_kind: str
    identity: torch.Tensor
    context: torch.Tensor
    state: ContextualEpisodicMemoryState


def issue411_protocol() -> dict[str, Any]:
    protocol = dict(fused_ficem_read_v26_3_protocol())
    protocol.update(
        {
            "probe_version": "aera-v26.3-issue411-synthetic-ficem-read-l4",
            "design_seed": DESIGN_SEED,
            "design_seed_is_scientific_seed": False,
            "d_model": D_MODEL,
            "time": TIME,
            "batch_sizes": list(BATCH_SIZES),
            "dtypes": list(DTYPE_NAMES),
            "validity_kinds": list(VALIDITY_KINDS),
            "warmup_calls": WARMUP_CALLS,
            "timed_rounds": TIMED_ROUNDS,
            "calls_per_round": CALLS_PER_ROUND,
            "fp32_atol": FP32_ATOL,
            "fp32_rtol": FP32_RTOL,
            "bfloat16_atol": BF16_ATOL,
            "bfloat16_rtol": BF16_RTOL,
            "max_geomean_latency_ratio_each_dtype": MAX_GEOMEAN_LATENCY_RATIO,
            "max_row_latency_ratio": MAX_ROW_LATENCY_RATIO,
            "max_full_read_cuda_event_ratio": MAX_FULL_EVENT_RATIO,
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
            "repair2_fixture_eligibility_only": True,
            "max_fixture_candidates_per_row": MAX_FIXTURE_CANDIDATES,
            "fixture_selection_reference_only": True,
            "fixture_selection_candidate_blind": True,
        }
    )
    return protocol


def cpu_contract_preflight() -> dict[str, Any]:
    if (D_MODEL, MEMORY_DIM, CAPACITY, TIME) != (200, 50, 48, 256):
        raise RuntimeError("issue411 production geometry drifted")
    if BATCH_SIZES != (8, 64):
        raise RuntimeError("issue411 batch sizes drifted")
    if DTYPE_NAMES != ("float32", "bfloat16"):
        raise RuntimeError("issue411 dtype set drifted")
    if VALIDITY_KINDS != ("mixed", "full"):
        raise RuntimeError("issue411 validity rows drifted")
    if MAX_FIXTURE_CANDIDATES != 32:
        raise RuntimeError("issue419 fixture-candidate cap drifted")
    if (WARMUP_CALLS, TIMED_ROUNDS, CALLS_PER_ROUND) != (10, 5, 100):
        raise RuntimeError("issue411 timing protocol drifted")
    if (FP32_ATOL, FP32_RTOL, BF16_ATOL, BF16_RTOL) != (
        1e-5,
        1e-5,
        1e-2,
        1e-2,
    ):
        raise RuntimeError("issue411 correctness tolerances drifted")
    if (
        MAX_GEOMEAN_LATENCY_RATIO != 0.90
        or MAX_ROW_LATENCY_RATIO != 1.05
        or MAX_FULL_EVENT_RATIO != 0.75
    ):
        raise RuntimeError("issue411 PASS thresholds drifted")
    return {
        "protocol": issue411_protocol(),
        "gpu_authorized_by_cpu_preflight": False,
        "synthetic_only": True,
        "scientific_seed_consumed": False,
    }


def build_memory(device: torch.device) -> CoalescedFICEMMemory:
    # Design-only initialization. No model/checkpoint/corpus is constructed.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(DESIGN_SEED + 1)
        raw = _RawFICEMSource()
        v25 = FactorizedIdentityContextEpisodicMemory(raw, capacity=CAPACITY)
        v251 = ExecutionEquivalentFactorizedIdentityContextMemory(v25)
        compact = StableCompactExecutionEquivalentFactorizedIdentityContextMemory(v251)
        memory = CoalescedFICEMMemory(compact)
    memory.set_differentiable_pretraining(False)
    return memory.to(device).eval()


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unsupported issue411 dtype: {name}")


def _precision_context(dtype_name: str):
    if dtype_name == "bfloat16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _cpu_randn(shape: tuple[int, ...], *, generator: torch.Generator, dtype: torch.dtype):
    return torch.randn(*shape, generator=generator).to(dtype=dtype)


def make_case(
    *,
    dtype_name: str,
    batch_size: int,
    validity_kind: str,
    generator: torch.Generator,
    device: torch.device,
) -> ReadCase:
    if dtype_name not in DTYPE_NAMES or batch_size not in BATCH_SIZES:
        raise ValueError("issue411 unsupported dtype/batch")
    if validity_kind not in VALIDITY_KINDS:
        raise ValueError("issue411 unsupported validity kind")
    dtype = _dtype_from_name(dtype_name)
    identity = _cpu_randn(
        (batch_size, TIME, D_MODEL), generator=generator, dtype=dtype
    ).to(device)
    context = _cpu_randn(
        (batch_size, TIME, D_MODEL), generator=generator, dtype=dtype
    ).to(device)
    keys = _cpu_randn(
        (batch_size, CAPACITY, MEMORY_DIM), generator=generator, dtype=dtype
    ).to(device)
    values = _cpu_randn(
        (batch_size, CAPACITY, MEMORY_DIM), generator=generator, dtype=dtype
    ).to(device)
    strengths = (
        torch.rand(batch_size, CAPACITY, generator=generator).mul_(0.95).add_(0.05)
    ).to(dtype=dtype, device=device)
    if validity_kind == "full":
        valid = torch.ones(batch_size, CAPACITY, dtype=torch.bool, device=device)
    else:
        valid_cpu = torch.rand(batch_size, CAPACITY, generator=generator) > 0.35
        valid_cpu[:, :8] = True
        valid = valid_cpu.to(device)
    state = ContextualEpisodicMemoryState(keys, values, strengths, valid)
    _set_known_empty_hint(state, False)
    return ReadCase(dtype_name, batch_size, validity_kind, identity, context, state)


def _clone_state(state: ContextualEpisodicMemoryState) -> ContextualEpisodicMemoryState:
    cloned = ContextualEpisodicMemoryState(
        state.keys.clone(),
        state.values.clone(),
        state.strengths.clone(),
        state.valid.clone(),
    )
    _set_known_empty_hint(cloned, bool(getattr(state, "_v25_1_known_empty", False)))
    return cloned


def _state_equal(a: ContextualEpisodicMemoryState, b: ContextualEpisodicMemoryState) -> bool:
    return (
        torch.equal(a.keys, b.keys)
        and torch.equal(a.values, b.values)
        and torch.equal(a.strengths, b.strengths)
        and torch.equal(a.valid, b.valid)
    )


def _reference_tail(
    similarity: torch.Tensor,
    state: ContextualEpisodicMemoryState,
) -> tuple[torch.Tensor, torch.Tensor]:
    strength_bias = torch.log(state.strengths.clamp(MIN_STRENGTH, 1.0))[:, None, :]
    logits = (similarity + strength_bias) / READ_TEMPERATURE
    masked = logits.masked_fill(~state.valid[:, None, :], -torch.inf)
    top_logits, top_indices = torch.topk(masked, k=READ_TOP_K, dim=-1)
    top_valid = state.valid[:, None, :].expand(-1, similarity.size(1), -1).gather(
        -1, top_indices
    )
    safe_logits = top_logits.masked_fill(~top_valid, -1e9)
    weights = torch.softmax(safe_logits.float(), dim=-1).to(similarity.dtype)
    weights = weights * top_valid.to(weights.dtype)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    expanded_values = state.values[:, None, :, :].expand(
        -1, similarity.size(1), -1, -1
    )
    gathered_values = expanded_values.gather(
        2,
        top_indices.unsqueeze(-1).expand(-1, -1, -1, MEMORY_DIM),
    )
    recalled = (weights.unsqueeze(-1) * gathered_values).sum(dim=2)
    return recalled, top_indices


def _tolerances(dtype_name: str) -> tuple[float, float]:
    return (FP32_ATOL, FP32_RTOL) if dtype_name == "float32" else (BF16_ATOL, BF16_RTOL)


def _full_read(backend, memory, case: ReadCase):
    with torch.no_grad(), _precision_context(case.dtype_name):
        return backend.read(memory, case.identity, case.context, case.state)


def _diagnostic_tail_inputs(memory: CoalescedFICEMMemory, case: ReadCase):
    with torch.no_grad(), _precision_context(case.dtype_name):
        _, _, query = memory.address_factors(case.identity, case.context)
        keys = F.normalize(case.state.keys, dim=-1)
        similarity = torch.einsum("btd,bsd->bts", query, keys)
    return query, keys, similarity.contiguous()


def _reference_boundary_gap(
    similarity: torch.Tensor,
    state: ContextualEpisodicMemoryState,
) -> tuple[bool, float]:
    """Return exact reference eligibility and minimum fourth-minus-fifth gap."""
    strength_bias = torch.log(state.strengths.clamp(MIN_STRENGTH, 1.0))[:, None, :]
    logits = (similarity + strength_bias) / READ_TEMPERATURE
    masked = logits.masked_fill(~state.valid[:, None, :], -torch.inf)
    values = torch.topk(masked, k=READ_TOP_K + 1, dim=-1).values
    gap = values[..., READ_TOP_K - 1] - values[..., READ_TOP_K]
    distinct = bool(torch.all(values[..., READ_TOP_K - 1] != values[..., READ_TOP_K]))
    return distinct, float(gap.float().min())


def _boundary_is_distinct(similarity: torch.Tensor, state: ContextualEpisodicMemoryState) -> bool:
    distinct, _ = _reference_boundary_gap(similarity, state)
    return distinct


def _row_id(dtype_name: str, batch_size: int, validity_kind: str) -> int:
    dtype_index = DTYPE_NAMES.index(dtype_name)
    batch_index = BATCH_SIZES.index(batch_size)
    validity_index = VALIDITY_KINDS.index(validity_kind)
    return dtype_index * (len(BATCH_SIZES) * len(VALIDITY_KINDS)) + batch_index * len(
        VALIDITY_KINDS
    ) + validity_index


def _row_seed(dtype_name: str, batch_size: int, validity_kind: str) -> int:
    """Stable design-only seed derived solely from DESIGN_SEED + frozen row identity."""
    return DESIGN_SEED + 10_000 * (_row_id(dtype_name, batch_size, validity_kind) + 1)


def _select_eligible_case(
    *,
    memory: CoalescedFICEMMemory,
    dtype_name: str,
    batch_size: int,
    validity_kind: str,
    device: torch.device,
) -> tuple[ReadCase, dict[str, Any]]:
    """Choose a tie-free fixture using the exact reference computation only.

    This function intentionally has no candidate backend argument and never calls
    Triton. Rejected fixtures are discarded before correctness, timing or profiling.
    """
    row_id = _row_id(dtype_name, batch_size, validity_kind)
    row_seed = _row_seed(dtype_name, batch_size, validity_kind)
    generator = torch.Generator().manual_seed(row_seed)
    for ordinal in range(1, MAX_FIXTURE_CANDIDATES + 1):
        case = make_case(
            dtype_name=dtype_name,
            batch_size=batch_size,
            validity_kind=validity_kind,
            generator=generator,
            device=device,
        )
        _, _, similarity = _diagnostic_tail_inputs(memory, case)
        distinct, min_gap = _reference_boundary_gap(similarity, case.state)
        if distinct:
            return case, {
                "row_id": row_id,
                "row_seed": row_seed,
                "selected_candidate_ordinal": ordinal,
                "max_fixture_candidates": MAX_FIXTURE_CANDIDATES,
                "min_reference_fourth_minus_fifth_logit_gap": min_gap,
                "reference_only": True,
                "candidate_output_used": False,
                "selected_before_correctness_timing_profiling": True,
            }
    raise RuntimeError(
        "issue419 found no tie-free reference fixture within the frozen 32-candidate cap "
        f"for {_row_key(dtype_name, batch_size, validity_kind)}"
    )


def correctness_row(
    memory: CoalescedFICEMMemory,
    case: ReadCase,
    reference: TorchFICEMReferenceBackend,
    candidate: TritonFICEMReadBackend,
) -> dict[str, Any]:
    identity_before = case.identity.clone()
    context_before = case.context.clone()
    state_before = _clone_state(case.state)

    reference_result = _full_read(reference, memory, case)
    candidate_result = _full_read(candidate, memory, case)
    torch.cuda.synchronize()

    query, keys, similarity = _diagnostic_tail_inputs(memory, case)
    if not _boundary_is_distinct(similarity, case.state):
        raise RuntimeError("issue411 synthetic row has a tied fourth/fifth read boundary")
    with torch.no_grad():
        reference_recalled, reference_indices = _reference_tail(similarity, case.state)
        candidate_recalled, candidate_indices = fused_ficem_read_tail(
            similarity,
            case.state.strengths,
            case.state.valid,
            case.state.values,
            return_top_indices=True,
        )
    torch.cuda.synchronize()
    if candidate_indices is None:
        raise RuntimeError("issue411 candidate did not return diagnostic top indices")

    atol, rtol = _tolerances(case.dtype_name)
    selected_set_exact = torch.equal(
        torch.sort(reference_indices, dim=-1).values,
        torch.sort(candidate_indices.to(torch.long), dim=-1).values,
    )
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
        and _state_equal(case.state, state_before)
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
        selected_set_exact
        and recalled_close
        and final_close
        and reuse_exact
        and source_unchanged
        and finite
        and meta_exact
    )
    return {
        "pass": passed,
        "selected_top4_set_exact": selected_set_exact,
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


def near_tie_correctness(dtype_name: str, device: torch.device) -> dict[str, Any]:
    dtype = _dtype_from_name(dtype_name)
    similarity = torch.full((1, 1, CAPACITY), -1.0, dtype=dtype, device=device)
    if dtype_name == "float32":
        values = [1.0, 0.999, 0.998, 0.997, 0.996]
    else:
        values = [1.0, 0.98, 0.96, 0.94, 0.92]
    similarity[0, 0, :5] = torch.tensor(values, dtype=dtype, device=device)
    strengths = torch.ones((1, CAPACITY), dtype=dtype, device=device)
    valid = torch.ones((1, CAPACITY), dtype=torch.bool, device=device)
    g = torch.Generator().manual_seed(DESIGN_SEED + (3 if dtype_name == "float32" else 4))
    payload = torch.randn(1, CAPACITY, MEMORY_DIM, generator=g).to(dtype=dtype, device=device)
    state = ContextualEpisodicMemoryState(
        keys=torch.zeros_like(payload), values=payload, strengths=strengths, valid=valid
    )
    reference_recalled, reference_indices = _reference_tail(similarity, state)
    candidate_recalled, candidate_indices = fused_ficem_read_tail(
        similarity, strengths, valid, payload, return_top_indices=True
    )
    torch.cuda.synchronize()
    if candidate_indices is None:
        raise RuntimeError("issue411 near-tie diagnostic missing indices")
    atol, rtol = _tolerances(dtype_name)
    return {
        "pass": bool(
            torch.equal(
                torch.sort(reference_indices, dim=-1).values,
                torch.sort(candidate_indices.to(torch.long), dim=-1).values,
            )
            and torch.allclose(
                reference_recalled, candidate_recalled, atol=atol, rtol=rtol
            )
        ),
        "selected_top4_set_exact": bool(
            torch.equal(
                torch.sort(reference_indices, dim=-1).values,
                torch.sort(candidate_indices.to(torch.long), dim=-1).values,
            )
        ),
        "max_abs_diff": float(
            (reference_recalled.float() - candidate_recalled.float()).abs().max()
        ),
    }


def _cuda_profile(call: Callable[[], Any]) -> dict[str, Any]:
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

    relevant = {
        "topk": 0,
        "softmax": 0,
        "gather": 0,
        "masked_fill": 0,
        "log": 0,
        "einsum": 0,
    }
    for item in profile.key_averages():
        key = str(item.key).lower()
        for token in relevant:
            if token in key:
                relevant[token] += int(item.count)
    return {
        "cuda_device_events": int(cuda_events),
        "triton_read_tail_events": int(triton_events),
        "relevant_operator_calls": relevant,
    }


def _timed_round_us(call: Callable[[], Any]) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    output = None
    for _ in range(CALLS_PER_ROUND):
        output = call()
    end.record()
    end.synchronize()
    if output is None:
        raise RuntimeError("issue411 timed loop produced no output")
    return float(start.elapsed_time(end)) * 1000.0 / CALLS_PER_ROUND


def _timed_summaries(calls: dict[str, Callable[[], Any]]) -> dict[str, dict[str, Any]]:
    for call in calls.values():
        for _ in range(WARMUP_CALLS):
            call()
    torch.cuda.synchronize()
    samples: dict[str, list[float]] = {name: [] for name in calls}
    names = list(calls)
    for round_index in range(TIMED_ROUNDS):
        order = list(reversed(names)) if round_index % 2 else names
        for name in order:
            samples[name].append(_timed_round_us(calls[name]))
    return {
        name: {
            "round_us_per_call": values,
            "median_us_per_call": float(statistics.median(values)),
            "mean_us_per_call": float(statistics.fmean(values)),
        }
        for name, values in samples.items()
    }


def _peak_vram(call: Callable[[], Any]) -> dict[str, float]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    output = call()
    torch.cuda.synchronize()
    result = {
        "peak_allocated_mb": float(torch.cuda.max_memory_allocated() / (1024 * 1024)),
        "peak_reserved_mb": float(torch.cuda.max_memory_reserved() / (1024 * 1024)),
    }
    del output
    return result


def known_empty_case(
    memory: CoalescedFICEMMemory,
    dtype_name: str,
    batch_size: int,
    device: torch.device,
    reference: TorchFICEMReferenceBackend,
    candidate: TritonFICEMReadBackend,
) -> dict[str, Any]:
    dtype = _dtype_from_name(dtype_name)
    g = torch.Generator().manual_seed(DESIGN_SEED + 1000 + batch_size)
    identity = torch.randn(batch_size, TIME, D_MODEL, generator=g).to(dtype=dtype, device=device)
    context = torch.randn(batch_size, TIME, D_MODEL, generator=g).to(dtype=dtype, device=device)
    state = memory.empty_state(batch_size, device, dtype)
    case = ReadCase(dtype_name, batch_size, "empty", identity, context, state)
    reference_result = _full_read(reference, memory, case)
    candidate_result = _full_read(candidate, memory, case)
    profile = _cuda_profile(lambda: _full_read(candidate, memory, case))
    return {
        "pass": bool(
            torch.equal(reference_result.recalled, candidate_result.recalled)
            and not bool(candidate_result.recalled.any())
            and candidate_result.projected_query is None
            and candidate_result.normalized_old_keys is None
            and profile["triton_read_tail_events"] == 0
        ),
        "zero_output_exact": bool(
            torch.equal(reference_result.recalled, candidate_result.recalled)
            and not bool(candidate_result.recalled.any())
        ),
        "reuse_is_none": bool(
            candidate_result.projected_query is None
            and candidate_result.normalized_old_keys is None
        ),
        "triton_read_tail_events": profile["triton_read_tail_events"],
    }


def _row_key(dtype_name: str, batch_size: int, validity_kind: str) -> str:
    return f"{dtype_name}_batch{batch_size}_{validity_kind}"


def run_ficem_read_probe() -> dict[str, Any]:
    cpu_contract_preflight()
    if not torch.cuda.is_available():
        raise RuntimeError("issue411 requires CUDA")
    device = torch.device("cuda")
    device_name = torch.cuda.get_device_name(0)
    if "L4" not in device_name.upper():
        raise RuntimeError(f"issue411 requires NVIDIA L4, found {device_name}")

    memory = build_memory(device)
    reference = TorchFICEMReferenceBackend()
    candidate = TritonFICEMReadBackend()

    near_tie = {
        dtype_name: near_tie_correctness(dtype_name, device)
        for dtype_name in DTYPE_NAMES
    }
    if not all(row["pass"] for row in near_tie.values()):
        raise RuntimeError("issue411 near-tie correctness failed")

    known_empty: dict[str, dict[str, Any]] = {}
    for dtype_name in DTYPE_NAMES:
        for batch_size in BATCH_SIZES:
            key = f"{dtype_name}_batch{batch_size}_empty"
            known_empty[key] = known_empty_case(
                memory, dtype_name, batch_size, device, reference, candidate
            )
            if not known_empty[key]["pass"]:
                raise RuntimeError(f"issue411 known-empty correctness failed for {key}")

    rows: dict[str, dict[str, Any]] = {}
    for dtype_name in DTYPE_NAMES:
        for batch_size in BATCH_SIZES:
            for validity_kind in VALIDITY_KINDS:
                key = _row_key(dtype_name, batch_size, validity_kind)
                case, eligibility = _select_eligible_case(
                    memory=memory,
                    dtype_name=dtype_name,
                    batch_size=batch_size,
                    validity_kind=validity_kind,
                    device=device,
                )
                correctness = correctness_row(memory, case, reference, candidate)
                if not correctness["pass"]:
                    raise RuntimeError(f"issue411 correctness failed for {key}")

                calls = {
                    "reference": lambda case=case: _full_read(reference, memory, case),
                    "candidate": lambda case=case: _full_read(candidate, memory, case),
                }
                timing = _timed_summaries(calls)
                reference_us = timing["reference"]["median_us_per_call"]
                candidate_us = timing["candidate"]["median_us_per_call"]
                latency_ratio = candidate_us / reference_us
                profiles = {name: _cuda_profile(call) for name, call in calls.items()}
                reference_events = profiles["reference"]["cuda_device_events"]
                candidate_events = profiles["candidate"]["cuda_device_events"]
                if reference_events <= 0 or candidate_events <= 0:
                    raise RuntimeError(f"issue411 profiler found no CUDA events for {key}")
                event_ratio = candidate_events / reference_events

                _, _, similarity = _diagnostic_tail_inputs(memory, case)
                tail_call = lambda similarity=similarity, case=case: fused_ficem_read_tail(
                    similarity,
                    case.state.strengths,
                    case.state.valid,
                    case.state.values,
                    return_top_indices=False,
                )
                tail_profile = _cuda_profile(tail_call)

                rows[key] = {
                    "dtype": dtype_name,
                    "batch_size": batch_size,
                    "validity_kind": validity_kind,
                    "eligibility": eligibility,
                    "correctness": correctness,
                    "timing": timing,
                    "latency_ratio_candidate_over_reference": float(latency_ratio),
                    "profiles": profiles,
                    "full_cuda_event_ratio_candidate_over_reference": float(event_ratio),
                    "candidate_tail_profile": tail_profile,
                    "vram": {name: _peak_vram(call) for name, call in calls.items()},
                    "row_latency_pass": bool(latency_ratio <= MAX_ROW_LATENCY_RATIO),
                    "full_event_ratio_pass": bool(event_ratio <= MAX_FULL_EVENT_RATIO),
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
        geomean_pass[dtype_name] = bool(geomean <= MAX_GEOMEAN_LATENCY_RATIO)

    correctness_pass = all(row["correctness"]["pass"] for row in rows.values())
    known_empty_pass = all(row["pass"] for row in known_empty.values())
    near_tie_pass = all(row["pass"] for row in near_tie.values())
    row_latency_pass = all(row["row_latency_pass"] for row in rows.values())
    event_ratio_pass = all(row["full_event_ratio_pass"] for row in rows.values())
    single_tail_kernel_pass = all(row["single_tail_kernel_pass"] for row in rows.values())
    no_reference_tail_ops_pass = all(
        row["candidate_no_reference_tail_ops_pass"] for row in rows.values()
    )
    eligibility_pass = all(
        row["eligibility"]["reference_only"]
        and not row["eligibility"]["candidate_output_used"]
        and row["eligibility"]["selected_before_correctness_timing_profiling"]
        and row["eligibility"]["selected_candidate_ordinal"] <= MAX_FIXTURE_CANDIDATES
        and row["eligibility"]["min_reference_fourth_minus_fifth_logit_gap"] > 0.0
        for row in rows.values()
    )
    overall_pass = bool(
        correctness_pass
        and known_empty_pass
        and near_tie_pass
        and eligibility_pass
        and row_latency_pass
        and event_ratio_pass
        and single_tail_kernel_pass
        and no_reference_tail_ops_pass
        and all(geomean_pass.values())
    )

    return {
        "protocol": issue411_protocol(),
        "device": device_name,
        "near_tie_correctness": near_tie,
        "known_empty_correctness": known_empty,
        "rows": rows,
        "geomean_latency_ratio_by_dtype": geomeans,
        "geomean_latency_pass_by_dtype": geomean_pass,
        "correctness_pass": correctness_pass,
        "known_empty_pass": known_empty_pass,
        "near_tie_pass": near_tie_pass,
        "eligibility_pass": eligibility_pass,
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
