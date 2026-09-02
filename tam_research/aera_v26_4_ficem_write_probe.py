from __future__ import annotations

"""Synthetic/non-scientific L4 gate for the AERA-v26.4 fused FICEM WRITE primitive.

Preregistered by issue #488.  This module does not load a model, checkpoint, corpus,
optimizer, or scientific seed.  It compares the exact merged production-shaped
`update_from_projected` path and the isolated post-similarity write tail against the
frozen PyTorch/stable-compaction semantics.
"""

from contextlib import nullcontext
from dataclasses import dataclass
import math
import statistics
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .aera_hardware_core_v24 import ContextualEpisodicMemoryState
from .aera_hardware_core_v25 import FactorizedIdentityContextEpisodicMemory
from .aera_hardware_core_v25_1 import ExecutionEquivalentFactorizedIdentityContextMemory
from .aera_hardware_core_v25_1_compact import StableCompactExecutionEquivalentFactorizedIdentityContextMemory
from .aera_hardware_core_v26 import CoalescedFICEMMemory, TorchFICEMReferenceBackend
from .aera_hardware_core_v26_4_ficem_write_triton import (
    TritonFICEMReadWriteBackend,
    fused_ficem_read_write_v26_4_protocol,
    fused_ficem_write_tail,
)

RESEARCH_ISSUE = 488
DESIGN_SEED = 408_487
D_MODEL = 200
WRITE_COUNT = 16
CAPACITY = 48
MEMORY_DIM = 50
DUPLICATE_THRESHOLD = 0.95
BATCH_SIZES = (8, 64)
DTYPE_NAMES = ("float32", "bfloat16")
VALIDITY_KINDS = ("mixed", "full")
EDGE_FIXTURES = (
    "empty_old_all_new_valid",
    "mixed_incoming_validity",
    "incoming_duplicate_newest_wins",
    "threshold_inclusive_and_below_control",
    "surviving_new_suppresses_old",
    "shadowed_new_does_not_suppress_old",
    "over_capacity_truncation",
    "invalid_retained_storage_order",
)
WARMUP_CALLS = 10
TIMED_ROUNDS = 5
CALLS_PER_ROUND = 100
FP32_ATOL = 1e-5
FP32_RTOL = 1e-5
BF16_ATOL = 1e-2
BF16_RTOL = 1e-2
MAX_GEOMEAN_LATENCY_RATIO = 0.90
MAX_ROW_LATENCY_RATIO = 1.05
MAX_TAIL_EVENT_RATIO = 0.75


class _RawFICEMSource(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.memory_dim = MEMORY_DIM
        self.q = nn.Linear(D_MODEL, MEMORY_DIM, bias=False)
        self.k = nn.Linear(D_MODEL, MEMORY_DIM, bias=False)
        self.v = nn.Linear(D_MODEL, MEMORY_DIM, bias=False)
        self.out = nn.Linear(MEMORY_DIM, D_MODEL, bias=False)
        self.differentiable_pretraining = False


@dataclass(frozen=True)
class WriteCase:
    dtype_name: str
    batch_size: int
    validity_kind: str
    projected_new_keys: torch.Tensor
    normalized_old_keys: torch.Tensor
    payload: torch.Tensor
    strength: torch.Tensor
    state: ContextualEpisodicMemoryState


@dataclass(frozen=True)
class TailInputs:
    incoming_similarity: torch.Tensor
    old_similarity: torch.Tensor
    new_keys: torch.Tensor
    new_values: torch.Tensor
    new_strengths: torch.Tensor
    new_valid: torch.Tensor
    state: ContextualEpisodicMemoryState


def issue488_protocol() -> dict[str, Any]:
    return {
        **fused_ficem_read_write_v26_4_protocol(),
        "probe_version": "aera-v26.4-issue488-synthetic-ficem-write-l4",
        "design_seed": DESIGN_SEED,
        "design_seed_is_scientific_seed": False,
        "write_count": WRITE_COUNT,
        "capacity": CAPACITY,
        "memory_dim": MEMORY_DIM,
        "duplicate_threshold": DUPLICATE_THRESHOLD,
        "batch_sizes": list(BATCH_SIZES),
        "dtypes": list(DTYPE_NAMES),
        "validity_kinds": list(VALIDITY_KINDS),
        "edge_fixtures": list(EDGE_FIXTURES),
        "warmup_calls": WARMUP_CALLS,
        "timed_rounds": TIMED_ROUNDS,
        "calls_per_round": CALLS_PER_ROUND,
        "fp32_atol": FP32_ATOL,
        "fp32_rtol": FP32_RTOL,
        "bfloat16_atol": BF16_ATOL,
        "bfloat16_rtol": BF16_RTOL,
        "max_geomean_latency_ratio_each_dtype": MAX_GEOMEAN_LATENCY_RATIO,
        "max_row_latency_ratio": MAX_ROW_LATENCY_RATIO,
        "max_tail_cuda_event_ratio": MAX_TAIL_EVENT_RATIO,
        "synthetic_only": True,
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


def cpu_contract_preflight() -> dict[str, Any]:
    assert DESIGN_SEED == 408_487
    assert (D_MODEL, WRITE_COUNT, CAPACITY, MEMORY_DIM) == (200, 16, 48, 50)
    assert DUPLICATE_THRESHOLD == 0.95
    assert BATCH_SIZES == (8, 64)
    assert DTYPE_NAMES == ("float32", "bfloat16")
    assert VALIDITY_KINDS == ("mixed", "full")
    assert len(EDGE_FIXTURES) == 8
    assert (WARMUP_CALLS, TIMED_ROUNDS, CALLS_PER_ROUND) == (10, 5, 100)
    assert (FP32_ATOL, FP32_RTOL, BF16_ATOL, BF16_RTOL) == (1e-5, 1e-5, 1e-2, 1e-2)
    assert (MAX_GEOMEAN_LATENCY_RATIO, MAX_ROW_LATENCY_RATIO, MAX_TAIL_EVENT_RATIO) == (0.90, 1.05, 0.75)
    return {
        "protocol": issue488_protocol(),
        "gpu_authorized_by_cpu_preflight": False,
        "synthetic_only": True,
        "scientific_seed_consumed": False,
    }


def _dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(name)


def _precision_context(dtype_name: str):
    if dtype_name == "bfloat16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def build_memory(device: torch.device) -> CoalescedFICEMMemory:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(DESIGN_SEED + 1)
        raw = _RawFICEMSource()
        v25 = FactorizedIdentityContextEpisodicMemory(raw, capacity=CAPACITY)
        v251 = ExecutionEquivalentFactorizedIdentityContextMemory(v25)
        compact = StableCompactExecutionEquivalentFactorizedIdentityContextMemory(v251)
        memory = CoalescedFICEMMemory(compact)
    memory.set_differentiable_pretraining(False)
    return memory.to(device).eval()


def _clone_state(state: ContextualEpisodicMemoryState) -> ContextualEpisodicMemoryState:
    return ContextualEpisodicMemoryState(
        keys=state.keys.clone(), values=state.values.clone(),
        strengths=state.strengths.clone(), valid=state.valid.clone()
    )


def _state_bit_equal(a: ContextualEpisodicMemoryState, b: ContextualEpisodicMemoryState) -> bool:
    return all(torch.equal(x, y) for x, y in (
        (a.keys, b.keys), (a.values, b.values), (a.strengths, b.strengths), (a.valid, b.valid)
    ))


def _cpu_randn(shape: tuple[int, ...], generator: torch.Generator, dtype: torch.dtype) -> torch.Tensor:
    return torch.randn(*shape, generator=generator).to(dtype=dtype)


def make_case(*, dtype_name: str, batch_size: int, validity_kind: str,
              generator: torch.Generator, device: torch.device) -> WriteCase:
    dtype = _dtype(dtype_name)
    new_keys = F.normalize(_cpu_randn((batch_size, WRITE_COUNT, MEMORY_DIM), generator, torch.float32), dim=-1).to(dtype=dtype, device=device)
    old_keys = _cpu_randn((batch_size, CAPACITY, MEMORY_DIM), generator, dtype).to(device)
    old_values = _cpu_randn((batch_size, CAPACITY, MEMORY_DIM), generator, dtype).to(device)
    old_strengths = torch.rand(batch_size, CAPACITY, generator=generator).mul_(0.9).add_(0.05).to(dtype=dtype, device=device)
    if validity_kind == "full":
        old_valid = torch.ones(batch_size, CAPACITY, dtype=torch.bool, device=device)
    else:
        old_valid = (torch.rand(batch_size, CAPACITY, generator=generator) > 0.35).to(device)
        old_valid[:, :8] = True
    state = ContextualEpisodicMemoryState(old_keys, old_values, old_strengths, old_valid)
    payload = _cpu_randn((batch_size, WRITE_COUNT, D_MODEL), generator, dtype).to(device)
    strength = torch.rand(batch_size, WRITE_COUNT, 1, generator=generator).to(dtype=dtype, device=device)
    strength[:, 0, 0] = 0.0
    normalized_old = F.normalize(old_keys, dim=-1)
    return WriteCase(dtype_name, batch_size, validity_kind, new_keys, normalized_old, payload, strength, state)


def _pre_tail(memory: CoalescedFICEMMemory, case: WriteCase) -> TailInputs:
    with torch.no_grad(), _precision_context(case.dtype_name):
        new_values = torch.tanh(memory.v(case.payload))
        new_strengths = case.strength[..., 0].clamp(0.0, 1.0)
        new_valid = new_strengths > 0.0
        incoming = torch.einsum("bkd,bjd->bkj", case.projected_new_keys, case.projected_new_keys)
        old_similarity = torch.einsum("bkd,bsd->bks", case.projected_new_keys, case.normalized_old_keys)
    return TailInputs(incoming.contiguous(), old_similarity.contiguous(), case.projected_new_keys.contiguous(),
                      new_values.contiguous(), new_strengths.contiguous(), new_valid.contiguous(), _clone_state(case.state))


def _reference_tail(memory: CoalescedFICEMMemory, inputs: TailInputs) -> ContextualEpisodicMemoryState:
    position = torch.arange(WRITE_COUNT, device=inputs.new_keys.device)
    later = position[None, :, None] < position[None, None, :]
    shadowed = (
        inputs.incoming_similarity.ge(DUPLICATE_THRESHOLD)
        & inputs.new_valid[:, :, None]
        & inputs.new_valid[:, None, :]
        & later
    ).any(dim=2)
    surviving_new = inputs.new_valid & ~shadowed
    duplicate_old = (
        inputs.old_similarity.ge(DUPLICATE_THRESHOLD)
        & surviving_new[:, :, None]
        & inputs.state.valid[:, None, :]
    ).any(dim=1)
    keep_old = inputs.state.valid & ~duplicate_old
    return memory._stable_compact_state(
        inputs.new_keys.flip(1), inputs.new_values.flip(1), inputs.new_strengths.flip(1),
        surviving_new.flip(1), inputs.state.keys, inputs.state.values,
        inputs.state.strengths, keep_old,
    )


def _candidate_tail(inputs: TailInputs) -> ContextualEpisodicMemoryState:
    return fused_ficem_write_tail(
        inputs.incoming_similarity, inputs.old_similarity, inputs.new_keys, inputs.new_values,
        inputs.new_strengths, inputs.new_valid, inputs.state,
    )


def _tolerances(dtype_name: str) -> tuple[float, float]:
    return (BF16_ATOL, BF16_RTOL) if dtype_name == "bfloat16" else (FP32_ATOL, FP32_RTOL)


def _compare_states(reference: ContextualEpisodicMemoryState,
                    candidate: ContextualEpisodicMemoryState,
                    *, dtype_name: str) -> dict[str, Any]:
    atol, rtol = _tolerances(dtype_name)
    float_close = {}
    max_abs = {}
    for name in ("keys", "values", "strengths"):
        ref = getattr(reference, name)
        cand = getattr(candidate, name)
        float_close[name] = bool(torch.allclose(ref, cand, atol=atol, rtol=rtol))
        max_abs[name] = float((ref.float() - cand.float()).abs().max())
    valid_exact = bool(torch.equal(reference.valid, candidate.valid))
    dtype_device_shape = all(
        getattr(reference, name).dtype == getattr(candidate, name).dtype
        and getattr(reference, name).device == getattr(candidate, name).device
        and getattr(reference, name).shape == getattr(candidate, name).shape
        for name in ("keys", "values", "strengths", "valid")
    )
    finite = all(bool(torch.isfinite(getattr(candidate, name)).all()) for name in ("keys", "values", "strengths"))
    passed = valid_exact and dtype_device_shape and finite and all(float_close.values())
    return {"pass": passed, "valid_exact": valid_exact, "float_close": float_close,
            "max_abs": max_abs, "dtype_device_shape_exact": dtype_device_shape, "finite": finite}


def projected_correctness(memory: CoalescedFICEMMemory, case: WriteCase,
                          candidate_backend: TritonFICEMReadWriteBackend,
                          reference_backend: TorchFICEMReferenceBackend) -> dict[str, Any]:
    source_before = _clone_state(case.state)
    tensors_before = [x.clone() for x in (case.projected_new_keys, case.normalized_old_keys, case.payload, case.strength)]
    with torch.no_grad(), _precision_context(case.dtype_name):
        reference = reference_backend.update_from_projected(
            memory, case.projected_new_keys, case.normalized_old_keys, case.payload, case.strength, _clone_state(case.state))
        candidate = candidate_backend.update_from_projected(
            memory, case.projected_new_keys, case.normalized_old_keys, case.payload, case.strength, _clone_state(case.state))
    comparison = _compare_states(reference, candidate, dtype_name=case.dtype_name)
    source_unchanged = _state_bit_equal(case.state, source_before) and all(
        torch.equal(now, old) for now, old in zip(
            (case.projected_new_keys, case.normalized_old_keys, case.payload, case.strength), tensors_before
        )
    )
    comparison["source_unchanged"] = source_unchanged
    comparison["pass"] = bool(comparison["pass"] and source_unchanged)
    return comparison


def tail_correctness(memory: CoalescedFICEMMemory, inputs: TailInputs, *, dtype_name: str) -> dict[str, Any]:
    source_before = TailInputs(inputs.incoming_similarity.clone(), inputs.old_similarity.clone(), inputs.new_keys.clone(),
                               inputs.new_values.clone(), inputs.new_strengths.clone(), inputs.new_valid.clone(), _clone_state(inputs.state))
    with torch.no_grad():
        reference = _reference_tail(memory, inputs)
        candidate = _candidate_tail(inputs)
    comparison = _compare_states(reference, candidate, dtype_name=dtype_name)
    source_unchanged = (
        torch.equal(inputs.incoming_similarity, source_before.incoming_similarity)
        and torch.equal(inputs.old_similarity, source_before.old_similarity)
        and torch.equal(inputs.new_keys, source_before.new_keys)
        and torch.equal(inputs.new_values, source_before.new_values)
        and torch.equal(inputs.new_strengths, source_before.new_strengths)
        and torch.equal(inputs.new_valid, source_before.new_valid)
        and _state_bit_equal(inputs.state, source_before.state)
    )
    comparison["source_unchanged"] = source_unchanged
    comparison["pass"] = bool(comparison["pass"] and source_unchanged)
    return comparison


def _base_edge_inputs(*, dtype_name: str, device: torch.device) -> TailInputs:
    dtype = _dtype(dtype_name)
    batch = 1
    new_keys = torch.arange(WRITE_COUNT * MEMORY_DIM, device=device, dtype=torch.float32).reshape(batch, WRITE_COUNT, MEMORY_DIM).div_(1000).to(dtype)
    new_values = (new_keys + 3).clone()
    new_strengths = torch.linspace(0.1, 0.9, WRITE_COUNT, device=device).reshape(1, -1).to(dtype)
    new_valid = torch.ones(batch, WRITE_COUNT, device=device, dtype=torch.bool)
    old_keys = torch.arange(CAPACITY * MEMORY_DIM, device=device, dtype=torch.float32).reshape(batch, CAPACITY, MEMORY_DIM).div_(2000).sub_(2).to(dtype)
    old_values = (old_keys - 5).clone()
    old_strengths = torch.linspace(0.05, 0.95, CAPACITY, device=device).reshape(1, -1).to(dtype)
    old_valid = torch.ones(batch, CAPACITY, device=device, dtype=torch.bool)
    incoming = torch.eye(WRITE_COUNT, device=device, dtype=dtype)
    old_similarity = torch.zeros(batch, WRITE_COUNT, CAPACITY, device=device, dtype=dtype)
    return TailInputs(incoming.unsqueeze(0).contiguous(), old_similarity, new_keys.contiguous(), new_values.contiguous(),
                      new_strengths.contiguous(), new_valid, ContextualEpisodicMemoryState(old_keys.contiguous(), old_values.contiguous(), old_strengths.contiguous(), old_valid))


def make_edge_fixture(name: str, *, dtype_name: str, device: torch.device) -> TailInputs:
    x = _base_edge_inputs(dtype_name=dtype_name, device=device)
    if name == "empty_old_all_new_valid":
        x.state.valid.zero_()
    elif name == "mixed_incoming_validity":
        x.new_valid[:, 1::2] = False
    elif name == "incoming_duplicate_newest_wins":
        x.incoming_similarity[:, 0, 1] = 1.0
        x.incoming_similarity[:, 1, 0] = 1.0
        x.incoming_similarity[:, 1, 2] = 1.0
        x.incoming_similarity[:, 2, 1] = 1.0
    elif name == "threshold_inclusive_and_below_control":
        x.incoming_similarity[:, 0, 1] = DUPLICATE_THRESHOLD
        x.incoming_similarity[:, 1, 0] = DUPLICATE_THRESHOLD
        x.incoming_similarity[:, 2, 3] = 0.949
        x.incoming_similarity[:, 3, 2] = 0.949
    elif name == "surviving_new_suppresses_old":
        x.old_similarity[:, 7, 3] = 1.0
    elif name == "shadowed_new_does_not_suppress_old":
        x.incoming_similarity[:, 0, 1] = 1.0
        x.incoming_similarity[:, 1, 0] = 1.0
        x.old_similarity[:, 0, 4] = 1.0
    elif name == "over_capacity_truncation":
        pass
    elif name == "invalid_retained_storage_order":
        x.new_valid[:, 0:6:2] = False
        x.state.valid[:, 0:8:2] = False
    else:
        raise ValueError(name)
    return x


def _event_ms(call: Callable[[], Any], calls: int = 1) -> float:
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(calls):
        call()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)) / calls


def _rotated_timing(candidate: Callable[[], Any], reference: Callable[[], Any]) -> tuple[float, float]:
    for _ in range(WARMUP_CALLS):
        reference(); candidate()
    cand_ms: list[float] = []
    ref_ms: list[float] = []
    for round_index in range(TIMED_ROUNDS):
        if round_index % 2 == 0:
            ref_ms.append(_event_ms(reference, CALLS_PER_ROUND))
            cand_ms.append(_event_ms(candidate, CALLS_PER_ROUND))
        else:
            cand_ms.append(_event_ms(candidate, CALLS_PER_ROUND))
            ref_ms.append(_event_ms(reference, CALLS_PER_ROUND))
    return statistics.median(cand_ms), statistics.median(ref_ms)


def _profile_candidate_tail(call: Callable[[], Any]) -> dict[str, Any]:
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
        call()
        torch.cuda.synchronize()
    rows = []
    for event in prof.key_averages():
        cuda_time = float(getattr(event, "device_time_total", getattr(event, "cuda_time_total", 0.0)) or 0.0)
        if cuda_time > 0:
            rows.append({"key": event.key, "cuda_time_total": cuda_time})
    keys = [row["key"] for row in rows]
    adjudicate = [key for key in keys if "_write_adjudicate_map_kernel" in key]
    materialize = [key for key in keys if "_write_materialize_kernel" in key]
    forbidden_fragments = ("aten::cat", "aten::stack", "aten::cumsum", "aten::scatter_add")
    forbidden = [key for key in keys if any(fragment in key for fragment in forbidden_fragments)]
    return {
        "events": rows,
        "adjudicate_kernel_events": adjudicate,
        "materialize_kernel_events": materialize,
        "forbidden_tail_events": forbidden,
        "pass": len(adjudicate) == 1 and len(materialize) == 1 and not forbidden,
    }


def run_ficem_write_probe() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("issue488 FICEM WRITE gate requires CUDA")
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    torch.manual_seed(DESIGN_SEED)
    torch.cuda.manual_seed_all(DESIGN_SEED)
    memory = build_memory(device)
    candidate_backend = TritonFICEMReadWriteBackend()
    reference_backend = TorchFICEMReferenceBackend()
    generator = torch.Generator().manual_seed(DESIGN_SEED)

    rows: dict[str, Any] = {}
    all_correct = True
    all_row_latency = True
    all_tail_ratio = True
    all_topology = True
    by_dtype: dict[str, list[float]] = {name: [] for name in DTYPE_NAMES}

    for dtype_name in DTYPE_NAMES:
        for batch_size in BATCH_SIZES:
            for validity_kind in VALIDITY_KINDS:
                key = f"{dtype_name}_batch{batch_size}_{validity_kind}"
                case = make_case(dtype_name=dtype_name, batch_size=batch_size, validity_kind=validity_kind,
                                 generator=generator, device=device)
                projected = projected_correctness(memory, case, candidate_backend, reference_backend)
                tail_inputs = _pre_tail(memory, case)
                tail = tail_correctness(memory, tail_inputs, dtype_name=dtype_name)

                def cand_projected() -> Any:
                    with torch.no_grad(), _precision_context(dtype_name):
                        return candidate_backend.update_from_projected(memory, case.projected_new_keys, case.normalized_old_keys,
                                                                       case.payload, case.strength, case.state)
                def ref_projected() -> Any:
                    with torch.no_grad(), _precision_context(dtype_name):
                        return reference_backend.update_from_projected(memory, case.projected_new_keys, case.normalized_old_keys,
                                                                       case.payload, case.strength, case.state)
                cand_ms, ref_ms = _rotated_timing(cand_projected, ref_projected)
                ratio = cand_ms / ref_ms
                by_dtype[dtype_name].append(ratio)

                def cand_tail() -> Any:
                    with torch.no_grad():
                        return _candidate_tail(tail_inputs)
                def ref_tail() -> Any:
                    with torch.no_grad():
                        return _reference_tail(memory, tail_inputs)
                cand_tail_ms, ref_tail_ms = _rotated_timing(cand_tail, ref_tail)
                tail_ratio = cand_tail_ms / ref_tail_ms
                topology = _profile_candidate_tail(cand_tail)
                row_correct = bool(projected["pass"] and tail["pass"])
                row_latency = ratio <= MAX_ROW_LATENCY_RATIO
                row_tail = tail_ratio <= MAX_TAIL_EVENT_RATIO
                rows[key] = {
                    "projected_correctness": projected,
                    "tail_correctness": tail,
                    "candidate_projected_ms": cand_ms,
                    "reference_projected_ms": ref_ms,
                    "latency_ratio_candidate_over_reference": ratio,
                    "candidate_tail_ms": cand_tail_ms,
                    "reference_tail_ms": ref_tail_ms,
                    "tail_cuda_event_ratio_candidate_over_reference": tail_ratio,
                    "topology": topology,
                    "pass": row_correct and row_latency and row_tail and topology["pass"],
                }
                all_correct &= row_correct
                all_row_latency &= row_latency
                all_tail_ratio &= row_tail
                all_topology &= bool(topology["pass"])

    edge_results: dict[str, Any] = {}
    for dtype_name in DTYPE_NAMES:
        for fixture_name in EDGE_FIXTURES:
            edge = make_edge_fixture(fixture_name, dtype_name=dtype_name, device=device)
            result = tail_correctness(memory, edge, dtype_name=dtype_name)
            edge_results[f"{dtype_name}_{fixture_name}"] = result
            all_correct &= bool(result["pass"])

    geomean = {
        dtype_name: math.exp(sum(math.log(value) for value in values) / len(values))
        for dtype_name, values in by_dtype.items()
    }
    geomean_pass = all(value <= MAX_GEOMEAN_LATENCY_RATIO for value in geomean.values())
    overall = all_correct and all_row_latency and all_tail_ratio and all_topology and geomean_pass
    result = {
        "decision": "PASS" if overall else "FAIL",
        "overall_pass": overall,
        "device": torch.cuda.get_device_name(0),
        "protocol": issue488_protocol(),
        "rows": rows,
        "edge_fixtures": edge_results,
        "correctness_pass": all_correct,
        "row_latency_pass": all_row_latency,
        "tail_event_ratio_pass": all_tail_ratio,
        "two_kernel_topology_pass": all_topology,
        "geomean_latency_ratio_by_dtype": geomean,
        "geomean_latency_pass": geomean_pass,
        "synthetic_only": True,
        "scientific_seed_consumed": False,
        "end_to_end_systems_authorized": False,
        "architecture_freeze_authorized": False,
        "s2_authorized": False,
        "fresh_scientific_seed_authorized": False,
        "100m_authorized": False,
        "breakthrough_proven": False,
    }
    return result
