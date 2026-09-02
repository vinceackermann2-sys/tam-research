from __future__ import annotations

"""Issue #487 production-shaped synthetic L4 gate for fused FICEM WRITE."""

from contextlib import nullcontext
from dataclasses import dataclass
import math
import statistics
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .aera_hardware_core_v24 import ContextualEpisodicMemoryState, DUPLICATE_SIMILARITY
from .aera_hardware_core_v25 import FactorizedIdentityContextEpisodicMemory
from .aera_hardware_core_v25_1 import ExecutionEquivalentFactorizedIdentityContextMemory
from .aera_hardware_core_v25_1_compact import (
    StableCompactExecutionEquivalentFactorizedIdentityContextMemory,
)
from .aera_hardware_core_v26 import CoalescedFICEMMemory, TorchFICEMReferenceBackend
from .aera_hardware_core_v26_4_ficem_write_triton import (
    TritonFICEMReadWriteBackend,
    WRITE_CAPACITY,
    WRITE_COUNT,
    WRITE_DUPLICATE_SIMILARITY,
    WRITE_MEMORY_DIM,
    _write_adjudicate_map_kernel,
    fused_ficem_read_write_v26_4_protocol,
    fused_ficem_write_tail,
)

RESEARCH_ISSUE = 487
DESIGN_SEED = 487_485
D_MODEL = 200
MEMORY_DIM = 50
CAPACITY = 48
WRITE_K = 16
BATCH_SIZES: tuple[int, ...] = (8, 64)
DTYPE_NAMES: tuple[str, ...] = ("float32", "bfloat16")
VALIDITY_KINDS: tuple[str, ...] = ("mixed", "full")
STRESS_KINDS: tuple[str, ...] = (
    "empty_old",
    "incoming_duplicate_heavy",
    "old_duplicate_heavy",
    "combined_duplicates",
    "near_threshold",
)

WARMUP_CALLS = 10
TIMED_ROUNDS = 5
CALLS_PER_ROUND = 100
MAX_GEOMEAN_LATENCY_RATIO = 0.90
MAX_ROW_LATENCY_RATIO = 1.05
MAX_TAIL_EVENT_RATIO = 0.25
NEAR_THRESHOLD_LOW = 0.94
NEAR_THRESHOLD_HIGH = 0.96

SOURCE_MAIN = "c0ee36ba66e11d24bb9990787e125e986171a46e"
WRITE_BACKEND_BLOB = "5d703bbba296328ca2f49407e56192d10541349d"
READ_BACKEND_BLOB = "263f68eb1186a8ac14a08fc4b4df1fc5b292c711"
V26_INTERFACE_BLOB = "268644ac4edee15a4cc4e29d3fed7f61eeb3caa7"
STABLE_REFERENCE_BLOB = "4e336b6e1a6238dac782fa320751d68281493ee1"


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
    payload_source: torch.Tensor
    write_strength: torch.Tensor
    state: ContextualEpisodicMemoryState


def issue487_protocol() -> dict[str, Any]:
    protocol = dict(fused_ficem_read_write_v26_4_protocol())
    protocol.update(
        {
            "probe_version": "aera-v26.4-issue487-synthetic-ficem-write-l4",
            "design_seed": DESIGN_SEED,
            "design_seed_is_scientific_seed": False,
            "d_model": D_MODEL,
            "memory_dim": MEMORY_DIM,
            "capacity": CAPACITY,
            "write_k": WRITE_K,
            "duplicate_similarity": WRITE_DUPLICATE_SIMILARITY,
            "batch_sizes": list(BATCH_SIZES),
            "dtypes": list(DTYPE_NAMES),
            "validity_kinds": list(VALIDITY_KINDS),
            "stress_kinds": list(STRESS_KINDS),
            "warmup_calls": WARMUP_CALLS,
            "timed_rounds": TIMED_ROUNDS,
            "calls_per_round": CALLS_PER_ROUND,
            "max_geomean_latency_ratio_each_dtype": MAX_GEOMEAN_LATENCY_RATIO,
            "max_row_latency_ratio": MAX_ROW_LATENCY_RATIO,
            "max_isolated_tail_event_ratio": MAX_TAIL_EVENT_RATIO,
            "near_threshold_low": NEAR_THRESHOLD_LOW,
            "near_threshold_high": NEAR_THRESHOLD_HIGH,
            "full_call_is_update_from_projected": True,
            "bit_exact_complete_durable_state_required": True,
            "invalid_retained_storage_bit_exact_required": True,
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
    )
    return protocol


def cpu_contract_preflight() -> dict[str, Any]:
    if (D_MODEL, MEMORY_DIM, CAPACITY, WRITE_K) != (200, 50, 48, 16):
        raise RuntimeError("issue487 production geometry drifted")
    if (WRITE_COUNT, WRITE_CAPACITY, WRITE_MEMORY_DIM) != (16, 48, 50):
        raise RuntimeError("issue487 candidate write geometry drifted")
    if WRITE_DUPLICATE_SIMILARITY != DUPLICATE_SIMILARITY or WRITE_DUPLICATE_SIMILARITY != 0.95:
        raise RuntimeError("issue487 duplicate threshold drifted")
    if BATCH_SIZES != (8, 64) or DTYPE_NAMES != ("float32", "bfloat16"):
        raise RuntimeError("issue487 benchmark rows drifted")
    if VALIDITY_KINDS != ("mixed", "full"):
        raise RuntimeError("issue487 validity rows drifted")
    if STRESS_KINDS != (
        "empty_old",
        "incoming_duplicate_heavy",
        "old_duplicate_heavy",
        "combined_duplicates",
        "near_threshold",
    ):
        raise RuntimeError("issue487 stress fixtures drifted")
    if (WARMUP_CALLS, TIMED_ROUNDS, CALLS_PER_ROUND) != (10, 5, 100):
        raise RuntimeError("issue487 timing protocol drifted")
    if (
        MAX_GEOMEAN_LATENCY_RATIO != 0.90
        or MAX_ROW_LATENCY_RATIO != 1.05
        or MAX_TAIL_EVENT_RATIO != 0.25
    ):
        raise RuntimeError("issue487 PASS thresholds drifted")
    return {
        "protocol": issue487_protocol(),
        "source_main": SOURCE_MAIN,
        "candidate_backend_blob": WRITE_BACKEND_BLOB,
        "repair5_read_backend_blob": READ_BACKEND_BLOB,
        "v26_interface_blob": V26_INTERFACE_BLOB,
        "stable_reference_blob": STABLE_REFERENCE_BLOB,
        "gpu_authorized_by_cpu_preflight": False,
        "synthetic_only": True,
        "scientific_seed_consumed": False,
    }


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unsupported issue487 dtype: {name}")


def _precision_context(dtype_name: str):
    if dtype_name == "bfloat16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def build_memory(device: torch.device, dtype_name: str) -> CoalescedFICEMMemory:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(DESIGN_SEED + 1)
        raw = _RawFICEMSource()
        v25 = FactorizedIdentityContextEpisodicMemory(raw, capacity=CAPACITY)
        v251 = ExecutionEquivalentFactorizedIdentityContextMemory(v25)
        compact = StableCompactExecutionEquivalentFactorizedIdentityContextMemory(v251)
        memory = CoalescedFICEMMemory(compact)
    memory.set_differentiable_pretraining(False)
    return memory.to(device=device, dtype=_dtype_from_name(dtype_name)).eval()


def _unit_rows(tensor: torch.Tensor) -> torch.Tensor:
    return F.normalize(tensor.float(), dim=-1).to(tensor.dtype)


def _state_clone(state: ContextualEpisodicMemoryState) -> ContextualEpisodicMemoryState:
    return ContextualEpisodicMemoryState(
        keys=state.keys.clone(),
        values=state.values.clone(),
        strengths=state.strengths.clone(),
        valid=state.valid.clone(),
    )


def _state_equal(a: ContextualEpisodicMemoryState, b: ContextualEpisodicMemoryState) -> bool:
    return (
        torch.equal(a.keys, b.keys)
        and torch.equal(a.values, b.values)
        and torch.equal(a.strengths, b.strengths)
        and torch.equal(a.valid, b.valid)
    )


def _all_finite(state: ContextualEpisodicMemoryState) -> bool:
    return bool(
        torch.isfinite(state.keys).all()
        and torch.isfinite(state.values).all()
        and torch.isfinite(state.strengths).all()
    )


def _copy_pairs_for_primary(keys: torch.Tensor, old_keys: torch.Tensor) -> None:
    # Exact-copy relationships are fixed before candidate execution.
    keys[:, 1] = keys[:, 0]
    keys[:, 5] = keys[:, 4]
    keys[:, 10] = keys[:, 9]
    old_keys[:, 0] = keys[:, 1]
    old_keys[:, 3] = keys[:, 5]
    old_keys[:, 7] = keys[:, 10]


def _make_primary_case(
    *,
    dtype_name: str,
    batch_size: int,
    validity_kind: str,
    generator: torch.Generator,
    device: torch.device,
) -> WriteCase:
    dtype = _dtype_from_name(dtype_name)
    projected = torch.randn(batch_size, WRITE_K, MEMORY_DIM, generator=generator)
    old_keys = torch.randn(batch_size, CAPACITY, MEMORY_DIM, generator=generator)
    projected = _unit_rows(projected).to(dtype=dtype, device=device)
    old_keys = _unit_rows(old_keys).to(dtype=dtype, device=device)
    _copy_pairs_for_primary(projected, old_keys)

    old_values = torch.randn(batch_size, CAPACITY, MEMORY_DIM, generator=generator).to(
        dtype=dtype, device=device
    )
    old_strengths = (0.05 + 0.90 * torch.rand(batch_size, CAPACITY, generator=generator)).to(
        dtype=dtype, device=device
    )
    if validity_kind == "full":
        old_valid = torch.ones(batch_size, CAPACITY, dtype=torch.bool, device=device)
        write_strength = (0.10 + 0.80 * torch.rand(batch_size, WRITE_K, 1, generator=generator)).to(
            dtype=dtype, device=device
        )
    elif validity_kind == "mixed":
        slot = torch.arange(CAPACITY)
        old_valid_cpu = ((slot % 3) != 0)[None, :].expand(batch_size, -1).clone()
        old_valid = old_valid_cpu.to(device=device)
        write_strength = (0.10 + 0.80 * torch.rand(batch_size, WRITE_K, 1, generator=generator)).to(
            dtype=dtype, device=device
        )
        write_strength[:, 3] = 0
        write_strength[:, 7] = 0
        write_strength[:, 12] = 0
    else:
        raise ValueError(f"unknown issue487 validity kind: {validity_kind}")

    payload = torch.randn(batch_size, WRITE_K, D_MODEL, generator=generator).to(
        dtype=dtype, device=device
    )
    state = ContextualEpisodicMemoryState(
        keys=old_keys.contiguous(),
        values=old_values.contiguous(),
        strengths=old_strengths.contiguous(),
        valid=old_valid.contiguous(),
    )
    with _precision_context(dtype_name):
        normalized_old = F.normalize(state.keys, dim=-1)
    return WriteCase(
        dtype_name=dtype_name,
        batch_size=batch_size,
        validity_kind=validity_kind,
        projected_new_keys=projected.contiguous(),
        normalized_old_keys=normalized_old.contiguous(),
        payload_source=payload.contiguous(),
        write_strength=write_strength.contiguous(),
        state=state,
    )


def _basis_pair(similarity: float, *, dim0: int, dim1: int) -> tuple[torch.Tensor, torch.Tensor]:
    a = torch.zeros(MEMORY_DIM)
    b = torch.zeros(MEMORY_DIM)
    a[dim0] = 1.0
    b[dim0] = similarity
    b[dim1] = math.sqrt(max(0.0, 1.0 - similarity * similarity))
    return a, b


def _make_stress_case(
    *,
    dtype_name: str,
    kind: str,
    device: torch.device,
) -> WriteCase:
    dtype = _dtype_from_name(dtype_name)
    kind_index = STRESS_KINDS.index(kind)
    generator = torch.Generator().manual_seed(DESIGN_SEED + 10_000 + 100 * kind_index + (0 if dtype_name == "float32" else 1))
    case = _make_primary_case(
        dtype_name=dtype_name,
        batch_size=8,
        validity_kind="mixed",
        generator=generator,
        device=device,
    )
    projected = case.projected_new_keys.clone()
    old_keys = case.state.keys.clone()
    old_values = case.state.values.clone()
    old_strengths = case.state.strengths.clone()
    old_valid = case.state.valid.clone()
    write_strength = case.write_strength.clone()

    if kind == "empty_old":
        old_valid.zero_()
    elif kind == "incoming_duplicate_heavy":
        for start in (0, 4, 8, 12):
            projected[:, start + 1 : start + 4] = projected[:, start : start + 1]
        write_strength.fill_(0.7)
    elif kind == "old_duplicate_heavy":
        old_valid[:, :16] = True
        old_keys[:, :16] = projected[:, :16]
        write_strength.fill_(0.7)
    elif kind == "combined_duplicates":
        projected[:, 1:4] = projected[:, 0:1]
        projected[:, 9:12] = projected[:, 8:9]
        old_valid[:, :8] = True
        old_keys[:, 0] = projected[:, 3]
        old_keys[:, 1] = projected[:, 11]
        write_strength.fill_(0.65)
        write_strength[:, 2] = 0
        write_strength[:, 10] = 0
    elif kind == "near_threshold":
        low_a, low_b = _basis_pair(NEAR_THRESHOLD_LOW, dim0=0, dim1=1)
        high_a, high_b = _basis_pair(NEAR_THRESHOLD_HIGH, dim0=2, dim1=3)
        projected[:, 0] = low_a.to(dtype=dtype, device=device)
        projected[:, 1] = low_b.to(dtype=dtype, device=device)
        projected[:, 2] = high_a.to(dtype=dtype, device=device)
        projected[:, 3] = high_b.to(dtype=dtype, device=device)
        old_valid[:, 0:2] = True
        old_keys[:, 0] = low_b.to(dtype=dtype, device=device)
        old_keys[:, 1] = high_b.to(dtype=dtype, device=device)
        write_strength.fill_(0.7)
    else:
        raise ValueError(f"unknown issue487 stress kind: {kind}")

    state = ContextualEpisodicMemoryState(
        keys=old_keys.contiguous(),
        values=old_values.contiguous(),
        strengths=old_strengths.contiguous(),
        valid=old_valid.contiguous(),
    )
    with _precision_context(dtype_name):
        normalized_old = F.normalize(state.keys, dim=-1)
    return WriteCase(
        dtype_name=dtype_name,
        batch_size=8,
        validity_kind=kind,
        projected_new_keys=projected.contiguous(),
        normalized_old_keys=normalized_old.contiguous(),
        payload_source=case.payload_source,
        write_strength=write_strength.contiguous(),
        state=state,
    )


def _full_projected_write(
    backend: Any,
    memory: CoalescedFICEMMemory,
    case: WriteCase,
) -> ContextualEpisodicMemoryState:
    with torch.inference_mode(), _precision_context(case.dtype_name):
        return backend.update_from_projected(
            memory,
            case.projected_new_keys,
            case.normalized_old_keys,
            case.payload_source,
            case.write_strength,
            case.state,
        )


def _ordinary_update_case(
    memory: CoalescedFICEMMemory,
    dtype_name: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, ContextualEpisodicMemoryState]:
    dtype = _dtype_from_name(dtype_name)
    generator = torch.Generator().manual_seed(DESIGN_SEED + 20_000 + (0 if dtype_name == "float32" else 1))
    identity = torch.randn(8, WRITE_K, D_MODEL, generator=generator).to(dtype=dtype, device=device)
    context = torch.randn(8, WRITE_K, D_MODEL, generator=generator).to(dtype=dtype, device=device)
    payload = torch.randn(8, WRITE_K, D_MODEL, generator=generator).to(dtype=dtype, device=device)
    strength = (0.1 + 0.8 * torch.rand(8, WRITE_K, 1, generator=generator)).to(dtype=dtype, device=device)
    keys = _unit_rows(torch.randn(8, CAPACITY, MEMORY_DIM, generator=generator)).to(dtype=dtype, device=device)
    values = torch.randn(8, CAPACITY, MEMORY_DIM, generator=generator).to(dtype=dtype, device=device)
    strengths = torch.rand(8, CAPACITY, generator=generator).to(dtype=dtype, device=device)
    valid = ((torch.arange(CAPACITY) % 4) != 0)[None, :].expand(8, -1).to(device=device)
    state = ContextualEpisodicMemoryState(keys=keys, values=values, strengths=strengths, valid=valid)
    return identity, context, payload, strength, state


def _ordinary_update_correctness(
    memory: CoalescedFICEMMemory,
    dtype_name: str,
    device: torch.device,
    reference: TorchFICEMReferenceBackend,
    candidate: TritonFICEMReadWriteBackend,
) -> dict[str, Any]:
    identity, context, payload, strength, state = _ordinary_update_case(memory, dtype_name, device)
    sources_before = (identity.clone(), context.clone(), payload.clone(), strength.clone(), _state_clone(state))
    with torch.inference_mode(), _precision_context(dtype_name):
        ref = reference.update(memory, identity, context, payload, strength, state)
        cand = candidate.update(memory, identity, context, payload, strength, state)
    torch.cuda.synchronize()
    source_exact = (
        torch.equal(identity, sources_before[0])
        and torch.equal(context, sources_before[1])
        and torch.equal(payload, sources_before[2])
        and torch.equal(strength, sources_before[3])
        and _state_equal(state, sources_before[4])
    )
    exact = _state_equal(ref, cand)
    return {
        "pass": bool(exact and source_exact and _all_finite(ref) and _all_finite(cand)),
        "complete_state_bit_exact": exact,
        "source_unchanged": source_exact,
        "finite": bool(_all_finite(ref) and _all_finite(cand)),
    }


def _reference_decisions(case: WriteCase, memory: CoalescedFICEMMemory) -> dict[str, torch.Tensor]:
    with torch.inference_mode(), _precision_context(case.dtype_name):
        new_values = torch.tanh(memory.v(case.payload_source))
        new_strengths = case.write_strength[..., 0].clamp(0.0, 1.0)
        new_valid = new_strengths > 0.0
        incoming_similarity = torch.einsum(
            "bkd,bjd->bkj", case.projected_new_keys, case.projected_new_keys
        )
        k_count = case.projected_new_keys.size(1)
        position = torch.arange(k_count, device=case.projected_new_keys.device)
        later = position[None, :, None] < position[None, None, :]
        shadowed = (
            incoming_similarity.ge(DUPLICATE_SIMILARITY)
            & new_valid[:, :, None]
            & new_valid[:, None, :]
            & later
        ).any(dim=2)
        surviving = new_valid & ~shadowed
        old_similarity = torch.einsum(
            "bkd,bsd->bks", case.projected_new_keys, case.normalized_old_keys
        )
        duplicate_old = (
            old_similarity.ge(DUPLICATE_SIMILARITY)
            & surviving[:, :, None]
            & case.state.valid[:, None, :]
        ).any(dim=1)
        keep_old = case.state.valid & ~duplicate_old
    return {
        "new_values": new_values.contiguous(),
        "new_strengths": new_strengths.contiguous(),
        "new_valid": new_valid.contiguous(),
        "incoming_similarity": incoming_similarity.contiguous(),
        "old_similarity": old_similarity.contiguous(),
        "shadowed": shadowed.contiguous(),
        "surviving": surviving.contiguous(),
        "keep_old": keep_old.contiguous(),
    }


def _reference_source_map(case: WriteCase, decisions: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    surviving = decisions["surviving"]
    keep_old = decisions["keep_old"]
    batch = case.batch_size
    source_map = torch.empty((batch, CAPACITY), dtype=torch.int32, device=case.projected_new_keys.device)
    durable_valid = torch.empty((batch, CAPACITY), dtype=torch.bool, device=case.projected_new_keys.device)
    # Correctness-only audit. Bounded B<=64, K16/C48; no timed path uses this loop.
    for b in range(batch):
        new_order = list(reversed(range(WRITE_K)))
        source_order = [(i, True, bool(surviving[b, i])) for i in new_order]
        source_order += [(i, False, bool(keep_old[b, i])) for i in range(CAPACITY)]
        valid_sources = [item for item in source_order if item[2]]
        invalid_sources = [item for item in source_order if not item[2]]
        retained = (valid_sources + invalid_sources)[:CAPACITY]
        encoded = [i if is_new else WRITE_K + i for i, is_new, _ in retained]
        source_map[b] = torch.tensor(encoded, dtype=torch.int32, device=source_map.device)
        valid_count = min(len(valid_sources), CAPACITY)
        durable_valid[b] = torch.arange(CAPACITY, device=source_map.device) < valid_count
    return source_map, durable_valid


def _candidate_source_map(case: WriteCase, decisions: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    if _write_adjudicate_map_kernel is None:
        raise RuntimeError("issue487 candidate map kernel unavailable")
    source_map = torch.empty((case.batch_size, CAPACITY), dtype=torch.int32, device=case.projected_new_keys.device)
    durable_valid = torch.empty((case.batch_size, CAPACITY), dtype=torch.bool, device=case.projected_new_keys.device)
    _write_adjudicate_map_kernel[(case.batch_size,)](
        decisions["incoming_similarity"],
        decisions["old_similarity"],
        decisions["new_valid"],
        case.state.valid,
        source_map,
        durable_valid,
        K=WRITE_K,
        CAPACITY=CAPACITY,
        K_BLOCK=16,
        CAPACITY_BLOCK=64,
        DUPLICATE_THRESHOLD=WRITE_DUPLICATE_SIMILARITY,
        num_warps=4,
    )
    return source_map, durable_valid


def correctness_row(
    memory: CoalescedFICEMMemory,
    case: WriteCase,
    reference: TorchFICEMReferenceBackend,
    candidate: TritonFICEMReadWriteBackend,
    *,
    audit_map: bool,
) -> dict[str, Any]:
    projected_before = case.projected_new_keys.clone()
    normalized_before = case.normalized_old_keys.clone()
    payload_before = case.payload_source.clone()
    strength_before = case.write_strength.clone()
    state_before = _state_clone(case.state)
    parameters_before = [p.detach().clone() for p in memory.parameters()]

    ref = _full_projected_write(reference, memory, case)
    cand = _full_projected_write(candidate, memory, case)
    torch.cuda.synchronize()

    exact_fields = {
        "keys": torch.equal(ref.keys, cand.keys),
        "values": torch.equal(ref.values, cand.values),
        "strengths": torch.equal(ref.strengths, cand.strengths),
        "valid": torch.equal(ref.valid, cand.valid),
    }
    meta_exact = (
        ref.keys.shape == cand.keys.shape
        and ref.values.shape == cand.values.shape
        and ref.strengths.shape == cand.strengths.shape
        and ref.valid.shape == cand.valid.shape
        and ref.keys.dtype == cand.keys.dtype
        and ref.values.dtype == cand.values.dtype
        and ref.strengths.dtype == cand.strengths.dtype
        and ref.valid.dtype == cand.valid.dtype
        and ref.keys.device == cand.keys.device
        and ref.values.device == cand.values.device
        and ref.strengths.device == cand.strengths.device
        and ref.valid.device == cand.valid.device
        and cand.keys.device.type == "cuda"
    )
    source_unchanged = (
        torch.equal(case.projected_new_keys, projected_before)
        and torch.equal(case.normalized_old_keys, normalized_before)
        and torch.equal(case.payload_source, payload_before)
        and torch.equal(case.write_strength, strength_before)
        and _state_equal(case.state, state_before)
        and all(torch.equal(p.detach(), before) for p, before in zip(memory.parameters(), parameters_before))
    )
    finite = _all_finite(ref) and _all_finite(cand)

    map_exact = True
    reference_shadowed = None
    reference_keep_old = None
    if audit_map:
        decisions = _reference_decisions(case, memory)
        ref_map, ref_valid = _reference_source_map(case, decisions)
        cand_map, cand_valid = _candidate_source_map(case, decisions)
        torch.cuda.synchronize()
        map_exact = bool(torch.equal(ref_map, cand_map) and torch.equal(ref_valid, cand_valid))
        reference_shadowed = int(decisions["shadowed"].sum().item())
        reference_keep_old = int(decisions["keep_old"].sum().item())

    passed = bool(all(exact_fields.values()) and meta_exact and source_unchanged and finite and map_exact)
    return {
        "pass": passed,
        "complete_state_bit_exact": bool(all(exact_fields.values())),
        "keys_bit_exact": bool(exact_fields["keys"]),
        "values_bit_exact": bool(exact_fields["values"]),
        "strengths_bit_exact": bool(exact_fields["strengths"]),
        "valid_exact": bool(exact_fields["valid"]),
        "candidate_source_map_exact": map_exact,
        "reference_shadowed_incoming_count": reference_shadowed,
        "reference_keep_old_count": reference_keep_old,
        "source_unchanged": source_unchanged,
        "finite": finite,
        "dtype_device_shape_exact": meta_exact,
    }


def _reference_tail_from_precomputed(
    memory: CoalescedFICEMMemory,
    case: WriteCase,
    decisions: dict[str, torch.Tensor],
) -> ContextualEpisodicMemoryState:
    # Exact reference post-similarity region: duplicate adjudication results are
    # represented by `surviving`/`keep_old`, followed by newest-first stable compact.
    return memory._stable_compact_state(
        case.projected_new_keys.flip(1),
        decisions["new_values"].flip(1),
        decisions["new_strengths"].flip(1),
        decisions["surviving"].flip(1),
        case.state.keys,
        case.state.values,
        case.state.strengths,
        decisions["keep_old"],
    )


def _candidate_tail_from_precomputed(
    case: WriteCase,
    decisions: dict[str, torch.Tensor],
) -> ContextualEpisodicMemoryState:
    return fused_ficem_write_tail(
        decisions["incoming_similarity"],
        decisions["old_similarity"],
        case.projected_new_keys,
        decisions["new_values"],
        decisions["new_strengths"],
        decisions["new_valid"],
        case.state,
    )


def _cuda_profile(call: Callable[[], Any]) -> dict[str, Any]:
    call()
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False,
    ) as profile:
        output = call()
    torch.cuda.synchronize()
    del output

    cuda_events = 0
    adjudicate_events = 0
    materialize_events = 0
    kernel_names: list[str] = []
    for event in profile.events():
        device_type = getattr(event, "device_type", None)
        if device_type == torch.autograd.DeviceType.CUDA or str(device_type).endswith("CUDA"):
            cuda_events += 1
            name = str(getattr(event, "name", ""))
            lower = name.lower()
            if "write_adjudicate_map_kernel" in lower:
                adjudicate_events += 1
                kernel_names.append(name)
            if "write_materialize_kernel" in lower:
                materialize_events += 1
                kernel_names.append(name)

    relevant = {token: 0 for token in ("cumsum", "scatter_add", "where", "cat", "stack", "topk", "einsum", "tanh")}
    for item in profile.key_averages():
        key = str(item.key).lower()
        for token in relevant:
            if token in key:
                relevant[token] += int(item.count)
    return {
        "cuda_device_events": int(cuda_events),
        "triton_adjudicate_events": int(adjudicate_events),
        "triton_materialize_events": int(materialize_events),
        "triton_write_tail_events": int(adjudicate_events + materialize_events),
        "triton_kernel_names": kernel_names,
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
        raise RuntimeError("issue487 timed loop produced no output")
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


def _row_key(dtype_name: str, batch_size: int, validity_kind: str) -> str:
    return f"{dtype_name}_batch{batch_size}_{validity_kind}"


def run_ficem_write_probe() -> dict[str, Any]:
    cpu_contract_preflight()
    if not torch.cuda.is_available():
        raise RuntimeError("issue487 requires CUDA")
    device = torch.device("cuda")
    device_name = torch.cuda.get_device_name(0)
    if "L4" not in device_name.upper():
        raise RuntimeError(f"issue487 requires NVIDIA L4, found {device_name}")

    reference = TorchFICEMReferenceBackend()
    candidate = TritonFICEMReadWriteBackend()
    generator = torch.Generator().manual_seed(DESIGN_SEED)

    stress: dict[str, dict[str, Any]] = {}
    ordinary_update: dict[str, dict[str, Any]] = {}
    for dtype_name in DTYPE_NAMES:
        memory = build_memory(device, dtype_name)
        ordinary_update[dtype_name] = _ordinary_update_correctness(
            memory, dtype_name, device, reference, candidate
        )
        if not ordinary_update[dtype_name]["pass"]:
            raise RuntimeError(f"issue487 ordinary update correctness failed for {dtype_name}")
        for kind in STRESS_KINDS:
            key = f"{dtype_name}_{kind}"
            case = _make_stress_case(dtype_name=dtype_name, kind=kind, device=device)
            stress[key] = correctness_row(
                memory, case, reference, candidate, audit_map=True
            )
            if not stress[key]["pass"]:
                raise RuntimeError(f"issue487 stress correctness failed for {key}")

    rows: dict[str, dict[str, Any]] = {}
    ratios_by_dtype: dict[str, list[float]] = {name: [] for name in DTYPE_NAMES}
    for dtype_name in DTYPE_NAMES:
        memory = build_memory(device, dtype_name)
        for batch_size in BATCH_SIZES:
            for validity_kind in VALIDITY_KINDS:
                case = _make_primary_case(
                    dtype_name=dtype_name,
                    batch_size=batch_size,
                    validity_kind=validity_kind,
                    generator=generator,
                    device=device,
                )
                key = _row_key(dtype_name, batch_size, validity_kind)
                correctness = correctness_row(
                    memory, case, reference, candidate, audit_map=True
                )
                if not correctness["pass"]:
                    raise RuntimeError(f"issue487 correctness failed for {key}")

                calls = {
                    "reference": lambda case=case, memory=memory: _full_projected_write(reference, memory, case),
                    "candidate": lambda case=case, memory=memory: _full_projected_write(candidate, memory, case),
                }
                timing = _timed_summaries(calls)
                reference_us = timing["reference"]["median_us_per_call"]
                candidate_us = timing["candidate"]["median_us_per_call"]
                latency_ratio = candidate_us / reference_us
                ratios_by_dtype[dtype_name].append(latency_ratio)

                decisions = _reference_decisions(case, memory)
                reference_tail = lambda case=case, memory=memory, decisions=decisions: _reference_tail_from_precomputed(memory, case, decisions)
                candidate_tail = lambda case=case, decisions=decisions: _candidate_tail_from_precomputed(case, decisions)
                tail_reference_state = reference_tail()
                tail_candidate_state = candidate_tail()
                torch.cuda.synchronize()
                tail_exact = _state_equal(tail_reference_state, tail_candidate_state)
                if not tail_exact:
                    raise RuntimeError(f"issue487 isolated tail correctness failed for {key}")

                full_profiles = {name: _cuda_profile(call) for name, call in calls.items()}
                tail_profiles = {
                    "reference": _cuda_profile(reference_tail),
                    "candidate": _cuda_profile(candidate_tail),
                }
                ref_tail_events = tail_profiles["reference"]["cuda_device_events"]
                cand_tail_events = tail_profiles["candidate"]["cuda_device_events"]
                if ref_tail_events <= 0 or cand_tail_events <= 0:
                    raise RuntimeError(f"issue487 tail profiler found no CUDA events for {key}")
                tail_event_ratio = cand_tail_events / ref_tail_events
                candidate_ops = tail_profiles["candidate"]["relevant_operator_calls"]
                candidate_tail_kernel_exact = (
                    tail_profiles["candidate"]["triton_adjudicate_events"] == 1
                    and tail_profiles["candidate"]["triton_materialize_events"] == 1
                    and tail_profiles["candidate"]["triton_write_tail_events"] == 2
                )
                candidate_tail_ops_clean = all(
                    candidate_ops[token] == 0
                    for token in ("cumsum", "scatter_add", "cat", "stack", "topk")
                )

                rows[key] = {
                    "correctness": correctness,
                    "timing": timing,
                    "latency_ratio_candidate_over_reference": latency_ratio,
                    "full_profiles": full_profiles,
                    "tail_profiles": tail_profiles,
                    "isolated_tail_state_bit_exact": tail_exact,
                    "isolated_tail_event_ratio_candidate_over_reference": tail_event_ratio,
                    "candidate_tail_kernel_exact": candidate_tail_kernel_exact,
                    "candidate_tail_ops_clean": candidate_tail_ops_clean,
                    "vram": {name: _peak_vram(call) for name, call in calls.items()},
                }

    geomean_by_dtype = {
        dtype_name: float(math.exp(statistics.fmean(math.log(value) for value in values)))
        for dtype_name, values in ratios_by_dtype.items()
    }
    correctness_pass = all(row["correctness"]["pass"] for row in rows.values()) and all(
        row["pass"] for row in stress.values()
    ) and all(row["pass"] for row in ordinary_update.values())
    row_latency_pass = all(
        row["latency_ratio_candidate_over_reference"] <= MAX_ROW_LATENCY_RATIO
        for row in rows.values()
    )
    geomean_latency_pass = all(
        value <= MAX_GEOMEAN_LATENCY_RATIO for value in geomean_by_dtype.values()
    )
    tail_kernel_pass = all(row["candidate_tail_kernel_exact"] for row in rows.values())
    tail_ops_pass = all(row["candidate_tail_ops_clean"] for row in rows.values())
    tail_event_ratio_pass = all(
        row["isolated_tail_event_ratio_candidate_over_reference"] <= MAX_TAIL_EVENT_RATIO
        for row in rows.values()
    )
    isolated_tail_exact_pass = all(row["isolated_tail_state_bit_exact"] for row in rows.values())
    overall_pass = bool(
        correctness_pass
        and row_latency_pass
        and geomean_latency_pass
        and tail_kernel_pass
        and tail_ops_pass
        and tail_event_ratio_pass
        and isolated_tail_exact_pass
    )
    return {
        "issue": RESEARCH_ISSUE,
        "decision": "PASS" if overall_pass else "FAIL",
        "overall_pass": overall_pass,
        "device": device_name,
        "protocol": issue487_protocol(),
        "rows": rows,
        "stress": stress,
        "ordinary_update": ordinary_update,
        "geomean_latency_ratio_by_dtype": geomean_by_dtype,
        "correctness_pass": correctness_pass,
        "row_latency_pass": row_latency_pass,
        "geomean_latency_pass": geomean_latency_pass,
        "isolated_tail_exact_pass": isolated_tail_exact_pass,
        "tail_kernel_pass": tail_kernel_pass,
        "tail_ops_pass": tail_ops_pass,
        "tail_event_ratio_pass": tail_event_ratio_pass,
        "synthetic_only": True,
        "scientific_seed_consumed": False,
        "end_to_end_systems_authorized": False,
        "architecture_freeze_authorized": False,
        "s2_authorized": False,
        "fresh_scientific_seed_authorized": False,
        "100m_authorized": False,
        "breakthrough_proven": False,
    }
