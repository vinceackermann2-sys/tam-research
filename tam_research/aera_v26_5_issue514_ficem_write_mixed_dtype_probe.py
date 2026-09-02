from __future__ import annotations

"""Issue #514 synthetic/non-scientific mixed-dtype FICEM WRITE L4 gate."""

from contextlib import nullcontext
import itertools
from typing import Any

import torch
import torch.nn.functional as F

from .aera_hardware_core_v24 import ContextualEpisodicMemoryState
from .aera_hardware_core_v25 import FactorizedIdentityContextEpisodicMemory
from .aera_hardware_core_v25_1 import ExecutionEquivalentFactorizedIdentityContextMemory
from .aera_hardware_core_v25_1_compact import (
    StableCompactExecutionEquivalentFactorizedIdentityContextMemory,
)
from .aera_hardware_core_v26 import CoalescedFICEMMemory, TorchFICEMReferenceBackend
from .aera_hardware_core_v26_5_ficem_write_mixed_dtype import (
    MixedDtypeTritonFICEMReadWriteBackend,
    fused_ficem_write_tail_mixed_dtype,
    mixed_dtype_ficem_read_write_v26_5_protocol,
)
from . import aera_v26_4_ficem_write_probe as historical

RESEARCH_ISSUE = 514
SOURCE_MAIN = "d9bda2bd3143308407c0d11e640d984385eb095a"
CANDIDATE_BLOB = "dab24c733eff7aa08e5f818614f7504eaac48dc3"
PREDECESSOR_BLOB = "e54570292489bd17570038dca7518419ac00418c"
READ_BACKEND_BLOB = "263f68eb1186a8ac14a08fc4b4df1fc5b292c711"
V26_INTERFACE_BLOB = "268644ac4edee15a4cc4e29d3fed7f61eeb3caa7"
STABLE_REFERENCE_BLOB = "4e336b6e1a6238dac782fa320751d68281493ee1"
HISTORICAL_PROBE_BLOB = "7d8c2c4990beb4c7b4a719d02d009ffefe94671f"
EXHAUSTED_508_LAUNCHER_BLOB = "5597dbbd79c782420d48ed538ef2669aebfe5fae"
EXHAUSTED_508_WORKFLOW_BLOB = "556ea59ebc2d95272caa774a9fef62efbf66a302"
EXHAUSTED_508_RUN = 33661498305
EXHAUSTED_508_JOB = 100352870198

DESIGN_SEED = 408_514
D_MODEL = 200
WRITE_COUNT = 16
CAPACITY = 48
MEMORY_DIM = 50
DUPLICATE_THRESHOLD = 0.95
BATCH_SIZES = (8, 64)
MATRIX_MASKS = tuple(range(256))
FLOAT_FIELD_NAMES = (
    "incoming_similarity",
    "old_similarity",
    "new_keys",
    "new_values",
    "new_strengths",
    "state_keys",
    "state_values",
    "state_strengths",
)
EDGE_FIXTURES = historical.EDGE_FIXTURES
EDGE_LAYOUTS = (
    "all_float32",
    "all_bfloat16",
    "compute_bfloat16_durable_float32",
    "compute_float32_durable_bfloat16",
)
PUBLIC_LAYOUTS = (
    "float32_inputs_float32_state",
    "bfloat16_inputs_bfloat16_state",
    "bfloat16_autocast_compute_float32_state",
)
FP32_ATOL = 1e-5
FP32_RTOL = 1e-5
BF16_ATOL = 1e-2
BF16_RTOL = 1e-2
TOPOLOGY_WARMUP_CALLS = 2
EXPECTED_DIRECT_CASES = 256
EXPECTED_EDGE_CASES = 32
EXPECTED_PUBLIC_ROWS = 6
EXPECTED_TOPOLOGY_ROWS = 4


def issue514_protocol() -> dict[str, Any]:
    return {
        **mixed_dtype_ficem_read_write_v26_5_protocol(),
        "gate_version": "aera-v26.5-issue514-mixed-dtype-write-correctness-topology",
        "gate_research_issue": RESEARCH_ISSUE,
        "source_main": SOURCE_MAIN,
        "candidate_blob": CANDIDATE_BLOB,
        "design_seed": DESIGN_SEED,
        "design_seed_is_scientific_seed": False,
        "geometry": {"d_model": D_MODEL, "write_count": WRITE_COUNT, "capacity": CAPACITY, "memory_dim": MEMORY_DIM},
        "duplicate_threshold": DUPLICATE_THRESHOLD,
        "batch_sizes": list(BATCH_SIZES),
        "float_fields": list(FLOAT_FIELD_NAMES),
        "matrix_mask_order": [0, 255],
        "matrix_case_count": EXPECTED_DIRECT_CASES,
        "edge_fixtures": list(EDGE_FIXTURES),
        "edge_layouts": list(EDGE_LAYOUTS),
        "edge_case_count": EXPECTED_EDGE_CASES,
        "public_layouts": list(PUBLIC_LAYOUTS),
        "public_row_count": EXPECTED_PUBLIC_ROWS,
        "topology_row_count": EXPECTED_TOPOLOGY_ROWS,
        "fp32_atol": FP32_ATOL,
        "fp32_rtol": FP32_RTOL,
        "bfloat16_atol": BF16_ATOL,
        "bfloat16_rtol": BF16_RTOL,
        "topology_warmup_calls": TOPOLOGY_WARMUP_CALLS,
        "performance_threshold_added": False,
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
        "independent_replication_credit": False,
        "100m_authorized": False,
        "breakthrough_proven": False,
    }


def cpu_contract_preflight() -> dict[str, Any]:
    assert DESIGN_SEED == 408_514
    assert (D_MODEL, WRITE_COUNT, CAPACITY, MEMORY_DIM) == (200, 16, 48, 50)
    assert DUPLICATE_THRESHOLD == 0.95
    assert BATCH_SIZES == (8, 64)
    assert MATRIX_MASKS == tuple(range(256))
    assert len(FLOAT_FIELD_NAMES) == 8
    assert EDGE_FIXTURES == (
        "empty_old_all_new_valid",
        "mixed_incoming_validity",
        "incoming_duplicate_newest_wins",
        "threshold_inclusive_and_below_control",
        "surviving_new_suppresses_old",
        "shadowed_new_does_not_suppress_old",
        "over_capacity_truncation",
        "invalid_retained_storage_order",
    )
    assert len(EDGE_LAYOUTS) == 4
    assert len(PUBLIC_LAYOUTS) == 3
    assert (EXPECTED_DIRECT_CASES, EXPECTED_EDGE_CASES, EXPECTED_PUBLIC_ROWS, EXPECTED_TOPOLOGY_ROWS) == (256, 32, 6, 4)
    return {
        "protocol": issue514_protocol(),
        "gpu_authorized_by_cpu_preflight": False,
        "synthetic_only": True,
        "scientific_seed_consumed": False,
    }


def _dtype_name(dtype: torch.dtype) -> str:
    if dtype is torch.float32:
        return "float32"
    if dtype is torch.bfloat16:
        return "bfloat16"
    raise TypeError(dtype)


def _dtype_from_bit(bit: int) -> torch.dtype:
    return torch.bfloat16 if bit else torch.float32


def _clone_state(state: ContextualEpisodicMemoryState) -> ContextualEpisodicMemoryState:
    return ContextualEpisodicMemoryState(
        keys=state.keys.clone(),
        values=state.values.clone(),
        strengths=state.strengths.clone(),
        valid=state.valid.clone(),
    )


def _clone_tail(inputs: historical.TailInputs) -> historical.TailInputs:
    return historical.TailInputs(
        incoming_similarity=inputs.incoming_similarity.clone(),
        old_similarity=inputs.old_similarity.clone(),
        new_keys=inputs.new_keys.clone(),
        new_values=inputs.new_values.clone(),
        new_strengths=inputs.new_strengths.clone(),
        new_valid=inputs.new_valid.clone(),
        state=_clone_state(inputs.state),
    )


def _tail_sources_unchanged(now: historical.TailInputs, before: historical.TailInputs) -> bool:
    return (
        torch.equal(now.incoming_similarity, before.incoming_similarity)
        and torch.equal(now.old_similarity, before.old_similarity)
        and torch.equal(now.new_keys, before.new_keys)
        and torch.equal(now.new_values, before.new_values)
        and torch.equal(now.new_strengths, before.new_strengths)
        and torch.equal(now.new_valid, before.new_valid)
        and torch.equal(now.state.keys, before.state.keys)
        and torch.equal(now.state.values, before.state.values)
        and torch.equal(now.state.strengths, before.state.strengths)
        and torch.equal(now.state.valid, before.state.valid)
    )


def _cast_tail_fields(
    base: historical.TailInputs,
    dtypes: tuple[torch.dtype, ...],
) -> historical.TailInputs:
    if len(dtypes) != 8:
        raise ValueError("issue514 requires eight floating field dtypes")
    return historical.TailInputs(
        incoming_similarity=base.incoming_similarity.to(dtype=dtypes[0]).contiguous(),
        old_similarity=base.old_similarity.to(dtype=dtypes[1]).contiguous(),
        new_keys=base.new_keys.to(dtype=dtypes[2]).contiguous(),
        new_values=base.new_values.to(dtype=dtypes[3]).contiguous(),
        new_strengths=base.new_strengths.to(dtype=dtypes[4]).contiguous(),
        new_valid=base.new_valid.clone().contiguous(),
        state=ContextualEpisodicMemoryState(
            keys=base.state.keys.to(dtype=dtypes[5]).contiguous(),
            values=base.state.values.to(dtype=dtypes[6]).contiguous(),
            strengths=base.state.strengths.to(dtype=dtypes[7]).contiguous(),
            valid=base.state.valid.clone().contiguous(),
        ),
    )


def _mask_dtypes(mask: int) -> tuple[torch.dtype, ...]:
    if mask < 0 or mask > 255:
        raise ValueError(mask)
    return tuple(_dtype_from_bit((mask >> index) & 1) for index in range(8))


def _layout_dtypes(layout: str) -> tuple[torch.dtype, ...]:
    if layout == "all_float32":
        return (torch.float32,) * 8
    if layout == "all_bfloat16":
        return (torch.bfloat16,) * 8
    if layout == "compute_bfloat16_durable_float32":
        return (torch.bfloat16,) * 5 + (torch.float32,) * 3
    if layout == "compute_float32_durable_bfloat16":
        return (torch.float32,) * 5 + (torch.bfloat16,) * 3
    raise ValueError(layout)


def _reference_tail_durable(
    memory: CoalescedFICEMMemory,
    inputs: historical.TailInputs,
) -> ContextualEpisodicMemoryState:
    reference = historical._reference_tail(memory, inputs)
    return ContextualEpisodicMemoryState(
        keys=reference.keys.to(dtype=inputs.state.keys.dtype),
        values=reference.values.to(dtype=inputs.state.values.dtype),
        strengths=reference.strengths.to(dtype=inputs.state.strengths.dtype),
        valid=reference.valid,
    )


def _candidate_tail(inputs: historical.TailInputs) -> ContextualEpisodicMemoryState:
    return fused_ficem_write_tail_mixed_dtype(
        inputs.incoming_similarity,
        inputs.old_similarity,
        inputs.new_keys,
        inputs.new_values,
        inputs.new_strengths,
        inputs.new_valid,
        inputs.state,
    )


def _field_tolerance(dtype: torch.dtype) -> tuple[float, float]:
    if dtype is torch.bfloat16:
        return BF16_ATOL, BF16_RTOL
    if dtype is torch.float32:
        return FP32_ATOL, FP32_RTOL
    raise TypeError(dtype)


def _compare_durable_states(
    reference: ContextualEpisodicMemoryState,
    candidate: ContextualEpisodicMemoryState,
    destination: ContextualEpisodicMemoryState,
) -> dict[str, Any]:
    float_close: dict[str, bool] = {}
    max_abs: dict[str, float] = {}
    dtype_exact: dict[str, bool] = {}
    for name in ("keys", "values", "strengths"):
        ref = getattr(reference, name)
        cand = getattr(candidate, name)
        target_dtype = getattr(destination, name).dtype
        atol, rtol = _field_tolerance(target_dtype)
        float_close[name] = bool(torch.allclose(ref, cand, atol=atol, rtol=rtol))
        max_abs[name] = float((ref.float() - cand.float()).abs().max())
        dtype_exact[name] = cand.dtype is target_dtype
    valid_exact = bool(torch.equal(reference.valid, candidate.valid))
    device_shape_exact = all(
        getattr(reference, name).device == getattr(candidate, name).device
        and getattr(reference, name).shape == getattr(candidate, name).shape
        for name in ("keys", "values", "strengths", "valid")
    )
    finite = all(
        bool(torch.isfinite(getattr(candidate, name)).all())
        for name in ("keys", "values", "strengths")
    )
    passed = (
        all(float_close.values())
        and all(dtype_exact.values())
        and valid_exact
        and device_shape_exact
        and finite
    )
    return {
        "pass": passed,
        "float_close": float_close,
        "max_abs": max_abs,
        "dtype_exact": dtype_exact,
        "valid_exact": valid_exact,
        "device_shape_exact": device_shape_exact,
        "finite": finite,
    }


def _run_tail_case(memory: CoalescedFICEMMemory, inputs: historical.TailInputs) -> dict[str, Any]:
    before = _clone_tail(inputs)
    try:
        with torch.no_grad():
            reference = _reference_tail_durable(memory, inputs)
            candidate = _candidate_tail(inputs)
        comparison = _compare_durable_states(reference, candidate, inputs.state)
        source_unchanged = _tail_sources_unchanged(inputs, before)
        comparison["source_unchanged"] = source_unchanged
        comparison["pass"] = bool(comparison["pass"] and source_unchanged)
        return comparison
    except Exception as exc:
        return {
            "pass": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "source_unchanged": _tail_sources_unchanged(inputs, before),
        }


def _build_memory(device: torch.device) -> CoalescedFICEMMemory:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(DESIGN_SEED + 1)
        raw = historical._RawFICEMSource()
        v25 = FactorizedIdentityContextEpisodicMemory(raw, capacity=CAPACITY)
        v251 = ExecutionEquivalentFactorizedIdentityContextMemory(v25)
        compact = StableCompactExecutionEquivalentFactorizedIdentityContextMemory(v251)
        memory = CoalescedFICEMMemory(compact)
    memory.set_differentiable_pretraining(False)
    return memory.to(device).eval()


def _cpu_randn(
    shape: tuple[int, ...],
    generator: torch.Generator,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    return torch.randn(*shape, generator=generator, dtype=torch.float32).to(dtype=dtype)


def _make_public_case(
    *,
    layout: str,
    batch_size: int,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, ContextualEpisodicMemoryState, bool]:
    if layout == "float32_inputs_float32_state":
        key_dtype = payload_dtype = strength_dtype = state_dtype = torch.float32
        use_autocast = False
    elif layout == "bfloat16_inputs_bfloat16_state":
        key_dtype = payload_dtype = strength_dtype = state_dtype = torch.bfloat16
        use_autocast = True
    elif layout == "bfloat16_autocast_compute_float32_state":
        key_dtype = payload_dtype = torch.bfloat16
        strength_dtype = torch.float32
        state_dtype = torch.float32
        use_autocast = True
    else:
        raise ValueError(layout)

    projected = F.normalize(
        _cpu_randn((batch_size, WRITE_COUNT, MEMORY_DIM), generator),
        dim=-1,
    ).to(device=device, dtype=key_dtype)
    old_keys = _cpu_randn((batch_size, CAPACITY, MEMORY_DIM), generator).to(
        device=device, dtype=state_dtype
    )
    old_values = _cpu_randn((batch_size, CAPACITY, MEMORY_DIM), generator).to(
        device=device, dtype=state_dtype
    )
    old_strengths = torch.rand(batch_size, CAPACITY, generator=generator).to(
        device=device, dtype=state_dtype
    )
    old_valid = (torch.rand(batch_size, CAPACITY, generator=generator) > 0.25).to(device)
    old_valid[:, :8] = True
    state = ContextualEpisodicMemoryState(
        keys=old_keys.contiguous(),
        values=old_values.contiguous(),
        strengths=old_strengths.contiguous(),
        valid=old_valid.contiguous(),
    )
    normalized_old = F.normalize(old_keys, dim=-1).contiguous()
    payload = _cpu_randn((batch_size, WRITE_COUNT, D_MODEL), generator).to(
        device=device, dtype=payload_dtype
    )
    strength = torch.rand(batch_size, WRITE_COUNT, 1, generator=generator).to(
        device=device, dtype=strength_dtype
    )
    strength[:, 0, 0] = 0.0
    return (
        projected.contiguous(),
        normalized_old,
        payload.contiguous(),
        strength.contiguous(),
        state,
        use_autocast,
    )


def _public_context(use_autocast: bool):
    if use_autocast:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _public_sources_unchanged(
    before: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, ContextualEpisodicMemoryState],
    after: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, ContextualEpisodicMemoryState],
) -> bool:
    for b, a in zip(before[:4], after[:4]):
        if not torch.equal(b, a):
            return False
    bstate, astate = before[4], after[4]
    return (
        torch.equal(bstate.keys, astate.keys)
        and torch.equal(bstate.values, astate.values)
        and torch.equal(bstate.strengths, astate.strengths)
        and torch.equal(bstate.valid, astate.valid)
    )


def _run_public_case(
    memory: CoalescedFICEMMemory,
    candidate_backend: MixedDtypeTritonFICEMReadWriteBackend,
    reference_backend: TorchFICEMReferenceBackend,
    *,
    layout: str,
    batch_size: int,
    generator: torch.Generator,
    device: torch.device,
) -> dict[str, Any]:
    projected, normalized_old, payload, strength, state, use_autocast = _make_public_case(
        layout=layout,
        batch_size=batch_size,
        generator=generator,
        device=device,
    )
    before = (
        projected.clone(),
        normalized_old.clone(),
        payload.clone(),
        strength.clone(),
        _clone_state(state),
    )
    try:
        with torch.no_grad(), _public_context(use_autocast):
            reference_raw = reference_backend.update_from_projected(
                memory,
                projected,
                normalized_old,
                payload,
                strength,
                _clone_state(state),
            )
            candidate = candidate_backend.update_from_projected(
                memory,
                projected,
                normalized_old,
                payload,
                strength,
                _clone_state(state),
            )
        reference = ContextualEpisodicMemoryState(
            keys=reference_raw.keys.to(dtype=state.keys.dtype),
            values=reference_raw.values.to(dtype=state.values.dtype),
            strengths=reference_raw.strengths.to(dtype=state.strengths.dtype),
            valid=reference_raw.valid,
        )
        comparison = _compare_durable_states(reference, candidate, state)
        after = (projected, normalized_old, payload, strength, state)
        source_unchanged = _public_sources_unchanged(before, after)
        comparison["source_unchanged"] = source_unchanged
        comparison["pass"] = bool(comparison["pass"] and source_unchanged)
        comparison["input_dtypes"] = {
            "projected": _dtype_name(projected.dtype),
            "normalized_old": _dtype_name(normalized_old.dtype),
            "payload": _dtype_name(payload.dtype),
            "strength": _dtype_name(strength.dtype),
            "state_keys": _dtype_name(state.keys.dtype),
            "state_values": _dtype_name(state.values.dtype),
            "state_strengths": _dtype_name(state.strengths.dtype),
        }
        return comparison
    except Exception as exc:
        return {
            "pass": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "input_dtypes": {
                "projected": _dtype_name(projected.dtype),
                "normalized_old": _dtype_name(normalized_old.dtype),
                "payload": _dtype_name(payload.dtype),
                "strength": _dtype_name(strength.dtype),
                "state_keys": _dtype_name(state.keys.dtype),
                "state_values": _dtype_name(state.values.dtype),
                "state_strengths": _dtype_name(state.strengths.dtype),
            },
        }


def _profile_candidate_tail(inputs: historical.TailInputs) -> dict[str, Any]:
    for _ in range(TOPOLOGY_WARMUP_CALLS):
        with torch.no_grad():
            _candidate_tail(inputs)
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        with torch.no_grad():
            _candidate_tail(inputs)
        torch.cuda.synchronize()
    device_events: list[dict[str, Any]] = []
    for event in prof.key_averages():
        device_time = float(
            getattr(event, "device_time_total", getattr(event, "cuda_time_total", 0.0)) or 0.0
        )
        if device_time > 0:
            device_events.append({"key": event.key, "device_time_total": device_time})
    keys = [row["key"] for row in device_events]
    adjudicate = [key for key in keys if "_write_adjudicate_map_kernel" in key]
    materialize = [key for key in keys if "_write_materialize_kernel" in key]
    forbidden_fragments = (
        "aten::to",
        "aten::_to_copy",
        "aten::copy_",
        "aten::cat",
        "aten::stack",
        "aten::cumsum",
        "aten::scatter_add",
        "aten::topk",
    )
    forbidden = [key for key in keys if any(fragment in key for fragment in forbidden_fragments)]
    allowed = set(adjudicate + materialize)
    extras = [key for key in keys if key not in allowed]
    passed = (
        len(adjudicate) == 1
        and len(materialize) == 1
        and len(device_events) == 2
        and not forbidden
        and not extras
    )
    return {
        "pass": passed,
        "device_events": device_events,
        "adjudicate_kernel_events": adjudicate,
        "materialize_kernel_events": materialize,
        "forbidden_cuda_events": forbidden,
        "extra_cuda_events": extras,
    }


def _base_matrix_inputs(device: torch.device) -> historical.TailInputs:
    return historical._base_edge_inputs(dtype_name="float32", device=device)


def run_mixed_dtype_write_probe() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("issue514 mixed-dtype FICEM WRITE gate requires CUDA")
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    torch.manual_seed(DESIGN_SEED)
    torch.cuda.manual_seed_all(DESIGN_SEED)

    memory = _build_memory(device)
    candidate_backend = MixedDtypeTritonFICEMReadWriteBackend()
    reference_backend = TorchFICEMReferenceBackend()

    direct_results: dict[str, Any] = {}
    direct_pass = True
    base = _base_matrix_inputs(device)
    for mask in MATRIX_MASKS:
        dtypes = _mask_dtypes(mask)
        inputs = _cast_tail_fields(base, dtypes)
        row = _run_tail_case(memory, inputs)
        row["dtypes"] = [_dtype_name(dtype) for dtype in dtypes]
        direct_results[f"mask_{mask:03d}"] = row
        direct_pass = bool(direct_pass and row["pass"])

    edge_results: dict[str, Any] = {}
    edge_pass = True
    representative_inputs: dict[str, historical.TailInputs] = {}
    for fixture_name in EDGE_FIXTURES:
        base_edge = historical.make_edge_fixture(
            fixture_name,
            dtype_name="float32",
            device=device,
        )
        for layout in EDGE_LAYOUTS:
            dtypes = _layout_dtypes(layout)
            inputs = _cast_tail_fields(base_edge, dtypes)
            key = f"{fixture_name}__{layout}"
            row = _run_tail_case(memory, inputs)
            row["dtypes"] = [_dtype_name(dtype) for dtype in dtypes]
            edge_results[key] = row
            edge_pass = bool(edge_pass and row["pass"])
            if fixture_name == EDGE_FIXTURES[0]:
                representative_inputs[layout] = inputs

    public_results: dict[str, Any] = {}
    public_pass = True
    generator = torch.Generator().manual_seed(DESIGN_SEED)
    for layout in PUBLIC_LAYOUTS:
        for batch_size in BATCH_SIZES:
            key = f"{layout}__batch{batch_size}"
            row = _run_public_case(
                memory,
                candidate_backend,
                reference_backend,
                layout=layout,
                batch_size=batch_size,
                generator=generator,
                device=device,
            )
            public_results[key] = row
            public_pass = bool(public_pass and row["pass"])

    topology_results: dict[str, Any] = {}
    topology_pass = True
    for layout in EDGE_LAYOUTS:
        try:
            row = _profile_candidate_tail(representative_inputs[layout])
        except Exception as exc:
            row = {"pass": False, "error_type": type(exc).__name__, "error": str(exc)}
        topology_results[layout] = row
        topology_pass = bool(topology_pass and row["pass"])

    overall_pass = bool(direct_pass and edge_pass and public_pass and topology_pass)
    return {
        "version": "aera-v26.5-issue514-mixed-dtype-write-correctness-topology",
        "research_issue": RESEARCH_ISSUE,
        "source_main": SOURCE_MAIN,
        "device": torch.cuda.get_device_name(0),
        "protocol": issue514_protocol(),
        "direct_matrix": direct_results,
        "direct_matrix_case_count": len(direct_results),
        "direct_matrix_pass": direct_pass,
        "edge_cases": edge_results,
        "edge_case_count": len(edge_results),
        "edge_cases_pass": edge_pass,
        "public_rows": public_results,
        "public_row_count": len(public_results),
        "public_rows_pass": public_pass,
        "topology_rows": topology_results,
        "topology_row_count": len(topology_results),
        "topology_pass": topology_pass,
        "overall_pass": overall_pass,
        "decision": "PASS" if overall_pass else "FAIL",
        "synthetic_only": True,
        "training_performed": False,
        "optimizer_created": False,
        "backward_performed": False,
        "corpus_accessed": False,
        "checkpoint_loaded": False,
        "checkpoint_written": False,
        "scientific_seed_consumed": False,
        "claims": {
            "mixed_dtype_write_primitive_passed": overall_pass,
            "successor_systems_preregistration_permitted": overall_pass,
            "end_to_end_systems_authorized": False,
            "architecture_freeze_authorized": False,
            "fresh_scientific_seed_authorized": False,
            "independent_replication_credit": False,
            "100m_authorized": False,
            "breakthrough_proven": False,
        },
    }
