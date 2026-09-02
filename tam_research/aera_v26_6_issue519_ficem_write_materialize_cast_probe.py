from __future__ import annotations

"""Issue #519 deterministic successor of the consumed #514 mixed-dtype WRITE gate."""

from typing import Any

import torch

from .aera_hardware_core_v24 import ContextualEpisodicMemoryState
from .aera_hardware_core_v26 import CoalescedFICEMMemory, TorchFICEMReferenceBackend
from .aera_hardware_core_v26_6_ficem_write_materialize_cast import (
    MaterializeCastTritonFICEMReadWriteBackend,
    fused_ficem_write_tail_materialize_cast,
    materialize_cast_ficem_read_write_v26_6_protocol,
)
from . import aera_v26_5_issue514_ficem_write_mixed_dtype_probe as frozen

RESEARCH_ISSUE = 519
SOURCE_MAIN = "cc9f401d7d3b5ed5c75dc8905ffc8f12df32616b"
CANDIDATE_BLOB = "d45c262314a0b4691f26812a279937a225043ad9"
PREDECESSOR_BLOB = frozen.PREDECESSOR_BLOB
READ_BACKEND_BLOB = frozen.READ_BACKEND_BLOB
V26_INTERFACE_BLOB = frozen.V26_INTERFACE_BLOB
STABLE_REFERENCE_BLOB = frozen.STABLE_REFERENCE_BLOB
HISTORICAL_PROBE_BLOB = frozen.HISTORICAL_PROBE_BLOB
FROZEN_ISSUE514_PROBE_BLOB = "e48dc308bef3b4ef9b6590ab0324db4a50c0f7af"
FROZEN_ISSUE514_LAUNCHER_BLOB = "1ab50f7b184feda61a6f6e1c7553296bed8863a6"
FROZEN_ISSUE514_WORKFLOW_BLOB = "5871b0a12e6168f16b59a1e7f1895feea6e8426c"
FROZEN_ISSUE514_RESULT_SHA256 = "c1a8936458c57e975787a27288d3caf494e360ec8ae8acb8d0f5742aef6bf505"
FROZEN_ISSUE514_RUN = 33664645415
FROZEN_ISSUE514_JOB = 100363263710
REPAIR_ISSUE = 517
REPAIR_PR = 518
REPAIR_CPU_HEAD = "c2edcfeb28efebe3818a92c5e00d53ea72689c75"
REPAIR_CPU_RUN = 33668780903
REPAIR_CPU_JOB = 100376942271
REPAIR_MERGE_MAIN = SOURCE_MAIN

# The decision surface is intentionally aliased to the frozen #514 probe so this
# repair gate cannot silently narrow or regenerate its deterministic fixtures.
DESIGN_SEED = frozen.DESIGN_SEED
D_MODEL = frozen.D_MODEL
WRITE_COUNT = frozen.WRITE_COUNT
CAPACITY = frozen.CAPACITY
MEMORY_DIM = frozen.MEMORY_DIM
DUPLICATE_THRESHOLD = frozen.DUPLICATE_THRESHOLD
BATCH_SIZES = frozen.BATCH_SIZES
MATRIX_MASKS = frozen.MATRIX_MASKS
FLOAT_FIELD_NAMES = frozen.FLOAT_FIELD_NAMES
EDGE_FIXTURES = frozen.EDGE_FIXTURES
EDGE_LAYOUTS = frozen.EDGE_LAYOUTS
PUBLIC_LAYOUTS = frozen.PUBLIC_LAYOUTS
FP32_ATOL = frozen.FP32_ATOL
FP32_RTOL = frozen.FP32_RTOL
BF16_ATOL = frozen.BF16_ATOL
BF16_RTOL = frozen.BF16_RTOL
TOPOLOGY_WARMUP_CALLS = frozen.TOPOLOGY_WARMUP_CALLS
EXPECTED_DIRECT_CASES = frozen.EXPECTED_DIRECT_CASES
EXPECTED_EDGE_CASES = frozen.EXPECTED_EDGE_CASES
EXPECTED_PUBLIC_ROWS = frozen.EXPECTED_PUBLIC_ROWS
EXPECTED_TOPOLOGY_ROWS = frozen.EXPECTED_TOPOLOGY_ROWS


def issue519_protocol() -> dict[str, Any]:
    return {
        **materialize_cast_ficem_read_write_v26_6_protocol(),
        "gate_version": "aera-v26.6-issue519-materialize-cast-write-correctness-topology",
        "gate_research_issue": RESEARCH_ISSUE,
        "source_main": SOURCE_MAIN,
        "candidate_blob": CANDIDATE_BLOB,
        "frozen_issue514_probe_blob": FROZEN_ISSUE514_PROBE_BLOB,
        "frozen_issue514_result_sha256": FROZEN_ISSUE514_RESULT_SHA256,
        "frozen_issue514_run": FROZEN_ISSUE514_RUN,
        "frozen_issue514_job": FROZEN_ISSUE514_JOB,
        "design_seed": DESIGN_SEED,
        "design_seed_is_scientific_seed": False,
        "decision_surface_reused_from_issue514": True,
        "geometry": {
            "d_model": D_MODEL,
            "write_count": WRITE_COUNT,
            "capacity": CAPACITY,
            "memory_dim": MEMORY_DIM,
        },
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
        "expected_adjudicate_kernel": "_write_adjudicate_map_kernel",
        "expected_materialize_kernel": "_write_materialize_cast_kernel",
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
    # Direct equality to the already-consumed #514 constants prevents any
    # candidate-dependent fixture adaptation after observing its result.
    assert DESIGN_SEED == frozen.DESIGN_SEED == 408_514
    assert (D_MODEL, WRITE_COUNT, CAPACITY, MEMORY_DIM) == (200, 16, 48, 50)
    assert DUPLICATE_THRESHOLD == frozen.DUPLICATE_THRESHOLD == 0.95
    assert BATCH_SIZES == frozen.BATCH_SIZES == (8, 64)
    assert MATRIX_MASKS == frozen.MATRIX_MASKS == tuple(range(256))
    assert FLOAT_FIELD_NAMES == frozen.FLOAT_FIELD_NAMES
    assert EDGE_FIXTURES == frozen.EDGE_FIXTURES
    assert EDGE_LAYOUTS == frozen.EDGE_LAYOUTS
    assert PUBLIC_LAYOUTS == frozen.PUBLIC_LAYOUTS
    assert (FP32_ATOL, FP32_RTOL) == (frozen.FP32_ATOL, frozen.FP32_RTOL) == (1e-5, 1e-5)
    assert (BF16_ATOL, BF16_RTOL) == (frozen.BF16_ATOL, frozen.BF16_RTOL) == (1e-2, 1e-2)
    assert (
        EXPECTED_DIRECT_CASES,
        EXPECTED_EDGE_CASES,
        EXPECTED_PUBLIC_ROWS,
        EXPECTED_TOPOLOGY_ROWS,
    ) == (256, 32, 6, 4)
    return {
        "protocol": issue519_protocol(),
        "gpu_authorized_by_cpu_preflight": False,
        "synthetic_only": True,
        "scientific_seed_consumed": False,
    }


def _candidate_tail(inputs: frozen.historical.TailInputs) -> ContextualEpisodicMemoryState:
    return fused_ficem_write_tail_materialize_cast(
        inputs.incoming_similarity,
        inputs.old_similarity,
        inputs.new_keys,
        inputs.new_values,
        inputs.new_strengths,
        inputs.new_valid,
        inputs.state,
    )


def _run_tail_case(
    memory: CoalescedFICEMMemory,
    inputs: frozen.historical.TailInputs,
) -> dict[str, Any]:
    before = frozen._clone_tail(inputs)
    try:
        with torch.no_grad():
            reference = frozen._reference_tail_durable(memory, inputs)
            candidate = _candidate_tail(inputs)
        comparison = frozen._compare_durable_states(reference, candidate, inputs.state)
        source_unchanged = frozen._tail_sources_unchanged(inputs, before)
        comparison["source_unchanged"] = source_unchanged
        comparison["pass"] = bool(comparison["pass"] and source_unchanged)
        return comparison
    except Exception as exc:
        return {
            "pass": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "source_unchanged": frozen._tail_sources_unchanged(inputs, before),
        }


def _run_public_case(
    memory: CoalescedFICEMMemory,
    candidate_backend: MaterializeCastTritonFICEMReadWriteBackend,
    reference_backend: TorchFICEMReferenceBackend,
    *,
    layout: str,
    batch_size: int,
    generator: torch.Generator,
    device: torch.device,
) -> dict[str, Any]:
    projected, normalized_old, payload, strength, state, use_autocast = frozen._make_public_case(
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
        frozen._clone_state(state),
    )
    try:
        with torch.no_grad(), frozen._public_context(use_autocast):
            reference_raw = reference_backend.update_from_projected(
                memory,
                projected,
                normalized_old,
                payload,
                strength,
                frozen._clone_state(state),
            )
            candidate = candidate_backend.update_from_projected(
                memory,
                projected,
                normalized_old,
                payload,
                strength,
                frozen._clone_state(state),
            )
        reference = ContextualEpisodicMemoryState(
            keys=reference_raw.keys.to(dtype=state.keys.dtype),
            values=reference_raw.values.to(dtype=state.values.dtype),
            strengths=reference_raw.strengths.to(dtype=state.strengths.dtype),
            valid=reference_raw.valid,
        )
        comparison = frozen._compare_durable_states(reference, candidate, state)
        after = (projected, normalized_old, payload, strength, state)
        source_unchanged = frozen._public_sources_unchanged(before, after)
        comparison["source_unchanged"] = source_unchanged
        comparison["pass"] = bool(comparison["pass"] and source_unchanged)
        comparison["input_dtypes"] = {
            "projected": frozen._dtype_name(projected.dtype),
            "normalized_old": frozen._dtype_name(normalized_old.dtype),
            "payload": frozen._dtype_name(payload.dtype),
            "strength": frozen._dtype_name(strength.dtype),
            "state_keys": frozen._dtype_name(state.keys.dtype),
            "state_values": frozen._dtype_name(state.values.dtype),
            "state_strengths": frozen._dtype_name(state.strengths.dtype),
        }
        return comparison
    except Exception as exc:
        return {
            "pass": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "input_dtypes": {
                "projected": frozen._dtype_name(projected.dtype),
                "normalized_old": frozen._dtype_name(normalized_old.dtype),
                "payload": frozen._dtype_name(payload.dtype),
                "strength": frozen._dtype_name(strength.dtype),
                "state_keys": frozen._dtype_name(state.keys.dtype),
                "state_values": frozen._dtype_name(state.values.dtype),
                "state_strengths": frozen._dtype_name(state.strengths.dtype),
            },
        }


def _profile_candidate_tail(inputs: frozen.historical.TailInputs) -> dict[str, Any]:
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
    materialize = [key for key in keys if "_write_materialize_cast_kernel" in key]
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


def run_materialize_cast_write_probe() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("issue519 v26.6 materialize-cast FICEM WRITE gate requires CUDA")
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    torch.manual_seed(DESIGN_SEED)
    torch.cuda.manual_seed_all(DESIGN_SEED)

    memory = frozen._build_memory(device)
    candidate_backend = MaterializeCastTritonFICEMReadWriteBackend()
    reference_backend = TorchFICEMReferenceBackend()

    direct_results: dict[str, Any] = {}
    direct_pass = True
    base = frozen._base_matrix_inputs(device)
    for mask in MATRIX_MASKS:
        dtypes = frozen._mask_dtypes(mask)
        inputs = frozen._cast_tail_fields(base, dtypes)
        row = _run_tail_case(memory, inputs)
        row["dtypes"] = [frozen._dtype_name(dtype) for dtype in dtypes]
        direct_results[f"mask_{mask:03d}"] = row
        direct_pass = bool(direct_pass and row["pass"])

    edge_results: dict[str, Any] = {}
    edge_pass = True
    representative_inputs: dict[str, frozen.historical.TailInputs] = {}
    for fixture_name in EDGE_FIXTURES:
        base_edge = frozen.historical.make_edge_fixture(
            fixture_name,
            dtype_name="float32",
            device=device,
        )
        for layout in EDGE_LAYOUTS:
            dtypes = frozen._layout_dtypes(layout)
            inputs = frozen._cast_tail_fields(base_edge, dtypes)
            key = f"{fixture_name}__{layout}"
            row = _run_tail_case(memory, inputs)
            row["dtypes"] = [frozen._dtype_name(dtype) for dtype in dtypes]
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
        "version": "aera-v26.6-issue519-materialize-cast-write-correctness-topology",
        "research_issue": RESEARCH_ISSUE,
        "source_main": SOURCE_MAIN,
        "device": torch.cuda.get_device_name(0),
        "protocol": issue519_protocol(),
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
            "materialize_cast_write_primitive_passed": overall_pass,
            "successor_systems_preregistration_permitted": overall_pass,
            "end_to_end_systems_authorized": False,
            "architecture_freeze_authorized": False,
            "fresh_scientific_seed_authorized": False,
            "independent_replication_credit": False,
            "100m_authorized": False,
            "breakthrough_proven": False,
        },
    }
