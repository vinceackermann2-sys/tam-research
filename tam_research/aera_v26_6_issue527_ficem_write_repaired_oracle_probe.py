from __future__ import annotations

"""Issue #527 one-shot repaired-oracle successor of the consumed #519 WRITE gate."""

from typing import Any

import torch

from .aera_hardware_core_v24 import ContextualEpisodicMemoryState
from . import aera_v26_6_issue519_ficem_write_materialize_cast_probe as frozen519
from .aera_v26_6_issue525_mixed_dtype_write_oracle import (
    durable_mixed_dtype_reference_tail,
)

RESEARCH_ISSUE = 527
SOURCE_MAIN = "e18aa12f1ddd96ba30f1b3f5e2be67d5f0922116"
CANDIDATE_BLOB = "d45c262314a0b4691f26812a279937a225043ad9"
FROZEN_ISSUE519_PROBE_BLOB = "ec22807434192f58e292bffc3de9828be2b44272"
FROZEN_ISSUE519_RESULT_SHA256 = (
    "b9fba0fca96644ef8db9bc46faf2c73d0c0cc1f1aaac6a321abe2411d3703cd5"
)
FROZEN_ISSUE519_RUN = 33672232063
FROZEN_ISSUE519_JOB = 100388368044
FROZEN_ISSUE519_ATTEMPT = 1
FROZEN_ISSUE522_RUN = 33675476637
FROZEN_ISSUE522_JOB = 100398984660
FROZEN_ISSUE522_ATTEMPT = 1
FROZEN_ISSUE522_EXCEPTION = "scatter(): Expected self.dtype to be equal to src.dtype"
ORACLE_BLOB = "8f472451af4024bb3faacb56d814f7d6bdb25cc9"
ORACLE_CPU_TEST_BLOB = "de3ae08b9db04803359d216f601d5c68dac3a542"
ORACLE_CPU_HEAD = "275fed17e0e0c855a8f9a5fd39bfa484e1b78ed2"
ORACLE_CPU_RUN = 33676365160
ORACLE_CPU_JOB = 100401938039
ORACLE_MERGE_MAIN = SOURCE_MAIN

# Reuse the consumed #519 decision surface exactly. No fixture regeneration.
DESIGN_SEED = frozen519.DESIGN_SEED
D_MODEL = frozen519.D_MODEL
WRITE_COUNT = frozen519.WRITE_COUNT
CAPACITY = frozen519.CAPACITY
MEMORY_DIM = frozen519.MEMORY_DIM
DUPLICATE_THRESHOLD = frozen519.DUPLICATE_THRESHOLD
BATCH_SIZES = frozen519.BATCH_SIZES
MATRIX_MASKS = frozen519.MATRIX_MASKS
FLOAT_FIELD_NAMES = frozen519.FLOAT_FIELD_NAMES
EDGE_FIXTURES = frozen519.EDGE_FIXTURES
EDGE_LAYOUTS = frozen519.EDGE_LAYOUTS
PUBLIC_LAYOUTS = frozen519.PUBLIC_LAYOUTS
FP32_ATOL = frozen519.FP32_ATOL
FP32_RTOL = frozen519.FP32_RTOL
BF16_ATOL = frozen519.BF16_ATOL
BF16_RTOL = frozen519.BF16_RTOL
TOPOLOGY_WARMUP_CALLS = frozen519.TOPOLOGY_WARMUP_CALLS
EXPECTED_DIRECT_CASES = frozen519.EXPECTED_DIRECT_CASES
EXPECTED_EDGE_CASES = frozen519.EXPECTED_EDGE_CASES
EXPECTED_PUBLIC_ROWS = frozen519.EXPECTED_PUBLIC_ROWS
EXPECTED_TOPOLOGY_ROWS = frozen519.EXPECTED_TOPOLOGY_ROWS


def issue527_protocol() -> dict[str, Any]:
    return {
        **frozen519.materialize_cast_ficem_read_write_v26_6_protocol(),
        "gate_version": "aera-v26.6-issue527-repaired-oracle-write-correctness-topology",
        "gate_research_issue": RESEARCH_ISSUE,
        "source_main": SOURCE_MAIN,
        "candidate_blob": CANDIDATE_BLOB,
        "frozen_issue519_probe_blob": FROZEN_ISSUE519_PROBE_BLOB,
        "frozen_issue519_result_sha256": FROZEN_ISSUE519_RESULT_SHA256,
        "frozen_issue519_run": FROZEN_ISSUE519_RUN,
        "frozen_issue519_job": FROZEN_ISSUE519_JOB,
        "frozen_issue519_attempt": FROZEN_ISSUE519_ATTEMPT,
        "frozen_issue522_run": FROZEN_ISSUE522_RUN,
        "frozen_issue522_job": FROZEN_ISSUE522_JOB,
        "oracle_blob": ORACLE_BLOB,
        "oracle_cpu_test_blob": ORACLE_CPU_TEST_BLOB,
        "oracle_cpu_head": ORACLE_CPU_HEAD,
        "oracle_cpu_run": ORACLE_CPU_RUN,
        "oracle_cpu_job": ORACLE_CPU_JOB,
        "oracle_merge_main": ORACLE_MERGE_MAIN,
        "direct_edge_oracle": "issue525_durable_mixed_dtype_reference_tail",
        "public_reference_reused_from_issue519": True,
        "topology_contract_reused_from_issue519": True,
        "decision_surface_reused_from_issue519": True,
        "design_seed": DESIGN_SEED,
        "design_seed_is_scientific_seed": False,
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
    assert DESIGN_SEED == frozen519.DESIGN_SEED == 408_514
    assert (D_MODEL, WRITE_COUNT, CAPACITY, MEMORY_DIM) == (200, 16, 48, 50)
    assert DUPLICATE_THRESHOLD == frozen519.DUPLICATE_THRESHOLD == 0.95
    assert BATCH_SIZES == frozen519.BATCH_SIZES == (8, 64)
    assert MATRIX_MASKS == frozen519.MATRIX_MASKS == tuple(range(256))
    assert FLOAT_FIELD_NAMES == frozen519.FLOAT_FIELD_NAMES
    assert EDGE_FIXTURES == frozen519.EDGE_FIXTURES
    assert EDGE_LAYOUTS == frozen519.EDGE_LAYOUTS
    assert PUBLIC_LAYOUTS == frozen519.PUBLIC_LAYOUTS
    assert (FP32_ATOL, FP32_RTOL) == (1e-5, 1e-5)
    assert (BF16_ATOL, BF16_RTOL) == (1e-2, 1e-2)
    assert (
        EXPECTED_DIRECT_CASES,
        EXPECTED_EDGE_CASES,
        EXPECTED_PUBLIC_ROWS,
        EXPECTED_TOPOLOGY_ROWS,
    ) == (256, 32, 6, 4)
    return {
        "protocol": issue527_protocol(),
        "gpu_authorized_by_cpu_preflight": False,
        "synthetic_only": True,
        "scientific_seed_consumed": False,
    }


def _run_tail_case(
    memory: frozen519.CoalescedFICEMMemory,
    inputs: frozen519.frozen.historical.TailInputs,
) -> dict[str, Any]:
    before = frozen519.frozen._clone_tail(inputs)
    try:
        with torch.no_grad():
            reference = durable_mixed_dtype_reference_tail(memory, inputs)
            candidate = frozen519._candidate_tail(inputs)
        comparison = frozen519.frozen._compare_durable_states(
            reference, candidate, inputs.state
        )
        source_unchanged = frozen519.frozen._tail_sources_unchanged(inputs, before)
        comparison["source_unchanged"] = source_unchanged
        comparison["pass"] = bool(comparison["pass"] and source_unchanged)
        return comparison
    except Exception as exc:
        return {
            "pass": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "source_unchanged": frozen519.frozen._tail_sources_unchanged(inputs, before),
        }


def run_repaired_oracle_write_probe() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("issue527 repaired-oracle FICEM WRITE gate requires CUDA")

    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    torch.manual_seed(DESIGN_SEED)
    torch.cuda.manual_seed_all(DESIGN_SEED)

    memory = frozen519.frozen._build_memory(device)
    candidate_backend = frozen519.MaterializeCastTritonFICEMReadWriteBackend()
    reference_backend = frozen519.TorchFICEMReferenceBackend()

    direct_results: dict[str, Any] = {}
    direct_pass = True
    base = frozen519.frozen._base_matrix_inputs(device)
    for mask in MATRIX_MASKS:
        dtypes = frozen519.frozen._mask_dtypes(mask)
        inputs = frozen519.frozen._cast_tail_fields(base, dtypes)
        row = _run_tail_case(memory, inputs)
        row["dtypes"] = [frozen519.frozen._dtype_name(dtype) for dtype in dtypes]
        direct_results[f"mask_{mask:03d}"] = row
        direct_pass = bool(direct_pass and row["pass"])

    edge_results: dict[str, Any] = {}
    edge_pass = True
    representative_inputs: dict[str, frozen519.frozen.historical.TailInputs] = {}
    for fixture_name in EDGE_FIXTURES:
        base_edge = frozen519.frozen.historical.make_edge_fixture(
            fixture_name,
            dtype_name="float32",
            device=device,
        )
        for layout in EDGE_LAYOUTS:
            dtypes = frozen519.frozen._layout_dtypes(layout)
            inputs = frozen519.frozen._cast_tail_fields(base_edge, dtypes)
            key = f"{fixture_name}__{layout}"
            row = _run_tail_case(memory, inputs)
            row["dtypes"] = [frozen519.frozen._dtype_name(dtype) for dtype in dtypes]
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
            row = frozen519._run_public_case(
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
            row = frozen519._profile_candidate_tail(representative_inputs[layout])
        except Exception as exc:
            row = {"pass": False, "error_type": type(exc).__name__, "error": str(exc)}
        topology_results[layout] = row
        topology_pass = bool(topology_pass and row["pass"])

    overall_pass = bool(direct_pass and edge_pass and public_pass and topology_pass)
    return {
        "version": "aera-v26.6-issue527-repaired-oracle-write-correctness-topology",
        "research_issue": RESEARCH_ISSUE,
        "source_main": SOURCE_MAIN,
        "device": torch.cuda.get_device_name(0),
        "protocol": issue527_protocol(),
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
            "repaired_oracle_write_primitive_passed": overall_pass,
            "successor_systems_preregistration_permitted": overall_pass,
            "end_to_end_systems_authorized": False,
            "architecture_freeze_authorized": False,
            "fresh_scientific_seed_authorized": False,
            "independent_replication_credit": False,
            "100m_authorized": False,
            "breakthrough_proven": False,
        },
    }
