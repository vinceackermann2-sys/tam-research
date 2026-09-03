from __future__ import annotations

"""Issue #558 successor probe for the merged AERA-v26.8 mixed READ repair.

The decision surface is the exact frozen #553 probe.  This shim changes only the
candidate backend/tail and the mixed-tail profiler's Triton event-name recognition
needed because v26.8's one mixed-only kernel has a new function name.  Fixtures,
generator order, correctness equations, timing, thresholds and historical surface
remain in the byte-frozen #553 module.
"""

from typing import Any, Callable

import torch

from . import aera_v26_7_issue553_ficem_read_mixed_dtype_probe as frozen553
from . import aera_hardware_core_v26_8_ficem_read_mixed_strength_precision as v26_8

RESEARCH_ISSUE = 558
SOURCE_MAIN = "ae25cb4133c1ff94bec1cdfa9aa58e4081c05c73"
SOURCE_TREE = "3b59aa070a98f873d728d2c30ab08156f73bec23"
V26_8_BACKEND_BLOB = "3575c58d1cd730be77649f087908c51dbf3e6088"
V26_8_CPU_TEST_BLOB = "443d36dcc61eb72f8a2f406a6f2ae1abfeb365c4"
FROZEN_ISSUE553_PROBE_BLOB = "ff9a47f510be07e8adeff018f327338147163cdb"
V26_7_BACKEND_BLOB = "d8133c6b204b1ee5f23955255fb2fb09d09bd723"
REPAIR5_BACKEND_BLOB = "263f68eb1186a8ac14a08fc4b4df1fc5b292c711"
V26_6_WRITE_BACKEND_BLOB = "d45c262314a0b4691f26812a279937a225043ad9"
HISTORICAL_PROBE_BLOB = "16aa99b9f6f0a1d11bd7bf5f36b2f0b1fb97047b"
REPAIR5_PROBE_BLOB = "6fd6518e10ed1ef4115863f98ac591ffd77ce903"
ISSUE530_SYSTEMS_BLOB = "9d5a3c31f4a3862f96b957540baa2e0ec6a84c6b"
V26_INTERFACE_BLOB = "268644ac4edee15a4cc4e29d3fed7f61eeb3caa7"
STABLE_REFERENCE_BLOB = "4e336b6e1a6238dac782fa320751d68281493ee1"

ISSUE557_HEAD = "783c5e2921d1d7fc7f598948d1bb968e51260440"
ISSUE557_CPU_RUN = 33730039451
ISSUE557_CPU_JOB = 100567632346
ISSUE557_MERGE = SOURCE_MAIN

ISSUE553_TRIGGER = 555
ISSUE553_RUN = 33727540468
ISSUE553_JOB = 100559866985
ISSUE553_RESULT_PATH = "/vol/aera-v26/issue553-ficem-read-mixed-dtype/result.json"
ISSUE553_RESULT_SHA256 = (
    "009af31baf70e46eb93b6e7489d62f356a02b727521d3fabe4a7dab2dcf5ab47"
)
ISSUE553_DECISION = "FAIL"

ISSUE545_TRIGGER = 550
ISSUE545_RUN = 33686037672
ISSUE545_JOB = 100433658768
ISSUE545_FAILURE = "FICEM read-tail floating dtypes must match"

ISSUE479_TRIGGER = 484
ISSUE479_RUN = 33618950619
ISSUE479_JOB = 100211244996

ISSUE529_TRIGGER = 529
ISSUE529_RUN = 33680028132
ISSUE529_JOB = 100414089065
ISSUE529_RESULT_SHA256 = (
    "a07790c2d55d7a696baf0e903a05dbdf925e3ef2a08b84c784c77d1fbdd31874"
)

DESIGN_SEED = frozen553.DESIGN_SEED
D_MODEL = frozen553.D_MODEL
MEMORY_DIM = frozen553.MEMORY_DIM
CAPACITY = frozen553.CAPACITY
TIME = frozen553.TIME
BATCH_SIZES = frozen553.BATCH_SIZES
DTYPE_NAMES = frozen553.DTYPE_NAMES
VALIDITY_KINDS = frozen553.VALIDITY_KINDS
WARMUP_CALLS = frozen553.WARMUP_CALLS
TIMED_ROUNDS = frozen553.TIMED_ROUNDS
CALLS_PER_ROUND = frozen553.CALLS_PER_ROUND
FP32_ATOL = frozen553.FP32_ATOL
FP32_RTOL = frozen553.FP32_RTOL
BF16_ATOL = frozen553.BF16_ATOL
BF16_RTOL = frozen553.BF16_RTOL
MAX_GEOMEAN_LATENCY_RATIO = frozen553.MAX_GEOMEAN_LATENCY_RATIO
MAX_ROW_LATENCY_RATIO = frozen553.MAX_ROW_LATENCY_RATIO
MAX_FULL_EVENT_RATIO = frozen553.MAX_FULL_EVENT_RATIO
MIXED_LAYOUTS = frozen553.MIXED_LAYOUTS


def issue558_protocol() -> dict[str, Any]:
    protocol = dict(v26_8.mixed_strength_precision_v26_8_protocol())
    protocol.update(
        {
            "probe_version": "aera-v26.8-issue558-ficem-read-mixed-strength-precision-l4",
            "research_issue": RESEARCH_ISSUE,
            "source_main_issue558": SOURCE_MAIN,
            "source_tree_issue558": SOURCE_TREE,
            "v26_8_backend_blob": V26_8_BACKEND_BLOB,
            "v26_8_cpu_test_blob": V26_8_CPU_TEST_BLOB,
            "frozen_issue553_probe_blob": FROZEN_ISSUE553_PROBE_BLOB,
            "v26_7_backend_blob": V26_7_BACKEND_BLOB,
            "repair5_backend_blob": REPAIR5_BACKEND_BLOB,
            "v26_6_write_backend_blob": V26_6_WRITE_BACKEND_BLOB,
            "historical_probe_blob": HISTORICAL_PROBE_BLOB,
            "repair5_probe_blob": REPAIR5_PROBE_BLOB,
            "issue530_systems_blob": ISSUE530_SYSTEMS_BLOB,
            "v26_interface_blob": V26_INTERFACE_BLOB,
            "stable_reference_blob": STABLE_REFERENCE_BLOB,
            "issue557_head": ISSUE557_HEAD,
            "issue557_cpu_run": ISSUE557_CPU_RUN,
            "issue557_cpu_job": ISSUE557_CPU_JOB,
            "issue557_merge": ISSUE557_MERGE,
            "issue553_trigger": ISSUE553_TRIGGER,
            "issue553_run": ISSUE553_RUN,
            "issue553_job": ISSUE553_JOB,
            "issue553_result_path": ISSUE553_RESULT_PATH,
            "issue553_result_sha256": ISSUE553_RESULT_SHA256,
            "issue553_decision": ISSUE553_DECISION,
            "issue553_consumed": True,
            "issue545_trigger": ISSUE545_TRIGGER,
            "issue545_run": ISSUE545_RUN,
            "issue545_job": ISSUE545_JOB,
            "issue545_failure": ISSUE545_FAILURE,
            "issue479_trigger": ISSUE479_TRIGGER,
            "issue479_run": ISSUE479_RUN,
            "issue479_job": ISSUE479_JOB,
            "issue529_trigger": ISSUE529_TRIGGER,
            "issue529_run": ISSUE529_RUN,
            "issue529_job": ISSUE529_JOB,
            "issue529_result_sha256": ISSUE529_RESULT_SHA256,
            "frozen_issue553_probe_logic_reused": True,
            "candidate_substitution_only": True,
            "mixed_tail_profiler_acceptance_changed": False,
            "mixed_tail_profiler_event_name_updated_only": True,
            "historical_surface_preserved": True,
            "historical_surface_candidate_is_v26_8": True,
            "mixed_layouts": [list(item) for item in MIXED_LAYOUTS],
            "mixed_regular_generator_continues_historical_stream": True,
            "mixed_timing_decision_bearing": False,
            "historical_timing_decision_bearing": True,
            "integration_bf16_compute_fp32_durable_full_backend_required": True,
            "complementary_fp32_compute_bf16_durable_full_backend_required": False,
            "design_seed": DESIGN_SEED,
            "design_seed_is_scientific_seed": False,
            "d_model": D_MODEL,
            "time": TIME,
            "capacity": CAPACITY,
            "memory_dim": MEMORY_DIM,
            "batch_sizes": list(BATCH_SIZES),
            "historical_dtypes": list(DTYPE_NAMES),
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
            "gpu_authorized_by_probe_module": False,
            "end_to_end_systems_authorized": False,
            "architecture_freeze_authorized": False,
            "s2_authorized": False,
            "fresh_scientific_seed_authorized": False,
            "independent_replication_credit": False,
            "100m_authorized": False,
            "breakthrough_proven": False,
            "scientific_seed_consumed": False,
        }
    )
    return protocol


def cpu_contract_preflight_issue558() -> dict[str, Any]:
    predecessor = frozen553.cpu_contract_preflight_issue553()
    candidate = v26_8.cpu_contract_preflight_issue556()

    if DESIGN_SEED != 408_411:
        raise RuntimeError("issue558 design seed drifted")
    if (D_MODEL, MEMORY_DIM, CAPACITY, TIME) != (200, 50, 48, 256):
        raise RuntimeError("issue558 geometry drifted")
    if BATCH_SIZES != (8, 64):
        raise RuntimeError("issue558 batch order drifted")
    if DTYPE_NAMES != ("float32", "bfloat16"):
        raise RuntimeError("issue558 historical dtype order drifted")
    if VALIDITY_KINDS != ("mixed", "full"):
        raise RuntimeError("issue558 validity order drifted")
    if MIXED_LAYOUTS != (("bfloat16", "float32"), ("float32", "bfloat16")):
        raise RuntimeError("issue558 mixed layout order drifted")
    if (WARMUP_CALLS, TIMED_ROUNDS, CALLS_PER_ROUND) != (10, 5, 100):
        raise RuntimeError("issue558 historical timing drifted")
    if (FP32_ATOL, FP32_RTOL, BF16_ATOL, BF16_RTOL) != (
        1e-5,
        1e-5,
        1e-2,
        1e-2,
    ):
        raise RuntimeError("issue558 tolerance drifted")
    if (
        MAX_GEOMEAN_LATENCY_RATIO != 0.90
        or MAX_ROW_LATENCY_RATIO != 1.05
        or MAX_FULL_EVENT_RATIO != 0.75
    ):
        raise RuntimeError("issue558 historical thresholds drifted")
    if predecessor["gpu_authorized_by_cpu_preflight"] is not False:
        raise RuntimeError("issue553 predecessor CPU contract drifted")
    if candidate["gpu_authorized_by_cpu_preflight"] is not False:
        raise RuntimeError("issue556 candidate CPU contract drifted")
    if frozen553.MixedDtypeTritonFICEMReadWriteBackend is v26_8.StrengthPrecisionTritonFICEMReadWriteBackend:
        raise RuntimeError("issue558 frozen #553 backend global was mutated before execution")
    if frozen553.fused_ficem_read_tail_mixed_dtype is v26_8.fused_ficem_read_tail_v26_8:
        raise RuntimeError("issue558 frozen #553 tail global was mutated before execution")

    protocol = issue558_protocol()
    forbidden = (
        "gpu_authorized_by_probe_module",
        "mixed_dtype_read_gpu_gate_authorized",
        "end_to_end_systems_authorized",
        "architecture_freeze_authorized",
        "s2_authorized",
        "fresh_scientific_seed_authorized",
        "independent_replication_credit",
        "100m_authorized",
        "breakthrough_proven",
        "scientific_seed_consumed",
    )
    if any(protocol[key] for key in forbidden):
        raise RuntimeError("issue558 CPU contract unexpectedly authorizes higher work")

    return {
        "frozen_issue553_contract": predecessor,
        "v26_8_candidate_contract": candidate,
        "protocol": protocol,
        "gpu_authorized_by_cpu_preflight": False,
        "synthetic_only": True,
        "scientific_seed_consumed": False,
    }


def _tail_profile_with_v26_8_event_name(call: Callable[[], Any]) -> dict[str, Any]:
    """Exact #553 mixed profiler, recognizing only the v26.8 mixed kernel name."""

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
            if "mixed_strength_precision_kernel" in str(getattr(event, "name", "")):
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


def run_ficem_read_probe_v26_8_issue558() -> dict[str, Any]:
    """Run the exact #553 gate with only the v26.8 candidate substitutions."""

    cpu_contract_preflight_issue558()

    original_backend = frozen553.MixedDtypeTritonFICEMReadWriteBackend
    original_tail = frozen553.fused_ficem_read_tail_mixed_dtype
    original_profiler = frozen553._tail_profile_with_cast_accounting
    try:
        frozen553.MixedDtypeTritonFICEMReadWriteBackend = (
            v26_8.StrengthPrecisionTritonFICEMReadWriteBackend
        )
        frozen553.fused_ficem_read_tail_mixed_dtype = v26_8.fused_ficem_read_tail_v26_8
        frozen553._tail_profile_with_cast_accounting = _tail_profile_with_v26_8_event_name
        result = frozen553.run_ficem_read_mixed_dtype_probe_v26_7()
    finally:
        frozen553.MixedDtypeTritonFICEMReadWriteBackend = original_backend
        frozen553.fused_ficem_read_tail_mixed_dtype = original_tail
        frozen553._tail_profile_with_cast_accounting = original_profiler

    frozen_protocol = result["protocol"]
    result["frozen_issue553_protocol"] = frozen_protocol
    result["protocol"] = issue558_protocol()
    result["issue558_probe_metadata"] = {
        "source_main": SOURCE_MAIN,
        "source_tree": SOURCE_TREE,
        "frozen_issue553_probe_blob": FROZEN_ISSUE553_PROBE_BLOB,
        "v26_8_backend_blob": V26_8_BACKEND_BLOB,
        "candidate_substitution_only": True,
        "mixed_tail_profiler_acceptance_changed": False,
        "mixed_tail_profiler_event_name_updated_only": True,
        "scientific_seed_consumed": False,
    }
    return result
