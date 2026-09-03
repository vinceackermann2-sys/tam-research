from __future__ import annotations

"""Issue #569 CPU-only memory-safe successor for the frozen v26.8 systems evaluator.

The scientific/system decision surface remains byte-frozen in #501/#530/#562.
This module changes only the implementation of two full-logit reductions:
`_logit_equivalence` and `_finite_output`.  Both are evaluated deterministically
one leading batch row at a time, preserving every element and the frozen
tolerances/metadata semantics while avoiding monolithic FP32 logit temporaries.

Issue #569 authorizes no GPU run.  A later separately preregistered one-shot
hardware gate is required after this adapter is CPU-green and merged.
"""

import math
from typing import Any

import torch

from . import aera_v26_5_end_to_end_systems as base
from . import aera_v26_8_issue562_end_to_end_systems as issue562

RESEARCH_ISSUE = 569
SOURCE_MAIN = "6e9471ca86fed0438bd25dd314040a16e637f2be"
SOURCE_TREE = "64f452228c58d5a33549dcf601912f6568f5701c"

BASE_SYSTEMS_BLOB = "c9731cae7e386f09b2a190b045532591c4fa00be"
REPAIR1_SYSTEMS_BLOB = "b3f7082b188644007b873db3733492f424d4941a"
ISSUE530_SYSTEMS_BLOB = "9d5a3c31f4a3862f96b957540baa2e0ec6a84c6b"
ISSUE562_ADAPTER_BLOB = "3534103eea21f7c4d9d31798ad34601fd47090d6"
ISSUE562_CPU_TEST_BLOB = "2893a86cbdf767cdfa92601503d107d5ca3912fb"
V26_8_CANDIDATE_BLOB = "3575c58d1cd730be77649f087908c51dbf3e6088"
ISSUE564_LAUNCHER_BLOB = "3b6268a905b4fd2707d1deaf5950c7a89682b4bd"
ISSUE564_WORKFLOW_BLOB = "376e693b2116c027d29211374928a8379667fa87"
ISSUE564_CPU_TEST_BLOB = "fdfd1d120651c92d4678a073d8e6dc67ea4c8b05"

ISSUE566_TRIGGER = 566
ISSUE566_BOUND_MAIN = SOURCE_MAIN
ISSUE566_RUN = 33737873193
ISSUE566_JOB = 100592625418
ISSUE566_ATTEMPT = 1
ISSUE566_AUTHORITATIVE_RESULT_EMITTED = False
ISSUE566_L4_STARTED = True
ISSUE566_FAILURE_TYPE = "torch.OutOfMemoryError"
ISSUE566_FAILURE = "CUDA out of memory. Tried to allocate 6.14 GiB."
ISSUE566_FAILURE_SITE = (
    "tam_research/aera_v26_5_end_to_end_systems.py::_logit_equivalence"
)
ISSUE566_FAILURE_EXPRESSION = (
    "float((reference.float() - candidate.float()).abs().max())"
)

ISSUE567_TRIGGER = 567
ISSUE567_RUN = 33737887818
ISSUE567_JOB = 100592679436
ISSUE567_CANONICAL = False
ISSUE567_GPU_STARTED = False

CHECKPOINT_HASHES = {
    "aera": "f8aa92421801e8f190247e420632be5f0c20bc5ea8bf6bdeefe06686b3a31b30",
    "transformer": "cdd5cab4439a709468d6607d45d82081b33e876b2e40d91d4a38ba139b219dd7",
}

CHUNK_BATCH_ROWS = 1

# Capture exact objects before the controlled runtime substitution.
_FROZEN_LOGIT_EQUIVALENCE = base._logit_equivalence
_FROZEN_FINITE_OUTPUT = base._finite_output
_FROZEN_ISSUE562_RUN = issue562.run_end_to_end_systems_v26_8


def issue569_systems_protocol() -> dict[str, Any]:
    """Describe the CPU-only evaluator-memory repair without changing thresholds."""

    predecessor = dict(issue562.issue562_systems_protocol())
    predecessor.update(
        {
            "version": "aera-v26.8-issue569-end-to-end-systems-memory-safe",
            "research_issue": RESEARCH_ISSUE,
            "source_main_issue569": SOURCE_MAIN,
            "source_tree_issue569": SOURCE_TREE,
            "base_systems_blob": BASE_SYSTEMS_BLOB,
            "repair1_systems_blob": REPAIR1_SYSTEMS_BLOB,
            "issue530_systems_blob": ISSUE530_SYSTEMS_BLOB,
            "issue562_adapter_blob": ISSUE562_ADAPTER_BLOB,
            "issue562_cpu_test_blob": ISSUE562_CPU_TEST_BLOB,
            "v26_8_candidate_blob": V26_8_CANDIDATE_BLOB,
            "issue564_launcher_blob": ISSUE564_LAUNCHER_BLOB,
            "issue564_workflow_blob": ISSUE564_WORKFLOW_BLOB,
            "issue564_cpu_test_blob": ISSUE564_CPU_TEST_BLOB,
            "chunk_batch_rows": CHUNK_BATCH_ROWS,
            "logit_equivalence_change": "deterministic_leading_batch_row_chunked_fp32_reduction",
            "finite_output_change": "deterministic_leading_batch_row_chunked_logits_finite_reduction",
            "all_elements_covered": True,
            "sampling_or_approximation": False,
            "result_dependent_chunk_sizing": False,
            "issue566_trigger": ISSUE566_TRIGGER,
            "issue566_bound_main": ISSUE566_BOUND_MAIN,
            "issue566_run": ISSUE566_RUN,
            "issue566_job": ISSUE566_JOB,
            "issue566_attempt": ISSUE566_ATTEMPT,
            "issue566_authoritative_result_emitted": ISSUE566_AUTHORITATIVE_RESULT_EMITTED,
            "issue566_l4_started": ISSUE566_L4_STARTED,
            "issue566_failure_type": ISSUE566_FAILURE_TYPE,
            "issue566_failure": ISSUE566_FAILURE,
            "issue566_failure_site": ISSUE566_FAILURE_SITE,
            "issue566_failure_expression": ISSUE566_FAILURE_EXPRESSION,
            "issue567_trigger": ISSUE567_TRIGGER,
            "issue567_run": ISSUE567_RUN,
            "issue567_job": ISSUE567_JOB,
            "issue567_canonical": ISSUE567_CANONICAL,
            "issue567_gpu_started": ISSUE567_GPU_STARTED,
            "checkpoint_hashes_issue566": dict(CHECKPOINT_HASHES),
            "gpu_authorized_by_issue569": False,
            "end_to_end_systems_executed_by_issue569": False,
            "architecture_freeze_authorized": False,
            "s2_authorized": False,
            "fresh_scientific_seed_authorized": False,
            "independent_replication_credit": False,
            "100m_authorized": False,
            "breakthrough_proven": False,
        }
    )
    return predecessor


def memory_safe_logit_equivalence(
    reference: torch.Tensor, candidate: torch.Tensor
) -> dict[str, Any]:
    """Frozen `_logit_equivalence`, reduced one leading batch row at a time."""

    exact_meta = (
        reference.dtype == candidate.dtype
        and reference.device == candidate.device
        and reference.shape == candidate.shape
    )

    if reference.ndim < 1 or candidate.ndim < 1:
        raise RuntimeError("issue569 requires batch-major logits with a leading batch dimension")
    if reference.size(0) != candidate.size(0):
        raise RuntimeError("issue569 reference/candidate leading batch dimensions differ")
    if reference.size(0) < 1:
        raise RuntimeError("issue569 requires non-empty batch-major logits")

    max_abs: float | None = None
    close = True
    for start in range(0, reference.size(0), CHUNK_BATCH_ROWS):
        stop = min(start + CHUNK_BATCH_ROWS, reference.size(0))
        reference_chunk = reference[start:stop].float()
        candidate_chunk = candidate[start:stop].float()
        chunk_max_abs = float((reference_chunk - candidate_chunk).abs().max())
        chunk_close = bool(
            torch.allclose(
                reference_chunk,
                candidate_chunk,
                atol=base.INTEGRATED_ATOL,
                rtol=base.INTEGRATED_RTOL,
            )
        )
        if max_abs is None or math.isnan(chunk_max_abs):
            max_abs = chunk_max_abs
        elif not math.isnan(max_abs):
            max_abs = max(max_abs, chunk_max_abs)
        close = bool(close and chunk_close)

    if max_abs is None:
        raise RuntimeError("issue569 logit reduction visited no rows")

    return {
        "pass": bool(exact_meta and close),
        "allclose": close,
        "dtype_device_shape_exact": exact_meta,
        "max_abs": max_abs,
        "atol": base.INTEGRATED_ATOL,
        "rtol": base.INTEGRATED_RTOL,
    }


def memory_safe_finite_output(output: dict[str, object]) -> bool:
    """Frozen `_finite_output` with only logits finiteness evaluated by row."""

    logits = output.get("logits")
    state = output.get("state")
    if not isinstance(logits, torch.Tensor) or not isinstance(state, base.HardwareAERAState):
        return False
    if logits.ndim < 1 or logits.size(0) < 1:
        raise RuntimeError("issue569 requires non-empty batch-major logits")

    logits_finite = True
    for start in range(0, logits.size(0), CHUNK_BATCH_ROWS):
        stop = min(start + CHUNK_BATCH_ROWS, logits.size(0))
        chunk_finite = bool(torch.isfinite(logits[start:stop]).all())
        logits_finite = bool(logits_finite and chunk_finite)
    if not logits_finite:
        return False

    # Preserve the frozen state-finiteness implementation exactly.
    for stage_state in state.stages:
        if not bool(torch.isfinite(stage_state.stream).all()):
            return False
        memory = stage_state.memory
        if not isinstance(memory, base.ContextualEpisodicMemoryState):
            return False
        if not all(
            bool(torch.isfinite(tensor).all())
            for tensor in (memory.keys, memory.values, memory.strengths)
        ):
            return False
    return True


def cpu_contract_preflight_issue569() -> dict[str, Any]:
    """CPU-only proof that #569 changes only two reduction implementations."""

    predecessor = issue562.cpu_contract_preflight_issue562()
    protocol = issue569_systems_protocol()

    if CHUNK_BATCH_ROWS != 1:
        raise RuntimeError("issue569 chunk_batch_rows drifted")
    if (base.INTEGRATED_ATOL, base.INTEGRATED_RTOL) != (1e-2, 1e-2):
        raise RuntimeError("issue569 integrated tolerances drifted")
    if base.SYSTEM_BATCH_SIZES != (8, 64):
        raise RuntimeError("issue569 batch sizes drifted")
    if (
        base.SYSTEM_WARMUP_CALLS,
        base.SYSTEM_TIMED_CALLS_PER_ROUND,
        base.SYSTEM_ROUNDS,
    ) != (3, 20, 5):
        raise RuntimeError("issue569 timing design drifted")
    if (base.BATCH8_MIN_FULL_SPEED_RATIO, base.BATCH64_MIN_FULL_SPEED_RATIO) != (
        0.25,
        1.25,
    ):
        raise RuntimeError("issue569 throughput thresholds drifted")
    if base.EXPECTED_STATE_BYTES != 77_760:
        raise RuntimeError("issue569 persistent-state bytes drifted")
    if (
        base.EXPECTED_SELECTED_WRITES,
        base.EXPECTED_CANDIDATES,
        base.EXPECTED_VECTOR_UPDATES,
    ) != (16, 255, 1):
        raise RuntimeError("issue569 write geometry drifted")

    if base._logit_equivalence is not _FROZEN_LOGIT_EQUIVALENCE:
        raise RuntimeError("issue569 frozen logit helper was mutated")
    if base._finite_output is not _FROZEN_FINITE_OUTPUT:
        raise RuntimeError("issue569 frozen finite-output helper was mutated")
    if issue562.run_end_to_end_systems_v26_8 is not _FROZEN_ISSUE562_RUN:
        raise RuntimeError("issue569 frozen #562 run function was mutated")

    if ISSUE566_AUTHORITATIVE_RESULT_EMITTED:
        raise RuntimeError("issue569 must preserve #566 as pre-RESULT consumed attempt")
    if not ISSUE566_L4_STARTED:
        raise RuntimeError("issue569 must preserve #566 L4-start evidence")
    if ISSUE567_CANONICAL or ISSUE567_GPU_STARTED:
        raise RuntimeError("issue569 must preserve #567 as inert pre-GPU duplicate")

    higher_false = (
        "gpu_authorized_by_issue569",
        "end_to_end_systems_executed_by_issue569",
        "architecture_freeze_authorized",
        "s2_authorized",
        "fresh_scientific_seed_authorized",
        "independent_replication_credit",
        "100m_authorized",
        "breakthrough_proven",
    )
    if any(protocol[key] for key in higher_false):
        raise RuntimeError("issue569 higher authorization drifted")

    return {
        "research_issue": RESEARCH_ISSUE,
        "source_main": SOURCE_MAIN,
        "source_tree": SOURCE_TREE,
        "protocol": protocol,
        "predecessor": predecessor,
        "gpu_authorized_by_cpu_preflight": False,
        "model_construction_performed": False,
        "checkpoint_loaded": False,
        "systems_measurement_performed": False,
        "scientific_seed_consumed": False,
        "architecture_freeze_authorized": False,
        "100m_authorized": False,
        "breakthrough_proven": False,
    }


def run_end_to_end_systems_v26_8_memory_safe(
    *, run_dir: str = base.CHECKPOINT_RELATIVE_DIR
) -> dict[str, Any]:
    """Run exact #562 systems execution with only two frozen helper substitutions.

    Issue #569 itself does not authorize invoking this on GPU.  A later separately
    preregistered one-shot gate must explicitly authorize the call.
    """

    if base._logit_equivalence is not _FROZEN_LOGIT_EQUIVALENCE:
        raise RuntimeError("issue569 refuses mutated frozen logit helper")
    if base._finite_output is not _FROZEN_FINITE_OUTPUT:
        raise RuntimeError("issue569 refuses mutated frozen finite-output helper")
    if issue562.run_end_to_end_systems_v26_8 is not _FROZEN_ISSUE562_RUN:
        raise RuntimeError("issue569 refuses mutated frozen #562 run function")

    base._logit_equivalence = memory_safe_logit_equivalence
    base._finite_output = memory_safe_finite_output
    try:
        result = _FROZEN_ISSUE562_RUN(run_dir=run_dir)
    finally:
        base._logit_equivalence = _FROZEN_LOGIT_EQUIVALENCE
        base._finite_output = _FROZEN_FINITE_OUTPUT

    frozen_scope = result.get("scope")
    result["scope"] = "aera_v26_8_issue569_memory_safe_frozen_issue562_systems_adapter"
    result["issue569_adapter_metadata"] = {
        "frozen_issue562_scope": frozen_scope,
        "source_main": SOURCE_MAIN,
        "source_tree": SOURCE_TREE,
        "chunk_batch_rows": CHUNK_BATCH_ROWS,
        "only_evaluator_implementation_changes": [
            "logit_equivalence_chunked_fp32_reduction",
            "finite_output_chunked_logits_finite_reduction",
        ],
        "issue566_consumed_pre_result": True,
        "issue567_inert_duplicate": True,
        "scientific_seed_consumed": False,
    }
    return result
