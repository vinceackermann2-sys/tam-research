from __future__ import annotations

"""Issue #562 CPU-first AERA-v26.8 end-to-end systems adapter.

The systems decision implementation remains byte-frozen in issue #530.  This
module changes only the candidate execution backend installed immediately after
the frozen #501 loader returns and before parameter snapshots or model calls.
The actual GPU systems run is *not* authorized by #562; a later separately
preregistered one-shot gate is required after this adapter is CPU-green/merged.
"""

from typing import Any

import torch

from . import aera_v26_6_issue530_end_to_end_systems as frozen530
from .aera_hardware_core_v26 import CoalescedFICEMMemory, TorchFICEMReferenceBackend
from .aera_hardware_core_v26_6_ficem_write_materialize_cast import (
    MaterializeCastTritonFICEMReadWriteBackend,
)
from .aera_hardware_core_v26_8_ficem_read_mixed_strength_precision import (
    StrengthPrecisionTritonFICEMReadWriteBackend,
    mixed_strength_precision_v26_8_protocol,
)

RESEARCH_ISSUE = 562
SOURCE_MAIN = "75987bfb7976c6a970d63801c6e81b5b4993f544"
SOURCE_TREE = "8f493dbdfe53392d47bbd1addfe2e61aa8dd132d"

BASE_SYSTEMS_BLOB = "c9731cae7e386f09b2a190b045532591c4fa00be"
REPAIR1_SYSTEMS_BLOB = "b3f7082b188644007b873db3733492f424d4941a"
ISSUE530_SYSTEMS_BLOB = "9d5a3c31f4a3862f96b957540baa2e0ec6a84c6b"
V26_8_CANDIDATE_BLOB = "3575c58d1cd730be77649f087908c51dbf3e6088"
V26_7_PREDECESSOR_BLOB = "d8133c6b204b1ee5f23955255fb2fb09d09bd723"
REPAIR5_READ_BLOB = "263f68eb1186a8ac14a08fc4b4df1fc5b292c711"
V26_6_WRITE_BLOB = "d45c262314a0b4691f26812a279937a225043ad9"
HISTORICAL_V26_4_WRITE_BLOB = "e54570292489bd17570038dca7518419ac00418c"
V26_INTERFACE_BLOB = "268644ac4edee15a4cc4e29d3fed7f61eeb3caa7"
STABLE_REFERENCE_BLOB = "4e336b6e1a6238dac782fa320751d68281493ee1"

ISSUE558_TRIGGER = 561
ISSUE558_RUN = 33733085825
ISSUE558_JOB = 100577290103
ISSUE558_ATTEMPT = 1
ISSUE558_BOUND_MAIN = SOURCE_MAIN
ISSUE558_RESULT_PATH = (
    "/vol/aera-v26/issue558-ficem-read-mixed-strength-precision/result.json"
)
ISSUE558_RESULT_SHA256 = (
    "e1fdc7e6b69a33084ca4b419b5489e755d7a98b12c367775ef19d1127700aa7e"
)
ISSUE558_DECISION = "PASS"

ISSUE553_TRIGGER = 555
ISSUE553_RUN = 33727540468
ISSUE553_JOB = 100559866985
ISSUE553_RESULT_SHA256 = (
    "009af31baf70e46eb93b6e7489d62f356a02b727521d3fabe4a7dab2dcf5ab47"
)
ISSUE553_DECISION = "FAIL"

ISSUE545_TRIGGER = 550
ISSUE545_RUN = 33686037672
ISSUE545_JOB = 100433658768
ISSUE545_ATTEMPT = 1
ISSUE545_FAILURE = "FICEM read-tail floating dtypes must match"
ISSUE545_AUTHORITATIVE_RESULT_EMITTED = False

ISSUE529_TRIGGER = 529
ISSUE529_RUN = 33680028132
ISSUE529_JOB = 100414089065
ISSUE529_RESULT_SHA256 = (
    "a07790c2d55d7a696baf0e903a05dbdf925e3ef2a08b84c784c77d1fbdd31874"
)

# Capture the exact #530 function objects before the controlled runtime swap.
_FROZEN_ISSUE530_PROTOCOL = frozen530.issue530_systems_protocol
_FROZEN_ISSUE530_LOADER = frozen530.load_models_v26_6
_FROZEN_ISSUE530_RUN = frozen530.run_end_to_end_systems_v26_6


def issue562_systems_protocol() -> dict[str, Any]:
    """Return frozen #530 protocol with only v26.8 candidate identity layered on."""

    protocol = dict(_FROZEN_ISSUE530_PROTOCOL())
    candidate_protocol = mixed_strength_precision_v26_8_protocol()
    protocol.update(
        {
            "version": "aera-v26.8-issue562-end-to-end-systems-adapter",
            "research_issue": RESEARCH_ISSUE,
            "source_main_issue562": SOURCE_MAIN,
            "source_tree_issue562": SOURCE_TREE,
            "base_systems_blob": BASE_SYSTEMS_BLOB,
            "repair1_systems_blob": REPAIR1_SYSTEMS_BLOB,
            "issue530_systems_blob": ISSUE530_SYSTEMS_BLOB,
            "v26_8_candidate_blob": V26_8_CANDIDATE_BLOB,
            "v26_7_predecessor_blob": V26_7_PREDECESSOR_BLOB,
            "repair5_read_blob": REPAIR5_READ_BLOB,
            "v26_6_write_blob": V26_6_WRITE_BLOB,
            "historical_v26_4_write_blob": HISTORICAL_V26_4_WRITE_BLOB,
            "v26_interface_blob": V26_INTERFACE_BLOB,
            "stable_reference_blob": STABLE_REFERENCE_BLOB,
            "candidate_backend": StrengthPrecisionTritonFICEMReadWriteBackend.name,
            "reference_backend": TorchFICEMReferenceBackend.name,
            "only_candidate_semantic_change_issue562": (
                "execution_backend_v26_6_to_v26_8_mixed_strength_precision"
            ),
            "frozen_issue530_run_function_reused": True,
            "frozen_issue530_loader_replaced_before_parameter_snapshot": True,
            "frozen_issue530_loader_replaced_before_any_model_call": True,
            "frozen_issue530_candidate_timing_label_retained": True,
            "candidate_v26_8_read_mixed_strength_precision": True,
            "candidate_v26_6_write_inherited": True,
            "candidate_write_backend_changed_by_v26_8": candidate_protocol[
                "write_backend_changed_by_v26_8"
            ],
            "candidate_training_backend_changed_by_v26_8": candidate_protocol[
                "training_backend_changed_by_v26_8"
            ],
            "candidate_same_dtype_dispatch": candidate_protocol[
                "same_dtype_dispatch"
            ],
            "candidate_mixed_new_triton_kernels": candidate_protocol[
                "mixed_new_triton_kernels"
            ],
            "candidate_mixed_tail_triton_launches_target": candidate_protocol[
                "mixed_tail_triton_launches_target"
            ],
            "candidate_mixed_host_pre_tail_cast_kernels": candidate_protocol[
                "mixed_host_pre_tail_cast_kernels"
            ],
            "issue558_trigger": ISSUE558_TRIGGER,
            "issue558_run": ISSUE558_RUN,
            "issue558_job": ISSUE558_JOB,
            "issue558_attempt": ISSUE558_ATTEMPT,
            "issue558_bound_main": ISSUE558_BOUND_MAIN,
            "issue558_result_path": ISSUE558_RESULT_PATH,
            "issue558_result_sha256": ISSUE558_RESULT_SHA256,
            "issue558_decision": ISSUE558_DECISION,
            "issue558_overall_pass": True,
            "issue558_historical_surface_pass": True,
            "issue558_mixed_rows_pass": [8, 8],
            "issue558_mixed_near_tie_pass": True,
            "issue558_mixed_known_empty_pass": True,
            "issue553_trigger": ISSUE553_TRIGGER,
            "issue553_run": ISSUE553_RUN,
            "issue553_job": ISSUE553_JOB,
            "issue553_result_sha256": ISSUE553_RESULT_SHA256,
            "issue553_decision": ISSUE553_DECISION,
            "issue545_trigger": ISSUE545_TRIGGER,
            "issue545_run": ISSUE545_RUN,
            "issue545_job": ISSUE545_JOB,
            "issue545_attempt": ISSUE545_ATTEMPT,
            "issue545_authoritative_result_emitted": False,
            "issue545_failure": ISSUE545_FAILURE,
            "issue529_trigger": ISSUE529_TRIGGER,
            "issue529_run": ISSUE529_RUN,
            "issue529_job": ISSUE529_JOB,
            "issue529_result_sha256": ISSUE529_RESULT_SHA256,
            "systems_gpu_authorized_by_issue562": False,
            "end_to_end_systems_executed_by_issue562": False,
            "architecture_freeze_authorized": False,
            "s2_authorized": False,
            "fresh_scientific_seed_authorized": False,
            "independent_replication_credit": False,
            "100m_authorized": False,
            "breakthrough_proven": False,
        }
    )
    return protocol


def cpu_contract_preflight_issue562() -> dict[str, Any]:
    """CPU-only proof that #562 preserves the exact frozen systems surface."""

    predecessor = frozen530.cpu_contract_preflight_issue530()
    protocol = issue562_systems_protocol()
    candidate_protocol = mixed_strength_precision_v26_8_protocol()
    base = frozen530.base

    if base.SYSTEM_BATCH_SIZES != (8, 64):
        raise RuntimeError("issue562 batch sizes drifted")
    if (
        base.SYSTEM_WARMUP_CALLS,
        base.SYSTEM_TIMED_CALLS_PER_ROUND,
        base.SYSTEM_ROUNDS,
    ) != (3, 20, 5):
        raise RuntimeError("issue562 timing design drifted")
    if (base.BATCH8_MIN_FULL_SPEED_RATIO, base.BATCH64_MIN_FULL_SPEED_RATIO) != (
        0.25,
        1.25,
    ):
        raise RuntimeError("issue562 throughput thresholds drifted")
    if (base.INTEGRATED_ATOL, base.INTEGRATED_RTOL) != (1e-2, 1e-2):
        raise RuntimeError("issue562 integrated tolerances drifted")
    if base.EXPECTED_STATE_BYTES != 77_760:
        raise RuntimeError("issue562 persistent-state bytes drifted")
    if (
        base.EXPECTED_SELECTED_WRITES,
        base.EXPECTED_CANDIDATES,
        base.EXPECTED_VECTOR_UPDATES,
    ) != (16, 255, 1):
        raise RuntimeError("issue562 write geometry drifted")
    if frozen530.run_end_to_end_systems_v26_6 is not _FROZEN_ISSUE530_RUN:
        raise RuntimeError("issue562 frozen #530 run function was mutated")
    if frozen530.load_models_v26_6 is not _FROZEN_ISSUE530_LOADER:
        raise RuntimeError("issue562 frozen #530 loader was mutated")
    if frozen530.issue530_systems_protocol is not _FROZEN_ISSUE530_PROTOCOL:
        raise RuntimeError("issue562 frozen #530 protocol function was mutated")
    if not issubclass(
        StrengthPrecisionTritonFICEMReadWriteBackend,
        MaterializeCastTritonFICEMReadWriteBackend,
    ):
        raise RuntimeError("issue562 v26.8 candidate no longer inherits v26.6 WRITE")

    required_candidate = {
        "same_dtype_dispatch": "historical-repair5",
        "same_dtype_arithmetic_changed_by_v26_8": False,
        "same_dtype_kernel_changed_by_v26_8": False,
        "mixed_new_triton_kernels": 1,
        "mixed_tail_triton_launches_target": 1,
        "mixed_strengths_values_dtype_equality_required": True,
        "mixed_host_pre_tail_cast_kernels": 0,
        "write_backend_changed_by_v26_8": False,
        "training_backend_changed_by_v26_8": False,
        "end_to_end_systems_authorized": False,
        "architecture_freeze_authorized": False,
        "fresh_scientific_seed_authorized": False,
        "100m_authorized": False,
        "breakthrough_proven": False,
    }
    for key, expected in required_candidate.items():
        if candidate_protocol.get(key) != expected:
            raise RuntimeError(
                f"issue562 candidate protocol drift: {key}="
                f"{candidate_protocol.get(key)!r} expected={expected!r}"
            )

    higher_false = (
        "systems_gpu_authorized_by_issue562",
        "end_to_end_systems_executed_by_issue562",
        "architecture_freeze_authorized",
        "s2_authorized",
        "fresh_scientific_seed_authorized",
        "independent_replication_credit",
        "100m_authorized",
        "breakthrough_proven",
    )
    if any(protocol[key] for key in higher_false):
        raise RuntimeError("issue562 higher authorization drifted")

    return {
        "research_issue": RESEARCH_ISSUE,
        "source_main": SOURCE_MAIN,
        "source_tree": SOURCE_TREE,
        "protocol": protocol,
        "predecessor": predecessor,
        "candidate_protocol": candidate_protocol,
        "gpu_authorized_by_cpu_preflight": False,
        "model_construction_performed": False,
        "checkpoint_loaded": False,
        "systems_measurement_performed": False,
        "scientific_seed_consumed": False,
        "architecture_freeze_authorized": False,
        "100m_authorized": False,
        "breakthrough_proven": False,
    }


def _install_v26_8_candidate_backend(candidate: torch.nn.Module) -> tuple[str, ...]:
    """Install v26.8 in every candidate memory stage before any model use."""

    names: list[str] = []
    stages = getattr(candidate, "stages", None)
    if stages is None:
        raise RuntimeError("issue562 candidate lacks stages")
    for stage in stages:
        memory = stage.memory
        if not isinstance(memory, CoalescedFICEMMemory):
            raise RuntimeError("issue562 candidate stage lacks CoalescedFICEMMemory")
        memory._execution_backend = StrengthPrecisionTritonFICEMReadWriteBackend()
        if memory.execution_backend_name != StrengthPrecisionTritonFICEMReadWriteBackend.name:
            raise RuntimeError("issue562 v26.8 candidate backend installation failed")
        names.append(memory.execution_backend_name)
    if not names:
        raise RuntimeError("issue562 candidate has no memory stages")
    return tuple(names)


def load_models_v26_8(*, run_dir: str, device: torch.device):
    """Use exact frozen #501 loading, then substitute v26.8 before any use."""

    reference, candidate, transformer = frozen530.base.load_models(
        run_dir=run_dir,
        device=device,
    )
    candidate_backend_names = _install_v26_8_candidate_backend(candidate)

    for stage in reference.stages:
        memory = stage.memory
        if not isinstance(memory, CoalescedFICEMMemory):
            raise RuntimeError("issue562 reference stage lacks CoalescedFICEMMemory")
        if memory.execution_backend_name != TorchFICEMReferenceBackend.name:
            raise RuntimeError("issue562 reference backend drifted from exact Torch reference")
    if any(
        name != StrengthPrecisionTritonFICEMReadWriteBackend.name
        for name in candidate_backend_names
    ):
        raise RuntimeError("issue562 candidate backend identity mismatch")
    return reference, candidate, transformer, candidate_backend_names


def run_end_to_end_systems_v26_8(
    *, run_dir: str = frozen530.base.CHECKPOINT_RELATIVE_DIR
) -> dict[str, Any]:
    """Execute byte-frozen #530 equations with only the v26.8 loader/protocol swapped.

    #562 itself does not authorize calling this function on GPU.  A later one-shot
    systems-gate preregistration must explicitly authorize the invocation.
    """

    if frozen530.run_end_to_end_systems_v26_6 is not _FROZEN_ISSUE530_RUN:
        raise RuntimeError("issue562 refuses mutated frozen #530 run function")
    if frozen530.load_models_v26_6 is not _FROZEN_ISSUE530_LOADER:
        raise RuntimeError("issue562 refuses mutated frozen #530 loader")
    if frozen530.issue530_systems_protocol is not _FROZEN_ISSUE530_PROTOCOL:
        raise RuntimeError("issue562 refuses mutated frozen #530 protocol")

    # The #530 run function looks up these two module globals at execution time.
    # Swap only those identities, run the byte-frozen equation body once, and
    # restore them even on failure.  Internal timing/result labels intentionally
    # remain the frozen `v26_6_*` names; protocol/backend_names record the actual
    # v26.8 candidate so reporting renames cannot perturb the decision surface.
    frozen530.load_models_v26_6 = load_models_v26_8
    frozen530.issue530_systems_protocol = issue562_systems_protocol
    try:
        result = _FROZEN_ISSUE530_RUN(run_dir=run_dir)
    finally:
        frozen530.load_models_v26_6 = _FROZEN_ISSUE530_LOADER
        frozen530.issue530_systems_protocol = _FROZEN_ISSUE530_PROTOCOL

    frozen_scope = result.get("scope")
    result["scope"] = "aera_v26_8_issue562_frozen_issue530_end_to_end_systems_adapter"
    result["issue562_adapter_metadata"] = {
        "frozen_issue530_scope": frozen_scope,
        "frozen_issue530_blob": ISSUE530_SYSTEMS_BLOB,
        "source_main": SOURCE_MAIN,
        "source_tree": SOURCE_TREE,
        "v26_8_candidate_blob": V26_8_CANDIDATE_BLOB,
        "candidate_backend": StrengthPrecisionTritonFICEMReadWriteBackend.name,
        "frozen_issue530_candidate_timing_label_retained": True,
        "only_candidate_semantic_change": (
            "execution_backend_v26_6_to_v26_8_mixed_strength_precision"
        ),
        "issue558_result_sha256": ISSUE558_RESULT_SHA256,
        "issue558_decision": ISSUE558_DECISION,
        "scientific_seed_consumed": False,
    }
    return result
