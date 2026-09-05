from __future__ import annotations

"""Issue #643 CPU-first v26.9 bounded-memory end-to-end systems adapter.

This module preserves the frozen #501/#503/#530/#562/#569 systems decision
surface.  It changes only the candidate execution backend from frozen v26.8 to
the exact v26.9 identity-weight-visibility backend and installs the already
merged #641 physical-sparse backend-identity evaluator only for the duration of
the single frozen #569 systems call.

Issue #643 itself authorizes no GPU or end-to-end execution.  A later separately
authorized one-shot L4 trigger is required after this adapter/harness is
CPU-green, merged, and live-audited.
"""

from typing import Any

import torch

from . import aera_v26_5_end_to_end_systems as base
from . import aera_v26_6_issue530_end_to_end_systems as frozen530
from . import aera_v26_8_issue562_end_to_end_systems as issue562
from . import aera_v26_8_issue569_end_to_end_systems_memory_safe as memory_safe
from . import aera_v26_9_issue641_physical_sparse_backend_identity_compat as issue641
from .aera_hardware_core_v26 import CoalescedFICEMMemory, TorchFICEMReferenceBackend
from .aera_hardware_core_v26_8_ficem_read_mixed_strength_precision import (
    StrengthPrecisionTritonFICEMReadWriteBackend,
)
from .aera_hardware_core_v26_9_ficem_read_identity_weight_visibility import (
    IdentityWeightVisibilityTritonFICEMReadWriteBackend,
    cpu_contract_preflight_issue600,
    identity_weight_visibility_v26_9_protocol,
)

RESEARCH_ISSUE = 643
SOURCE_MAIN = "ef93e787e6d01585307f05f046d7fd3806374511"
SOURCE_TREE = "a44bcdb61b3124494e58902cad3d233cf7926cff"

BASE_SYSTEMS_BLOB = "c9731cae7e386f09b2a190b045532591c4fa00be"
REPAIR1_SYSTEMS_BLOB = "b3f7082b188644007b873db3733492f424d4941a"
ISSUE530_SYSTEMS_BLOB = "9d5a3c31f4a3862f96b957540baa2e0ec6a84c6b"
ISSUE562_SYSTEMS_BLOB = "3534103eea21f7c4d9d31798ad34601fd47090d6"
ISSUE569_SYSTEMS_BLOB = "1eeaa80adb2ae960e2a8dad06031c4ed5ca99203"
V26_9_BACKEND_BLOB = "b81cc209f5d95abbe1fb8bd620c78e87c067bc19"
ISSUE625_REPAIR_BLOB = "92d06a4954bca1b302355e81f5bf09b06fcee222"
ISSUE641_ADAPTER_BLOB = "5ea1919d15904add0f9e0fb714757f32b11442cb"
ISSUE641_CPU_TEST_BLOB = "e620a9874958bda78d586269f597095f5cf70670"

ISSUE571_TRIGGER = 573
ISSUE571_RUN = 33741700781
ISSUE571_JOB = 100604889696
ISSUE571_RESULT_PATH = "/vol/aera-v26/issue571-memory-safe-end-to-end-systems/result.json"
ISSUE571_RESULT_SHA256 = "afeeb62351cc4fb97d272c5b55c9621839e26f83753ae1fb237733d58a5ee472"
ISSUE571_DECISION = "FAIL_FROZEN_E2E_SYSTEMS_GATE"

ISSUE630_TRIGGER = 640
ISSUE630_RUN = 33985543569
ISSUE630_JOB = 101358253857
ISSUE630_RESULT_PATH = "/vol/aera-v26/issue630-runner-allocation-continuation/result.json"
ISSUE630_RESULT_SHA256 = "ef92c85b55484b3ce191cd4016be86bf52da997a153f737194976164b29554b4"
ISSUE630_DECISION = "PASS"
ISSUE630_DESIGN_SEED = 891475817

ISSUE642_HEAD = "504b5b3fdd254645eeb31bcb059831a8a6ee3164"
ISSUE642_CPU_RUN = 33989770634
ISSUE642_CPU_JOB = 101369805445
ISSUE642_MERGE = SOURCE_MAIN

CHECKPOINT_HASHES = {
    "aera": "f8aa92421801e8f190247e420632be5f0c20bc5ea8bf6bdeefe06686b3a31b30",
    "transformer": "cdd5cab4439a709468d6607d45d82081b33e876b2e40d91d4a38ba139b219dd7",
}

_FROZEN_ISSUE562_LOADER = issue562.load_models_v26_8
_FROZEN_ISSUE562_PROTOCOL = issue562.issue562_systems_protocol
_FROZEN_ISSUE562_RUN = issue562.run_end_to_end_systems_v26_8
_FROZEN_ISSUE569_RUN = memory_safe.run_end_to_end_systems_v26_8_memory_safe
_FROZEN_ISSUE641_WRAPPER = issue641.with_v26_9_physical_sparse_evaluator


def issue643_candidate_protocol() -> dict[str, Any]:
    """Frozen #562 protocol with only exact v26.9 candidate identity layered on."""

    protocol = dict(_FROZEN_ISSUE562_PROTOCOL())
    candidate = identity_weight_visibility_v26_9_protocol()
    protocol.update(
        {
            "version": "aera-v26.9-issue643-bounded-memory-end-to-end-systems",
            "research_issue": RESEARCH_ISSUE,
            "source_main_issue643": SOURCE_MAIN,
            "source_tree_issue643": SOURCE_TREE,
            "base_systems_blob": BASE_SYSTEMS_BLOB,
            "repair1_systems_blob": REPAIR1_SYSTEMS_BLOB,
            "issue530_systems_blob": ISSUE530_SYSTEMS_BLOB,
            "issue562_systems_blob": ISSUE562_SYSTEMS_BLOB,
            "issue569_systems_blob": ISSUE569_SYSTEMS_BLOB,
            "v26_9_candidate_blob": V26_9_BACKEND_BLOB,
            "issue625_repair_blob": ISSUE625_REPAIR_BLOB,
            "issue641_adapter_blob": ISSUE641_ADAPTER_BLOB,
            "candidate_backend": IdentityWeightVisibilityTritonFICEMReadWriteBackend.name,
            "reference_backend": TorchFICEMReferenceBackend.name,
            "only_candidate_semantic_change_issue643": (
                "execution_backend_v26_8_to_v26_9_identity_weight_visibility"
            ),
            "frozen_issue530_run_function_reused": True,
            "frozen_issue530_loader_replaced_before_parameter_snapshot": True,
            "frozen_issue530_loader_replaced_before_any_model_call": True,
            "frozen_issue569_memory_safe_reductions_reused": True,
            "issue641_physical_sparse_identity_adapter_scoped": True,
            "candidate_v26_9_identity_weight_visibility": True,
            "candidate_v26_6_write_inherited": True,
            "candidate_write_backend_changed_by_v26_9": candidate[
                "write_backend_changed_by_v26_9"
            ],
            "candidate_training_backend_changed_by_v26_9": candidate[
                "training_backend_changed_by_v26_9"
            ],
            "candidate_same_dtype_dispatch": candidate["same_dtype_dispatch"],
            "candidate_mixed_new_triton_kernels": candidate[
                "mixed_new_triton_kernels"
            ],
            "candidate_mixed_tail_triton_launches_target": candidate[
                "mixed_tail_triton_launches_target"
            ],
            "issue571_trigger": ISSUE571_TRIGGER,
            "issue571_run": ISSUE571_RUN,
            "issue571_job": ISSUE571_JOB,
            "issue571_result_path": ISSUE571_RESULT_PATH,
            "issue571_result_sha256": ISSUE571_RESULT_SHA256,
            "issue571_decision": ISSUE571_DECISION,
            "issue630_trigger": ISSUE630_TRIGGER,
            "issue630_run": ISSUE630_RUN,
            "issue630_job": ISSUE630_JOB,
            "issue630_result_path": ISSUE630_RESULT_PATH,
            "issue630_result_sha256": ISSUE630_RESULT_SHA256,
            "issue630_decision": ISSUE630_DECISION,
            "issue630_design_seed_consumed": ISSUE630_DESIGN_SEED,
            "issue642_head": ISSUE642_HEAD,
            "issue642_cpu_run": ISSUE642_CPU_RUN,
            "issue642_cpu_job": ISSUE642_CPU_JOB,
            "issue642_merge": ISSUE642_MERGE,
            "systems_gpu_authorized_by_issue643": False,
            "end_to_end_systems_executed_by_issue643": False,
            "architecture_freeze_authorized": False,
            "s2_authorized": False,
            "fresh_scientific_seed_authorized": False,
            "independent_replication_credit": False,
            "100m_authorized": False,
            "breakthrough_proven": False,
        }
    )
    return protocol


def _install_v26_9_candidate_backend(candidate: torch.nn.Module) -> tuple[str, ...]:
    """Install exact v26.9 in every candidate memory stage before model use."""

    names: list[str] = []
    stages = getattr(candidate, "stages", None)
    if stages is None:
        raise RuntimeError("issue643 candidate lacks stages")
    for stage in stages:
        memory = stage.memory
        if not isinstance(memory, CoalescedFICEMMemory):
            raise RuntimeError("issue643 candidate stage lacks CoalescedFICEMMemory")
        memory._execution_backend = IdentityWeightVisibilityTritonFICEMReadWriteBackend()
        if (
            memory.execution_backend_name
            != IdentityWeightVisibilityTritonFICEMReadWriteBackend.name
        ):
            raise RuntimeError("issue643 v26.9 candidate backend installation failed")
        names.append(memory.execution_backend_name)
    if not names:
        raise RuntimeError("issue643 candidate has no memory stages")
    return tuple(names)


def load_models_v26_9(*, run_dir: str, device: torch.device):
    """Use exact frozen #501 loading, then substitute v26.9 before any model use."""

    reference, candidate, transformer = frozen530.base.load_models(
        run_dir=run_dir,
        device=device,
    )
    candidate_backend_names = _install_v26_9_candidate_backend(candidate)

    for stage in reference.stages:
        memory = stage.memory
        if not isinstance(memory, CoalescedFICEMMemory):
            raise RuntimeError("issue643 reference stage lacks CoalescedFICEMMemory")
        if memory.execution_backend_name != TorchFICEMReferenceBackend.name:
            raise RuntimeError("issue643 reference backend drifted from exact Torch reference")
    if any(
        name != IdentityWeightVisibilityTritonFICEMReadWriteBackend.name
        for name in candidate_backend_names
    ):
        raise RuntimeError("issue643 candidate backend identity mismatch")
    return reference, candidate, transformer, candidate_backend_names


def cpu_contract_preflight_issue643() -> dict[str, Any]:
    """CPU-only proof that #643 preserves the frozen systems surface."""

    predecessor = memory_safe.cpu_contract_preflight_issue569()
    candidate_preflight = cpu_contract_preflight_issue600()
    issue641_protocol = issue641.issue641_protocol()
    protocol = issue643_candidate_protocol()

    if issue562.load_models_v26_8 is not _FROZEN_ISSUE562_LOADER:
        raise RuntimeError("issue643 frozen #562 loader was mutated")
    if issue562.issue562_systems_protocol is not _FROZEN_ISSUE562_PROTOCOL:
        raise RuntimeError("issue643 frozen #562 protocol was mutated")
    if issue562.run_end_to_end_systems_v26_8 is not _FROZEN_ISSUE562_RUN:
        raise RuntimeError("issue643 frozen #562 run function was mutated")
    if memory_safe.run_end_to_end_systems_v26_8_memory_safe is not _FROZEN_ISSUE569_RUN:
        raise RuntimeError("issue643 frozen #569 run function was mutated")
    if issue641.with_v26_9_physical_sparse_evaluator is not _FROZEN_ISSUE641_WRAPPER:
        raise RuntimeError("issue643 frozen #641 evaluator wrapper was mutated")

    if not issubclass(
        IdentityWeightVisibilityTritonFICEMReadWriteBackend,
        StrengthPrecisionTritonFICEMReadWriteBackend,
    ):
        raise RuntimeError("issue643 v26.9 no longer inherits frozen v26.8/v26.6 behavior")
    if (
        IdentityWeightVisibilityTritonFICEMReadWriteBackend.name
        != issue641.V26_9_BACKEND_NAME
    ):
        raise RuntimeError("issue643 v26.9 backend name drifted from #641 exact identity")
    if issue641_protocol["issue630_result_sha256"] != ISSUE630_RESULT_SHA256:
        raise RuntimeError("issue643 #640 primitive authority drifted")
    if issue641_protocol["issue571_result_sha256"] != ISSUE571_RESULT_SHA256:
        raise RuntimeError("issue643 #571 historical authority drifted")

    if base.SYSTEM_BATCH_SIZES != (8, 64):
        raise RuntimeError("issue643 batch sizes drifted")
    if (
        base.SYSTEM_WARMUP_CALLS,
        base.SYSTEM_TIMED_CALLS_PER_ROUND,
        base.SYSTEM_ROUNDS,
    ) != (3, 20, 5):
        raise RuntimeError("issue643 timing design drifted")
    if (base.BATCH8_MIN_FULL_SPEED_RATIO, base.BATCH64_MIN_FULL_SPEED_RATIO) != (
        0.25,
        1.25,
    ):
        raise RuntimeError("issue643 throughput thresholds drifted")
    if (base.INTEGRATED_ATOL, base.INTEGRATED_RTOL) != (1e-2, 1e-2):
        raise RuntimeError("issue643 integrated tolerances drifted")
    if base.EXPECTED_STATE_BYTES != 77_760:
        raise RuntimeError("issue643 persistent-state bytes drifted")
    if (
        base.EXPECTED_SELECTED_WRITES,
        base.EXPECTED_CANDIDATES,
        base.EXPECTED_VECTOR_UPDATES,
    ) != (16, 255, 1):
        raise RuntimeError("issue643 write geometry drifted")
    if base.CHECKPOINT_RELATIVE_DIR != "/vol/aera-real-language/v25-dev-seed8471":
        raise RuntimeError("issue643 checkpoint path drifted")

    required_candidate = {
        "same_dtype_dispatch": "historical-repair5",
        "mixed_new_triton_kernels": 1,
        "mixed_tail_triton_launches_target": 1,
        "softmax_weight_visibility_controlled_by_identity_dtype": True,
        "identity_weight_visibility_control_separate": True,
        "write_backend_changed_by_v26_9": False,
        "training_backend_changed_by_v26_9": False,
        "gpu_authorized_by_issue600": False,
        "end_to_end_systems_authorized": False,
        "architecture_freeze_authorized": False,
        "fresh_scientific_seed_authorized": False,
        "100m_authorized": False,
        "breakthrough_proven": False,
    }
    candidate_protocol = candidate_preflight["protocol"]
    for key, expected in required_candidate.items():
        if candidate_protocol.get(key) != expected:
            raise RuntimeError(
                f"issue643 candidate protocol drift: {key}="
                f"{candidate_protocol.get(key)!r} expected={expected!r}"
            )

    inherited_protocol = memory_safe.issue569_systems_protocol()
    inherited_required = {
        "batch_sizes": [8, 64],
        "random_token_seed_rule": "138471 + 10000 + batch_size",
        "hard": True,
        "route_mode": "hard_sparse",
        "physically_real_sparse_required": True,
        "dense_masked_sparse_credit": False,
        "persistent_state_bytes_per_session": 77_760,
        "production_write_geometry": [16, 255, 1],
        "chunk_batch_rows": 1,
        "all_elements_covered": True,
        "sampling_or_approximation": False,
        "result_dependent_chunk_sizing": False,
    }
    for key, expected in inherited_required.items():
        if inherited_protocol.get(key) != expected:
            raise RuntimeError(
                f"issue643 frozen #569 protocol drift: {key}="
                f"{inherited_protocol.get(key)!r} expected={expected!r}"
            )

    higher_false = (
        "systems_gpu_authorized_by_issue643",
        "end_to_end_systems_executed_by_issue643",
        "architecture_freeze_authorized",
        "s2_authorized",
        "fresh_scientific_seed_authorized",
        "independent_replication_credit",
        "100m_authorized",
        "breakthrough_proven",
    )
    if any(protocol[key] for key in higher_false):
        raise RuntimeError("issue643 higher authorization drifted")

    return {
        "research_issue": RESEARCH_ISSUE,
        "source_main": SOURCE_MAIN,
        "source_tree": SOURCE_TREE,
        "protocol": protocol,
        "predecessor": predecessor,
        "candidate_preflight": candidate_preflight,
        "issue641_protocol": issue641_protocol,
        "checkpoint_path": base.CHECKPOINT_RELATIVE_DIR,
        "checkpoint_hashes_expected": dict(CHECKPOINT_HASHES),
        "gpu_authorized_by_cpu_preflight": False,
        "model_construction_performed": False,
        "checkpoint_loaded": False,
        "systems_measurement_performed": False,
        "scientific_seed_consumed": False,
        "architecture_freeze_authorized": False,
        "s2_authorized": False,
        "fresh_scientific_seed_authorized": False,
        "independent_replication_credit": False,
        "100m_authorized": False,
        "breakthrough_proven": False,
    }


def run_end_to_end_systems_v26_9_bounded_memory(
    *, run_dir: str = base.CHECKPOINT_RELATIVE_DIR
) -> dict[str, Any]:
    """Run frozen #569 once with only v26.9 candidate + #641 identity adapter.

    #643 itself does not authorize invoking this on GPU.  A later exact-main
    one-shot trigger must separately authorize the call.
    """

    if issue562.load_models_v26_8 is not _FROZEN_ISSUE562_LOADER:
        raise RuntimeError("issue643 refuses mutated frozen #562 loader")
    if issue562.issue562_systems_protocol is not _FROZEN_ISSUE562_PROTOCOL:
        raise RuntimeError("issue643 refuses mutated frozen #562 protocol")
    if issue562.run_end_to_end_systems_v26_8 is not _FROZEN_ISSUE562_RUN:
        raise RuntimeError("issue643 refuses mutated frozen #562 run function")
    if memory_safe.run_end_to_end_systems_v26_8_memory_safe is not _FROZEN_ISSUE569_RUN:
        raise RuntimeError("issue643 refuses mutated frozen #569 run function")
    if issue641.with_v26_9_physical_sparse_evaluator is not _FROZEN_ISSUE641_WRAPPER:
        raise RuntimeError("issue643 refuses mutated frozen #641 evaluator wrapper")

    issue562.load_models_v26_8 = load_models_v26_9
    issue562.issue562_systems_protocol = issue643_candidate_protocol
    try:
        result = _FROZEN_ISSUE641_WRAPPER(
            lambda: _FROZEN_ISSUE569_RUN(run_dir=run_dir)
        )
    finally:
        issue562.load_models_v26_8 = _FROZEN_ISSUE562_LOADER
        issue562.issue562_systems_protocol = _FROZEN_ISSUE562_PROTOCOL

    names = result.get("candidate_backend_names")
    if not isinstance(names, (list, tuple)) or not names:
        raise RuntimeError("issue643 systems result lacks candidate backend names")
    if any(
        name != IdentityWeightVisibilityTritonFICEMReadWriteBackend.name
        for name in names
    ):
        raise RuntimeError("issue643 systems result candidate backend identity drifted")

    frozen_scope = result.get("scope")
    result["scope"] = "aera_v26_9_issue643_bounded_memory_frozen_issue569_systems_adapter"
    result["issue643_adapter_metadata"] = {
        "frozen_issue569_scope": frozen_scope,
        "source_main": SOURCE_MAIN,
        "source_tree": SOURCE_TREE,
        "base_systems_blob": BASE_SYSTEMS_BLOB,
        "repair1_systems_blob": REPAIR1_SYSTEMS_BLOB,
        "issue530_systems_blob": ISSUE530_SYSTEMS_BLOB,
        "issue562_systems_blob": ISSUE562_SYSTEMS_BLOB,
        "issue569_systems_blob": ISSUE569_SYSTEMS_BLOB,
        "v26_9_candidate_blob": V26_9_BACKEND_BLOB,
        "issue641_adapter_blob": ISSUE641_ADAPTER_BLOB,
        "candidate_backend": IdentityWeightVisibilityTritonFICEMReadWriteBackend.name,
        "reference_backend": TorchFICEMReferenceBackend.name,
        "only_candidate_change": (
            "v26_8_to_v26_9_identity_weight_visibility_execution_backend"
        ),
        "physical_sparse_evaluator_change": (
            "merged_issue641_exact_v26_9_backend_identity_only"
        ),
        "chunk_batch_rows": 1,
        "issue571_historical_result_sha256": ISSUE571_RESULT_SHA256,
        "issue571_historical_decision": ISSUE571_DECISION,
        "issue630_primitive_result_sha256": ISSUE630_RESULT_SHA256,
        "issue630_primitive_decision": ISSUE630_DECISION,
        "issue630_design_seed_consumed": ISSUE630_DESIGN_SEED,
        "scientific_seed_consumed": False,
        "architecture_freeze_authorized": False,
        "s2_authorized": False,
        "fresh_scientific_seed_authorized": False,
        "independent_replication_credit": False,
        "100m_authorized": False,
        "breakthrough_proven": False,
    }
    return result
