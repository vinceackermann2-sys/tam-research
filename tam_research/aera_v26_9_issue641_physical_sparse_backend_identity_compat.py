from __future__ import annotations

"""CPU-only v26.9 physical-sparse evaluator identity compatibility for issue #641.

This adapter preserves the frozen issue #501 physical-sparse proof and changes
only the version-specific backend-name identity decision for a v26.9 candidate.
It authorizes no end-to-end execution or hardware run.
"""

from typing import Any, Callable, TypeVar

from . import aera_v26_5_end_to_end_systems as base
from .aera_hardware_core_v26_9_ficem_read_identity_weight_visibility import (
    IdentityWeightVisibilityTritonFICEMReadWriteBackend,
)

RESEARCH_ISSUE = 641
SOURCE_MAIN = "a5708223f2bba17c0eb931d63507fee93e98605b"
SOURCE_TREE = "388597aa7a07b919a691156694ea2732abdbe9f1"

ISSUE501_SYSTEMS_BLOB = "c9731cae7e386f09b2a190b045532591c4fa00be"
ISSUE530_SYSTEMS_BLOB = "9d5a3c31f4a3862f96b957540baa2e0ec6a84c6b"
ISSUE562_SYSTEMS_BLOB = "3534103eea21f7c4d9d31798ad34601fd47090d6"
ISSUE569_SYSTEMS_BLOB = "1eeaa80adb2ae960e2a8dad06031c4ed5ca99203"
V26_9_BACKEND_BLOB = "b81cc209f5d95abbe1fb8bd620c78e87c067bc19"
ISSUE625_REPAIR_BLOB = "92d06a4954bca1b302355e81f5bf09b06fcee222"

ISSUE571_RESULT_PATH = "/vol/aera-v26/issue571-memory-safe-end-to-end-systems/result.json"
ISSUE571_RESULT_SHA256 = "afeeb62351cc4fb97d272c5b55c9621839e26f83753ae1fb237733d58a5ee472"
ISSUE571_RUN = 33741700781
ISSUE571_JOB = 100604889696

ISSUE630_TRIGGER = 640
ISSUE630_RUN = 33985543569
ISSUE630_JOB = 101358253857
ISSUE630_RESULT_PATH = "/vol/aera-v26/issue630-runner-allocation-continuation/result.json"
ISSUE630_RESULT_SHA256 = "ef92c85b55484b3ce191cd4016be86bf52da997a153f737194976164b29554b4"

HISTORICAL_BACKEND_NAME = base.TritonFICEMReadWriteBackend.name
V26_9_BACKEND_NAME = IdentityWeightVisibilityTritonFICEMReadWriteBackend.name

_FROZEN_PHYSICAL_SPARSE_PROOF = base._physical_sparse_proof
_FROZEN_RESULT_KEYS = frozenset(
    {
        "pass",
        "optional_executed_fractions",
        "sparse_route_exercised",
        "coalesced_select_merge_positive",
        "backend_activity_positive",
        "backend_names_exact",
        "backend_names",
        "backend_read_calls",
        "backend_update_calls",
        "backend_projected_update_calls",
        "coalesced_float_state_select_calls",
        "coalesced_valid_select_calls",
        "coalesced_float_state_merge_calls",
        "coalesced_valid_merge_calls",
        "dense_masked_sparse_credit",
    }
)

_T = TypeVar("_T")


def issue641_protocol() -> dict[str, Any]:
    return {
        "research_issue": RESEARCH_ISSUE,
        "source_main": SOURCE_MAIN,
        "source_tree": SOURCE_TREE,
        "issue501_systems_blob": ISSUE501_SYSTEMS_BLOB,
        "issue530_systems_blob": ISSUE530_SYSTEMS_BLOB,
        "issue562_systems_blob": ISSUE562_SYSTEMS_BLOB,
        "issue569_systems_blob": ISSUE569_SYSTEMS_BLOB,
        "v26_9_backend_blob": V26_9_BACKEND_BLOB,
        "issue625_repair_blob": ISSUE625_REPAIR_BLOB,
        "issue571_result_path": ISSUE571_RESULT_PATH,
        "issue571_result_sha256": ISSUE571_RESULT_SHA256,
        "issue571_run": ISSUE571_RUN,
        "issue571_job": ISSUE571_JOB,
        "issue630_trigger": ISSUE630_TRIGGER,
        "issue630_run": ISSUE630_RUN,
        "issue630_job": ISSUE630_JOB,
        "issue630_result_path": ISSUE630_RESULT_PATH,
        "issue630_result_sha256": ISSUE630_RESULT_SHA256,
        "historical_backend_name": HISTORICAL_BACKEND_NAME,
        "v26_9_backend_name": V26_9_BACKEND_NAME,
        "cpu_only": True,
        "end_to_end_systems_authorized": False,
        "gpu_authorized": False,
        "architecture_freeze_authorized": False,
        "s2_authorized": False,
        "fresh_scientific_seed_authorized": False,
        "independent_replication_credit": False,
        "100m_authorized": False,
        "breakthrough_proven": False,
    }


def _validate_frozen_result_shape(historical: dict[str, Any]) -> None:
    if frozenset(historical) != _FROZEN_RESULT_KEYS:
        raise RuntimeError("issue641 frozen physical-sparse result schema drifted")
    for key in (
        "pass",
        "sparse_route_exercised",
        "coalesced_select_merge_positive",
        "backend_activity_positive",
        "backend_names_exact",
        "dense_masked_sparse_credit",
    ):
        if not isinstance(historical[key], bool):
            raise RuntimeError(f"issue641 frozen physical-sparse bool field drifted: {key}")
    if historical["dense_masked_sparse_credit"] is not False:
        raise RuntimeError("issue641 refuses dense-masked sparse credit")
    names = historical["backend_names"]
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise RuntimeError("issue641 frozen backend_names schema drifted")
    expected_historical_pass = bool(
        historical["sparse_route_exercised"]
        and historical["coalesced_select_merge_positive"]
        and historical["backend_activity_positive"]
        and historical["backend_names_exact"]
    )
    if historical["pass"] is not expected_historical_pass:
        raise RuntimeError("issue641 frozen physical-sparse pass equation drifted")


def adapt_frozen_physical_sparse_result_v26_9(
    historical: dict[str, Any],
) -> dict[str, Any]:
    """Recompute only backend identity and the dependent top-level PASS."""

    _validate_frozen_result_shape(historical)
    names = historical["backend_names"]
    v26_9_backend_names_exact = bool(
        names and all(name == V26_9_BACKEND_NAME for name in names)
    )

    adapted = dict(historical)
    adapted["historical_backend_names_exact"] = historical["backend_names_exact"]
    adapted["backend_names_exact"] = v26_9_backend_names_exact
    adapted["pass"] = bool(
        historical["sparse_route_exercised"]
        and historical["coalesced_select_merge_positive"]
        and historical["backend_activity_positive"]
        and v26_9_backend_names_exact
    )
    return adapted


def physical_sparse_proof_v26_9(model: Any, output: dict[str, object]) -> dict[str, Any]:
    """Call the exact frozen #501 proof once, then adapt backend identity only."""

    historical = _FROZEN_PHYSICAL_SPARSE_PROOF(model, output)
    return adapt_frozen_physical_sparse_result_v26_9(historical)


def with_v26_9_physical_sparse_evaluator(supplied_callable: Callable[[], _T]) -> _T:
    """Temporarily install only the issue #641 evaluator around one callable."""

    if base._physical_sparse_proof is not _FROZEN_PHYSICAL_SPARSE_PROOF:
        raise RuntimeError("issue641 refuses pre-existing frozen evaluator drift")
    base._physical_sparse_proof = physical_sparse_proof_v26_9
    try:
        return supplied_callable()
    finally:
        base._physical_sparse_proof = _FROZEN_PHYSICAL_SPARSE_PROOF
