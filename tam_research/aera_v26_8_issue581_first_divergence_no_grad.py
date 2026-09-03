from __future__ import annotations

"""Issue #581 CPU-first execution-context repair for consumed #580.

The #578 first-divergence localizer is historical evidence and remains byte-frozen.
Its only pre-result blocker was that the entire function was decorated with
``torch.inference_mode()``, so the models it constructed contained inference
parameters without version counters while the frozen body intentionally checks
``parameter._version`` before and after localization.

This module does not copy or change that body. It calls the exact undecorated
``__wrapped__`` body once under ``torch.no_grad()``. That keeps autograd disabled
while preserving ordinary tensor version counters. An outer inference-mode caller
is rejected rather than silently changing the execution contract.
"""

from typing import Any

import torch

from . import aera_v26_8_issue578_first_divergence_localizer as issue578


RESEARCH_ISSUE = 581
SOURCE_MAIN = "eba8d04ed262c9bf539a99d256af5e99eb6a87d1"
SOURCE_TREE = "e0275443a58a0dd8528e1329988713dbb2045bd0"

FROZEN_ISSUE578_LOCALIZER_BLOB = "5933d1113b950348bb63e5f3eb8713ad36dbf580"
FROZEN_ISSUE578_LAUNCHER_BLOB = "cd47e1252bed5617556998659eadfe6a61637d39"
FROZEN_ISSUE578_WORKFLOW_BLOB = "b76282733903d220e7118ede283f789db0eb56ba"
FROZEN_ISSUE578_CPU_TEST_BLOB = "6dd02b5a25514ad9987d7617e4a4b1ddbb1e6f0a"

ISSUE580_TRIGGER = 580
ISSUE580_RUN = 33748196657
ISSUE580_JOB = 100625461189
ISSUE580_ATTEMPT = 1
ISSUE580_BOUND_MAIN = "eba8d04ed262c9bf539a99d256af5e99eb6a87d1"
ISSUE580_L4_STARTED = True
ISSUE580_AUTHORITATIVE_RESULT_EMITTED = False
ISSUE580_FAILURE = "Inference tensors do not track version counter."


def issue581_execution_context_protocol() -> dict[str, Any]:
    return {
        "research_issue": RESEARCH_ISSUE,
        "source_main": SOURCE_MAIN,
        "source_tree": SOURCE_TREE,
        "frozen_issue578_localizer_blob": FROZEN_ISSUE578_LOCALIZER_BLOB,
        "frozen_issue578_launcher_blob": FROZEN_ISSUE578_LAUNCHER_BLOB,
        "frozen_issue578_workflow_blob": FROZEN_ISSUE578_WORKFLOW_BLOB,
        "frozen_issue578_cpu_test_blob": FROZEN_ISSUE578_CPU_TEST_BLOB,
        "issue580_trigger": ISSUE580_TRIGGER,
        "issue580_run": ISSUE580_RUN,
        "issue580_job": ISSUE580_JOB,
        "issue580_attempt": ISSUE580_ATTEMPT,
        "issue580_bound_main": ISSUE580_BOUND_MAIN,
        "issue580_l4_started": ISSUE580_L4_STARTED,
        "issue580_authoritative_result_emitted": ISSUE580_AUTHORITATIVE_RESULT_EMITTED,
        "issue580_failure": ISSUE580_FAILURE,
        "frozen_body_reused_via_wrapped": True,
        "execution_context": "torch.no_grad",
        "outer_inference_mode_allowed": False,
        "grad_enabled_during_successor_entry": False,
        "frozen_before_after_parameter_version_checks_preserved": True,
        "gpu_authorized_by_issue581": False,
        "localization_executed_by_issue581": False,
        "repair_authorized": False,
        "architecture_freeze_authorized": False,
        "s2_authorized": False,
        "fresh_scientific_seed_authorized": False,
        "independent_replication_credit": False,
        "100m_authorized": False,
        "breakthrough_proven": False,
    }


def cpu_contract_preflight_issue581() -> dict[str, Any]:
    target = issue578.run_first_divergence_localization
    undecorated = getattr(target, "__wrapped__", None)
    if undecorated is None or not callable(undecorated):
        raise RuntimeError("issue581 frozen #578 localizer no longer exposes callable __wrapped__")
    protocol = issue581_execution_context_protocol()
    higher = (
        "gpu_authorized_by_issue581",
        "localization_executed_by_issue581",
        "repair_authorized",
        "architecture_freeze_authorized",
        "s2_authorized",
        "fresh_scientific_seed_authorized",
        "independent_replication_credit",
        "100m_authorized",
        "breakthrough_proven",
    )
    if any(protocol[key] for key in higher):
        raise RuntimeError("issue581 higher authorization drifted")
    return {
        "research_issue": RESEARCH_ISSUE,
        "source_main": SOURCE_MAIN,
        "source_tree": SOURCE_TREE,
        "protocol": protocol,
        "frozen_wrapped_body_available": True,
        "model_construction_performed": False,
        "checkpoint_loaded": False,
        "localization_measurement_performed": False,
        "gpu_authorized_by_cpu_preflight": False,
        "scientific_seed_consumed": False,
    }


@torch.no_grad()
def run_first_divergence_localization_issue581(
    *,
    run_dir: str = issue578.base.CHECKPOINT_RELATIVE_DIR,
) -> dict[str, Any]:
    if torch.is_inference_mode_enabled():
        raise RuntimeError(
            "issue581 refuses outer torch.inference_mode because frozen _version checks "
            "require ordinary tensor version counters"
        )
    if torch.is_grad_enabled():
        raise RuntimeError("issue581 no-grad execution contract unexpectedly enabled gradients")
    target = issue578.run_first_divergence_localization
    undecorated = getattr(target, "__wrapped__", None)
    if undecorated is None or not callable(undecorated):
        raise RuntimeError("issue581 frozen #578 localizer no longer exposes callable __wrapped__")
    return undecorated(run_dir=run_dir)
