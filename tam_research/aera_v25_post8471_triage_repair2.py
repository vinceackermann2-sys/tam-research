from __future__ import annotations

"""Preflight-schema-only repair for frozen v25 checkpoint triage.

Preregistered by #377 after #376 failed before L4 allocation because its
preflight incorrectly expected a top-level seed in the durable result.json.
The actual checkpoint diagnostic remains exactly repair1's implementation.
"""

from typing import Any

from .aera_v25_post8471_triage_repair1 import (
    LOSS_TIME_SLICE,
    run_checkpoint_triage_repair1,
)

REPAIR_ISSUE = 377
SOURCE_REPAIR_ISSUE = 372
SOURCE_FAILED_TRIGGER = 376
SOURCE_FAILED_ACTIONS_RUN = 32858651011
SOURCE_SEED = 8471


def validate_source_result_seed(payload: dict[str, Any]) -> dict[str, int]:
    if not isinstance(payload, dict):
        raise RuntimeError("source result must be a dictionary")
    protocol = payload.get("protocol")
    aera = payload.get("aera")
    transformer = payload.get("transformer")
    if not isinstance(protocol, dict):
        raise RuntimeError("source result missing protocol dictionary")
    if not isinstance(aera, dict):
        raise RuntimeError("source result missing aera summary")
    if not isinstance(transformer, dict):
        raise RuntimeError("source result missing transformer summary")
    values = {
        "protocol_development_seed": protocol.get("development_seed"),
        "aera_seed": aera.get("seed"),
        "transformer_seed": transformer.get("seed"),
    }
    if any(value != SOURCE_SEED for value in values.values()):
        raise RuntimeError(f"source result seed schema mismatch: {values}")
    return {key: int(value) for key, value in values.items()}


def repair2_protocol() -> dict[str, Any]:
    return {
        "research_issue": 369,
        "repair1_issue": SOURCE_REPAIR_ISSUE,
        "repair2_issue": REPAIR_ISSUE,
        "source_failed_trigger": SOURCE_FAILED_TRIGGER,
        "source_failed_actions_run": SOURCE_FAILED_ACTIONS_RUN,
        "source_checkpoint_seed": SOURCE_SEED,
        "diagnostic_sampling_seed": 138471,
        "diagnostic_implementation": "run_checkpoint_triage_repair1_unmodified",
        "loss_time_slice_tokens": LOSS_TIME_SLICE,
        "semantic_change": "source_result_seed_schema_preflight_only",
        "training_performed": False,
        "checkpoint_mutation_authorized": False,
        "scientific_protocol_changed": False,
        "100m_authorized": False,
    }


__all__ = [
    "LOSS_TIME_SLICE",
    "REPAIR_ISSUE",
    "SOURCE_SEED",
    "repair2_protocol",
    "run_checkpoint_triage_repair1",
    "validate_source_result_seed",
]
