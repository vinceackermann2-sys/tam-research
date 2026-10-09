"""Zero-GPU, in-memory rehearsal of a one-shot Conv7 engineering dispatch.

There is NO real dispatch here: no Modal, subprocess, networking, disk writes,
credentials, persistent reservation, GPU allocation, or authorized engineering seed.
Events are explicitly synthetic; the result can never authorize paid work.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

from scripts.cortex_attention8_conv7_readiness_v1 import check_readiness

ROOT = Path(__file__).resolve().parents[1]
PROPOSAL_PATH = ROOT / "research/reduced_attention/attention8_conv7_engineering_proposal_v1.json"
SYNTHETIC_TAG = "SYNTHETIC_ONE_SHOT_DISPATCH_REHEARSAL_V1"
MODELS = (
    "fresh_transformer",
    "reduced_attention_dense_v2_attention8",
    "attention8_causal_depthwise_conv7",
)
ACTIONS = frozenset(("reserve", "start", "complete", "fail"))


def preflight_dry_run(root: Path = ROOT) -> dict[str, Any]:
    source = check_readiness(root)
    proposal = json.loads(
        (root / "research/reduced_attention/attention8_conv7_engineering_proposal_v1.json")
        .read_text(encoding="utf-8")
    )
    pilot = proposal["draft_engineering_pilot"]
    if source["classification"] != "CONV7_THREE_WAY_CPU_READINESS_PASS_PAID_EXECUTION_BLOCKED":
        raise RuntimeError("source readiness did not block paid execution")
    if proposal["stage"] != "ZERO_GPU_PROPOSAL_NOT_PREREGISTERED":
        raise RuntimeError("unexpected authorization stage")
    if tuple(pilot["comparators"]) != MODELS:
        raise RuntimeError("model order drift")
    if any(value is not False for value in proposal["authority"].values()):
        raise RuntimeError("authority flag is not false")
    if pilot["engineering_seed"] is not None:
        raise RuntimeError("an engineering seed was assigned without new authorization")
    if pilot["strict_max_gpu_spend_usd"] is not None:
        raise RuntimeError("a GPU spending cap was assigned outside a new authorization")
    if pilot["modal_account"] != "primary":
        raise RuntimeError("unauthorized Modal account")
    return {
        "classification": "CONV7_DISPATCH_DRY_RUN_READY_REAL_DISPATCH_BLOCKED",
        "synthetic_only": True,
        "source_locked": True,
        "models": list(MODELS),
        "per_model_steps": pilot["per_model_optimizer_steps"],
        "full_horizon_steps": pilot["full_horizon_optimizer_steps"],
        "engineering_seed": None,
        "spending_cap_usd": None,
        "durable_reservation_made": False,
        "paid_execution_permitted": False,
        "gpu_allocated": False,
        "writes_performed": False,
        "seed_consumed": False,
        "retry_permitted": False,
        "resume_permitted": False,
        "breakthrough_claim_permitted": False,
    }


def replay_synthetic_dispatch(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Rehearse a state machine only; no source of real authorization is accepted.

    Each synthetic event has exactly marker, action, and model.
    A reserve event has model=None; the three model attempts must be sequential.
    Failure is terminal; no restart, retry, or resume is accepted.
    """
    if not isinstance(events, (list, tuple)):
        raise TypeError("events must be a list/tuple of synthetic event mappings")
    state = "UNRESERVED"
    model_index = 0
    started = 0
    completed = 0
    mock_reserved = False
    rejection: str | None = None
    for event in events:
        if not isinstance(event, Mapping):
            rejection = "event_not_mapping"
            break
        if set(event) != {"marker", "action", "model"}:
            rejection = "event_schema_mismatch"
            break
        if event["marker"] != SYNTHETIC_TAG:
            rejection = "event_not_synthetic"
            break
        action = event["action"]
        model = event["model"]
        if action not in ACTIONS:
            rejection = "unsupported_action"
            break
        if state in ("COMPLETE", "TERMINAL_FAILED"):
            rejection = "event_after_terminal"
            break
        if action == "reserve":
            if model is not None or state != "UNRESERVED":
                rejection = "reservation_order_mismatch"
                break
            mock_reserved = True
            state = "READY"
            continue
        if state == "UNRESERVED":
            rejection = "attempt_without_reservation"
            break
        if model_index >= len(MODELS) or model != MODELS[model_index]:
            rejection = "model_order_or_identity_mismatch"
            break
        if action == "start":
            if state != "READY":
                rejection = "duplicate_or_concurrent_start"
                break
            started += 1
            state = "RUNNING"
        elif action == "complete":
            if state != "RUNNING":
                rejection = "complete_without_start"
                break
            completed += 1
            model_index += 1
            state = "COMPLETE" if model_index == len(MODELS) else "READY"
        elif action == "fail":
            if state != "RUNNING":
                rejection = "failure_without_start"
                break
            state = "TERMINAL_FAILED"
    envelope: dict[str, Any] = {
        "synthetic_only": True,
        "real_reservation_created": False,
        "gpu_allocated": False,
        "writes_performed": False,
        "real_seed_consumed": False,
        "paid_execution_permitted": False,
        "retry_permitted": False,
        "resume_permitted": False,
        "scientific_evidence": False,
        "breakthrough_claim_permitted": False,
        "mock_reservation_observed": mock_reserved,
        "synthetic_attempts_started": started,
        "synthetic_attempts_completed": completed,
    }
    if rejection is not None:
        return {**envelope, "classification": "SYNTHETIC_DISPATCH_REJECTED", "reason": rejection}
    if state == "COMPLETE":
        return {**envelope, "classification": "SYNTHETIC_DISPATCH_ALL_THREE_COMPLETE_NOT_EVIDENCE"}
    if state == "TERMINAL_FAILED":
        return {**envelope, "classification": "SYNTHETIC_DISPATCH_TERMINAL_FAIL_NO_RETRY"}
    return {**envelope, "classification": "SYNTHETIC_DISPATCH_INCOMPLETE_BLOCKED"}


def event(action: str, model: str | None = None) -> dict[str, Any]:
    """Build a clearly marked synthetic event; never a live job."""
    return {"marker": SYNTHETIC_TAG, "action": action, "model": model}


if __name__ == "__main__":
    # Static source check only; absolutely no dispatch or durable marker writes.
    print(json.dumps(preflight_dry_run(), sort_keys=True))
