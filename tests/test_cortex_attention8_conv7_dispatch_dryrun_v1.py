"""CPU-only test suite for the Conv7 one-shot dispatch simulation."""
from __future__ import annotations

from pathlib import Path

import pytest

from tam_research.cortex_attention8_conv7_dispatch_dryrun_v1 import (
    MODELS,
    event,
    preflight_dry_run,
    replay_synthetic_dispatch,
)


def _success_events():
    events = [event("reserve")]
    for model in MODELS:
        events.extend((event("start", model), event("complete", model)))
    return events


def test_source_locked_preflight_remains_paid_execution_blocked():
    report = preflight_dry_run()
    assert report["classification"] == "CONV7_DISPATCH_DRY_RUN_READY_REAL_DISPATCH_BLOCKED"
    assert report["source_locked"] and report["synthetic_only"]
    assert report["models"] == list(MODELS)
    assert report["per_model_steps"] == 3052 and report["full_horizon_steps"] == 30518
    assert report["engineering_seed"] is None and report["spending_cap_usd"] is None
    for flag in ("durable_reservation_made", "paid_execution_permitted", "gpu_allocated",
                 "writes_performed", "seed_consumed", "retry_permitted",
                 "resume_permitted", "breakthrough_claim_permitted"):
        assert report[flag] is False


def test_synthetic_success_never_becomes_scientific_or_paid_evidence():
    out = replay_synthetic_dispatch(_success_events())
    assert out["classification"] == "SYNTHETIC_DISPATCH_ALL_THREE_COMPLETE_NOT_EVIDENCE"
    assert out["mock_reservation_observed"]
    assert out["synthetic_attempts_started"] == 3
    assert out["synthetic_attempts_completed"] == 3
    for flag in ("real_reservation_created", "gpu_allocated", "writes_performed",
                 "real_seed_consumed", "paid_execution_permitted", "retry_permitted",
                 "resume_permitted", "scientific_evidence", "breakthrough_claim_permitted"):
        assert out[flag] is False


@pytest.mark.parametrize(
    ("events", "reason"),
    [
        ([event("start", MODELS[0])], "attempt_without_reservation"),
        ([event("complete", MODELS[0])], "attempt_without_reservation"),
        ([event("reserve"), event("reserve")], "reservation_order_mismatch"),
        ([event("reserve"), event("start", MODELS[1])], "model_order_or_identity_mismatch"),
        ([event("reserve"), event("complete", MODELS[0])], "complete_without_start"),
        ([event("reserve"), event("fail", MODELS[0])], "failure_without_start"),
        ([event("reserve"), event("start", MODELS[0]), event("start", MODELS[0])], "duplicate_or_concurrent_start"),
        ([event("reserve"), event("start", MODELS[0]), event("start", MODELS[1])], "model_order_or_identity_mismatch"),
        ([event("reserve"), event("start", MODELS[0]), event("complete", MODELS[0]), event("complete", MODELS[1])], "complete_without_start"),
        (_success_events() + [event("reserve")], "event_after_terminal"),
        ([event("reserve", MODELS[0])], "reservation_order_mismatch"),
        ([{"marker": "REAL", "action": "reserve", "model": None}], "event_not_synthetic"),
        ([{"marker": "x", "action": "start", "model": MODELS[0]}], "event_not_synthetic"),
        ([{"marker": "SYNTHETIC_ONE_SHOT_DISPATCH_REHEARSAL_V1", "action": "retry", "model": MODELS[0]}], "unsupported_action"),
        ([{"marker": "SYNTHETIC_ONE_SHOT_DISPATCH_REHEARSAL_V1", "action": "reserve", "model": None, "authorized": True}], "event_schema_mismatch"),
    ],
)
def test_invalid_transition_rejected(events, reason):
    out = replay_synthetic_dispatch(events)
    assert out["classification"] == "SYNTHETIC_DISPATCH_REJECTED"
    assert out["reason"] == reason
    assert out["paid_execution_permitted"] is False


@pytest.mark.parametrize("failure_model_index", [0, 1, 2])
def test_error_is_terminal_and_never_resumes(failure_model_index):
    events = [event("reserve")]
    for i in range(failure_model_index):
        events.extend((event("start", MODELS[i]), event("complete", MODELS[i])))
    events.extend((event("start", MODELS[failure_model_index]), event("fail", MODELS[failure_model_index])))
    out = replay_synthetic_dispatch(events)
    assert out["classification"] == "SYNTHETIC_DISPATCH_TERMINAL_FAIL_NO_RETRY"
    assert out["synthetic_attempts_started"] == failure_model_index + 1
    assert out["synthetic_attempts_completed"] == failure_model_index
    refused = replay_synthetic_dispatch(events + [event("start", MODELS[failure_model_index])])
    assert refused["classification"] == "SYNTHETIC_DISPATCH_REJECTED"
    assert refused["reason"] == "event_after_terminal"


def test_missing_reservation_or_incomplete_series_stays_blocked():
    assert replay_synthetic_dispatch([])["classification"] == "SYNTHETIC_DISPATCH_INCOMPLETE_BLOCKED"
    assert replay_synthetic_dispatch([event("reserve")])["classification"] == "SYNTHETIC_DISPATCH_INCOMPLETE_BLOCKED"
    assert replay_synthetic_dispatch([event("reserve"), event("start", MODELS[0])])["classification"] == "SYNTHETIC_DISPATCH_INCOMPLETE_BLOCKED"


def test_sources_contain_no_live_dispatch_or_persistent_side_effects():
    root = Path(__file__).resolve().parents[1]
    source = (root / "tam_research/cortex_attention8_conv7_dispatch_dryrun_v1.py").read_text(encoding="utf-8")
    for banned in ("import modal", "modal run", ".remote(", "gpu=", "cuda", "subprocess.",
                   "requests.", "urllib.", "volume.commit", ".write_text(",
                   ".write_bytes(", ".mkdir(", ".replace(", "optimizer.step("):
        assert banned not in source
    with pytest.raises(TypeError):
        replay_synthetic_dispatch("not a list")
