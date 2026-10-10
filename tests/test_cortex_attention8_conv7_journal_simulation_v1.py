"""CPU-only synthetic journal failure and race tests."""
from concurrent.futures import ThreadPoolExecutor
import json
import pytest
from tam_research.cortex_attention8_conv7_journal_simulation_v1 import SyntheticJournal, MODELS


def test_all_models_complete_without_paid_authority(tmp_path):
    j = SyntheticJournal(tmp_path)
    j.reserve()
    for name in MODELS:
        j.start(name)
        j.finish(name, "COMPLETE")
    state = j.snapshot()
    assert state["state"] == "SYNTHETIC_ALL_COMPLETE_NOT_EVIDENCE"
    assert set(state["statuses"].values()) == {"COMPLETE"}
    for forbidden in ("real_reservation", "paid_authority", "real_seed_consumed",
                      "gpu_allocated", "scientific_evidence"):
        assert state[forbidden] is False
    with pytest.raises(RuntimeError):
        j.start(MODELS[0])


@pytest.mark.parametrize("failure_index", range(3))
def test_terminal_error_blocks_every_later_attempt(tmp_path, failure_index):
    j = SyntheticJournal(tmp_path)
    j.reserve()
    for name in MODELS[:failure_index]:
        j.start(name)
        j.finish(name, "COMPLETE")
    failed = MODELS[failure_index]
    j.start(failed)
    j.finish(failed, "ERROR")
    assert j.snapshot()["state"] == "TERMINAL_FAILURE_NO_RETRY"
    with pytest.raises(RuntimeError):
        j.start(failed)
    with pytest.raises(RuntimeError):
        j.finish(failed, "COMPLETE")
    if failure_index < 2:
        with pytest.raises(RuntimeError):
            j.start(MODELS[failure_index+1])


def test_open_attempt_survives_reopen_and_cannot_restart(tmp_path):
    j = SyntheticJournal(tmp_path)
    j.reserve()
    j.start(MODELS[0])
    recovered = SyntheticJournal(tmp_path)
    assert recovered.snapshot()["state"] == "INCOMPLETE_START_NO_REPLAY"
    with pytest.raises(RuntimeError):
        recovered.start(MODELS[0])
    with pytest.raises(RuntimeError):
        recovered.start(MODELS[1])


def test_concurrent_reservations_only_one_wins(tmp_path):
    def action(_):
        try:
            SyntheticJournal(tmp_path).reserve()
            return "ok"
        except RuntimeError:
            return "rejected"
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(action, [0, 1])) == ["ok", "rejected"]


def test_concurrent_start_and_terminal_writers_are_exclusive(tmp_path):
    j = SyntheticJournal(tmp_path)
    j.reserve()
    def start(_):
        try:
            SyntheticJournal(tmp_path).start(MODELS[0])
            return "ok"
        except RuntimeError:
            return "rejected"
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(start, [0, 1])) == ["ok", "rejected"]
    def finish(outcome):
        try:
            SyntheticJournal(tmp_path).finish(MODELS[0], outcome)
            return "ok"
        except RuntimeError:
            return "rejected"
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(finish, ["COMPLETE", "ERROR"])) == ["ok", "rejected"]
    assert j.snapshot()["statuses"][MODELS[0]] in ("COMPLETE", "ERROR")


def test_partial_marker_fails_closed(tmp_path):
    j = SyntheticJournal(tmp_path)
    j.reserve()
    (tmp_path / "00-start.json").write_bytes(b"{")
    with pytest.raises(RuntimeError, match="corrupt or partial"):
        j.snapshot()
    with pytest.raises(RuntimeError):
        j.start(MODELS[0])


def test_forged_authority_or_wrong_order_blocked(tmp_path):
    j = SyntheticJournal(tmp_path)
    j.reserve()
    with pytest.raises(RuntimeError):
        j.start(MODELS[1])
    f = tmp_path / "reservation.json"
    payload = json.loads(f.read_text())
    payload["paid_authority"] = True
    f.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="cannot authorize GPU"):
        j.snapshot()


def test_no_live_dispatch_hooks_present():
    from pathlib import Path
    source = (Path(__file__).resolve().parents[1] /
              "tam_research/cortex_attention8_conv7_journal_simulation_v1.py").read_text()
    for x in ("import modal", "modal run", ".remote(", "torch.", "subprocess.",
              "requests.", "gpu=", "optimizer.step(", "h100_train_one("):
        assert x not in source
