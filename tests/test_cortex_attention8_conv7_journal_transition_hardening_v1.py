"""Adversarial CPU journal transition tests; no Modal or training."""
import json

import pytest
from tam_research.cortex_attention8_conv7_dispatch_dryrun_v1 import MODELS
from tam_research.cortex_attention8_conv7_journal_simulation_v1 import MARK, SyntheticJournal


def forged_start(root, model):
    i = MODELS.index(model)
    (root / f"{i:02d}-start.json").write_text(
        json.dumps({"mode": MARK, "model": model, "action": "start", "paid_authority": False}),
        encoding="utf-8",
    )


@pytest.mark.parametrize("bad_followup", ("STARTED", "COMPLETE", "ERROR"))
def test_valid_looking_later_marker_after_error_must_be_rejected(tmp_path, bad_followup):
    journal = SyntheticJournal(tmp_path)
    journal.reserve()
    journal.start(MODELS[0])
    journal.finish(MODELS[0], "ERROR")
    forged_start(tmp_path, MODELS[1])
    if bad_followup != "STARTED":
        (tmp_path / "01-terminal.json").write_text(
            json.dumps({"mode": MARK, "model": MODELS[1], "action": "terminal",
                        "outcome": bad_followup, "paid_authority": False}),
            encoding="utf-8",
        )
    with pytest.raises(RuntimeError, match="out of order"):
        journal.snapshot()
    with pytest.raises(RuntimeError):
        journal.start(MODELS[2])


@pytest.mark.parametrize("later_index", (1, 2))
def test_incomplete_first_model_cannot_hide_later_attempt(tmp_path, later_index):
    journal = SyntheticJournal(tmp_path)
    journal.reserve()
    journal.start(MODELS[0])
    forged_start(tmp_path, MODELS[later_index])
    with pytest.raises(RuntimeError, match="out of order"):
        journal.snapshot()


def test_empty_predecessor_must_not_hide_skipped_model(tmp_path):
    journal = SyntheticJournal(tmp_path)
    journal.reserve()
    forged_start(tmp_path, MODELS[2])
    with pytest.raises(RuntimeError, match="out of order"):
        journal.snapshot()


def test_valid_symlink_marker_rejected(tmp_path):
    journal = SyntheticJournal(tmp_path)
    journal.reserve()
    (tmp_path / "00-start.json").symlink_to(tmp_path / "reservation.json")
    with pytest.raises(RuntimeError, match="symlink"):
        journal.snapshot()


def test_dangling_symlink_does_not_look_unreserved(tmp_path):
    journal = SyntheticJournal(tmp_path)
    (tmp_path / "reservation.json").symlink_to(tmp_path / "absent-target.json")
    with pytest.raises(RuntimeError, match="symlink"):
        journal.snapshot()
    with pytest.raises(RuntimeError):
        journal.reserve()


def test_good_three_model_path_still_valid(tmp_path):
    journal = SyntheticJournal(tmp_path)
    journal.reserve()
    for model in MODELS:
        journal.start(model)
        journal.finish(model, "COMPLETE")
    result = journal.snapshot()
    assert result["state"] == "SYNTHETIC_ALL_COMPLETE_NOT_EVIDENCE"
    assert not result["paid_authority"]
    assert not result["gpu_allocated"]
    assert not result["real_seed_consumed"]


def test_no_modal_or_gpu_dispatch_hooks():
    from pathlib import Path
    text = (Path(__file__).resolve().parents[1] /
            "tam_research/cortex_attention8_conv7_journal_simulation_v1.py").read_text()
    for prohibited in ("import modal", ".remote(", "modal run", "gpu=", "cuda",
                       "requests.", "subprocess.", "optimizer.step(", "h100_train_one("):
        assert prohibited not in text
