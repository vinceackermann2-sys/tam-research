"""Offline Colab/Conv7 migration blocker and tampering tests."""
from pathlib import Path
import shutil

import pytest

from scripts.cortex_attention8_conv7_colab_readiness_audit_v1 import (
    COMPUTE_GUIDE, CONV7_JOB, PINNED, ROOT, WORKER, audit_colab_migration,
)


def copy_fixture(dest):
    for path in set(PINNED) | {COMPUTE_GUIDE}:
        target = dest / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / path, target)


def test_current_status_reports_execution_blocked_not_breakthrough():
    report = audit_colab_migration()
    assert report["classification"] == "CONV7_COLAB_MIGRATION_CPU_AUDIT_PASS_PAID_EXECUTION_BLOCKED"
    assert report["protected_blobs_checked"] == 4
    assert report["colab_worker_max_timeout_sec"] == 300
    assert report["colab_worker_default_timeout_sec"] == 180
    assert "colab_conv7_standalone_job_missing" in report["blockers"]
    assert len(report["blockers"]) == 8
    for field in ("real_colab_dataset_verified", "real_colab_worker_live_status_known",
                  "real_colab_training_done", "seed_reserved", "paid_execution_permitted",
                  "scientific_evidence", "breakthrough_claim_permitted"):
        assert report[field] is False


@pytest.mark.parametrize("path", sorted(PINNED))
def test_frozen_sources_cannot_change_silently(tmp_path, path):
    copy_fixture(tmp_path)
    target = tmp_path / path
    target.write_bytes(target.read_bytes() + b"\n# synthetic drift")
    with pytest.raises(RuntimeError, match="protected"):
        audit_colab_migration(tmp_path)


def test_standalone_job_creation_requires_fresh_audit(tmp_path):
    copy_fixture(tmp_path)
    job = tmp_path / CONV7_JOB
    job.parent.mkdir(parents=True)
    job.write_text("# not a reviewed executable")
    with pytest.raises(RuntimeError, match="new source-bound runner audit"):
        audit_colab_migration(tmp_path)


def test_never_launch_a_gpu_or_issue_from_preflight():
    source = (ROOT / "scripts/cortex_attention8_conv7_colab_readiness_audit_v1.py").read_text()
    for keyword in ("import modal", "colab run", "gh issue create", "gh issue edit",
                    "subprocess.", "optimizer.step(", ".remote(", "gpu=",
                    ".write_text(", ".write_bytes(", "requests."):
        assert keyword not in source


def test_historical_runner_not_confused_with_colab():
    out = audit_colab_migration()
    assert out["historical_modal_lock_not_colab_authority"]
    assert not out["paid_execution_permitted"]
