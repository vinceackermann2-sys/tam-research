"""Pure Git subprocess persistence tests with fake files, zero training/GPU."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from experiments.rlt.analysis.checkpoint_archive_guard import (
    CheckpointNotPersisted, validate_staged_checkpoints, validate_persisted_checkpoints,
)


def git(root: Path, *args):
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


def repo(tmp_path: Path):
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.name", "CPU Guard Test")
    git(tmp_path, "config", "user.email", "guard@example.test")
    (tmp_path / ".gitignore").write_text("*.pt\n")
    git(tmp_path, "add", ".gitignore")
    git(tmp_path, "commit", "-qm", "initial")
    cp_path = Path("checkpoints/trained.pt")
    (tmp_path / cp_path).parent.mkdir()
    cp_content = b"fake checkpoint bytes for test only"
    (tmp_path / cp_path).write_bytes(cp_content)
    report = {"status": "complete", "results": [{"name": "mock", "checkpoint": {
        "path": cp_path.as_posix(), "sha256": hashlib.sha256(cp_content).hexdigest(),
        "bytes": len(cp_content)
    }}]}
    result_path = Path("results/mock.json")
    (tmp_path / result_path).parent.mkdir()
    (tmp_path / result_path).write_text(json.dumps(report))
    git(tmp_path, "add", result_path.as_posix())
    return result_path, cp_path


def test_guard_rejects_successful_json_without_staged_ignored_checkpoint(tmp_path):
    result_path, cp_path = repo(tmp_path)
    with pytest.raises(CheckpointNotPersisted, match="NOT staged"):
        validate_staged_checkpoints(tmp_path, result_path)
    git(tmp_path, "commit", "-qm", "result-only-incorrectly-green")
    with pytest.raises(CheckpointNotPersisted, match="failed"):
        validate_persisted_checkpoints(tmp_path, result_path, "HEAD")


def test_explicit_force_stage_and_remote_tree_readback(tmp_path):
    result_path, cp_path = repo(tmp_path)
    git(tmp_path, "add", "-f", cp_path.as_posix())
    assert validate_staged_checkpoints(tmp_path, result_path) == 1
    git(tmp_path, "commit", "-qm", "durably-archive-fake-checkpoint")
    assert validate_persisted_checkpoints(tmp_path, result_path, "HEAD") == 1
    # Disk tampering doesn't change immutable Git commit bytes.
    (tmp_path / cp_path).write_bytes(b"tampered")
    assert validate_persisted_checkpoints(tmp_path, result_path, "HEAD") == 1


def test_wrong_sha256_never_counts_as_persisted(tmp_path):
    result_path, cp_path = repo(tmp_path)
    git(tmp_path, "add", "-f", cp_path.as_posix())
    data = json.loads((tmp_path / result_path).read_text())
    data["results"][0]["checkpoint"]["sha256"] = "0" * 64
    (tmp_path / result_path).write_text(json.dumps(data))
    with pytest.raises(CheckpointNotPersisted, match="mismatch"):
        validate_staged_checkpoints(tmp_path, result_path)
    git(tmp_path, "add", result_path.as_posix())
    git(tmp_path, "commit", "-qm", "wrong checksum")
    with pytest.raises(CheckpointNotPersisted, match="mismatched"):
        validate_persisted_checkpoints(tmp_path, result_path, "HEAD")
