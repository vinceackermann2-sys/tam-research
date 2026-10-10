"""RLT future-only checkpoint persistence guard; never reruns experiments.

An output JSON containing local checkpoint paths and SHA256 hashes is not
durable evidence. A successful run must prove persisted bytes in a commit ref
or a downloadable, checksum-verified artifact with a permanent inventory.

This module implements the Git-commit variant. It checks the STAGED paths before
commit and the REMOTE FETCHED ref after push. It is pure inspection; does not
add files, commit, push, train, or call any GPU.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


class CheckpointNotPersisted(RuntimeError):
    pass


def _git(repo: Path, *args: str) -> bytes:
    result = subprocess.run(["git", *args], cwd=repo, capture_output=True)
    if result.returncode:
        raise CheckpointNotPersisted(
            f"git {' '.join(args[:2])} failed: {result.stderr.decode(errors='replace')[:500]}"
        )
    return result.stdout


def _manifest(result_json: Path) -> list[tuple[Path, str, int]]:
    data = json.loads(result_json.read_text())
    if data.get("status") != "complete":
        raise CheckpointNotPersisted("terminal result JSON required")
    items = []
    for model in data.get("results", []):
        cp = model.get("checkpoint")
        if not cp:
            raise CheckpointNotPersisted("checkpoint entry missing")
        path = Path(cp["path"])
        digest = cp["sha256"].lower()
        size = cp["bytes"]
        if path.is_absolute() or ".." in path.parts or len(digest) != 64 or not isinstance(size, int) or size <= 0:
            raise CheckpointNotPersisted(f"invalid checkpoint metadata {path}")
        if any(ch not in "0123456789abcdef" for ch in digest):
            raise CheckpointNotPersisted("invalid SHA256")
        items.append((path, digest, size))
    if not items:
        raise CheckpointNotPersisted("no checkpoint entries")
    if len({str(path) for path, _, _ in items}) != len(items):
        raise CheckpointNotPersisted("duplicate checkpoint entry")
    return items


def validate_staged_checkpoints(repo: Path, result_json: Path) -> int:
    """Call immediately after git add (use -f for ignored files if intended).

    Detects .pt files that were written locally but silently omitted by git add.
    """
    repo = repo.resolve()
    entries = _manifest(repo / result_json)
    staged = set(_git(repo, "diff", "--cached", "--name-only", "--diff-filter=ACMR").decode().splitlines())
    for path, sha, length in entries:
        raw_path = repo / path
        if not raw_path.is_file():
            raise CheckpointNotPersisted(f"checkpoint missing locally: {path}")
        raw = raw_path.read_bytes()
        if len(raw) != length or hashlib.sha256(raw).hexdigest() != sha:
            raise CheckpointNotPersisted(f"checkpoint hash/size mismatch locally: {path}")
        if path.as_posix() not in staged:
            raise CheckpointNotPersisted(f"checkpoint NOT staged (gitignore?): {path}")
    return len(entries)


def validate_persisted_checkpoints(repo: Path, result_json: Path, commit_ref: str) -> int:
    """Call after fetching the pushed ref. Fail if Git lacks exact checkpoint bytes.

    commit_ref must be a trusted commit SHA or explicit fetched ref; do not
    pass arbitrary untrusted user strings. This does not fetch over the network.
    """
    repo = repo.resolve()
    if not commit_ref or any(c.isspace() for c in commit_ref) or ":" in commit_ref:
        raise CheckpointNotPersisted("invalid commit ref")
    entries = _manifest(repo / result_json)
    for path, sha, length in entries:
        stored = _git(repo, "show", f"{commit_ref}:{path.as_posix()}")
        if len(stored) != length or hashlib.sha256(stored).hexdigest() != sha:
            raise CheckpointNotPersisted(f"checkpoint missing or mismatched in {commit_ref}: {path}")
    return len(entries)
