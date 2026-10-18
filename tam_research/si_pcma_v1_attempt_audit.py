"""Conservative read-only original-SI attempt-document audit for PCMA-v1.

This is NOT a reservation service, an availability checker, a scientific
launcher, or proof that a candidate seed was used. It inventories *mentions*
of seed-shaped integers in original-SI protocols/results/addenda. Every
documented candidate is treated as a possible conflict until adjudicated.
A new-looking seed is NEVER approved: missing cross-branch issues, local raw
attempts, frozen PR heads, or worker/storage logs could conceal consumption.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Sequence


SOURCE_PREFIX = "SI_ORIGINAL_"
DOCUMENT_TYPES = ("PROTOCOL", "RESULTS", "ADDENDUM")
# 4-6 digit IDs are what this particular original SI track historically used.
# This deliberately also picks up training budgets; false positives are
# clearly labelled as candidates and may only deny permission, never grant it.
NUMBER = re.compile(r"(?<![0-9A-Za-z])([1-9][0-9]{3,5})(?![0-9A-Za-z])")
RANGE = re.compile(r"(?<![0-9A-Za-z])([1-9][0-9]{3,5})\s*[-–—]\s*([1-9][0-9]{3,5})(?![0-9A-Za-z])")
SEED_CONTEXT = re.compile(
    r"\bseed(?:s|ed)?\b|\battempt(?:s|ed)?\b|\breplac(?:ement|ed|ing)\b"
    r"|\brerun\b|\bretr(?:y|ies|ied)\b|\bconsumed\b",
    re.IGNORECASE,
)
KNOWN_CONSUMED_V26 = frozenset((118010, 118211, 118212, 118213))
KNOWN_CONSUMED_V25 = frozenset((117010, 117211, 117212, 117213, 117214))
KNOWN_INTERRUPTED = frozenset((99611, 100211, 103211, 103212,
                               115213, 116211, 117212))
# These are anchored specific negative records, not exhaustive by design.
MANDATORY_ANCHORS = KNOWN_CONSUMED_V26 | KNOWN_CONSUMED_V25 | KNOWN_INTERRUPTED


@dataclass(frozen=True)
class DocumentaryMention:
    candidate_id: int
    file_path: str
    line: int
    document_git_blob: str
    evidence: str
    evidence_class: str = "POTENTIAL_SEED_ONLY"


@dataclass(frozen=True)
class AttemptAudit:
    source_file_count: int
    result_file_count: int
    protocol_file_count: int
    mention_count: int
    candidate_ids: tuple[int, ...]
    missing_anchor_ids: tuple[int, ...]
    document_blobs: tuple[tuple[str, str], ...]
    source_coverage: str
    global_attempt_ledger_covered: bool
    may_reserve_or_execute: bool


@dataclass(frozen=True)
class SeedRequestReview:
    proposed_id: int
    conflicting_mentions: tuple[DocumentaryMention, ...]
    known_consumed_or_interrupted: bool
    verdict: str
    reasons: tuple[str, ...]
    can_reserve: bool


def _git_blob_bytes(data: bytes) -> str:
    payload = b"blob " + str(len(data)).encode("ascii") + b"\0" + data
    return hashlib.sha1(payload).hexdigest()


def _is_source_doc(path: Path) -> bool:
    return (path.suffix.lower() == ".md"
            and path.name.startswith(SOURCE_PREFIX)
            and any(token in path.stem for token in DOCUMENT_TYPES))


def _numbers_with_bounded_ranges(line: str) -> set[int]:
    """Never allocate giant ranges from numeric text."""
    values = {int(x) for x in NUMBER.findall(line)}
    for start, end in RANGE.findall(line):
        lo, hi = int(start), int(end)
        if 0 <= hi - lo <= 16:
            values.update(range(lo, hi + 1))
    return values


def documented_mentions(repo_root: Path) -> tuple[DocumentaryMention, ...]:
    """Read only repository documents; do not query/modify cloud resources."""
    root = Path(repo_root).resolve()
    docs = root / "docs"
    if not docs.is_dir():
        raise FileNotFoundError("docs directory absent")
    mentions: list[DocumentaryMention] = []
    for file in sorted(docs.glob("SI_ORIGINAL_*.md")):
        if not _is_source_doc(file):
            continue
        if not file.is_file() or file.is_symlink():
            raise ValueError("unexpected evidence file type")
        data = file.read_bytes()
        if len(data) > 1024 * 1024:
            raise ValueError("oversized historical evidence file")
        blob = _git_blob_bytes(data)
        content = data.decode("utf-8")
        for line_num, line in enumerate(content.splitlines(), start=1):
            if not SEED_CONTEXT.search(line):
                continue
            ids = _numbers_with_bounded_ranges(line)
            # Avoid extremely long line snippets; the original source file
            # + line number + blob is the authoritative evidence locator.
            snippet = line.strip()[:220]
            for candidate in sorted(ids):
                mentions.append(DocumentaryMention(
                    candidate, str(file.relative_to(root)), line_num,
                    blob, snippet,
                ))
    return tuple(mentions)


def audit_original_si_docs(repo_root: Path) -> AttemptAudit:
    root = Path(repo_root).resolve()
    docs = [p for p in (root / "docs").glob("SI_ORIGINAL_*.md")
            if _is_source_doc(p)]
    mentions = documented_mentions(root)
    candidates = frozenset(m.candidate_id for m in mentions)
    document_blobs = tuple(sorted({
        (m.file_path, m.document_git_blob) for m in mentions
    }))
    return AttemptAudit(
        source_file_count=len(docs),
        result_file_count=sum("RESULTS" in p.name for p in docs),
        protocol_file_count=sum("PROTOCOL" in p.name for p in docs),
        mention_count=len(mentions),
        candidate_ids=tuple(sorted(candidates)),
        missing_anchor_ids=tuple(sorted(MANDATORY_ANCHORS - candidates)),
        document_blobs=document_blobs,
        source_coverage="SI_ORIGINAL_PROTOCOL_RESULT_ADDENDUM_WORKTREE_ONLY",
        global_attempt_ledger_covered=False,
        may_reserve_or_execute=False,
    )


def review_seed_request(
    repo_root: Path, proposed_id: int,
    *, evidence: Sequence[DocumentaryMention] | None = None,
) -> SeedRequestReview:
    """ALWAYS fail closed; a no-hit is neither available nor approved."""
    if type(proposed_id) is not int or not 1000 <= proposed_id <= 999999:
        raise ValueError("invalid scientific seed ID")
    records = tuple(evidence) if evidence is not None else documented_mentions(repo_root)
    matching = tuple(x for x in records if x.candidate_id == proposed_id)
    known = proposed_id in MANDATORY_ANCHORS
    reasons = [
        "Global GitHub issue/PR comments and worker/local attempt logs are not exhaustively audited.",
        "No durable atomic attempt reservation exists.",
        "No frozen independently reviewed SI/PCMA scientific protocol or authorization exists.",
    ]
    if matching:
        reasons.insert(0, "Historical SI documentation mentions this candidate; usage requires manual adjudication.")
    if known:
        reasons.insert(0, "ID belongs to a documented consumed/interrupted original-SI attempt.")
    if not matching and not known:
        reasons.insert(0, "No local mention found is NOT proof of an unused seed.")
    return SeedRequestReview(
        proposed_id=proposed_id,
        conflicting_mentions=matching,
        known_consumed_or_interrupted=known,
        verdict="BLOCKED_UNVERIFIED",
        reasons=tuple(reasons),
        can_reserve=False,
    )


def audit_for_display(repo_root: Path) -> dict[str, object]:
    """Compact, deterministic, nonauthorizing local audit summary."""
    result = audit_original_si_docs(repo_root)
    return {
        "status": "UNVERIFIED_HISTORICAL_DOCUMENT_SCAN",
        "source_coverage": result.source_coverage,
        "document_count": result.source_file_count,
        "results_count": result.result_file_count,
        "protocols_count": result.protocol_file_count,
        "mention_count": result.mention_count,
        "candidate_count": len(result.candidate_ids),
        "missing_mandatory_anchors": list(result.missing_anchor_ids),
        "global_ledger_verified": False,
        "may_reserve_or_execute": False,
    }
