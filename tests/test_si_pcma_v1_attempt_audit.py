"""Read-only SI source evidence inventory; NO scientific work executed."""
from pathlib import Path

import pytest

from tam_research.si_pcma_v1_attempt_audit import (
    MANDATORY_ANCHORS, KNOWN_CONSUMED_V25, KNOWN_CONSUMED_V26,
    _git_blob_bytes, audit_for_display, audit_original_si_docs,
    documented_mentions, review_seed_request,
)


def test_git_blob_is_content_addressed_and_matches_git_object_format():
    import hashlib
    text = b"sample\n"
    expected = hashlib.sha1(b"blob 7\0sample\n").hexdigest()
    assert _git_blob_bytes(text) == expected


def test_fixture_reports_possible_reuse_not_confirmed_attempt(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "SI_ORIGINAL_X_RESULTS.md").write_text(
        "original seeds 117211/117212\n"
        "attempt seed 118211 was reserved; 118212–118213 interrupted\n"
        "train tokens 140000 for reproducible steps\n"
        "IDs 99999 are not evidence\n",
        encoding="utf-8")
    mentions = documented_mentions(tmp_path)
    found = {m.candidate_id for m in mentions}
    assert {117211, 117212, 118211, 118212, 118213} <= found
    assert 140000 not in found and 99999 not in found
    assert all(m.file_path == "docs/SI_ORIGINAL_X_RESULTS.md" and
               len(m.document_git_blob) == 40 for m in mentions)
    audit = audit_original_si_docs(tmp_path)
    assert audit.source_file_count == 1 and audit.result_file_count == 1
    assert audit.global_attempt_ledger_covered is False
    assert audit.may_reserve_or_execute is False


def test_developer_fixture_range_expansion_is_strictly_bounded(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "SI_ORIGINAL_TEST_ADDENDUM.md").write_text(
        "engineering seeds 107010–107014\n"
        "engineering seeds 105000–200000 unverified interval\n",
        encoding="utf-8")
    numbers = {x.candidate_id for x in documented_mentions(tmp_path)}
    assert {107010,107011,107012,107013,107014} <= numbers
    assert 199999 not in numbers
    assert 105000 in numbers and 200000 in numbers


def test_known_consumed_attempts_are_never_recommended(tmp_path):
    (tmp_path / "docs").mkdir()
    for candidate in sorted(MANDATORY_ANCHORS):
        verdict = review_seed_request(tmp_path, candidate)
        assert verdict.known_consumed_or_interrupted
        assert verdict.verdict == "BLOCKED_UNVERIFIED"
        assert not verdict.can_reserve
        assert any("consumed/interrupted" in reason for reason in verdict.reasons)


def test_seed_with_no_document_hits_is_still_denied(tmp_path):
    (tmp_path / "docs").mkdir()
    verdict = review_seed_request(tmp_path, 333333)
    assert verdict.conflicting_mentions == ()
    assert not verdict.known_consumed_or_interrupted
    assert verdict.verdict == "BLOCKED_UNVERIFIED"
    assert not verdict.can_reserve
    assert any("NOT proof" in msg for msg in verdict.reasons)


def test_report_path_provenance_line_and_no_approval(tmp_path):
    (tmp_path / "docs").mkdir()
    f = tmp_path / "docs" / "SI_ORIGINAL_A_PROTOCOL.md"
    f.write_text("headline\nseed 100214 was registered\n", encoding="utf-8")
    m = documented_mentions(tmp_path)
    assert len(m) == 1 and m[0].line == 2
    assert m[0].candidate_id == 100214
    assert m[0].evidence_class == "POTENTIAL_SEED_ONLY"
    review = review_seed_request(tmp_path, 100214)
    assert len(review.conflicting_mentions) == 1
    assert not review.can_reserve
    display = audit_for_display(tmp_path)
    assert display["global_ledger_verified"] is False
    assert display["may_reserve_or_execute"] is False


def test_result_must_not_be_confused_with_scientific_seed(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "SI_ORIGINAL_TEST_RESULTS.md").write_text(
        "accuracy result 100211 was NOT a seed event here\n",
        encoding="utf-8")
    records = documented_mentions(tmp_path)
    assert records and records[0].candidate_id == 100211
    assert all(x.evidence_class == "POTENTIAL_SEED_ONLY" for x in records)
    # Conservative false positive -> block anyway. A document mention never
    # permits making a positive assertion about actual execution.


@pytest.mark.parametrize("bad", [-1, 0, 3.0, True, 999, 1000000])
def test_seed_request_requires_bounded_nonboolean_integer(tmp_path, bad):
    (tmp_path / "docs").mkdir()
    with pytest.raises(ValueError):
        review_seed_request(tmp_path, bad)


def test_missing_document_root_is_a_hard_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        documented_mentions(tmp_path)


def test_real_repo_report_has_historical_stops_and_documented_replacements():
    root = Path(__file__).resolve().parents[1]
    audit = audit_original_si_docs(root)
    records = documented_mentions(root)
    assert audit.source_file_count >= 30
    assert audit.result_file_count >= 20
    assert audit.protocol_file_count >= 20
    assert audit.mention_count == len(records) >= 100
    # Verify a deliberately narrow set of well-documented historical anchors.
    assert {117211,117212,117213,117214,118211,118212,118213} <= set(audit.candidate_ids)
    assert {99611,100211,103211,103212,115213,116211} <= set(audit.candidate_ids)
    assert not audit.global_attempt_ledger_covered
    assert not audit.may_reserve_or_execute
    assert review_seed_request(root, 118213).verdict == "BLOCKED_UNVERIFIED"
    assert review_seed_request(root, 999999).verdict == "BLOCKED_UNVERIFIED"
    assert len(audit.document_blobs) >= 20


def test_reference_stop_and_quarantine_are_retained():
    assert {117211,117212,117213,117214} <= KNOWN_CONSUMED_V25
    assert {118211,118212,118213} <= KNOWN_CONSUMED_V26
