"""PCMA-v1 read-only scientific preflight contract tests, NOT attempts."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from tam_research.si_pcma_v1_science_preflight import (
    CANONICAL_ORIGINAL_HYBRID, CANONICAL_ORIGINAL_PATH, KNOWN_CONSUMED_SI,
    _CODE_PATHS, draft_manifest, git_blob, seal_draft_document,
    validate_review_manifest,
)
from tam_research.si_pcma_v1_cost_profile import (
    profile_public_shape, summarize_nominal_parity, DiagnosticCPUCost,
)


DUMMY_PIN = "0" * 40
FAKE_PARENT_COMMIT = "1" * 40


def _review_doc():
    paths = {name: DUMMY_PIN for name in _CODE_PATHS}
    paths[CANONICAL_ORIGINAL_PATH] = CANONICAL_ORIGINAL_HYBRID
    return draft_manifest(FAKE_PARENT_COMMIT, paths)


def test_draft_manifest_has_no_seed_or_authorization_ever():
    manifest = _review_doc()
    assert manifest["status"] == "DRAFT_UNAPPROVED"
    assert manifest["scientific_seeds"] == []
    assert manifest["reserved_attempts"] == []
    assert manifest["explicit_authorization"] is False
    verdict = validate_review_manifest(manifest)
    assert verdict.status == "BLOCKED"
    assert verdict.may_execute is False
    assert verdict.new_scientific_seeds_reserved == 0
    assert verdict.paid_compute_authorized is False
    assert any("ledger" in x for x in verdict.blockers)
    assert any("heldout" in x for x in verdict.blockers)


def test_cannot_self_approve_or_erase_negative_science():
    m = _review_doc()
    m["explicit_authorization"] = True
    m["status"] = "APPROVED"
    m["negative_evidence"] = {
        "original_v26": "GO", "v25_seed_117214_duplicate": "VALID"}
    m["scientific_seeds"] = [117214]
    m["reserved_attempts"] = [{"seed": 117214}]
    verdict = validate_review_manifest(m)
    assert verdict.status == "BLOCKED"
    assert not verdict.may_execute
    assert any("consumed" in x for x in verdict.blockers)
    assert any("negative scientific evidence" in x for x in verdict.blockers)
    assert any("approval cannot be" in x for x in verdict.blockers)


def test_original_source_is_immutable_and_git_hash_works():
    root = Path(__file__).resolve().parents[1]
    actual = git_blob(root / CANONICAL_ORIGINAL_PATH, repo_root=root)
    assert actual == CANONICAL_ORIGINAL_HYBRID


def test_source_tampering_and_unknown_fields_are_blocked():
    m = _review_doc()
    m["source_blobs"][CANONICAL_ORIGINAL_PATH] = DUMMY_PIN
    m["unexpected_override"] = True
    verdict = validate_review_manifest(m)
    assert any("original canonical SI" in x for x in verdict.blockers)
    assert any("unapproved extra" in x for x in verdict.blockers)
    with pytest.raises(ValueError):
        draft_manifest(FAKE_PARENT_COMMIT, m["source_blobs"])


def test_missing_pin_and_partial_registration_cannot_execute():
    m = _review_doc()
    m["source_blobs"].pop("tam_research/si_pcma_v1_neural_probe.py")
    m["training_recipe"] = {"updates": 9999}
    verdict = validate_review_manifest(m)
    assert verdict.status == "BLOCKED"
    assert any("source file pins" in x for x in verdict.blockers)
    assert any("unapproved field" in x for x in verdict.blockers)
    assert verdict.source_files_checked == 0


def test_seed_exclusion_list_is_a_floor_not_exhaustive_permission():
    assert {117211, 117212, 117213, 117214, 118211, 118212, 118213} <= KNOWN_CONSUMED_SI
    m = _review_doc()
    m["scientific_seeds"] = [999_999_999]  # Not proof of available!
    verdict = validate_review_manifest(m)
    assert verdict.status == "BLOCKED"
    assert any("seeds cannot" in x for x in verdict.blockers)
    assert any("not exhaustively" in x for x in verdict.blockers)


def test_nonexecutable_draft_seal_is_content_addressed():
    m = _review_doc()
    blob = seal_draft_document(m)
    doc = json.loads(blob)
    assert doc["sha256"] == hashlib.sha256(
        json.dumps(m, sort_keys=True, separators=(",", ":"),
                   allow_nan=False).encode()).hexdigest()
    assert b"NO SCIENTIFIC EXECUTION AUTHORIZED" in blob
    assert blob == seal_draft_document(m)


def test_bare_input_cannot_claim_experiment_permission():
    assert validate_review_manifest({}).may_execute is False
    assert validate_review_manifest(None).status == "REJECTED"


def test_cpu_forward_profiles_all_three_with_explicitly_partial_flops():
    import torch
    from tam_research.si_pcma_v1_factual_learning import FactualRewardPredictor
    from tam_research.si_pcma_v1_fair_comparison import PCMAGRURewardPredictor
    torch.manual_seed(0)  # Engineering fixture; not scientific attempt seed.
    torch.set_num_threads(2)
    models = {
        "hybrid": FactualRewardPredictor("hybrid"),
        "transformer": FactualRewardPredictor("transformer"),
        "gru": PCMAGRURewardPredictor(),
    }
    reports = {name: profile_public_shape(name, model, tokens=2)
               for name, model in models.items()}
    assert all(r.tokens == 2 and r.total_parameters > 0 for r in reports.values())
    assert all(r.parameter_bytes > 0 and r.elapsed_cpu_ns > 0 for r in reports.values())
    assert all(r.flops_complete is False and
               r.equal_active_compute_verified is False for r in reports.values())
    details = summarize_nominal_parity(reports)
    assert details["same_synthetic_input_length"]
    assert not details["active_flops_matched"]
    assert not details["matched_scientific_training_compute"]
    assert not details["valid_scientific_quality_result"]
    # Source v26 parity was close; wrapper heads are also identical shapes.
    assert details["hybrid_vs_transformer_below_0_1pct_parameter_difference"]


def test_cost_profiler_denies_unbounded_shapes_and_invented_flop_certification():
    import torch
    from tam_research.si_pcma_v1_fair_comparison import PCMAGRURewardPredictor
    model = PCMAGRURewardPredictor()
    with pytest.raises(ValueError):
        profile_public_shape("gru", model, tokens=17)
    with pytest.raises(ValueError):
        profile_public_shape("gru", model, tokens=0)
    fake = DiagnosticCPUCost(
        "gru", 1, 50, 200, 0, 10, 10, 1, False, False)
    with pytest.raises(ValueError):
        summarize_nominal_parity({"gru": fake})
    invented = replace(fake, architecture="hybrid",
                       flops_complete=True, equal_active_compute_verified=True)
    with pytest.raises(ValueError, match="cannot certify"):
        summarize_nominal_parity({
            "gru": fake, "hybrid": invented,
            "transformer": replace(fake, architecture="transformer"),
        })
