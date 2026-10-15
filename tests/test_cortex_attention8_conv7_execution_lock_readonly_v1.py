"""CPU-only source lock contract and fail-closed negative tests."""
import json
import hashlib
from pathlib import Path
import shutil

import pytest

from scripts.cortex_attention8_conv7_execution_lock_readonly_v1 import (
    ROOT, LOCK_PATH, PROPOSAL_PATH, blob_sha, check_execution_lock,
)


def clone_fixture(tmp_path):
    obj = json.loads((ROOT / LOCK_PATH).read_text())
    for source in set(obj["trusted_source_blobs"]) | {LOCK_PATH}:
        path = tmp_path / source
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / source, path)
    return obj


def test_read_only_manifest_pass_denies_all_real_execution():
    report = check_execution_lock()
    assert report["classification"] == "CONV7_EXECUTION_LOCK_CPU_PASS_GPU_BLOCKED"
    assert report["source_blobs_checked"] == 13
    assert report["three_model_token_exposures"] == 600047616
    assert report["heldout_tokens_per_model"] == 819200
    assert len(report["source_lock_sha256"]) == 64
    for field in ("real_corpus_bytes_verified", "real_modal_volume_reservation_verified",
                  "paid_dispatch_permitted", "seed_consumed", "scientific_evidence",
                  "breakthrough_claim_permitted"):
        assert report[field] is False


def test_modified_real_source_blob_fails_closed(tmp_path):
    clone_fixture(tmp_path)
    path = tmp_path / "architectures/cortex_s/attention8_causal_conv7_v1.py"
    path.write_bytes(path.read_bytes() + b"\n# synthetic tamper\n")
    with pytest.raises(RuntimeError, match="protected source drift"):
        check_execution_lock(tmp_path)


@pytest.mark.parametrize("key,value", [
    ("seed",123456),("max_spend_usd",3.0),("approved",True),
    ("paid_dispatch_permitted",True),("real_reservation_created",True),
    ("resume",True),("retries",1),("job_id","fake-job")
])
def test_modified_manifest_authority_rejected(tmp_path,key,value):
    obj = clone_fixture(tmp_path)
    obj["execution"][key] = value
    (tmp_path / LOCK_PATH).write_text(json.dumps(obj))
    with pytest.raises(RuntimeError, match="protected source-lock manifest drift"):
        check_execution_lock(tmp_path)


def test_wrong_heldout_token_count_rejected(tmp_path):
    obj = clone_fixture(tmp_path)
    obj["pilot"]["final_eval_tokens"] = 1
    (tmp_path / LOCK_PATH).write_text(json.dumps(obj))
    with pytest.raises(RuntimeError, match="protected source-lock manifest drift"):
        check_execution_lock(tmp_path)


def test_modified_frozen_proposal_rejected(tmp_path):
    clone_fixture(tmp_path)
    path = tmp_path / PROPOSAL_PATH
    proposal = json.loads(path.read_text())
    proposal["authority"]["gpu_authorized"] = True
    path.write_text(json.dumps(proposal))
    with pytest.raises(RuntimeError, match="protected source drift"):
        check_execution_lock(tmp_path)


def replace_candidate_source_lock(tmp_path):
    obj = clone_fixture(tmp_path)
    candidate = "architectures/cortex_s/attention8_causal_conv7_v1.py"
    path = tmp_path / candidate
    path.write_bytes(path.read_bytes() + b"\n# synthetic source mutation\n")
    obj["trusted_source_blobs"].pop(candidate)
    replacement = "synthetic-placeholder.txt"
    payload = b"synthetic unrelated source-lock fixture\n"
    (tmp_path / replacement).write_bytes(payload)
    obj["trusted_source_blobs"][replacement] = blob_sha(payload)
    assert len(obj["trusted_source_blobs"]) == 13
    return obj


def test_removed_candidate_replaced_by_valid_hash_cannot_hide_source_mutation(tmp_path):
    obj = replace_candidate_source_lock(tmp_path)
    # The old checker accepted this self-consistent thirteen-entry map while
    # never examining the modified candidate. Its manifest is not authority.
    (tmp_path / LOCK_PATH).write_text(json.dumps(obj))
    with pytest.raises(RuntimeError, match="protected source-lock manifest drift"):
        check_execution_lock(tmp_path)


def test_exact_source_paths_required_even_with_a_new_manifest_digest(tmp_path, monkeypatch):
    import scripts.cortex_attention8_conv7_execution_lock_readonly_v1 as validator
    obj = replace_candidate_source_lock(tmp_path)
    raw = json.dumps(obj).encode()
    (tmp_path / LOCK_PATH).write_bytes(raw)
    # Exercise the independent path-set guard if a later review updates the
    # manifest digest; removing the candidate must still fail closed.
    monkeypatch.setattr(validator, "EXPECTED_LOCK_SHA256", hashlib.sha256(raw).hexdigest())
    with pytest.raises(RuntimeError, match="exact protected source paths"):
        check_execution_lock(tmp_path)


def test_self_updated_proposal_and_manifest_cannot_authorize_100x_lr_change(tmp_path):
    obj = clone_fixture(tmp_path)
    path = tmp_path / PROPOSAL_PATH
    proposal = json.loads(path.read_text())
    proposal["draft_engineering_pilot"]["learning_rate"] = 0.03
    obj["pilot"]["lr_peak"] = 0.03
    payload = json.dumps(proposal).encode()
    path.write_bytes(payload)
    obj["trusted_source_blobs"][PROPOSAL_PATH] = blob_sha(payload)
    (tmp_path / LOCK_PATH).write_text(json.dumps(obj))
    # Internal proposal/hash/schedule consistency cannot authenticate a changed
    # scientific setting or replace the immutable reviewed manifest.
    with pytest.raises(RuntimeError, match="protected source-lock manifest drift"):
        check_execution_lock(tmp_path)


def test_validator_contains_no_paid_dispatch_path():
    source = (ROOT / "scripts/cortex_attention8_conv7_execution_lock_readonly_v1.py").read_text()
    for token in ("import modal", ".remote(", "modal run", "gpu=", "cuda",
                  "optimizer.step(", ".write_text(", ".write_bytes(", "subprocess.", "requests."):
        assert token not in source
