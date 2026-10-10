"""CPU-only source lock contract and fail-closed negative tests."""
import json
from pathlib import Path
import shutil

import pytest

from scripts.cortex_attention8_conv7_execution_lock_readonly_v1 import (
    ROOT, LOCK_PATH, PROPOSAL_PATH, check_execution_lock,
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
    with pytest.raises(RuntimeError, match="execution policy"):
        check_execution_lock(tmp_path)


def test_wrong_heldout_token_count_rejected(tmp_path):
    obj = clone_fixture(tmp_path)
    obj["pilot"]["final_eval_tokens"] = 1
    (tmp_path / LOCK_PATH).write_text(json.dumps(obj))
    with pytest.raises(RuntimeError, match="final heldout"):
        check_execution_lock(tmp_path)


def test_modified_frozen_proposal_rejected(tmp_path):
    clone_fixture(tmp_path)
    path = tmp_path / PROPOSAL_PATH
    proposal = json.loads(path.read_text())
    proposal["authority"]["gpu_authorized"] = True
    path.write_text(json.dumps(proposal))
    with pytest.raises(RuntimeError, match="protected source drift"):
        check_execution_lock(tmp_path)


def test_validator_contains_no_paid_dispatch_path():
    source = (ROOT / "scripts/cortex_attention8_conv7_execution_lock_readonly_v1.py").read_text()
    for token in ("import modal", ".remote(", "modal run", "gpu=", "cuda",
                  "optimizer.step(", ".write_text(", ".write_bytes(", "subprocess.", "requests."):
        assert token not in source
