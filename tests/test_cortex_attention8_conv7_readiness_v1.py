"""Tests are CPU-only; there must be no path to training or paid allocation."""
from pathlib import Path
import pytest
from scripts.cortex_attention8_conv7_readiness_v1 import (
    EXPECTED_BLOBS,
    _git_blob_sha,
    check_readiness,
)


def test_readiness_reports_frozen_source_and_unfunded_execution():
    report = check_readiness()
    assert report["classification"] == "CONV7_THREE_WAY_CPU_READINESS_PASS_PAID_EXECUTION_BLOCKED"
    assert report["source_locked"] is True
    assert report["architecture_cpu_contract_verified"] is True
    assert report["per_model_optimizer_steps"] == 3052
    assert report["per_model_token_exposures"] == 200015872
    assert report["full_horizon_optimizer_steps"] == 30518
    assert 9.8 < report["full_horizon_lr_at_pilot_endpoint"] / report["incorrect_short_horizon_lr_at_pilot_endpoint"] < 10
    assert report["comparators"] == [
        "fresh_transformer",
        "reduced_attention_dense_v2_attention8",
        "attention8_causal_depthwise_conv7",
    ]
    for flag in ("engineering_seed_assigned", "strict_gpu_spend_cap_assigned",
                 "gpu_launch_permitted", "writes_performed", "gpu_allocated", "seed_consumed"):
        assert report[flag] is False
    assert report["git_blobs"] == EXPECTED_BLOBS


def test_source_lock_rejects_any_modified_scientific_input(tmp_path):
    for filename in EXPECTED_BLOBS:
        source = Path(__file__).resolve().parents[1] / filename
        dst = tmp_path / filename
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(source.read_bytes())
    changed = tmp_path / "research/reduced_attention/attention8_conv7_engineering_proposal_v1.json"
    changed.write_bytes(changed.read_bytes() + b" ")
    with pytest.raises(RuntimeError, match="source-lock mismatch"):
        check_readiness(tmp_path)


def test_readiness_uses_no_modal_or_training_entrypoints():
    source = (Path(__file__).resolve().parents[1] /
              "scripts/cortex_attention8_conv7_readiness_v1.py").read_text(encoding="utf-8")
    for prohibited in ("import modal", "gpu=", "cuda", ".remote(", "modal run",
                       "optimizer.step(", "backward(", "reserve_pair_dispatch(",
                       ".write_bytes(", ".write_text(", ".commit("):
        assert prohibited not in source
    assert _git_blob_sha(b"") == "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"
