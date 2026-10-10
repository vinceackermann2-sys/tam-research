"""CPU-only synthetic TokenBin sampling parity, never a real engineering result."""
from __future__ import annotations
import numpy as np
import pytest

from tam_research.cortex_attention8_conv7_data_stream_parity_v1 import (
    MODELS, SYNTHETIC_SEED, audit_frozen_data_stream_source,
    compare_synthetic_three_way_streams, synthetic_cpu_stream_digest,
)


@pytest.fixture
def shards(tmp_path):
    # Synthetic, small, disjoint token IDs, no frozen data, no network access.
    train = tmp_path / "mock_train.bin"
    heldout = tmp_path / "mock_val.bin"
    (np.arange(4096, dtype=np.uint16) % 100).tofile(train)
    (np.arange(4096, dtype=np.uint16) % 100 + 500).tofile(heldout)
    return train, heldout


def test_source_bound_rng_offsets_and_exposure_contract():
    result = audit_frozen_data_stream_source()
    assert result["classification"] == "CONV7_SOURCE_BOUND_SAMPLER_PARITY_CPU_ONLY"
    assert result["train_rng_offset"] == 10000
    assert result["compile_warmup_rng_offset"] == 99999
    assert result["final_eval_rng_offset"] == 30000
    assert result["training_tokens_per_optimizer_step"] == 65536
    assert result["final_heldout_tokens_per_model"] == 819200
    assert result["frozen_blobs_verified"]
    for key in ("engineering_seed_assigned", "gpu_allocated",
                "paid_run_authorized", "scientific_evidence"):
        assert result[key] is False


def test_three_model_sampler_parity_heldout_isolation_and_warmup_independence(shards):
    result = compare_synthetic_three_way_streams(*shards)
    assert result["classification"] == "CONV7_SYNTHETIC_THREE_MODEL_STREAM_PARITY_PASS_NOT_EVIDENCE"
    assert set(result["train_digests"]) == set(MODELS)
    assert set(result["heldout_digests"]) == set(MODELS)
    assert len(set(result["train_digests"].values())) == 1
    assert len(set(result["heldout_digests"].values())) == 1
    assert result["train_digests"][MODELS[0]] != result["heldout_digests"][MODELS[0]]
    assert result["synthetic_seed_consumes_science"] is False
    assert result["gpu_allocated"] is False
    assert result["scientific_evidence"] is False


def test_independent_sampler_is_restart_deterministic(shards):
    train, val = shards
    before = synthetic_cpu_stream_digest(train, 10000)
    eval_digest = synthetic_cpu_stream_digest(val, 30000)
    warmup_digest = synthetic_cpu_stream_digest(train, 99999)
    after = synthetic_cpu_stream_digest(train, 10000)
    assert before == after
    assert before != warmup_digest
    assert before != eval_digest
    assert SYNTHETIC_SEED not in (60231, 60232)


@pytest.mark.parametrize("offset", (-1, 0, 20000, 60232))
def test_invalid_sampling_seed_offset_rejected(shards, offset):
    with pytest.raises(ValueError, match="offset"):
        synthetic_cpu_stream_digest(shards[0], offset)


def test_cannot_confuse_training_and_validation_shards(shards):
    with pytest.raises(ValueError, match="distinct"):
        compare_synthetic_three_way_streams(shards[0], shards[0])


def test_no_modal_gpu_or_training_dispatch():
    from pathlib import Path
    source = (Path(__file__).resolve().parents[1] /
              "tam_research/cortex_attention8_conv7_data_stream_parity_v1.py").read_text()
    # Check actual executable interfaces instead of self-scanning this test.
    for forbidden in ("import modal", "modal run", ".remote(", "gpu=",
                      "torch.compile(", "optimizer.step(", "h100_train_one(",
                      ".write_text(", ".write_bytes(", "subprocess."):
        assert forbidden not in source
