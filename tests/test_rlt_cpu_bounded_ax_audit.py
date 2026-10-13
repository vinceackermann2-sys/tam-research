"""AX zero-GPU counterfactual and inspection tests, not AW reruns."""
import math

import torch

from experiments.rlt.cpu_bounded_ax_audit import (
    frozen_aw_class_bias, prove_unreachable_counterfactuals, scan_retention_trace,
)
from experiments.rlt.cpu_lastwrite_probe_ar import make_batch
from experiments.rlt.model_sliding_window_ax import SlidingWindowTransformer
from experiments.rlt.model_streaming_bounded import BoundedChunkGatedRLT, tiny_streaming_config


def test_final_causal_15_token_suffix_is_provably_uninformative():
    torch.manual_seed(234)
    rows = prove_unreachable_counterfactuals(SlidingWindowTransformer())
    assert set(rows) == {"16", "64", "127"}
    for row in rows.values():
        assert row == {
            "paired_examples": 4, "identical_final_tokens": 15,
            "opposite_labels": True, "invariant_transformer_logits": True,
        }


def test_frozen_aw_bias_read_only():
    rows = frozen_aw_class_bias()
    assert rows["127"]["accuracy"] == 0.5390625
    assert rows["127"]["class0_accuracy"] == 0.9453125
    assert rows["127"]["class1_accuracy"] == 0.1328125
    assert rows["127"]["class_gap"] > 0.8


def test_initialization_only_gate_diagnostics_are_bounded_and_finite():
    torch.manual_seed(987)
    model = BoundedChunkGatedRLT(tiny_streaming_config())
    x, _, _ = make_batch(2, (64,), 20331081, seq_len=128)
    r = scan_retention_trace(model, x, last_write_distance=64)
    assert r["trace_type"] == "weights_supplied_by_caller_not_aw_checkpoint"
    assert r["state_shape"] == [2, 32]
    assert len(r["state_norm_by_token"]) == len(r["write_norm_by_token"]) == 128
    assert 0 <= r["mean_gate"] <= 1
    assert 0 <= r["mean_direct_retention_over_gap"] <= 1
    assert all(math.isfinite(v) for v in r["state_norm_by_token"])
    assert all(math.isfinite(v) for v in r["write_norm_by_token"])
