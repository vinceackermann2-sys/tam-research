"""Constructor validation only; no forward pass, optimizer or research run."""
from __future__ import annotations

import pytest

from experiments.rlt.model import parameter_count
from experiments.rlt.model_sliding_window_ax import (
    SlidingWindowConfig, SlidingWindowTransformer, tiny_sliding_window_config,
)
from experiments.rlt.model_streaming_bounded import (
    BoundedChunkGatedRLT, BoundedStreamingConfig, tiny_streaming_config,
)


INVALID_DIMENSIONS = [
    True, False, 1.0, 1.5, 8.0, None, "8", float("nan"), float("inf"),
]


def forbidden_model_initialization(*args, **kwargs):
    raise AssertionError("invalid configuration reached model initialization")


@pytest.mark.parametrize("chunk_size", INVALID_DIMENSIONS)
def test_bad_chunk_type_is_rejected_before_weight_allocation(monkeypatch, chunk_size):
    monkeypatch.setattr(BoundedChunkGatedRLT, "__init__", forbidden_model_initialization)
    with pytest.raises(ValueError, match="chunk_size must be a positive integer"):
        BoundedChunkGatedRLT(BoundedStreamingConfig(
            tiny_streaming_config().rlt, chunk_size=chunk_size,
        ))


@pytest.mark.parametrize("window", INVALID_DIMENSIONS)
def test_bad_window_type_is_rejected_before_weight_allocation(monkeypatch, window):
    monkeypatch.setattr(SlidingWindowTransformer, "__init__", forbidden_model_initialization)
    with pytest.raises(ValueError, match="window must be a positive integer"):
        SlidingWindowTransformer(SlidingWindowConfig(window=window))


@pytest.mark.parametrize("chunk_size", [1, 4, 8])
def test_valid_bounded_chunk_sizes_preserve_config(chunk_size):
    original = tiny_streaming_config()
    cfg = BoundedStreamingConfig(original.rlt, chunk_size=chunk_size)
    assert cfg.rlt == original.rlt
    assert cfg.chunk_size == chunk_size


@pytest.mark.parametrize("window", [1, 4, 8])
def test_valid_sliding_windows_preserve_config(window):
    cfg = SlidingWindowConfig(window=window)
    assert cfg.window == window
    assert cfg.width == 32 and cfg.layers == 2 and cfg.ff_inner == 145


def test_historical_tiny_configs_preserve_architecture_and_parameter_counts():
    cfg = tiny_streaming_config()
    carry = BoundedChunkGatedRLT(cfg, carry_between_chunks=True)
    reset = BoundedChunkGatedRLT(cfg, carry_between_chunks=False)
    sliding = SlidingWindowTransformer(tiny_sliding_window_config())
    assert cfg.chunk_size == cfg.rlt.max_seq_len == cfg.rlt.swa_window == 8
    assert parameter_count(carry) == parameter_count(reset) == 27_712
    assert parameter_count(sliding) == 27_680
    assert sliding.cfg.window == 8 and sliding.receptive_distance == 14
