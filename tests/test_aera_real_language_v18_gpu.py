from __future__ import annotations

import torch

from tam_research import aera_real_language_v18_gpu as v18g


def test_v18_gpu_development_constants_are_frozen_and_seed_is_fresh() -> None:
    assert v18g.SEED == 8351
    assert v18g.EVAL_SEED == 98_351
    assert v18g.MEMORY_EVAL_SEED == 108_351
    assert v18g.SYSTEMS_EVAL_SEED == 118_351
    assert v18g.SYSTEM_BATCH_SIZES == (8, 64)
    assert v18g.MEMORY_EVAL_BATCHES == 16
    assert v18g.MEMORY_EVAL_BATCH_SIZE == 8
    expected_rates = torch.tensor(
        [0.5, 1.0 / 3.0, 1.0 / 6.0], dtype=v18g.TARGET_RATES.dtype
    )
    assert torch.equal(v18g.TARGET_RATES, expected_rates)


def test_v18_gpu_thresholds_are_preregistered() -> None:
    assert v18g.QUALITY_GAP_MAX_NLL == 0.50
    assert v18g.MEMORY_SECOND_CHUNK_MIN_ADVANTAGE_NLL == 0.005
    assert v18g.MEMORY_OVERALL_MIN_ADVANTAGE_NLL == 0.0
    assert v18g.WRITE_MEAN_MIN == 0.01
    assert v18g.WRITE_MEAN_MAX == 0.95
    assert v18g.WRITE_SPREAD_MIN == 0.01
    assert v18g.OPTIONAL_STAGE_TARGET_MAE_MAX == 0.12
    assert v18g.OPTIONAL_STAGE_MIN_RUN_FRACTION == 0.05
    assert v18g.TOTAL_STAGE_EXEC_MIN == 0.35
    assert v18g.TOTAL_STAGE_EXEC_MAX == 0.70
    assert v18g.BATCH8_MIN_SPEED_RATIO == 0.25
    assert v18g.BATCH64_MIN_SPEED_RATIO == 1.25


def test_gate_stats_detects_variation_and_empty_input() -> None:
    empty = v18g._gate_stats([])
    assert empty == {"mean": 0.0, "std": 0.0, "p10": 0.0, "p90": 0.0, "count": 0.0}

    stats = v18g._gate_stats([torch.tensor([[0.1], [0.3], [0.7], [0.9]])])
    assert 0.49 < stats["mean"] < 0.51
    assert stats["p90"] > stats["p10"]
    assert stats["count"] == 4.0


def test_second_chunk_nll_uses_only_later_chunk() -> None:
    chunk = 2
    vocab = v18g.VOCAB_SIZE
    y = torch.tensor([[0, 0, 1, 2]])
    logits = torch.zeros(1, 4, vocab)
    # Make first chunk deliberately terrible; second chunk effectively perfect.
    logits[:, :chunk, 2] = 40.0
    logits[0, 2, 1] = 40.0
    logits[0, 3, 2] = 40.0
    nll = v18g._second_chunk_nll(logits, y, chunk)
    assert nll < 1e-6
