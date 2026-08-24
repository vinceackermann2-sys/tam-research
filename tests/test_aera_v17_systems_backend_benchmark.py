from __future__ import annotations

import torch

from tam_research.aera_v17_systems_backend_benchmark import (
    BATCH_SIZES,
    CHECKPOINT_SEED,
    LOGIT_COMPARE_BATCH_SLICE,
    PROBE_SEED,
    _compare_logits_memory_bounded,
)


def test_backend_benchmark_constants_are_frozen() -> None:
    assert CHECKPOINT_SEED == 8331
    assert PROBE_SEED == 118_331
    assert BATCH_SIZES == (8, 16, 32, 64)
    assert tuple(sorted(set(BATCH_SIZES))) == BATCH_SIZES
    assert LOGIT_COMPARE_BATCH_SLICE == 1


def test_memory_bounded_logit_comparison_exact_cpu() -> None:
    torch.manual_seed(1)
    reference = torch.randn(5, 7, 11, dtype=torch.bfloat16)
    optimized = reference.clone()
    reference_argmax = reference.argmax(dim=-1)
    delta, agreement = _compare_logits_memory_bounded(
        reference,
        optimized,
        reference_argmax,
        batch_slice=1,
    )
    assert delta == 0.0
    assert agreement == 1.0


def test_memory_bounded_logit_comparison_detects_delta_and_argmax_change_cpu() -> None:
    reference = torch.zeros(3, 2, 5, dtype=torch.float32)
    reference[..., 0] = 1.0
    optimized = reference.clone()
    optimized[1, 0, 0] = -2.0
    optimized[1, 0, 1] = 3.0
    reference_argmax = reference.argmax(dim=-1)
    delta, agreement = _compare_logits_memory_bounded(
        reference,
        optimized,
        reference_argmax,
        batch_slice=1,
    )
    assert delta == 3.0
    assert agreement == 5.0 / 6.0


def test_memory_bounded_logit_comparison_rejects_nonpositive_slice() -> None:
    x = torch.zeros(1, 2, 3)
    try:
        _compare_logits_memory_bounded(x, x, x.argmax(dim=-1), batch_slice=0)
    except ValueError as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("expected ValueError")
