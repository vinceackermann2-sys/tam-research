from __future__ import annotations

from tam_research.aera_v17_systems_backend_benchmark import (
    BATCH_SIZES,
    CHECKPOINT_SEED,
    PROBE_SEED,
)


def test_backend_benchmark_constants_are_frozen() -> None:
    assert CHECKPOINT_SEED == 8331
    assert PROBE_SEED == 118_331
    assert BATCH_SIZES == (8, 16, 32, 64)
    assert tuple(sorted(set(BATCH_SIZES))) == BATCH_SIZES
