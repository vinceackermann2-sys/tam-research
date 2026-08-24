from __future__ import annotations

import pytest
import torch

from tam_research import aera_v17_systems_probe as probe


def test_probe_is_frozen_to_seed8331_checkpoint() -> None:
    assert probe.CHECKPOINT_SEED == 8331
    assert probe.PROBE_SEED == 108_331
    assert probe.BATCH_SIZES == (1, 2, 4, 8, 16, 32)
    assert probe.WARMUP_ITERS == 3
    assert probe.TIMED_ITERS == 10


def test_wrong_checkpoint_seed_is_refused() -> None:
    with pytest.raises(RuntimeError, match="checkpoint seed mismatch"):
        probe._require_checkpoint_seed({"seed": 8332})


def test_cuda_benchmark_refuses_cpu() -> None:
    with pytest.raises(RuntimeError, match="requires CUDA"):
        probe._benchmark_cuda(lambda: None, device=torch.device("cpu"))


def test_batch_sweep_is_unique_ascending_and_bounded() -> None:
    assert tuple(sorted(set(probe.BATCH_SIZES))) == probe.BATCH_SIZES
    assert probe.BATCH_SIZES[0] == 1
    assert probe.BATCH_SIZES[-1] == 32
