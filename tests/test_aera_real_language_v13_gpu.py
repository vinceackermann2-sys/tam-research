from __future__ import annotations

import pytest

from tam_research.aera_real_language_v13_gpu import SEED, train_matched_pair


def test_v13_gpu_harness_is_frozen_to_fresh_development_seed8251():
    assert SEED == 8251
    with pytest.raises(ValueError, match="frozen to seed 8251"):
        train_matched_pair(data_dir="unused", run_dir="unused", seed=8252)
