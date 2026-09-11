from __future__ import annotations

import pytest
import torch

from tam_research.chm_v1_small_lm import RETRIEVAL_HOPS, SCIENTIFIC_SEEDS
from tam_research.chm_v1_small_lm_protocol import (
    GRAD_ACCUM,
    LOCAL_WINDOW,
    MICRO_BATCH,
    SESSION_LEN,
    TOKENS_PER_STEP,
    TOTAL_STEPS,
    build_scientific_pair,
    protocol_preflight,
)


def test_protocol_preserves_inherited_global_token_accounting() -> None:
    assert LOCAL_WINDOW == 512
    assert RETRIEVAL_HOPS == 2
    assert SESSION_LEN == 1024
    assert MICRO_BATCH == 4
    assert GRAD_ACCUM == 4
    assert TOKENS_PER_STEP == 8 * 512 * 4 == 16_384
    assert TOTAL_STEPS == 512


def test_protocol_preflight_is_zero_credit_parameter_matched_and_two_hop() -> None:
    result = protocol_preflight()
    assert result["classification"] == "IMPLEMENTATION_PREFLIGHT_ONLY_NO_GPU_AUTHORITY"
    assert result["gpu_authorized"] is False
    assert result["token_budget_per_model"] == 8_388_608
    assert result["tokens_per_step"] == 16_384
    assert result["retrieval_hops"] == 2
    assert result["parameter_accounting"]["within_preregistered_one_percent"] is True


def test_reserved_scientific_seeds_require_separate_authorization() -> None:
    for seed in SCIENTIFIC_SEEDS:
        with pytest.raises(RuntimeError, match="separate paid-run authorization"):
            build_scientific_pair(seed, torch.device("cpu"))
