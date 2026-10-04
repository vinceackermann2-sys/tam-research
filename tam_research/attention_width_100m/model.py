from __future__ import annotations

from tam_research.models import ModelConfig, ResearchLM, parameter_count
from tam_research.attention_width_factorial.model import (
    WidthFactorialConfig,
    WidthFactorialResearchLM,
)

from .protocol import (
    BASE_FF_HIDDEN,
    D_MODEL,
    MAX_SEQ_LEN,
    N_HEADS,
    N_LAYERS,
    REDUCED_ATTN_INNER,
    REDUCED_FF_HIDDEN,
    VOCAB_SIZE,
)


def baseline_100m_config(*, max_seq_len: int = MAX_SEQ_LEN) -> ModelConfig:
    cfg = ModelConfig(
        architecture="transformer",
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        max_seq_len=max_seq_len,
        ff_mult=4,
    )
    if BASE_FF_HIDDEN != 4 * D_MODEL:
        raise AssertionError("baseline FF geometry drifted")
    return cfg


def reduced_100m_config(*, max_seq_len: int = MAX_SEQ_LEN) -> WidthFactorialConfig:
    return WidthFactorialConfig(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        max_seq_len=max_seq_len,
        attention_inner=REDUCED_ATTN_INNER,
        ff_hidden=REDUCED_FF_HIDDEN,
        chunk_size=64,
        locality="global",
        architecture="attention_width_factorial",
    )


def build_baseline_100m(*, max_seq_len: int = MAX_SEQ_LEN) -> ResearchLM:
    return ResearchLM(baseline_100m_config(max_seq_len=max_seq_len))


def build_reduced_100m(
    *,
    max_seq_len: int = MAX_SEQ_LEN,
) -> WidthFactorialResearchLM:
    return WidthFactorialResearchLM(
        reduced_100m_config(max_seq_len=max_seq_len)
    )


def baseline_100m_parameter_count() -> int:
    return parameter_count(build_baseline_100m())


def reduced_100m_parameter_count() -> int:
    return parameter_count(build_reduced_100m())
