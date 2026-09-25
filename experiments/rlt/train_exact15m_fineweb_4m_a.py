from __future__ import annotations

from typing import Any

from experiments.rlt import train_compiled_pair_1m as base
from experiments.rlt.model import RLTConfig
from tam_research.models import ModelConfig


SCIENTIFIC_SEED = 20_261_014
PAIRED_BATCH_SEED = 20_271_014
PAIRED_EVAL_SEED = 20_291_014
COMPILE_PROBE_SEED = 20_301_014
TOKEN_BUDGET = 4_194_304
EVAL_BATCHES = 64
EXPECTED_PARAMETERS = 15_129_344


def rlt_config_exact15m() -> RLTConfig:
    return RLTConfig(
        vocab_size=50_257,
        d_model=256,
        n_heads=8,
        n_stages=2,
        max_seq_len=128,
        ff_mult=4,
        swa_window=32,
    )


def transformer_config_exact15m() -> ModelConfig:
    return ModelConfig(
        vocab_size=50_257,
        d_model=256,
        n_layers=3,
        n_heads=8,
        max_seq_len=128,
        ff_mult=4,
        ff_inner=938,
        architecture="transformer",
    )


def train_exact15m_fineweb_4m_a(**kwargs: Any) -> dict[str, Any]:
    base.SCIENTIFIC_SEED = SCIENTIFIC_SEED
    base.PAIRED_BATCH_SEED = PAIRED_BATCH_SEED
    base.PAIRED_EVAL_SEED = PAIRED_EVAL_SEED
    base.COMPILE_PROBE_SEED = COMPILE_PROBE_SEED
    base.TOKEN_BUDGET = TOKEN_BUDGET
    base.EVAL_BATCHES = EVAL_BATCHES
    base.EXPECTED_RLT_PARAMETERS = EXPECTED_PARAMETERS
    base.EXPECTED_TRANSFORMER_PARAMETERS = EXPECTED_PARAMETERS
    base.MAX_PARAMETER_DELTA_FRACTION = 0.0
    base.rlt_config = rlt_config_exact15m
    base.transformer_config = transformer_config_exact15m

    result = base.train_compiled_pair_1m(**kwargs)
    delta = float(result["derived"]["rlt_minus_transformer_final_nll"])
    if result["rlt"]["parameters"] != EXPECTED_PARAMETERS:
        raise RuntimeError("exact15m RLT parameter count drift")
    if result["transformer"]["parameters"] != EXPECTED_PARAMETERS:
        raise RuntimeError("exact15m Transformer parameter count drift")
    if float(result["derived"]["transformer_parameter_delta_fraction"]) != 0.0:
        raise RuntimeError("exact15m pair is not exactly parameter matched")

    result["classification"] = (
        "PAIRED_EXACT15M_FINEWEB_4M_RLT_LOWER_NLL_SINGLE_RUN_HINT"
        if delta < 0
        else "PAIRED_EXACT15M_FINEWEB_4M_NO_RLT_QUALITY_ADVANTAGE"
    )
    result["breakthrough_claim_supported"] = False
    result["scale_test"] = {
        "rlt_parameters": EXPECTED_PARAMETERS,
        "transformer_parameters": EXPECTED_PARAMETERS,
        "parameter_delta": 0,
        "rlt_d_model": 256,
        "transformer_d_model": 256,
        "rlt_stages": 2,
        "transformer_layers": 3,
        "transformer_ff_inner": 938,
        "reference_small_model_parameters": 7_007_616,
    }
    result["interpretation_ceiling"] = (
        "Single exactly parameter-matched 15,129,344-parameter pair trained for ~4M tokens "
        "on the same FineWeb-Edu token shard used by the prior 7M D run. This tests whether "
        "the replicated small-model RLT quality signal survives roughly 2.16x parameter scale. "
        "One scale seed cannot establish a general scaling law or breakthrough."
    )
    return result
