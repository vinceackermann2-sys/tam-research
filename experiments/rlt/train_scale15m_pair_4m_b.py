from __future__ import annotations

from typing import Any

from experiments.rlt import train_compiled_pair_1m as base
from experiments.rlt.model import RLTConfig
from tam_research.models import ModelConfig

SCIENTIFIC_SEED = 20_261_015
PAIRED_BATCH_SEED = 20_271_015
PAIRED_EVAL_SEED = 20_291_015
COMPILE_PROBE_SEED = 20_301_015
TOKEN_BUDGET = 4_194_304
EVAL_BATCHES = 64
EXPECTED_PARAMETERS = 15_129_344


def scale_rlt_config() -> RLTConfig:
    return RLTConfig(
        vocab_size=50_257,
        d_model=256,
        n_heads=8,
        n_stages=2,
        max_seq_len=128,
        ff_mult=4,
        swa_window=32,
    )


def scale_transformer_config() -> ModelConfig:
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


def train_scale15m_pair_4m_b(**kwargs: Any) -> dict[str, Any]:
    base.SCIENTIFIC_SEED = SCIENTIFIC_SEED
    base.PAIRED_BATCH_SEED = PAIRED_BATCH_SEED
    base.PAIRED_EVAL_SEED = PAIRED_EVAL_SEED
    base.COMPILE_PROBE_SEED = COMPILE_PROBE_SEED
    base.TOKEN_BUDGET = TOKEN_BUDGET
    base.EVAL_BATCHES = EVAL_BATCHES
    base.rlt_config = scale_rlt_config
    base.transformer_config = scale_transformer_config
    base.EXPECTED_RLT_PARAMETERS = EXPECTED_PARAMETERS
    base.EXPECTED_TRANSFORMER_PARAMETERS = EXPECTED_PARAMETERS

    result = base.train_compiled_pair_1m(**kwargs)
    rp = int(result["rlt"]["parameters"])
    tp = int(result["transformer"]["parameters"])
    if rp != EXPECTED_PARAMETERS or tp != EXPECTED_PARAMETERS or rp != tp:
        raise RuntimeError(f"exact scale parameter match failed: rlt={rp}, transformer={tp}")

    delta = float(result["derived"]["rlt_minus_transformer_final_nll"])
    result["classification"] = (
        "PAIRED_SCALE15M_4M_RLT_LOWER_NLL_SINGLE_RUN_HINT"
        if delta < 0
        else "PAIRED_SCALE15M_4M_NO_RLT_QUALITY_ADVANTAGE"
    )
    result["breakthrough_claim_supported"] = False
    result["scale_test"] = {
        "rlt_parameters": rp,
        "transformer_parameters": tp,
        "parameter_delta": tp - rp,
        "parameter_delta_fraction": 0.0,
        "rlt_d_model": 256,
        "rlt_stages": 2,
        "transformer_d_model": 256,
        "transformer_layers": 3,
        "transformer_ff_inner": 938,
    }
    result["interpretation_ceiling"] = (
        "Fresh-seed exact-parameter-matched ~15.1M scale test at the established 4M-token budget. "
        "A positive result would show the small-model quality effect survives roughly 2.16x "
        "parameter scale without parameter-count confounding. It remains a single seed and "
        "cannot support a breakthrough claim until independently replicated and tested at "
        "larger training-token and compute-matched scales."
    )
    return result
