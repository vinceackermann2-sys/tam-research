from __future__ import annotations

from typing import Any

from experiments.rlt import train_compiled_pair_1m as base

SCIENTIFIC_SEED = 20_261_011
PAIRED_BATCH_SEED = 20_271_011
PAIRED_EVAL_SEED = 20_291_011
COMPILE_PROBE_SEED = 20_301_011
TOKEN_BUDGET = 4_194_304
EVAL_BATCHES = 64


def train_compiled_pair_4m_d(**kwargs: Any) -> dict[str, Any]:
    base.SCIENTIFIC_SEED = SCIENTIFIC_SEED
    base.PAIRED_BATCH_SEED = PAIRED_BATCH_SEED
    base.PAIRED_EVAL_SEED = PAIRED_EVAL_SEED
    base.COMPILE_PROBE_SEED = COMPILE_PROBE_SEED
    base.TOKEN_BUDGET = TOKEN_BUDGET
    base.EVAL_BATCHES = EVAL_BATCHES
    result = base.train_compiled_pair_1m(**kwargs)
    delta = float(result["derived"]["rlt_minus_transformer_final_nll"])
    result["classification"] = (
        "PAIRED_4M_INDEPENDENT_TRAIN_SHARD_RLT_LOWER_NLL_SINGLE_RUN_HINT"
        if delta < 0
        else "PAIRED_4M_INDEPENDENT_TRAIN_SHARD_NO_RLT_QUALITY_ADVANTAGE"
    )
    result["breakthrough_claim_supported"] = False
    result["interpretation_ceiling"] = (
        "Single matched ~4M-token run on a newly shuffled FineWeb-Edu training shard at "
        "~7M parameters. A lower RLT validation NLL would show that the replicated A/B/C "
        "effect is not tied to the original seed-1234 training-token sample, but it remains "
        "one dataset family and cannot establish a general architecture breakthrough."
    )
    return result
