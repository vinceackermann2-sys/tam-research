from __future__ import annotations

from typing import Any

from experiments.rlt import train_compiled_pair_1m as base

SCIENTIFIC_SEED = 20_261_010
PAIRED_BATCH_SEED = 20_271_010
PAIRED_EVAL_SEED = 20_291_010
COMPILE_PROBE_SEED = 20_301_010
TOKEN_BUDGET = 4_194_304
EVAL_BATCHES = 64


def train_compiled_pair_4m_c(**kwargs: Any) -> dict[str, Any]:
    base.SCIENTIFIC_SEED = SCIENTIFIC_SEED
    base.PAIRED_BATCH_SEED = PAIRED_BATCH_SEED
    base.PAIRED_EVAL_SEED = PAIRED_EVAL_SEED
    base.COMPILE_PROBE_SEED = COMPILE_PROBE_SEED
    base.TOKEN_BUDGET = TOKEN_BUDGET
    base.EVAL_BATCHES = EVAL_BATCHES
    result = base.train_compiled_pair_1m(**kwargs)
    delta = float(result["derived"]["rlt_minus_transformer_final_nll"])
    result["classification"] = (
        "PAIRED_4M_RLT_LOWER_NLL_SINGLE_RUN_HINT"
        if delta < 0
        else "PAIRED_4M_NO_RLT_QUALITY_ADVANTAGE"
    )
    result["breakthrough_claim_supported"] = False
    result["interpretation_ceiling"] = (
        "Third independently seeded matched ~4M-token run at ~7M parameters. Agreement with "
        "the first two 4M runs would materially strengthen seed-level replication of the "
        "small-model NLL effect, but it still cannot establish a general breakthrough "
        "without broader model, data, and scale evidence."
    )
    return result
