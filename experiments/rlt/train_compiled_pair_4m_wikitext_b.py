from __future__ import annotations

from typing import Any

from experiments.rlt import train_compiled_pair_1m as base

SCIENTIFIC_SEED = 20_261_013
PAIRED_BATCH_SEED = 20_271_013
PAIRED_EVAL_SEED = 20_291_013
COMPILE_PROBE_SEED = 20_301_013
TOKEN_BUDGET = 4_194_304
EVAL_BATCHES = 64


def train_compiled_pair_4m_wikitext_b(**kwargs: Any) -> dict[str, Any]:
    base.SCIENTIFIC_SEED = SCIENTIFIC_SEED
    base.PAIRED_BATCH_SEED = PAIRED_BATCH_SEED
    base.PAIRED_EVAL_SEED = PAIRED_EVAL_SEED
    base.COMPILE_PROBE_SEED = COMPILE_PROBE_SEED
    base.TOKEN_BUDGET = TOKEN_BUDGET
    base.EVAL_BATCHES = EVAL_BATCHES
    result = base.train_compiled_pair_1m(**kwargs)
    delta = float(result["derived"]["rlt_minus_transformer_final_nll"])
    result["classification"] = (
        "PAIRED_4M_WIKITEXT_RLT_LOWER_NLL_SINGLE_RUN_HINT"
        if delta < 0
        else "PAIRED_4M_WIKITEXT_NO_RLT_QUALITY_ADVANTAGE"
    )
    result["breakthrough_claim_supported"] = False
    result["interpretation_ceiling"] = (
        "Second independent matched ~4M-token run trained and validated in-domain on the same "
        "pinned WikiText-103 raw bytes at ~7M parameters. Agreement with run A would establish "
        "two-seed corpus-level replication, but broader seed, scale, and corpus evidence remains required."
    )
    return result
