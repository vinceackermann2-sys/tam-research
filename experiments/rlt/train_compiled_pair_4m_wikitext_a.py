from __future__ import annotations

from typing import Any

from experiments.rlt import train_compiled_pair_1m as base

SCIENTIFIC_SEED = 20_261_012
PAIRED_BATCH_SEED = 20_271_012
PAIRED_EVAL_SEED = 20_291_012
COMPILE_PROBE_SEED = 20_301_012
TOKEN_BUDGET = 4_194_304
EVAL_BATCHES = 64


def train_compiled_pair_4m_wikitext_a(**kwargs: Any) -> dict[str, Any]:
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
        "Single matched ~4M-token run trained and validated in-domain on pinned WikiText-103 raw "
        "at ~7M parameters. This directly tests whether the FineWeb-Edu RLT advantage transfers "
        "when both models are trained on a different corpus. A single WikiText seed is not enough "
        "for a general architecture claim and requires independent-seed replication if positive."
    )
    return result
