from __future__ import annotations

from typing import Any

from experiments.rlt import train_compiled_pair_1m as base

SCIENTIFIC_SEED = 20_261_007
PAIRED_BATCH_SEED = 20_271_007
PAIRED_EVAL_SEED = 20_291_007
COMPILE_PROBE_SEED = 20_301_007

def train_compiled_pair_1m_c(**kwargs: Any) -> dict[str, Any]:
    base.SCIENTIFIC_SEED = SCIENTIFIC_SEED
    base.PAIRED_BATCH_SEED = PAIRED_BATCH_SEED
    base.PAIRED_EVAL_SEED = PAIRED_EVAL_SEED
    base.COMPILE_PROBE_SEED = COMPILE_PROBE_SEED
    return base.train_compiled_pair_1m(**kwargs)
