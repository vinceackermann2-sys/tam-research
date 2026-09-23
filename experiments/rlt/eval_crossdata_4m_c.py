from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from experiments.rlt.eval_crossdata_4m_ab import (
    BATCH_SIZE,
    DATA_SEED,
    EVAL_BATCHES,
    EVAL_TARGET_TOKENS,
    REQUIRED_VAL_TOKENS,
    SEQ_LEN,
    _eval_pair,
)

CHECKPOINT_JOB_ID = "rlt-compiled-paired-4m-modal-20260923-c"


def evaluate_crossdata_4m_c(*, checkpoint_root: str, val_path: str) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("cross-data checkpoint evaluation requires CUDA")
    torch.set_float32_matmul_precision("high")

    pair = _eval_pair(
        label="C",
        job_id=CHECKPOINT_JOB_ID,
        checkpoint_root=Path(checkpoint_root),
        val_path=Path(val_path),
    )
    delta = float(pair["rlt_minus_transformer_nll"])
    classification = (
        "CROSSDATA_4M_C_RLT_LOWER_NLL"
        if delta < 0
        else "CROSSDATA_4M_C_NO_RLT_QUALITY_ADVANTAGE"
    )
    return {
        "status": "complete",
        "classification": classification,
        "breakthrough_claim_supported": False,
        "fresh_data": {
            "dataset": "HuggingFaceFW/fineweb-edu",
            "dataset_config": "sample-10BT",
            "shuffle_seed": DATA_SEED,
            "eval_target_tokens": EVAL_TARGET_TOKENS,
            "required_val_tokens": REQUIRED_VAL_TOKENS,
            "sampling": "deterministic non-overlapping 64-token next-token sequences from the same fresh validation shard used for 4M A/B",
        },
        "pair": pair,
        "interpretation_ceiling": (
            "Fresh-data evaluation of the third independently trained ~4M-token model-seed pair "
            "on the same held-out shard definition used for A/B. Agreement would permit a "
            "three-seed fresh-data aggregate, but this remains a ~7M-parameter single-dataset result."
        ),
    }
