from __future__ import annotations

import math
from pathlib import Path
import statistics
from typing import Any

import torch

from experiments.rlt.eval_crossdata_4m_ab import (
    BATCH_SIZE,
    EVAL_BATCHES,
    EVAL_TARGET_TOKENS,
    REQUIRED_VAL_TOKENS,
    SEQ_LEN,
    _eval_pair,
)

C4_DATASET_ID = "allenai/c4"
C4_CONFIG = "en"
C4_REVISION = "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"
C4_SPLIT = "validation"
TOKENIZER_ID = "gpt2"

FAMILIES = {
    "fineweb_trained": (
        ("FW-A", "rlt-compiled-paired-4m-modal-20260920-a"),
        ("FW-B", "rlt-compiled-paired-4m-modal-20260922-b"),
    ),
    "wikitext_trained": (
        ("WT-A", "rlt-compiled-paired-4m-wikitext-modal-20260923-a"),
        ("WT-B", "rlt-compiled-paired-4m-wikitext-modal-20260925-b"),
    ),
}


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    deltas = [float(x["rlt_minus_transformer_nll"]) for x in rows]
    mean = sum(deltas) / len(deltas)
    sd = statistics.stdev(deltas)
    se = sd / math.sqrt(len(deltas))
    t95 = 12.706204736432095  # df=1, two-sided 95%
    return {
        "pair_count": len(rows),
        "rlt_lower_pair_count": sum(x < 0 for x in deltas),
        "mean_rlt_minus_transformer_nll": mean,
        "sample_sd_pair_delta_nll": sd,
        "se_pair_delta_nll": se,
        "t95_ci_pair_delta_nll_df1": [mean - t95 * se, mean + t95 * se],
    }


def evaluate_c4_balanced_4m(*, checkpoint_root: str, c4_val_path: str) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("C4 balanced evaluation requires CUDA")
    torch.set_float32_matmul_precision("high")

    root = Path(checkpoint_root)
    family_results: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []

    for family, pairs in FAMILIES.items():
        rows = [
            _eval_pair(
                label=label,
                job_id=job_id,
                checkpoint_root=root,
                val_path=Path(c4_val_path),
            )
            for label, job_id in pairs
        ]
        agg = _aggregate(rows)
        family_results[family] = {"pairs": rows, "aggregate": agg}
        all_rows.extend(rows)

    fw = family_results["fineweb_trained"]["aggregate"]
    wt = family_results["wikitext_trained"]["aggregate"]
    fw_both = fw["rlt_lower_pair_count"] == 2
    wt_both = wt["rlt_lower_pair_count"] == 2

    if fw_both and wt_both:
        classification = "C4_BALANCED_BOTH_TRAINING_FAMILIES_ALL_RLT_LOWER_NLL"
    elif fw_both and not wt_both:
        classification = "C4_BALANCED_FINEWEB_FAMILY_RLT_LOWER_WIKITEXT_MIXED"
    elif wt_both and not fw_both:
        classification = "C4_BALANCED_WIKITEXT_FAMILY_RLT_LOWER_FINEWEB_MIXED"
    elif fw["mean_rlt_minus_transformer_nll"] < 0 and wt["mean_rlt_minus_transformer_nll"] < 0:
        classification = "C4_BALANCED_BOTH_FAMILY_MEANS_RLT_LOWER_MIXED_SEEDS"
    else:
        classification = "C4_BALANCED_NO_CONSISTENT_CROSS_FAMILY_RLT_ADVANTAGE"

    family_mean_difference = (
        wt["mean_rlt_minus_transformer_nll"] - fw["mean_rlt_minus_transformer_nll"]
    )

    return {
        "status": "complete",
        "classification": classification,
        "breakthrough_claim_supported": False,
        "c4": {
            "dataset": C4_DATASET_ID,
            "dataset_config": C4_CONFIG,
            "dataset_revision": C4_REVISION,
            "split": C4_SPLIT,
            "tokenizer": TOKENIZER_ID,
            "eval_target_tokens_per_pair": EVAL_TARGET_TOKENS,
            "required_tokens": REQUIRED_VAL_TOKENS,
            "batch_size": BATCH_SIZE,
            "seq_len": SEQ_LEN,
            "eval_batches": EVAL_BATCHES,
            "sampling": "deterministic contiguous GPT-2 token stream from pinned C4 English validation, streamed in repository order",
        },
        "families": family_results,
        "family_comparison": {
            "wikitext_trained_mean_minus_fineweb_trained_mean_relative_delta_nll": family_mean_difference,
        },
        "interpretation_ceiling": (
            "Balanced third-domain evaluation of two FineWeb-Edu-trained and two WikiText-trained "
            "~7M-parameter paired checkpoints on one pinned C4 validation token stream. It tests "
            "whether RLT-vs-Transformer quality differences persist on a neutral web-text corpus "
            "and whether transfer behavior depends on training corpus. Two seeds per family and "
            "one model scale remain insufficient for a general architecture claim."
        ),
    }
