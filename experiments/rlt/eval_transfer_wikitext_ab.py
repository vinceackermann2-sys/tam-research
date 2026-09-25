from __future__ import annotations

import math
from pathlib import Path
import statistics
from typing import Any

import torch

from experiments.rlt.eval_crossdata_4m_ab import _eval_pair as eval_fineweb_pair
from experiments.rlt.eval_crosscorpus_wikitext_4m_abcd import _eval_pair as eval_wikitext_pair


PAIR_JOBS = (
    ("A", "rlt-compiled-paired-4m-wikitext-modal-20260923-a"),
    ("B", "rlt-compiled-paired-4m-wikitext-modal-20260925-b"),
)


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    deltas = [float(x["rlt_minus_transformer_nll"]) for x in rows]
    mean = sum(deltas) / len(deltas)
    sd = statistics.stdev(deltas)
    se = sd / math.sqrt(len(deltas))
    t95 = 12.706204736432095  # df=1
    return {
        "pair_count": len(rows),
        "rlt_lower_pair_count": sum(x < 0 for x in deltas),
        "mean_rlt_minus_transformer_nll": mean,
        "sample_sd_pair_delta_nll": sd,
        "se_pair_delta_nll": se,
        "t95_ci_pair_delta_nll_df1": [mean - t95 * se, mean + t95 * se],
    }


def evaluate_transfer_wikitext_ab(
    *,
    checkpoint_root: str,
    wikitext_test_path: str,
    fineweb_val_path: str,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("symmetric transfer evaluation requires CUDA")
    torch.set_float32_matmul_precision("high")

    root = Path(checkpoint_root)
    wiki_rows: list[dict[str, Any]] = []
    fine_rows: list[dict[str, Any]] = []
    transfer_rows: list[dict[str, Any]] = []

    for label, job_id in PAIR_JOBS:
        wiki = eval_wikitext_pair(
            label=label,
            job_id=job_id,
            checkpoint_root=root,
            test_path=Path(wikitext_test_path),
        )
        fine = eval_fineweb_pair(
            label=label,
            job_id=job_id,
            checkpoint_root=root,
            val_path=Path(fineweb_val_path),
        )
        wiki_rows.append(wiki)
        fine_rows.append(fine)
        transfer_rows.append({
            "label": label,
            "checkpoint_job_id": job_id,
            "wikitext_in_domain_rlt_minus_transformer_nll": wiki["rlt_minus_transformer_nll"],
            "fineweb_out_of_domain_rlt_minus_transformer_nll": fine["rlt_minus_transformer_nll"],
            "ood_minus_indomain_relative_delta_nll": (
                fine["rlt_minus_transformer_nll"] - wiki["rlt_minus_transformer_nll"]
            ),
        })

    wiki_agg = _aggregate(wiki_rows)
    fine_agg = _aggregate(fine_rows)
    wiki_both = wiki_agg["rlt_lower_pair_count"] == 2
    fine_both = fine_agg["rlt_lower_pair_count"] == 2

    if wiki_both and fine_both:
        classification = "TRANSFER_WIKITEXT_AB_RLT_LOWER_BOTH_DOMAINS"
    elif wiki_both and not fine_both:
        classification = "TRANSFER_WIKITEXT_AB_RLT_LOWER_INDOMAIN_NOT_ROBUST_OOD"
    elif not wiki_both and fine_both:
        classification = "TRANSFER_WIKITEXT_AB_RLT_LOWER_OOD_NOT_CONSISTENT_INDOMAIN"
    else:
        classification = "TRANSFER_WIKITEXT_AB_NO_CONSISTENT_TWO_SEED_ADVANTAGE"

    transfer_gaps = [float(x["ood_minus_indomain_relative_delta_nll"]) for x in transfer_rows]
    mean_gap = sum(transfer_gaps) / len(transfer_gaps)
    gap_sd = statistics.stdev(transfer_gaps)
    gap_se = gap_sd / math.sqrt(len(transfer_gaps))
    t95 = 12.706204736432095

    return {
        "status": "complete",
        "classification": classification,
        "breakthrough_claim_supported": False,
        "wikitext_indomain": {
            "pairs": wiki_rows,
            "aggregate": wiki_agg,
        },
        "fineweb_out_of_domain": {
            "pairs": fine_rows,
            "aggregate": fine_agg,
        },
        "transfer": {
            "pairs": transfer_rows,
            "mean_ood_minus_indomain_relative_delta_nll": mean_gap,
            "sample_sd_gap": gap_sd,
            "se_gap": gap_se,
            "t95_ci_gap_df1": [mean_gap - t95 * gap_se, mean_gap + t95 * gap_se],
        },
        "interpretation_ceiling": (
            "Two independently seeded WikiText-trained ~7M-parameter model pairs evaluated both "
            "in-domain on pinned WikiText-103 test text and out-of-domain on a fresh FineWeb-Edu "
            "shard. This directly measures whether the RLT-vs-Transformer quality difference "
            "changes under distribution shift. Two seeds remain too few for a general claim."
        ),
    }
