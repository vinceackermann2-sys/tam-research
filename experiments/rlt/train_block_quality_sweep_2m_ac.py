from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from experiments.rlt import train_compiled_pair_1m as base
from experiments.rlt.model import RLTConfig
from experiments.rlt.model_block import BlockRecurrentLoopedTransformer
from tam_research.models import ModelConfig

SCIENTIFIC_SEED = 20_261_039
PAIRED_BATCH_SEED = 20_271_039
PAIRED_EVAL_SEED = 20_291_039
COMPILE_PROBE_SEED = 20_301_039
TOKEN_BUDGET = 2_097_152
EVAL_BATCHES = 64
BLOCK_SIZES = (1, 2, 4, 8)
EXPECTED_PARAMETERS = 15_129_344


class CompilableBlockRLT(nn.Module):
    def __init__(self, model: BlockRecurrentLoopedTransformer):
        super().__init__()
        self.base = model

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.base(tokens)


def rlt_config() -> RLTConfig:
    return RLTConfig(
        vocab_size=50_257,
        d_model=256,
        n_heads=8,
        n_stages=2,
        max_seq_len=128,
        ff_mult=4,
        swa_window=32,
    )


def transformer_config() -> ModelConfig:
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


def _fixed_block_class(block_size: int):
    class FixedBlockRLT(BlockRecurrentLoopedTransformer):
        def __init__(self, cfg: RLTConfig):
            super().__init__(cfg, block_size=block_size)
    FixedBlockRLT.__name__ = f"FixedBlock{block_size}RLT"
    return FixedBlockRLT


def train_block_quality_sweep_2m_ac(
    *,
    data_dir: str,
    checkpoint_dir: str,
    token_budget: int,
    seq_len: int,
    micro_batch_size: int,
    grad_accum_steps: int,
) -> dict[str, Any]:
    if token_budget != TOKEN_BUDGET:
        raise RuntimeError(f"token budget drift: {token_budget}")

    base.SCIENTIFIC_SEED = SCIENTIFIC_SEED
    base.PAIRED_BATCH_SEED = PAIRED_BATCH_SEED
    base.PAIRED_EVAL_SEED = PAIRED_EVAL_SEED
    base.COMPILE_PROBE_SEED = COMPILE_PROBE_SEED
    base.TOKEN_BUDGET = TOKEN_BUDGET
    base.SEQ_LEN = seq_len
    base.MICRO_BATCH_SIZE = micro_batch_size
    base.GRAD_ACCUM_STEPS = grad_accum_steps
    base.EVAL_BATCHES = EVAL_BATCHES
    base.rlt_config = rlt_config
    base.transformer_config = transformer_config
    base.EXPECTED_RLT_PARAMETERS = EXPECTED_PARAMETERS
    base.EXPECTED_TRANSFORMER_PARAMETERS = EXPECTED_PARAMETERS
    base.CompilableRLT = CompilableBlockRLT

    rows: list[dict[str, Any]] = []
    for block_size in BLOCK_SIZES:
        base.RecurrentLoopedTransformer = _fixed_block_class(block_size)
        pair = base.train_compiled_pair_1m(
            data_dir=data_dir,
            checkpoint_dir=f"{checkpoint_dir}/block{block_size}",
            token_budget=token_budget,
            seq_len=seq_len,
            micro_batch_size=micro_batch_size,
            grad_accum_steps=grad_accum_steps,
        )
        if pair.get("status") != "complete":
            raise RuntimeError(f"block{block_size} pair incomplete: {pair}")
        if pair.get("semantic_equivalence", {}).get("passed") is not True:
            raise RuntimeError(
                f"block{block_size} compile-wrapper equivalence failed: "
                f"{pair.get('semantic_equivalence')}"
            )
        rp = int(pair["rlt"]["parameters"])
        tp = int(pair["transformer"]["parameters"])
        if rp != EXPECTED_PARAMETERS or tp != EXPECTED_PARAMETERS or rp != tp:
            raise RuntimeError(
                f"block{block_size} parameter mismatch: rlt={rp}, transformer={tp}"
            )
        rows.append({
            "block_size": block_size,
            "block_recurrences_per_seq64": 64 // block_size,
            "pair": pair,
        })

    transformer_nlls = [
        float(row["pair"]["transformer"]["final_eval"]["nll"]) for row in rows
    ]
    transformer_spread = max(transformer_nlls) - min(transformer_nlls)
    if transformer_spread > 1e-4:
        raise RuntimeError(
            f"repeated same-seed Transformer controls diverged: "
            f"{transformer_nlls}, spread={transformer_spread}"
        )

    block1_nll = float(rows[0]["pair"]["rlt"]["final_eval"]["nll"])
    transformer_nll = sum(transformer_nlls) / len(transformer_nlls)
    summaries: list[dict[str, Any]] = []
    for row in rows:
        pair = row["pair"]
        rlt_nll = float(pair["rlt"]["final_eval"]["nll"])
        summaries.append({
            "block_size": row["block_size"],
            "block_recurrences_per_seq64": row["block_recurrences_per_seq64"],
            "rlt_final_nll": rlt_nll,
            "transformer_final_nll": float(pair["transformer"]["final_eval"]["nll"]),
            "rlt_minus_transformer_final_nll": (
                rlt_nll - float(pair["transformer"]["final_eval"]["nll"])
            ),
            "quality_penalty_vs_block1_nll": rlt_nll - block1_nll,
            "rlt_tokens_per_second": float(
                pair["rlt"]["tokens_per_second_excluding_compile_and_eval"]
            ),
            "transformer_tokens_per_second": float(
                pair["transformer"]["tokens_per_second_excluding_compile_and_eval"]
            ),
            "semantic_equivalence": pair["semantic_equivalence"],
        })

    best_non1 = min(
        (x for x in summaries if x["block_size"] != 1),
        key=lambda x: x["quality_penalty_vs_block1_nll"],
    )
    return {
        "status": "complete",
        "classification": "BLOCK_QUALITY_SWEEP_2M_COMPLETE",
        "breakthrough_claim_supported": False,
        "scientific_seed": SCIENTIFIC_SEED,
        "paired_batch_seed": PAIRED_BATCH_SEED,
        "paired_eval_seed": PAIRED_EVAL_SEED,
        "compile_probe_seed": COMPILE_PROBE_SEED,
        "matched_conditions": {
            "token_budget": TOKEN_BUDGET,
            "seq_len": seq_len,
            "micro_batch_size": micro_batch_size,
            "grad_accum_steps": grad_accum_steps,
            "block_sizes": list(BLOCK_SIZES),
            "same_model_seed": True,
            "same_batch_stream": True,
            "same_eval_stream": True,
            "same_data_bytes": True,
            "expected_parameters_each": EXPECTED_PARAMETERS,
        },
        "transformer_control_consistency": {
            "nlls": transformer_nlls,
            "max_minus_min_nll": transformer_spread,
            "passed": transformer_spread <= 1e-4,
            "mean_nll": transformer_nll,
        },
        "summaries": summaries,
        "best_non1_by_quality_penalty": best_non1,
        "pairs": {str(row["block_size"]): row["pair"] for row in rows},
        "interpretation_ceiling": (
            "Same-seed 2M-token architecture-selection sweep. Its purpose is to isolate "
            "quality change caused by recurrence granularity by comparing block sizes "
            "1/2/4/8 under identical initialization, data, batch stream, eval stream, "
            "optimizer protocol, and exact parameter count. It is selection evidence, "
            "not a replicated breakthrough result."
        ),
    }
