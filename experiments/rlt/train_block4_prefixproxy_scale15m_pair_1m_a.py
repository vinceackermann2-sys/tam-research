from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from experiments.rlt import train_compiled_pair_1m as base
from experiments.rlt.model import RLTConfig
from experiments.rlt.model_block_prefix_proxy import PrefixProxyBlockRecurrentLoopedTransformer
from tam_research.models import ModelConfig

SCIENTIFIC_SEED = 20_261_040
PAIRED_BATCH_SEED = 20_271_040
PAIRED_EVAL_SEED = 20_291_040
COMPILE_PROBE_SEED = 20_301_040
TOKEN_BUDGET = 1_048_576
EVAL_BATCHES = 64
BLOCK_SIZE = 4
EXPECTED_PARAMETERS = 15_129_344


class FixedBlock4PrefixProxyRLT(PrefixProxyBlockRecurrentLoopedTransformer):
    def __init__(self, cfg: RLTConfig):
        super().__init__(cfg, block_size=BLOCK_SIZE)


class CompilableBlockRLT(nn.Module):
    """Thin compile wrapper: preserves BlockRecurrentLoopedTransformer.forward exactly."""

    def __init__(self, model: BlockRecurrentLoopedTransformer):
        super().__init__()
        self.base = model

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.base(tokens)


def block_rlt_config() -> RLTConfig:
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


def train_block4_prefixproxy_scale15m_pair_1m_a(**kwargs: Any) -> dict[str, Any]:
    # Reuse the frozen paired-training harness while swapping only the RLT
    # architecture constructor and compile wrapper.
    base.SCIENTIFIC_SEED = SCIENTIFIC_SEED
    base.PAIRED_BATCH_SEED = PAIRED_BATCH_SEED
    base.PAIRED_EVAL_SEED = PAIRED_EVAL_SEED
    base.COMPILE_PROBE_SEED = COMPILE_PROBE_SEED
    base.TOKEN_BUDGET = TOKEN_BUDGET
    base.EVAL_BATCHES = EVAL_BATCHES
    base.rlt_config = block_rlt_config
    base.transformer_config = scale_transformer_config
    base.EXPECTED_RLT_PARAMETERS = EXPECTED_PARAMETERS
    base.EXPECTED_TRANSFORMER_PARAMETERS = EXPECTED_PARAMETERS

    # BlockRecurrentLoopedTransformer defaults to block_size=4.
    base.RecurrentLoopedTransformer = BlockRecurrentLoopedTransformer
    base.CompilableRLT = CompilableBlockRLT

    result = base.train_compiled_pair_1m(**kwargs)

    rp = int(result["rlt"]["parameters"])
    tp = int(result["transformer"]["parameters"])
    if rp != EXPECTED_PARAMETERS or tp != EXPECTED_PARAMETERS or rp != tp:
        raise RuntimeError(
            f"exact block4 scale parameter match failed: block8={rp}, transformer={tp}"
        )

    equivalence = result.get("semantic_equivalence") or {}
    if equivalence.get("passed") is not True:
        raise RuntimeError(
            f"block4 compile-wrapper equivalence failed: {equivalence}"
        )

    delta = float(result["derived"]["rlt_minus_transformer_final_nll"])
    result["classification"] = (
        "PAIRED_BLOCK4_PREFIXPROXY_SCALE15M_1M_RLT_LOWER_NLL_SINGLE_RUN_HINT"
        if delta < 0
        else "PAIRED_BLOCK4_PREFIXPROXY_SCALE15M_1M_NO_RLT_QUALITY_ADVANTAGE"
    )
    result["breakthrough_claim_supported"] = False
    result["architecture_test"] = {
        "variant": "block_recurrent_rlt_prefix_proxy",
        "block_size": BLOCK_SIZE,
        "block4_prefixproxy_parameters": rp,
        "transformer_parameters": tp,
        "parameter_delta": tp - rp,
        "parameter_delta_fraction": 0.0,
        "rlt_d_model": 256,
        "rlt_stages": 2,
        "transformer_d_model": 256,
        "transformer_layers": 3,
        "transformer_ff_inner": 938,
        "block_recurrences_per_seq64": 64 // BLOCK_SIZE,
        "token_recurrences_in_frozen_rlt_per_seq64": 64,
    }
    result["interpretation_ceiling"] = (
        "Fresh-seed exact-parameter/equal-token 4M comparison of the parameter-neutral "
        "block-recurrent RLT variant (block_size=4) against the matched Transformer. "
        "This tests whether the systems speedup preserves useful quality at the established "
        "15.1M scale. A single positive seed is only a hint; it requires replication and "
        "longer-budget testing before any practical or breakthrough claim."
    )
    return result
