from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from experiments.rlt import train_compiled_pair_1m as base
from experiments.rlt.model import RLTConfig
from experiments.rlt.model_light_state import LightStateRecurrentTransformer
from tam_research.models import ModelConfig

SCIENTIFIC_SEED = 20_261_042
PAIRED_BATCH_SEED = 20_271_042
PAIRED_EVAL_SEED = 20_291_042
COMPILE_PROBE_SEED = 20_301_042
TOKEN_BUDGET = 4_194_304
EVAL_BATCHES = 64
EXPECTED_PARAMETERS = 15_129_344


class CompilableLightStateRLT(nn.Module):
    """Thin compile wrapper preserving LightStateRecurrentTransformer.forward."""

    def __init__(self, model: LightStateRecurrentTransformer):
        super().__init__()
        self.base = model

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.base(tokens)


def light_state_config() -> RLTConfig:
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


def train_lightstate_scale15m_pair_4m_b(**kwargs: Any) -> dict[str, Any]:
    base.SCIENTIFIC_SEED = SCIENTIFIC_SEED
    base.PAIRED_BATCH_SEED = PAIRED_BATCH_SEED
    base.PAIRED_EVAL_SEED = PAIRED_EVAL_SEED
    base.COMPILE_PROBE_SEED = COMPILE_PROBE_SEED
    base.TOKEN_BUDGET = TOKEN_BUDGET
    base.EVAL_BATCHES = EVAL_BATCHES
    base.rlt_config = light_state_config
    base.transformer_config = scale_transformer_config
    base.EXPECTED_RLT_PARAMETERS = EXPECTED_PARAMETERS
    base.EXPECTED_TRANSFORMER_PARAMETERS = EXPECTED_PARAMETERS
    base.RecurrentLoopedTransformer = LightStateRecurrentTransformer
    base.CompilableRLT = CompilableLightStateRLT

    result = base.train_compiled_pair_1m(**kwargs)

    rp = int(result["rlt"]["parameters"])
    tp = int(result["transformer"]["parameters"])
    if rp != EXPECTED_PARAMETERS or tp != EXPECTED_PARAMETERS or rp != tp:
        raise RuntimeError(
            f"exact light-state parameter match failed: light_state={rp}, transformer={tp}"
        )

    equivalence = result.get("semantic_equivalence") or {}
    if equivalence.get("passed") is not True:
        raise RuntimeError(
            f"light-state compile-wrapper equivalence failed: {equivalence}"
        )

    delta = float(result["derived"]["rlt_minus_transformer_final_nll"])
    result["classification"] = (
        "PAIRED_LIGHTSTATE_SCALE15M_4M_RLT_LOWER_NLL_SINGLE_RUN_HINT"
        if delta < 0.0
        else "PAIRED_LIGHTSTATE_SCALE15M_4M_NO_RLT_QUALITY_ADVANTAGE"
    )
    result["breakthrough_claim_supported"] = False
    result["architecture_test"] = {
        "variant": "light_state_recurrent_rlt",
        "parameters": rp,
        "transformer_parameters": tp,
        "parameter_delta": tp - rp,
        "parameter_delta_fraction": 0.0,
        "rlt_d_model": 256,
        "rlt_stages": 2,
        "transformer_d_model": 256,
        "transformer_layers": 3,
        "transformer_ff_inner": 938,
        "token_recurrent_updates_per_seq64": 64,
        "serial_transition": "merge_plus_rmsnorm_only",
        "heavy_decoder_execution": "parallel_causal_full_sequence",
    }
    result["interpretation_ceiling"] = (
        "Fresh-seed exact-parameter/equal-token 4M comparison of the parameter-neutral "
        "light-state recurrent RLT against the matched Transformer. The architecture retains "
        "64 token-level recurrent state updates while moving attention, cross-attention and FFN "
        "outside the serial loop. A single positive seed is only a hint and requires replication "
        "and longer-budget validation before any practical or breakthrough claim."
    )
    return result
