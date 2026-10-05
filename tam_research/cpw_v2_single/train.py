from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from tam_research.cpw_v1.model import CPWV1Config
from tam_research.cpw_v1_pairwise.train import train_pairwise_candidate
from tam_research.train import train_language_model

from .model import SINGLE_ARMS, SinglePredictorCPWResearchLM


def train_single_candidate(
    *,
    arm: str,
    seed: int,
    data_dir: str,
    run_root: str,
    token_budget: int,
    seq_len: int,
    micro_batch_size: int,
    grad_accum_steps: int,
) -> dict[str, Any]:
    if arm == "transformer":
        return train_language_model(
            architecture="transformer",
            seed=seed,
            data_dir=data_dir,
            run_root=str(Path(run_root) / "transformer"),
            token_budget=token_budget,
            seq_len=seq_len,
            micro_batch_size=micro_batch_size,
            grad_accum_steps=grad_accum_steps,
            eval_every_tokens=token_budget,
            checkpoint_every_tokens=token_budget,
            resume=False,
            compile_model=False,
        )
    if arm == "sequence_memory":
        return train_pairwise_candidate(
            arm="sequence_memory",
            seed=seed,
            data_dir=data_dir,
            run_root=str(Path(run_root) / "sequence-memory"),
            token_budget=token_budget,
            seq_len=seq_len,
            micro_batch_size=micro_batch_size,
            grad_accum_steps=grad_accum_steps,
        )
    if arm not in SINGLE_ARMS:
        raise ValueError(f"unknown single panel arm: {arm}")

    from tam_research import train as train_module

    original_config = train_module.ModelConfig
    original_model = train_module.ResearchLM

    def cfg_factory(*, architecture: str, max_seq_len: int) -> CPWV1Config:
        return replace(
            CPWV1Config(),
            architecture="cpwv1",
            max_seq_len=max_seq_len,
        )

    class BoundSingleModel(SinglePredictorCPWResearchLM):
        def __init__(self, cfg: CPWV1Config):
            super().__init__(arm, cfg)

    try:
        train_module.ModelConfig = cfg_factory  # type: ignore[assignment]
        train_module.ResearchLM = BoundSingleModel  # type: ignore[assignment]
        return train_module.train_language_model(
            architecture=f"cpwv2_{arm}",
            seed=seed,
            data_dir=data_dir,
            run_root=str(Path(run_root) / arm),
            token_budget=token_budget,
            seq_len=seq_len,
            micro_batch_size=micro_batch_size,
            grad_accum_steps=grad_accum_steps,
            eval_every_tokens=token_budget,
            checkpoint_every_tokens=token_budget,
            resume=False,
            compile_model=False,
        )
    finally:
        train_module.ModelConfig = original_config
        train_module.ResearchLM = original_model
