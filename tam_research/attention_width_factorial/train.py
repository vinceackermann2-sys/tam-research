from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from tam_research.train import train_language_model
from tam_research.pgw_core_mechanism.model import (
    ChunkLocal256Config,
    ChunkLocal256ResearchLM,
)

from .model import WidthFactorialConfig, WidthFactorialResearchLM
from .protocol import ARMS


def train_attention_width_arm(
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
    arm = arm.lower()
    if arm not in ARMS:
        raise ValueError(f"unsupported attention-width arm: {arm}")

    from tam_research import train as train_module

    if arm == "global256_ff1024":
        return train_language_model(
            architecture="transformer",
            seed=seed,
            data_dir=data_dir,
            run_root=str(Path(run_root) / arm),
            token_budget=token_budget,
            seq_len=seq_len,
            micro_batch_size=micro_batch_size,
            grad_accum_steps=grad_accum_steps,
            resume=False,
            compile_model=False,
        )

    original_config = train_module.ModelConfig
    original_model = train_module.ResearchLM

    if arm == "local256_ff1024":
        model_cls = ChunkLocal256ResearchLM

        def config_factory(*, architecture: str, max_seq_len: int) -> ChunkLocal256Config:
            if architecture != arm:
                raise ValueError("attention-width arm/config mismatch")
            return replace(ChunkLocal256Config(), max_seq_len=max_seq_len)

    else:
        locality = {
            "global224_ff1088": "global",
            "local224_ff1088": "local",
        }[arm]
        model_cls = WidthFactorialResearchLM

        def config_factory(*, architecture: str, max_seq_len: int) -> WidthFactorialConfig:
            if architecture != arm:
                raise ValueError("attention-width arm/config mismatch")
            return replace(
                WidthFactorialConfig(),
                max_seq_len=max_seq_len,
                locality=locality,
            )

    try:
        train_module.ModelConfig = config_factory  # type: ignore[assignment]
        train_module.ResearchLM = model_cls  # type: ignore[assignment]
        return train_module.train_language_model(
            architecture=arm,
            seed=seed,
            data_dir=data_dir,
            run_root=str(Path(run_root) / arm),
            token_budget=token_budget,
            seq_len=seq_len,
            micro_batch_size=micro_batch_size,
            grad_accum_steps=grad_accum_steps,
            resume=False,
            compile_model=False,
        )
    finally:
        train_module.ModelConfig = original_config
        train_module.ResearchLM = original_model
