from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from tam_research.train import train_language_model
from tam_research.pgw_v2_mechanism.model import (
    PGWControlConfig,
    PGWControlResearchLM,
)

from .model import (
    ChunkLocal256Config,
    ChunkLocal256ResearchLM,
    PGWCoreConfig,
    PGWCoreResearchLM,
)
from .protocol import ARMS


def train_core_mechanism_arm(
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
        raise ValueError(f"unsupported core mechanism arm: {arm}")

    from tam_research import train as train_module

    if arm == "transformer":
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

    if arm == "chunk_local_256":
        model_cls = ChunkLocal256ResearchLM

        def config_factory(*, architecture: str, max_seq_len: int) -> ChunkLocal256Config:
            if architecture != arm:
                raise ValueError("core arm/config mismatch")
            return replace(ChunkLocal256Config(), max_seq_len=max_seq_len)

    elif arm == "local224_predictor_carry":
        model_cls = PGWControlResearchLM

        def config_factory(*, architecture: str, max_seq_len: int) -> PGWControlConfig:
            if architecture != arm:
                raise ValueError("core arm/config mismatch")
            return replace(
                PGWControlConfig(),
                max_seq_len=max_seq_len,
                control_mode="no_workspace",
            )

    else:
        core_mode = {
            "local224_predictor_reset": "predictor_reset",
            "local224_only": "local_only",
        }[arm]
        model_cls = PGWCoreResearchLM

        def config_factory(*, architecture: str, max_seq_len: int) -> PGWCoreConfig:
            if architecture != arm:
                raise ValueError("core arm/config mismatch")
            return replace(
                PGWCoreConfig(),
                max_seq_len=max_seq_len,
                core_mode=core_mode,
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
