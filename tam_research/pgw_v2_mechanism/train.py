from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from tam_research.train import train_language_model
from tam_research.pgw_v1.model import PGWV1Config, PGWV1ResearchLM
from tam_research.pgw_v2.model import PGWV2Config, PGWV2ResearchLM

from .model import PGWControlConfig, PGWControlResearchLM
from .protocol import ARMS


def train_mechanism_arm(
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
        raise ValueError(f"unsupported mechanism arm: {arm}")

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

    if arm == "mean_read_predictive":
        model_cls = PGWV1ResearchLM

        def config_factory(*, architecture: str, max_seq_len: int) -> PGWV1Config:
            if architecture != arm:
                raise ValueError("mechanism arm/config mismatch")
            return replace(PGWV1Config(), max_seq_len=max_seq_len)

    elif arm == "token_read_predictive":
        model_cls = PGWV2ResearchLM

        def config_factory(*, architecture: str, max_seq_len: int) -> PGWV2Config:
            if architecture != arm:
                raise ValueError("mechanism arm/config mismatch")
            return replace(PGWV2Config(), max_seq_len=max_seq_len)

    else:
        control_mode = {
            "token_read_fixed_random": "fixed_random",
            "token_read_recency": "recency",
            "no_workspace": "no_workspace",
        }[arm]
        model_cls = PGWControlResearchLM

        def config_factory(*, architecture: str, max_seq_len: int) -> PGWControlConfig:
            if architecture != arm:
                raise ValueError("mechanism arm/config mismatch")
            return replace(
                PGWControlConfig(),
                max_seq_len=max_seq_len,
                control_mode=control_mode,
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
