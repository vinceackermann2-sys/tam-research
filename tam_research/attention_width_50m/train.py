from __future__ import annotations

from pathlib import Path
from typing import Any

from tam_research.train import train_language_model

from .model import (
    baseline_50m_config,
    reduced_50m_config,
)
from tam_research.models import ResearchLM
from tam_research.attention_width_factorial.model import WidthFactorialResearchLM
from .protocol import ARMS


def train_attention_width_50m_arm(
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
        raise ValueError(f"unsupported 50M attention-width arm: {arm}")

    from tam_research import train as train_module

    original_config = train_module.ModelConfig
    original_model = train_module.ResearchLM

    if arm == "global384_ff1536":
        model_cls = ResearchLM

        def config_factory(*, architecture: str, max_seq_len: int):
            if architecture != arm:
                raise ValueError("50M arm/config mismatch")
            return baseline_50m_config(max_seq_len=max_seq_len)

    else:
        model_cls = WidthFactorialResearchLM

        def config_factory(*, architecture: str, max_seq_len: int):
            if architecture != arm:
                raise ValueError("50M arm/config mismatch")
            return reduced_50m_config(max_seq_len=max_seq_len)

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
