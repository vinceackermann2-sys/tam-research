from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from tam_research.cpw_v1.model import CPWV1Config
from tam_research.train import train_language_model

from .model import CPWV1FastResearchLM


def train_cpw_v1_fast(
    *,
    seed: int,
    data_dir: str,
    run_root: str,
    token_budget: int,
    seq_len: int,
    micro_batch_size: int,
    grad_accum_steps: int,
) -> dict[str, Any]:
    from tam_research import train as train_module

    original_config = train_module.ModelConfig
    original_model = train_module.ResearchLM

    def fast_config(*, architecture: str, max_seq_len: int) -> CPWV1Config:
        if architecture != "cpwv1fast":
            raise ValueError("fast constructor got non-fast architecture")
        return replace(
            CPWV1Config(),
            architecture="cpwv1",
            max_seq_len=max_seq_len,
        )

    try:
        train_module.ModelConfig = fast_config  # type: ignore[assignment]
        train_module.ResearchLM = CPWV1FastResearchLM  # type: ignore[assignment]
        return train_module.train_language_model(
            architecture="cpwv1fast",
            seed=seed,
            data_dir=data_dir,
            run_root=str(Path(run_root) / "cpwv1fast"),
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
