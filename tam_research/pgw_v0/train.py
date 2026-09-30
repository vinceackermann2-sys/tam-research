from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from tam_research.train import train_language_model

from .model import PGWConfig, PGWResearchLM


def train_pgw_candidate(
    *,
    architecture: str,
    seed: int,
    data_dir: str,
    run_root: str,
    token_budget: int,
    seq_len: int,
    micro_batch_size: int,
    grad_accum_steps: int,
) -> dict[str, Any]:
    """Run both arms through the repository's established trainer."""
    architecture = architecture.lower()
    if architecture not in {"transformer", "pgw"}:
        raise ValueError("architecture must be transformer or pgw")

    from tam_research import train as train_module

    if architecture == "transformer":
        return train_language_model(
            architecture="transformer",
            seed=seed,
            data_dir=data_dir,
            run_root=str(Path(run_root) / "transformer"),
            token_budget=token_budget,
            seq_len=seq_len,
            micro_batch_size=micro_batch_size,
            grad_accum_steps=grad_accum_steps,
            resume=False,
            compile_model=False,
        )

    original_config = train_module.ModelConfig
    original_model = train_module.ResearchLM

    def pgw_config(*, architecture: str, max_seq_len: int) -> PGWConfig:
        if architecture != "pgw":
            raise ValueError("PGW constructor substitution received non-PGW architecture")
        return replace(PGWConfig(), max_seq_len=max_seq_len)

    try:
        train_module.ModelConfig = pgw_config  # type: ignore[assignment]
        train_module.ResearchLM = PGWResearchLM  # type: ignore[assignment]
        result = train_module.train_language_model(
            architecture="pgw",
            seed=seed,
            data_dir=data_dir,
            run_root=str(Path(run_root) / "pgw"),
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

    return result
