from __future__ import annotations

from pathlib import Path
from typing import Any

from .scales import SCALE_SPECS, model_config_for_scale


def train_scaled_language_model(
    *,
    architecture: str,
    model_scale: str,
    seed: int,
    data_dir: str,
    run_root: str,
    token_budget: int,
    seq_len: int,
    micro_batch_size: int,
    grad_accum_steps: int,
    compile_model: bool = True,
    eval_every_tokens: int = 5_000_000,
    checkpoint_every_tokens: int = 10_000_000,
) -> dict[str, Any]:
    """Run the established training loop with a scale-specific matched config.

    The baseline trainer intentionally remains untouched so historical evidence
    stays reproducible. Modal runs execute in isolated containers, so temporarily
    substituting the ModelConfig constructor here is process-local. Scale- and
    batching-specific run roots prevent checkpoint/resume collisions.

    ``eval_every_tokens`` and ``checkpoint_every_tokens`` default to the historical
    values. Long budget-capped production-style runs may raise them to reduce
    evaluation/checkpoint I/O without changing the training objective or optimizer.
    """
    scale = model_scale.lower()
    if scale not in SCALE_SPECS:
        raise ValueError(f"unknown model scale: {model_scale!r}")
    if architecture not in {"transformer", "tamv3"}:
        raise ValueError("scaling grid supports only transformer and tamv3")
    if eval_every_tokens <= 0 or checkpoint_every_tokens <= 0:
        raise ValueError("evaluation and checkpoint intervals must be positive")

    from . import train as train_module

    original_model_config = train_module.ModelConfig

    def scaled_config(*, architecture: str, max_seq_len: int):
        return model_config_for_scale(
            architecture,
            scale,
            max_seq_len=max_seq_len,
        )

    scale_root = str(
        Path(run_root)
        / scale
        / f"ctx{seq_len}-mb{micro_batch_size}-ga{grad_accum_steps}"
    )
    try:
        train_module.ModelConfig = scaled_config  # type: ignore[assignment]
        result = train_module.train_language_model(
            architecture=architecture,
            seed=seed,
            data_dir=data_dir,
            run_root=scale_root,
            token_budget=token_budget,
            seq_len=seq_len,
            micro_batch_size=micro_batch_size,
            grad_accum_steps=grad_accum_steps,
            eval_every_tokens=eval_every_tokens,
            checkpoint_every_tokens=checkpoint_every_tokens,
            compile_model=compile_model,
        )
    finally:
        train_module.ModelConfig = original_model_config

    result["model_scale"] = scale
    result["scale_spec"] = {
        "d_model": SCALE_SPECS[scale].d_model,
        "n_layers": SCALE_SPECS[scale].n_layers,
        "n_heads": SCALE_SPECS[scale].n_heads,
        "state_size": SCALE_SPECS[scale].state_size,
        "tamv3_attention_inner": SCALE_SPECS[scale].tamv3_attention_inner,
    }
    return result
