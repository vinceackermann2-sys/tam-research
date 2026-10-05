from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any

from tam_research.cpw_v1.model import CPWV1Config
from tam_research.train import train_language_model

from .model import CPWV2SequenceLM
from .protocol import EVAL_EVERY_TOKENS


def _curve_from_metrics(run_dir: Path) -> list[dict[str, float | int]]:
    path = run_dir / "metrics.jsonl"
    rows: list[dict[str, float | int]] = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        record = json.loads(line)
        if record.get("type") == "eval":
            rows.append({
                "tokens_seen": int(record["tokens_seen"]),
                "nll": float(record["nll"]),
                "perplexity": float(record["perplexity"]),
            })
    return rows


def train_sequence_candidate(
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
    architecture = architecture.lower()
    root = Path(run_root)

    if architecture == "transformer":
        result = train_language_model(
            architecture="transformer",
            seed=seed,
            data_dir=data_dir,
            run_root=str(root / "transformer"),
            token_budget=token_budget,
            seq_len=seq_len,
            micro_batch_size=micro_batch_size,
            grad_accum_steps=grad_accum_steps,
            eval_every_tokens=min(EVAL_EVERY_TOKENS, token_budget),
            checkpoint_every_tokens=token_budget,
            resume=False,
            compile_model=False,
        )
        run_dir = root / "transformer" / f"transformer-25m-eager-seed{seed}"
        result["learning_curve"] = _curve_from_metrics(run_dir)
        return result

    if architecture != "cpwv2seq":
        raise ValueError("architecture must be transformer or cpwv2seq")

    from tam_research import train as train_module

    original_config = train_module.ModelConfig
    original_model = train_module.ResearchLM

    def cfg_factory(*, architecture: str, max_seq_len: int) -> CPWV1Config:
        if architecture != "cpwv2seq":
            raise ValueError("unexpected architecture")
        return replace(
            CPWV1Config(),
            architecture="cpwv1",
            max_seq_len=max_seq_len,
        )

    try:
        train_module.ModelConfig = cfg_factory  # type: ignore[assignment]
        train_module.ResearchLM = CPWV2SequenceLM  # type: ignore[assignment]
        result = train_module.train_language_model(
            architecture="cpwv2seq",
            seed=seed,
            data_dir=data_dir,
            run_root=str(root / "sequence"),
            token_budget=token_budget,
            seq_len=seq_len,
            micro_batch_size=micro_batch_size,
            grad_accum_steps=grad_accum_steps,
            eval_every_tokens=min(EVAL_EVERY_TOKENS, token_budget),
            checkpoint_every_tokens=token_budget,
            resume=False,
            compile_model=False,
        )
    finally:
        train_module.ModelConfig = original_config
        train_module.ResearchLM = original_model

    run_dir = root / "sequence" / f"cpwv2seq-25m-eager-seed{seed}"
    result["learning_curve"] = _curve_from_metrics(run_dir)
    return result
