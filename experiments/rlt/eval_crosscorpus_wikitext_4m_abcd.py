from __future__ import annotations

import math
from pathlib import Path
import statistics
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from experiments.rlt.model import RLTConfig, RecurrentLoopedTransformer, parameter_count as rlt_parameter_count
from tam_research.models import ModelConfig, ResearchLM, parameter_count as transformer_parameter_count


DATASET_ID = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-103-raw-v1"
DATASET_REVISION = "00aa25585682d4957f9e86edc73f59be7419af99"
DATASET_SPLIT = "test"
TOKENIZER_ID = "gpt2"
SEQ_LEN = 64
BATCH_SIZE = 64
EVAL_BATCHES = 32
EVAL_TARGET_TOKENS = BATCH_SIZE * SEQ_LEN * EVAL_BATCHES
REQUIRED_TEST_TOKENS = BATCH_SIZE * (SEQ_LEN + 1) * EVAL_BATCHES
EXPECTED_RLT_PARAMETERS = 7_007_616
EXPECTED_TRANSFORMER_PARAMETERS = 7_040_896
PAIR_JOBS = (
    ("A", "rlt-compiled-paired-4m-modal-20260920-a"),
    ("B", "rlt-compiled-paired-4m-modal-20260922-b"),
    ("C", "rlt-compiled-paired-4m-modal-20260923-c"),
    ("D", "rlt-compiled-paired-4m-modal-20260923-d"),
)


def rlt_config() -> RLTConfig:
    return RLTConfig(
        vocab_size=50_257,
        d_model=128,
        n_heads=4,
        n_stages=2,
        max_seq_len=128,
        ff_mult=4,
        swa_window=32,
    )


def transformer_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=50_257,
        d_model=128,
        n_layers=3,
        n_heads=4,
        max_seq_len=128,
        ff_mult=4,
        architecture="transformer",
    )


def _load_pair(checkpoint_root: Path, job_id: str, device: torch.device) -> tuple[RecurrentLoopedTransformer, ResearchLM]:
    rlt_path = checkpoint_root / job_id / "rlt_final.pt"
    transformer_path = checkpoint_root / job_id / "transformer_final.pt"
    if not rlt_path.exists() or not transformer_path.exists():
        raise FileNotFoundError(f"missing checkpoint pair for {job_id}")

    rlt_ckpt = torch.load(rlt_path, map_location="cpu", weights_only=True)
    transformer_ckpt = torch.load(transformer_path, map_location="cpu", weights_only=True)

    rlt = RecurrentLoopedTransformer(rlt_config()).to(device)
    transformer = ResearchLM(transformer_config()).to(device)
    rlt.load_state_dict(rlt_ckpt["model_state_dict"], strict=True)
    transformer.load_state_dict(transformer_ckpt["model_state_dict"], strict=True)

    if rlt_parameter_count(rlt) != EXPECTED_RLT_PARAMETERS:
        raise RuntimeError("RLT parameter drift")
    if transformer_parameter_count(transformer) != EXPECTED_TRANSFORMER_PARAMETERS:
        raise RuntimeError("Transformer parameter drift")

    rlt.eval()
    transformer.eval()
    return rlt, transformer


def _batch_at(data: np.memmap, batch_index: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    span = BATCH_SIZE * (SEQ_LEN + 1)
    start = batch_index * span
    end = start + span
    chunk = np.asarray(data[start:end], dtype=np.int64).reshape(BATCH_SIZE, SEQ_LEN + 1)
    x = torch.from_numpy(chunk[:, :-1].copy()).to(device=device, non_blocking=True)
    y = torch.from_numpy(chunk[:, 1:].copy()).to(device=device, non_blocking=True)
    return x, y


def _eval_pair(*, label: str, job_id: str, checkpoint_root: Path, test_path: Path) -> dict[str, Any]:
    device = torch.device("cuda")
    rlt, transformer = _load_pair(checkpoint_root, job_id, device)
    data = np.memmap(test_path, dtype=np.uint16, mode="r")
    if len(data) < REQUIRED_TEST_TOKENS:
        raise RuntimeError(f"WikiText test shard too short: {len(data)} < {REQUIRED_TEST_TOKENS}")

    amp = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    rlt_losses: list[float] = []
    transformer_losses: list[float] = []
    deltas: list[float] = []
    rlt_lower_batches = 0

    with torch.no_grad():
        for batch_index in range(EVAL_BATCHES):
            x, y = _batch_at(data, batch_index, device)
            with torch.autocast(device_type="cuda", dtype=amp):
                rlt_logits = rlt(x)
                transformer_logits = transformer(x)
            rlt_loss = float(F.cross_entropy(
                rlt_logits.float().reshape(-1, 50_257), y.reshape(-1), reduction="mean"
            ).item())
            transformer_loss = float(F.cross_entropy(
                transformer_logits.float().reshape(-1, 50_257), y.reshape(-1), reduction="mean"
            ).item())
            delta = rlt_loss - transformer_loss
            rlt_losses.append(rlt_loss)
            transformer_losses.append(transformer_loss)
            deltas.append(delta)
            if delta < 0:
                rlt_lower_batches += 1

    torch.cuda.synchronize(device)
    rlt_nll = sum(rlt_losses) / len(rlt_losses)
    transformer_nll = sum(transformer_losses) / len(transformer_losses)
    mean_delta = sum(deltas) / len(deltas)
    sd_delta = statistics.stdev(deltas)
    se_delta = sd_delta / math.sqrt(len(deltas))
    ci_half = 1.959963984540054 * se_delta
    out = {
        "label": label,
        "checkpoint_job_id": job_id,
        "eval_target_tokens": EVAL_TARGET_TOKENS,
        "eval_batches": EVAL_BATCHES,
        "batch_size": BATCH_SIZE,
        "seq_len": SEQ_LEN,
        "rlt_nll": rlt_nll,
        "rlt_perplexity": math.exp(min(rlt_nll, 20.0)),
        "transformer_nll": transformer_nll,
        "transformer_perplexity": math.exp(min(transformer_nll, 20.0)),
        "rlt_minus_transformer_nll": mean_delta,
        "rlt_lower_batch_count": rlt_lower_batches,
        "paired_batch_delta_sd": sd_delta,
        "paired_batch_delta_se": se_delta,
        "paired_batch_delta_normal95_ci": [mean_delta - ci_half, mean_delta + ci_half],
    }
    del rlt, transformer
    torch.cuda.empty_cache()
    return out


def evaluate_wikitext_4m_abcd(*, checkpoint_root: str, test_path: str) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("WikiText cross-corpus evaluation requires CUDA")
    torch.set_float32_matmul_precision("high")

    results = [
        _eval_pair(
            label=label,
            job_id=job_id,
            checkpoint_root=Path(checkpoint_root),
            test_path=Path(test_path),
        )
        for label, job_id in PAIR_JOBS
    ]
    deltas = [float(x["rlt_minus_transformer_nll"]) for x in results]
    mean_delta = sum(deltas) / len(deltas)
    sd = statistics.stdev(deltas)
    se = sd / math.sqrt(len(deltas))
    t95 = 3.182446305284263  # df=3, two-sided 95%
    ci = [mean_delta - t95 * se, mean_delta + t95 * se]
    count_lower = sum(x < 0 for x in deltas)

    if count_lower == len(deltas):
        classification = "CROSSCORPUS_WIKITEXT_4M_ABCD_ALL_FOUR_RLT_LOWER_NLL"
    elif mean_delta < 0:
        classification = "CROSSCORPUS_WIKITEXT_4M_ABCD_MEAN_RLT_LOWER_NLL_MIXED"
    else:
        classification = "CROSSCORPUS_WIKITEXT_4M_ABCD_NO_RLT_QUALITY_ADVANTAGE"

    return {
        "status": "complete",
        "classification": classification,
        "breakthrough_claim_supported": False,
        "cross_corpus": {
            "dataset": DATASET_ID,
            "dataset_config": DATASET_CONFIG,
            "dataset_revision": DATASET_REVISION,
            "split": DATASET_SPLIT,
            "tokenizer": TOKENIZER_ID,
            "eval_target_tokens_per_pair": EVAL_TARGET_TOKENS,
            "required_test_tokens": REQUIRED_TEST_TOKENS,
            "sampling": "deterministic contiguous GPT-2 token stream from the pinned WikiText-103 raw test split",
        },
        "pairs": results,
        "aggregate_across_model_pairs": {
            "pair_count": len(results),
            "rlt_lower_pair_count": count_lower,
            "mean_rlt_minus_transformer_nll": mean_delta,
            "sample_sd_pair_delta_nll": sd,
            "se_pair_delta_nll": se,
            "t95_ci_pair_delta_nll_df3": ci,
        },
        "interpretation_ceiling": (
            "Out-of-domain evaluation of four ~4M-token FineWeb-Edu-trained checkpoint pairs on "
            "a pinned WikiText-103 raw test split. Agreement would show the quality difference "
            "is not restricted to FineWeb-Edu validation text, but it would not show that RLT "
            "retains an advantage when trained on a different corpus or at larger model scale."
        ),
    }
