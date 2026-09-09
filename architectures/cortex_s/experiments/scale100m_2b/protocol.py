from __future__ import annotations

from dataclasses import asdict
import math
from typing import Any

from architectures.cortex_s.language_model import CortexSLMConfig


EXPERIMENT_ID = "cortex-s-v0-100m-2b-paired8100-v3-grouped"
PROJECT_NAMESPACE = "cortex-s-v0/100m-2b-v3-grouped"
PRODUCTION_MOE_BACKEND = "physical_padded_grouped_bf16"
LOGICAL_EXPERT_HIDDEN = 338
PHYSICAL_EXPERT_HIDDEN = 344

# Historical matched Transformer control already completed in repo issue #140.
PAIRED_SEED = 8_100
BASELINE_ISSUE = 140
BASELINE_TRANSFORMER = {
    "parameters": 101_803_520,
    # The trainer's nominal budget is 2B, but the final optimizer step is a full
    # batch. Literal exposures are therefore the separately frozen value below.
    "pretrain_tokens": 2_000_000_000,
    "pretrain_token_budget": 2_000_000_000,
    "full_batch_token_exposures": 2_000_027_648,
    "optimizer_steps": 30_518,
    "seq_len": 512,
    "micro_batch_size": 64,
    "grad_accum_steps": 2,
    "global_batch": 128,
    "final_nll": 2.7115590302149455,
    "final_perplexity": 15.053722884190283,
    "training_seconds": 6_227.609573988244,
    "total_compute_seconds": 6_478.327432424761,
    "training_tokens_per_second": 321_151.5755348581,
    "peak_vram_gb": 11.896,
}

# Exact immutable pretraining source consumed by the historical Transformer.
DATA_DIR = "/vol/data/tam100m-2b-curated-v1"
TRAIN_TOKENS = 2_000_000_000
VAL_TOKENS = 5_000_000
SEQ_LEN = 512
MICRO_BATCH_SIZE = 64
GRAD_ACCUM_STEPS = 2
GLOBAL_BATCH = MICRO_BATCH_SIZE * GRAD_ACCUM_STEPS
TOKENS_PER_OPTIMIZER_STEP = MICRO_BATCH_SIZE * SEQ_LEN * GRAD_ACCUM_STEPS
TOTAL_OPTIMIZER_STEPS = math.ceil(TRAIN_TOKENS / TOKENS_PER_OPTIMIZER_STEP)
FULL_BATCH_TOKEN_EXPOSURES = TOTAL_OPTIMIZER_STEPS * TOKENS_PER_OPTIMIZER_STEP
EVAL_EVERY_TOKENS = 200_000_000
CHECKPOINT_EVERY_TOKENS = 200_000_000
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.1
WARMUP_RATIO = 0.02

# CPU-only fingerprint-v2 evidence: issue #767 / workflow 34271912952.
TRAIN_SHA256 = "93e9cb0b7076a4ddd855fc696f657ea62592b8a03a05220be99c402f9043265b"
VAL_SHA256 = "ae0bc5adf36d0aa8e55e5e3903401d3f114b93037f43221944b4e88c5d1a5760"
META_SHA256 = "14bbbcf0ab0b8cba374074ef8ccb80a04ecead06c74780f35a1becd5aef1b8f3"
FINGERPRINT_EVIDENCE_ISSUE = 767
FINGERPRINT_EVIDENCE_RUN = 34_271_912_952

# Systems history. The original/v2 calibration path was too slow and is consumed.
V2_PREFLIGHT_ISSUE = 769
V2_PREFLIGHT_SOURCE_SHA = "6cb1882aa5b02e4f8917a7cae44e6d1925b9e290"
V2_CALIBRATION_TPS = 133_384.4585
V2_PROJECTED_FULL_SECONDS = 16_676.53
REPAIR4_EVIDENCE_ISSUE = 799
REPAIR4_EVIDENCE_RUN = 34_339_214_619
REPAIR4_EVIDENCE_JOB = 102_425_759_653
REPAIR4_SOURCE_SHA = "9f4196b10674c6eb0454dc2d3459a7c283f16637"
REPAIR4_MEASURED_SPEEDUP = 1.3018190376470742
REPAIR4_GROUPED_TPS = 182_118.51594019053
REPAIR4_GROUPED_PEAK_VRAM_GIB = 45.2682785987854
REPAIR4_SEMANTIC_LOSS_DELTA = 0.004322052001953125

# All prior engineering seeds are consumed and may never be reused. The v3 grouped
# production preflight gets one fresh engineering seed; it is not scientific data.
CONSUMED_ENGINEERING_SEEDS = (910_001, 2_026_090_901, 2_026_090_902, 2_026_090_903, 2_026_090_904)
CALIBRATION_SEED = 2_026_090_905
CALIBRATION_STEPS = 40
CALIBRATION_WARMUP_STEPS = 5

# Budget lock. A separately triggered full run may not start unless this exact
# production grouped graph projects below the frozen wall-clock envelope.
H100_USD_PER_SECOND_SNAPSHOT = 0.001097
STARTER_CPU_USD_PER_CORE_SECOND_SNAPSHOT = 0.00003942
STARTER_MEMORY_USD_PER_GIB_SECOND_SNAPSHOT = 0.00000667
FULL_CPU_CORES = 8
FULL_MEMORY_GIB = 32
MAX_PROJECTED_FULL_SECONDS = 8_500
HARD_FULL_TIMEOUT_SECONDS = 10_000
USER_CREDIT_ENVELOPE_USD = 29.0

# 100M CORTEX-S remains matched on total trainable parameters, not active FLOPs.
CORTEX_100M_CONFIG = CortexSLMConfig(
    vocab_size=50_257,
    d_model=512,
    n_layers=24,
    n_heads=16,
    max_seq_len=1024,
    state_size=128,
    num_experts=8,
    top_k=2,
    expert_hidden=LOGICAL_EXPERT_HIDDEN,
    attention_every=6,
)
EXPECTED_CORTEX_PARAMS = 101_778_112
EXPECTED_TRANSFORMER_PARAMS = 101_803_520
MAX_PARAMETER_GAP_FRACTION = 0.005

# Reserved independent scientific replication seeds remain untouched.
RESERVED_FRESH_SEEDS = (48_131, 48_132, 48_133)


def projected_full_cost_usd(seconds: float) -> dict[str, float]:
    """Conservative Starter-plan compute estimate from the frozen pricing snapshot."""

    gpu = seconds * H100_USD_PER_SECOND_SNAPSHOT
    cpu = seconds * FULL_CPU_CORES * STARTER_CPU_USD_PER_CORE_SECOND_SNAPSHOT
    memory = seconds * FULL_MEMORY_GIB * STARTER_MEMORY_USD_PER_GIB_SECOND_SNAPSHOT
    return {
        "gpu_usd": gpu,
        "cpu_usd_conservative": cpu,
        "memory_usd_conservative": memory,
        "total_usd_conservative": gpu + cpu + memory,
    }


def _transformer_100m_parameter_count() -> int:
    vocab = 50_257
    d_model = 512
    layers = 24
    max_seq_len = 1024
    token = vocab * d_model
    position = max_seq_len * d_model
    per_block = 4 * d_model * d_model + 8 * d_model * d_model + 4 * d_model
    final_norm = 2 * d_model
    return token + position + layers * per_block + final_norm


def validate_protocol(*, actual_cortex_params: int | None = None) -> dict[str, Any]:
    CORTEX_100M_CONFIG.validate()
    baseline_params = _transformer_100m_parameter_count()
    if baseline_params != EXPECTED_TRANSFORMER_PARAMS:
        raise RuntimeError(
            f"Transformer parameter formula drifted: {baseline_params:,} != {EXPECTED_TRANSFORMER_PARAMS:,}"
        )
    cortex_params = EXPECTED_CORTEX_PARAMS if actual_cortex_params is None else actual_cortex_params
    if cortex_params != EXPECTED_CORTEX_PARAMS:
        raise RuntimeError(
            f"CORTEX-S parameter count drifted: {cortex_params:,} != {EXPECTED_CORTEX_PARAMS:,}"
        )
    parameter_gap = abs(cortex_params - baseline_params) / baseline_params
    if parameter_gap > MAX_PARAMETER_GAP_FRACTION:
        raise RuntimeError(f"parameter gap {parameter_gap:.6%} exceeds 0.5%")
    if GLOBAL_BATCH != 128 or MICRO_BATCH_SIZE * GRAD_ACCUM_STEPS != GLOBAL_BATCH:
        raise RuntimeError("global batch protocol drift")
    if TRAIN_TOKENS != BASELINE_TRANSFORMER["pretrain_token_budget"]:
        raise RuntimeError("token budget no longer matches historical Transformer")
    if TOTAL_OPTIMIZER_STEPS != BASELINE_TRANSFORMER["optimizer_steps"]:
        raise RuntimeError("optimizer-step count no longer matches historical Transformer")
    if FULL_BATCH_TOKEN_EXPOSURES != BASELINE_TRANSFORMER["full_batch_token_exposures"]:
        raise RuntimeError("full-batch token exposures no longer match historical Transformer")
    if SEQ_LEN != BASELINE_TRANSFORMER["seq_len"]:
        raise RuntimeError("context length no longer matches historical Transformer")
    if CORTEX_100M_CONFIG.top_k / CORTEX_100M_CONFIG.num_experts > 0.5:
        raise RuntimeError("sparse execution fraction exceeds preregistered ceiling")
    if CORTEX_100M_CONFIG.expert_hidden != LOGICAL_EXPERT_HIDDEN:
        raise RuntimeError("logical expert width drift")
    if PHYSICAL_EXPERT_HIDDEN != 344:
        raise RuntimeError("grouped physical expert width drift")
    forbidden = set(CONSUMED_ENGINEERING_SEEDS) | {PAIRED_SEED, *RESERVED_FRESH_SEEDS}
    if CALIBRATION_SEED in forbidden:
        raise RuntimeError("v3 calibration seed is consumed or scientifically reserved")
    hard_cost = projected_full_cost_usd(HARD_FULL_TIMEOUT_SECONDS)["total_usd_conservative"]
    if hard_cost >= USER_CREDIT_ENVELOPE_USD:
        raise RuntimeError(
            f"hard timeout could consume ${hard_cost:.2f}, exceeding the user credit envelope"
        )
    return {
        "experiment_id": EXPERIMENT_ID,
        "cortex_parameters": cortex_params,
        "transformer_parameters": baseline_params,
        "parameter_gap_fraction": parameter_gap,
        "parameter_gap_percent": 100.0 * parameter_gap,
        "executed_expert_fraction": CORTEX_100M_CONFIG.top_k / CORTEX_100M_CONFIG.num_experts,
        "attention_layers": CORTEX_100M_CONFIG.n_layers // CORTEX_100M_CONFIG.attention_every,
        "token_budget": TRAIN_TOKENS,
        "optimizer_steps": TOTAL_OPTIMIZER_STEPS,
        "full_batch_token_exposures": FULL_BATCH_TOKEN_EXPOSURES,
        "full_batch_overshoot_tokens": FULL_BATCH_TOKEN_EXPOSURES - TRAIN_TOKENS,
        "production_moe_backend": PRODUCTION_MOE_BACKEND,
        "logical_expert_hidden": LOGICAL_EXPERT_HIDDEN,
        "physical_expert_hidden": PHYSICAL_EXPERT_HIDDEN,
        "projected_cost_at_gate": projected_full_cost_usd(MAX_PROJECTED_FULL_SECONDS),
        "hard_timeout_cost_ceiling": projected_full_cost_usd(HARD_FULL_TIMEOUT_SECONDS),
    }


def protocol_snapshot() -> dict[str, Any]:
    validated = validate_protocol()
    return {
        **validated,
        "project_namespace": PROJECT_NAMESPACE,
        "paired_seed": PAIRED_SEED,
        "calibration_seed": CALIBRATION_SEED,
        "consumed_engineering_seeds": list(CONSUMED_ENGINEERING_SEEDS),
        "reserved_fresh_seeds": list(RESERVED_FRESH_SEEDS),
        "data_dir": DATA_DIR,
        "train_sha256": TRAIN_SHA256,
        "val_sha256": VAL_SHA256,
        "meta_sha256": META_SHA256,
        "fingerprint_evidence_issue": FINGERPRINT_EVIDENCE_ISSUE,
        "fingerprint_evidence_run": FINGERPRINT_EVIDENCE_RUN,
        "v2_preflight": {
            "issue": V2_PREFLIGHT_ISSUE,
            "source_sha": V2_PREFLIGHT_SOURCE_SHA,
            "measured_tps": V2_CALIBRATION_TPS,
            "projected_full_seconds": V2_PROJECTED_FULL_SECONDS,
            "full_run_launched": False,
        },
        "repair4_engineering_evidence": {
            "issue": REPAIR4_EVIDENCE_ISSUE,
            "run": REPAIR4_EVIDENCE_RUN,
            "job": REPAIR4_EVIDENCE_JOB,
            "source_sha": REPAIR4_SOURCE_SHA,
            "speedup": REPAIR4_MEASURED_SPEEDUP,
            "grouped_tps": REPAIR4_GROUPED_TPS,
            "grouped_peak_vram_gib": REPAIR4_GROUPED_PEAK_VRAM_GIB,
            "semantic_loss_delta": REPAIR4_SEMANTIC_LOSS_DELTA,
            "scientific_evidence": False,
        },
        "train_token_budget": TRAIN_TOKENS,
        "val_tokens": VAL_TOKENS,
        "seq_len": SEQ_LEN,
        "micro_batch_size": MICRO_BATCH_SIZE,
        "grad_accum_steps": GRAD_ACCUM_STEPS,
        "global_batch": GLOBAL_BATCH,
        "eval_every_tokens": EVAL_EVERY_TOKENS,
        "checkpoint_every_tokens": CHECKPOINT_EVERY_TOKENS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO,
        "max_projected_full_seconds": MAX_PROJECTED_FULL_SECONDS,
        "hard_full_timeout_seconds": HARD_FULL_TIMEOUT_SECONDS,
        "pricing_snapshot": {
            "h100_usd_per_second": H100_USD_PER_SECOND_SNAPSHOT,
            "starter_cpu_usd_per_core_second": STARTER_CPU_USD_PER_CORE_SECOND_SNAPSHOT,
            "starter_memory_usd_per_gib_second": STARTER_MEMORY_USD_PER_GIB_SECOND_SNAPSHOT,
        },
        "cortex_config": asdict(CORTEX_100M_CONFIG),
        "historical_transformer": dict(BASELINE_TRANSFORMER),
        "classification": "PAIRED_HISTORICAL_CONTROL_ADAPTIVE_EXPERIMENT",
        "breakthrough_claim_allowed": False,
        "continual_learning_claim_allowed": False,
    }
