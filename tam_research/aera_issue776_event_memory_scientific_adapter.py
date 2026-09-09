from __future__ import annotations

"""#776 repaired event-memory scientific adapter.

This module reuses the frozen #748 training/evaluation/decision contract and
changes only the model-construction seam so A/B/C/D instantiate the merged
event-aligned IntegratedEventMemoryCapabilityLM at the original scientific size.

Importing this module allocates no GPU and consumes no scientific seed.
"""

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch

from tam_research import aera_issue748_memory_capability_seed1_harness_base as base
from tam_research import aera_memory_capability_gate_v1 as gate
from tam_research.aera_issue770_integrated_event_memory_cpu import (
    IntegratedEventMemoryCapabilityLM,
)

RESEARCH_ISSUE = 776
PARENT_INTEGRATION_ISSUE = 770
PARENT_CORRECTION_ISSUE = 773
SOURCE_MAIN = "0a838b5e20a867ddf886589445fb7e856644d109"
SOURCE_TREE = "88b06324ae046510f88e037667b6304c6e74c33e"
IMPLEMENTATION_BASE_MAIN = "5c53d405af9d76e20a1a3e10979fe3c88380157d"
IMPLEMENTATION_BASE_TREE = "81264d140d51cdba30a9c9f51abd9dc9dacabcb6"
CANDIDATE_MODEL_SEED = 27_641
CONSUMED_MODEL_SEED = 17_641
FUTURE_MODEL_SEED_LOCKED = 37_641

PREAUTH_PREFIX = "[aera-event-memory-repair-seed27641-preauth-v1]"
L4_PREFIX = "[aera-event-memory-repair-seed27641-l4-v1]"
RESULT_PATH = "/vol/aera-capability/event-memory-repair-v1-seed27641/result.json"
CHECKPOINT_DIR = "/vol/aera-capability/event-memory-repair-v1-seed27641/checkpoints"

TRAINED_VARIANTS = tuple(base.TRAINED_VARIANTS)
REFERENCE_VARIANTS = tuple(base.REFERENCE_VARIANTS)

GPU_AUTHORIZED = False
MODAL_SCIENTIFIC_EXECUTION_AUTHORIZED = False
SCIENTIFIC_TRAINING_AUTHORIZED = False
SCIENTIFIC_EVALUATION_AUTHORIZED = False
SCIENTIFIC_SEED_27641_AUTHORIZED = False
SEED_37641_AUTHORIZED = False
CHECKPOINT_WRITE_AUTHORIZED = False
RESULT_WRITE_AUTHORIZED = False
SYSTEMS_OPTIMIZATION_AUTHORIZED = False
ARCHITECTURE_FREEZE_AUTHORIZED = False
S2_REPLICATION_AUTHORIZED = False
SCALING_AUTHORIZED = False
BREAKTHROUGH_PROVEN = False

if CANDIDATE_MODEL_SEED != base.SCIENTIFIC_MODEL_SEEDS[1]:
    raise RuntimeError("#776 candidate seed drift")
if CONSUMED_MODEL_SEED != base.SCIENTIFIC_MODEL_SEEDS[0]:
    raise RuntimeError("#776 consumed seed drift")
if FUTURE_MODEL_SEED_LOCKED != base.SCIENTIFIC_MODEL_SEEDS[2]:
    raise RuntimeError("#776 future seed drift")


def _assert_candidate_seed_allowed(seed: int, *, scientific_seed_authorized: bool) -> None:
    seed = int(seed)
    if seed in base.SCIENTIFIC_MODEL_SEEDS and not scientific_seed_authorized:
        raise PermissionError("#776 scientific model seed requires separate explicit authorization")
    if scientific_seed_authorized and seed != CANDIDATE_MODEL_SEED:
        raise PermissionError("#776 authorization is scoped only to scientific seed 27641")


def build_repaired_model(
    variant_name: str,
    model_seed: int,
    *,
    scientific_seed_authorized: bool = False,
    device: torch.device | str = "cpu",
) -> IntegratedEventMemoryCapabilityLM:
    if variant_name not in TRAINED_VARIANTS and variant_name != "F_backbone_plus_recurrent_state":
        raise ValueError("#776 repaired scientific scaffold supports A/B/C/D and F only")
    _assert_candidate_seed_allowed(
        model_seed, scientific_seed_authorized=scientific_seed_authorized
    )
    torch.manual_seed(int(model_seed))
    model = IntegratedEventMemoryCapabilityLM(variant_name, base.scientific_config())
    return model.to(device)


@contextmanager
def _patched_base_builder() -> Iterator[None]:
    """Temporarily replace only the frozen #748 model-construction seam."""
    original = base.build_model
    if original is build_repaired_model:
        raise RuntimeError("#776 nested builder patch is forbidden")
    base.build_model = build_repaired_model
    try:
        yield
    finally:
        base.build_model = original


def train_variant(
    variant_name: str,
    model_seed: int,
    *,
    device: torch.device | str,
    scientific_seed_authorized: bool = False,
    max_steps: int | None = None,
    checkpoint_dir: str | Path | None = None,
):
    _assert_candidate_seed_allowed(
        model_seed, scientific_seed_authorized=scientific_seed_authorized
    )
    with _patched_base_builder():
        return base.train_variant(
            variant_name,
            model_seed,
            device=device,
            scientific_seed_authorized=scientific_seed_authorized,
            max_steps=max_steps,
            checkpoint_dir=checkpoint_dir,
        )


def load_model_checkpoint(
    variant_name: str,
    model_seed: int,
    checkpoint_path: str | Path,
    *,
    device: torch.device | str,
    scientific_seed_authorized: bool = False,
) -> IntegratedEventMemoryCapabilityLM:
    _assert_candidate_seed_allowed(
        model_seed, scientific_seed_authorized=scientific_seed_authorized
    )
    with _patched_base_builder():
        model = base.load_model_checkpoint(
            variant_name,
            model_seed,
            checkpoint_path,
            device=device,
            scientific_seed_authorized=scientific_seed_authorized,
        )
    if not isinstance(model, IntegratedEventMemoryCapabilityLM):
        raise RuntimeError("#776 checkpoint reload did not construct repaired model")
    return model


# Re-export the exact frozen data/evaluation/decision functions rather than copy them.
evaluation_cases = base.evaluation_cases
evaluate_model = base.evaluate_model
evaluate_simple_retrieval = base.evaluate_simple_retrieval
measure_model_inference_latency_ms = base.measure_model_inference_latency_ms
measure_simple_retrieval_latency_ms = base.measure_simple_retrieval_latency_ms
select_equal_gpu_time_checkpoints = base.select_equal_gpu_time_checkpoints
model_accounting = base.model_accounting
seed_decision = base.seed1_decision
validate_evaluation_metrics = base.validate_evaluation_metrics


def authority_snapshot() -> dict[str, bool]:
    return {
        "modal_scientific_execution": MODAL_SCIENTIFIC_EXECUTION_AUTHORIZED,
        "gpu": GPU_AUTHORIZED,
        "scientific_training": SCIENTIFIC_TRAINING_AUTHORIZED,
        "scientific_evaluation": SCIENTIFIC_EVALUATION_AUTHORIZED,
        "scientific_seed_27641": SCIENTIFIC_SEED_27641_AUTHORIZED,
        "seed_37641": SEED_37641_AUTHORIZED,
        "checkpoint_write": CHECKPOINT_WRITE_AUTHORIZED,
        "result_write": RESULT_WRITE_AUTHORIZED,
        "systems_optimization": SYSTEMS_OPTIMIZATION_AUTHORIZED,
        "architecture_freeze": ARCHITECTURE_FREEZE_AUTHORIZED,
        "s2_replication": S2_REPLICATION_AUTHORIZED,
        "scaling": SCALING_AUTHORIZED,
        "breakthrough_claim": BREAKTHROUGH_PROVEN,
    }


def protocol_snapshot() -> dict[str, Any]:
    return {
        "protocol_version": "aera-event-memory-repair-scientific-v1",
        "research_issue": RESEARCH_ISSUE,
        "parent_integration_issue": PARENT_INTEGRATION_ISSUE,
        "parent_correction_issue": PARENT_CORRECTION_ISSUE,
        "source_main": SOURCE_MAIN,
        "source_tree": SOURCE_TREE,
        "implementation_base_main": IMPLEMENTATION_BASE_MAIN,
        "implementation_base_tree": IMPLEMENTATION_BASE_TREE,
        "candidate_model_seed": CANDIDATE_MODEL_SEED,
        "consumed_model_seed": CONSUMED_MODEL_SEED,
        "future_model_seed_locked": FUTURE_MODEL_SEED_LOCKED,
        "preauth_prefix": PREAUTH_PREFIX,
        "l4_prefix": L4_PREFIX,
        "result_path": RESULT_PATH,
        "checkpoint_dir": CHECKPOINT_DIR,
        "scientific_model_config": dict(base.MODEL_CONFIG),
        "training": {
            "trained_variants": list(TRAINED_VARIANTS),
            "token_budget_per_variant": gate.TOKEN_BUDGET_PER_TRAINED_VARIANT,
            "optimizer_steps": base.OPTIMIZER_STEPS,
            "checkpoint_interval_tokens": gate.CHECKPOINT_INTERVAL_TOKENS,
            "checkpoint_every_steps": base.CHECKPOINT_EVERY_STEPS,
            "max_gpu_seconds_per_variant": gate.MAX_GPU_SECONDS_PER_VARIANT_SEED,
            "microbatch_size": base.MICROBATCH_SIZE,
            "grad_accum_steps": base.GRAD_ACCUM_STEPS,
            "sequence_chunks": base.TRAIN_SEQUENCE_CHUNKS,
            "tokens_per_optimizer_step": base.TOKENS_PER_OPTIMIZER_STEP,
            "adamw_lr": base.ADAMW_LR,
            "adamw_min_lr": base.LR_MIN,
            "warmup_steps": base.LR_WARMUP_STEPS,
            "betas": list(base.ADAMW_BETAS),
            "weight_decay": base.ADAMW_WEIGHT_DECAY,
            "grad_clip": base.GRAD_CLIP_NORM,
        },
        "evaluation": {
            "cases": 432,
            "unique_case_ids": 432,
            "nonreset_per_distance": {"2": 108, "8": 108, "32": 108},
            "reset_per_distance": {"8": 54, "32": 54},
            "batch_size": base.EVAL_BATCH_SIZE,
            "bootstrap_resamples": gate.BOOTSTRAP_RESAMPLES,
            "latency_batch_size": base.LATENCY_BATCH_SIZE,
            "latency_warmup_calls": base.LATENCY_WARMUP_CALLS,
            "latency_timed_calls": base.LATENCY_TIMED_CALLS,
            "equal_token_required": True,
            "equal_gpu_time_required": True,
        },
        "thresholds": {
            "long_distance_accuracy_gain_min": gate.LONG_DISTANCE_ACCURACY_GAIN_MIN,
            "paired_ci_lower_accuracy_gain_min": gate.PAIRED_CI_LOWER_ACCURACY_GAIN_MIN,
            "correction_accuracy_gain_min": gate.CORRECTION_ACCURACY_GAIN_MIN,
            "stale_value_error_reduction_min": gate.STALE_VALUE_ERROR_REDUCTION_MIN,
            "routing_control_accuracy_margin_min": gate.ROUTING_CONTROL_ACCURACY_MARGIN_MIN,
            "session_reset_leakage_max": gate.SESSION_RESET_LEAKAGE_MAX,
            "simple_retrieval_dominance_margin": gate.SIMPLE_RETRIEVAL_DOMINANCE_MARGIN,
            "memory_inference_latency_multiplier_max": gate.MEMORY_INFERENCE_LATENCY_MULTIPLIER_MAX,
            "memory_train_time_multiplier_max": gate.MEMORY_TRAIN_TIME_MULTIPLIER_MAX,
            "parameter_delta_fraction_max": gate.PARAMETER_DELTA_FRACTION_MAX,
        },
        "authority": authority_snapshot(),
    }


def normalize_repaired_decision(decision: dict[str, Any]) -> dict[str, Any]:
    """Preserve frozen threshold checks while making the next-step wording seed-aware."""
    out = dict(decision)
    legacy = str(out.get("recommendation", ""))
    out["legacy_frozen_recommendation"] = legacy
    if legacy == "CONFIRM_SEEDS_2_3_RECOMMENDED_REQUIRES_SEPARATE_AUTHORIZATION":
        out["recommendation"] = (
            "REPAIRED_SEED27641_POSITIVE_CONFIRM_SEED37641_REQUIRES_SEPARATE_AUTHORIZATION"
        )
    return out


def validate_repaired_result_schema(result: dict[str, Any]) -> None:
    required = {
        "research_issue",
        "source_main",
        "source_tree",
        "model_seed",
        "equal_token",
        "equal_gpu_time",
        "training",
        "inference_latency_ms",
        "simple_retrieval",
        "equal_gpu_time_checkpoint_selection",
        "decision",
        "authority",
    }
    missing = sorted(required - set(result))
    if missing:
        raise ValueError(f"#776 result missing keys: {missing}")
    if int(result["research_issue"]) != RESEARCH_ISSUE:
        raise ValueError("#776 result research issue drift")
    if int(result["model_seed"]) != CANDIDATE_MODEL_SEED:
        raise ValueError("#776 result seed drift")
    for surface_name in ("equal_token", "equal_gpu_time"):
        surface = result[surface_name]
        if set(surface) != set(TRAINED_VARIANTS):
            raise ValueError(f"#776 {surface_name} variant schema drift")
        for metrics in surface.values():
            validate_evaluation_metrics(metrics)
    validate_evaluation_metrics(result["simple_retrieval"])
    if set(result["training"]) != set(TRAINED_VARIANTS):
        raise ValueError("#776 training variant schema drift")
    if set(result["inference_latency_ms"]) != set(TRAINED_VARIANTS) | {"E_simple_retrieval"}:
        raise ValueError("#776 latency schema drift")
    authority = result["authority"]
    for key in (
        "seed_37641_authorized",
        "systems_optimization_authorized",
        "architecture_freeze_authorized",
        "s2_replication_authorized",
        "scaling_authorized",
        "breakthrough_proven",
    ):
        if authority.get(key) is not False:
            raise ValueError(f"#776 result authority drift: {key}")
