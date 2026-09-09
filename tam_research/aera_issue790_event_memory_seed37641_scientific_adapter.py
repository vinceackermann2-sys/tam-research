from __future__ import annotations

"""#790 seed37641 repaired event-memory scientific adapter.

This successor preserves the frozen #748 training/evaluation/decision contract,
uses the merged event-aligned model, and binds held-out materialization through
the authoritative #785 -> #750 bridge. Importing this module allocates no GPU
and consumes no scientific seed.
"""

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch

from tam_research import aera_issue785_eval_materialization_bridge_cpu as bridge
from tam_research.aera_issue770_integrated_event_memory_cpu import (
    IntegratedEventMemoryCapabilityLM,
)

old = bridge.repaired
base = old.base
gate = old.gate

RESEARCH_ISSUE = 790
PARENT_RUNNER_ISSUE = 776
PARENT_BRIDGE_ISSUE = 785
SOURCE_MAIN = "5e5d746c70380eb4819534eb5a061a2d142fa26f"
SOURCE_TREE = "2348be0e21c107d254146a7426b432f6f808687f"
IMPLEMENTATION_BASE_MAIN = "1c0ecd321a530daee32a4603a9cd6fb201be2d8a"
IMPLEMENTATION_BASE_TREE = "00dd4e489f5ad70fc68b5d195a9ba820ac299a54"

CANDIDATE_MODEL_SEED = 37_641
CONSUMED_MODEL_SEEDS = (17_641, 27_641)

PREAUTH_PREFIX = "[aera-event-memory-repair-seed37641-preauth-v1]"
L4_PREFIX = "[aera-event-memory-repair-seed37641-l4-v1]"
RESULT_PATH = "/vol/aera-capability/event-memory-repair-v2-seed37641/result.json"
CHECKPOINT_DIR = "/vol/aera-capability/event-memory-repair-v2-seed37641/checkpoints"

TRAINED_VARIANTS = tuple(base.TRAINED_VARIANTS)
REFERENCE_VARIANTS = tuple(base.REFERENCE_VARIANTS)

MODAL_SCIENTIFIC_EXECUTION_AUTHORIZED = False
GPU_AUTHORIZED = False
SCIENTIFIC_TRAINING_AUTHORIZED = False
SCIENTIFIC_EVALUATION_AUTHORIZED = False
SCIENTIFIC_SEED_37641_AUTHORIZED = False
CHECKPOINT_WRITE_AUTHORIZED = False
RESULT_WRITE_AUTHORIZED = False
SYSTEMS_OPTIMIZATION_AUTHORIZED = False
ARCHITECTURE_FREEZE_AUTHORIZED = False
S2_REPLICATION_AUTHORIZED = False
SCALING_AUTHORIZED = False
BREAKTHROUGH_PROVEN = False

if tuple(base.SCIENTIFIC_MODEL_SEEDS) != (17_641, 27_641, 37_641):
    raise RuntimeError("#790 frozen scientific seed set drift")
if bridge.evaluation_cases is not bridge.corrected_harness.evaluation_cases:
    raise RuntimeError("#790 bridge materializer drift")
if base.evaluation_cases is not bridge.evaluation_cases:
    raise RuntimeError("#790 corrected evaluator is not bound into frozen base runtime")


def _assert_candidate_seed_allowed(seed: int, *, scientific_seed_authorized: bool) -> None:
    seed = int(seed)
    if seed in CONSUMED_MODEL_SEEDS:
        raise PermissionError("#790 consumed scientific seeds 17641/27641 are permanently rejected")
    if seed in base.SCIENTIFIC_MODEL_SEEDS and not scientific_seed_authorized:
        raise PermissionError("#790 scientific model seed requires separate explicit authorization")
    if scientific_seed_authorized and seed != CANDIDATE_MODEL_SEED:
        raise PermissionError("#790 authorization is scoped only to scientific seed 37641")


def build_repaired_model(
    variant_name: str,
    model_seed: int,
    *,
    scientific_seed_authorized: bool = False,
    device: torch.device | str = "cpu",
) -> IntegratedEventMemoryCapabilityLM:
    if variant_name not in TRAINED_VARIANTS and variant_name != "F_backbone_plus_recurrent_state":
        raise ValueError("#790 repaired scientific scaffold supports A/B/C/D and F only")
    _assert_candidate_seed_allowed(
        model_seed, scientific_seed_authorized=scientific_seed_authorized
    )
    torch.manual_seed(int(model_seed))
    model = IntegratedEventMemoryCapabilityLM(variant_name, base.scientific_config())
    return model.to(device)


@contextmanager
def _patched_base_builder() -> Iterator[None]:
    original = base.build_model
    if original is build_repaired_model:
        raise RuntimeError("#790 nested builder patch is forbidden")
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
        raise RuntimeError("#790 checkpoint reload did not construct repaired model")
    return model


evaluation_cases = bridge.evaluation_cases
evaluate_model = bridge.evaluate_model
evaluate_simple_retrieval = bridge.evaluate_simple_retrieval
measure_model_inference_latency_ms = bridge.measure_model_inference_latency_ms
measure_simple_retrieval_latency_ms = bridge.measure_simple_retrieval_latency_ms
select_equal_gpu_time_checkpoints = bridge.select_equal_gpu_time_checkpoints
model_accounting = bridge.model_accounting
seed_decision = bridge.seed_decision
validate_evaluation_metrics = bridge.validate_evaluation_metrics


def authority_snapshot() -> dict[str, bool]:
    return {
        "modal_scientific_execution": MODAL_SCIENTIFIC_EXECUTION_AUTHORIZED,
        "gpu": GPU_AUTHORIZED,
        "scientific_training": SCIENTIFIC_TRAINING_AUTHORIZED,
        "scientific_evaluation": SCIENTIFIC_EVALUATION_AUTHORIZED,
        "scientific_seed_37641": SCIENTIFIC_SEED_37641_AUTHORIZED,
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
        "protocol_version": "aera-event-memory-repair-seed37641-scientific-v1",
        "research_issue": RESEARCH_ISSUE,
        "parent_runner_issue": PARENT_RUNNER_ISSUE,
        "parent_bridge_issue": PARENT_BRIDGE_ISSUE,
        "source_main": SOURCE_MAIN,
        "source_tree": SOURCE_TREE,
        "implementation_base_main": IMPLEMENTATION_BASE_MAIN,
        "implementation_base_tree": IMPLEMENTATION_BASE_TREE,
        "candidate_model_seed": CANDIDATE_MODEL_SEED,
        "consumed_model_seeds": list(CONSUMED_MODEL_SEEDS),
        "preauth_prefix": PREAUTH_PREFIX,
        "l4_prefix": L4_PREFIX,
        "result_path": RESULT_PATH,
        "checkpoint_dir": CHECKPOINT_DIR,
        "scientific_model_config": dict(base.MODEL_CONFIG),
        "materialization": {
            "bridge_issue": 785,
            "bridge_function": "evaluation_cases",
            "corrected_harness_issue": 750,
            "cases": 432,
            "unique_case_ids": 432,
            "nonreset_per_distance": {"2": 108, "8": 108, "32": 108},
            "reset_per_distance": {"8": 54, "32": 54},
            "raw_base_materializer_used": False,
        },
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


def normalize_final_seed_decision(decision: dict[str, Any]) -> dict[str, Any]:
    out = dict(decision)
    legacy = str(out.get("recommendation", ""))
    out["legacy_frozen_recommendation"] = legacy
    if legacy == "CONFIRM_SEEDS_2_3_RECOMMENDED_REQUIRES_SEPARATE_AUTHORIZATION":
        out["recommendation"] = (
            "REPAIRED_SEED37641_POSITIVE_BOUNDED_CAPABILITY_EVIDENCE_REVIEW_REQUIRED"
        )
    return out


def validate_seed37641_result_schema(result: dict[str, Any]) -> None:
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
        raise ValueError(f"#790 result missing keys: {missing}")
    if int(result["research_issue"]) != RESEARCH_ISSUE:
        raise ValueError("#790 result research issue drift")
    if int(result["model_seed"]) != CANDIDATE_MODEL_SEED:
        raise ValueError("#790 result seed drift")
    for surface_name in ("equal_token", "equal_gpu_time"):
        surface = result[surface_name]
        if set(surface) != set(TRAINED_VARIANTS):
            raise ValueError(f"#790 {surface_name} variant schema drift")
        for metrics in surface.values():
            validate_evaluation_metrics(metrics)
    validate_evaluation_metrics(result["simple_retrieval"])
    if set(result["training"]) != set(TRAINED_VARIANTS):
        raise ValueError("#790 training variant schema drift")
    if set(result["inference_latency_ms"]) != set(TRAINED_VARIANTS) | {"E_simple_retrieval"}:
        raise ValueError("#790 latency schema drift")
    authority = result["authority"]
    for key in (
        "systems_optimization_authorized",
        "architecture_freeze_authorized",
        "s2_replication_authorized",
        "scaling_authorized",
        "breakthrough_proven",
    ):
        if authority.get(key) is not False:
            raise ValueError(f"#790 result authority drift: {key}")
