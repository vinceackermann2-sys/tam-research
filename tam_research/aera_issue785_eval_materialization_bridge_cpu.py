from __future__ import annotations

"""#785 CPU-only bridge from repaired #776 model surface to #750 evaluation materialization.

This module deliberately performs no model construction, training, evaluation,
checkpoint/result I/O, CUDA use, or Modal execution at import time. It exists
only to make the authoritative #750 held-out case materializer the evaluation
source for the repaired #776 scientific surface.
"""

from typing import Any

from tam_research import aera_issue776_event_memory_scientific_adapter as repaired
from tam_research import aera_issue748_memory_capability_seed1_harness as corrected_harness

RESEARCH_ISSUE = 785
PARENT_RESEARCH_ISSUE = 776
CONSUMED_L4_ISSUE = 783

ISSUE_CREATION_MAIN = "a781647a0203c5023c5781a7718fa277947130b7"
ISSUE_CREATION_TREE = "ed4e95fd120d3c634854d94c977a11d3a951696b"
IMPLEMENTATION_BASE_MAIN = "f2869318ac9668b8255ad548afa82ab513c0a024"
IMPLEMENTATION_BASE_TREE = "2ea29d80157718f3358a2d2ba49fdefbec03f8ea"

CORRECTED_HARNESS_BLOB = "afc939a69633f68ded05eb585c95a599a8f5c981"
REPAIRED_ADAPTER_BLOB = "1489df753f2eee0f3acc5aba049c0eeb40fce10d"
FROZEN_BASE_BLOB = "40003b68987d026265b1d53fcb637f18277f9cac"
FROZEN_GATE_BLOB = "981602432b684989f7a5011ff953e6965368c5e5"
INTEGRATED_REPAIRED_MODEL_BLOB = "b695b1b7b7476be3a16433c96dff32870f2e1f49"
V18_CORE_BLOB = "97861a2407876f62665b13140c2135b4a11d4597"
DELTA_MEMORY_BLOB = "ec0b5d29b3d4ac27bd60fd9c152480b9f177c3e9"

CONSUMED_SCIENTIFIC_SEEDS = (17_641, 27_641)
FUTURE_SCIENTIFIC_SEED_LOCKED = 37_641

MODAL_EXECUTION_AUTHORIZED = False
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

if repaired.base.evaluation_cases is not corrected_harness.evaluation_cases:
    raise RuntimeError("#785 corrected evaluation materializer did not bind into frozen base runtime")

evaluation_cases = corrected_harness.evaluation_cases

build_repaired_model = repaired.build_repaired_model
train_variant = repaired.train_variant
load_model_checkpoint = repaired.load_model_checkpoint
evaluate_model = repaired.evaluate_model
evaluate_simple_retrieval = repaired.evaluate_simple_retrieval
measure_model_inference_latency_ms = repaired.measure_model_inference_latency_ms
measure_simple_retrieval_latency_ms = repaired.measure_simple_retrieval_latency_ms
select_equal_gpu_time_checkpoints = repaired.select_equal_gpu_time_checkpoints
model_accounting = repaired.model_accounting
seed_decision = repaired.seed_decision
validate_evaluation_metrics = repaired.validate_evaluation_metrics
normalize_repaired_decision = repaired.normalize_repaired_decision
validate_repaired_result_schema = repaired.validate_repaired_result_schema


def authority_snapshot() -> dict[str, bool]:
    return {
        "modal_execution": MODAL_EXECUTION_AUTHORIZED,
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


def bridge_protocol_snapshot() -> dict[str, Any]:
    gate = repaired.base.gate
    return {
        "protocol_version": "aera-issue785-eval-materialization-bridge-cpu-v1",
        "research_issue": RESEARCH_ISSUE,
        "parent_research_issue": PARENT_RESEARCH_ISSUE,
        "consumed_l4_issue": CONSUMED_L4_ISSUE,
        "issue_creation_main": ISSUE_CREATION_MAIN,
        "issue_creation_tree": ISSUE_CREATION_TREE,
        "implementation_base_main": IMPLEMENTATION_BASE_MAIN,
        "implementation_base_tree": IMPLEMENTATION_BASE_TREE,
        "lineage_blobs": {
            "corrected_harness": CORRECTED_HARNESS_BLOB,
            "repaired_adapter": REPAIRED_ADAPTER_BLOB,
            "frozen_base": FROZEN_BASE_BLOB,
            "frozen_gate": FROZEN_GATE_BLOB,
            "integrated_repaired_model": INTEGRATED_REPAIRED_MODEL_BLOB,
            "v18_core": V18_CORE_BLOB,
            "delta_memory": DELTA_MEMORY_BLOB,
        },
        "repair": {
            "materializer_module": "tam_research.aera_issue748_memory_capability_seed1_harness",
            "materializer_function": "evaluation_cases",
            "capacity_error": "records exceed chunk capacity",
            "max_attempts": 4096,
            "copies_retry_algorithm": False,
            "raw_base_materializer_used": False,
        },
        "evaluation": {
            "total_cases": 432,
            "unique_sample_ids": 432,
            "nonreset_per_distance": {"2": 108, "8": 108, "32": 108},
            "reset_per_distance": {"8": 54, "32": 54},
            "split": "eval",
            "heldout_pair_rule": "(key_index + value_index) % 5 == 4",
            "retention_distances": list(gate.EVAL_RETENTION_DISTANCES),
        },
        "scientific_seeds": {
            "consumed": list(CONSUMED_SCIENTIFIC_SEEDS),
            "future_locked": FUTURE_SCIENTIFIC_SEED_LOCKED,
        },
        "authority": authority_snapshot(),
    }
