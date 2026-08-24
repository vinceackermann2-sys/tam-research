from __future__ import annotations

"""Guarded AERA-v19 real-language token-wise-memory development harness.

This deliberately reuses the frozen v18 training/evaluation implementation and
thresholds. The only architecture binding changed is v18 -> v19, whose CPU gates
prove that routing, predictive stream, memory write equation/timing, checkpoint
layout and parameter count are unchanged while prior-memory reads become token-wise.

Seed 8371 is development-only and cannot count toward independent replication.
"""

import json
from pathlib import Path
from typing import Any

from . import aera_real_language_v18_gpu as base
from . import aera_real_language_v19 as v19

SEED = 8371
EVAL_SEED = 98_371
MEMORY_EVAL_SEED = 108_371
SYSTEMS_EVAL_SEED = 118_371

# These are aliases to the already-preregistered v18 gates. V19 is not allowed to
# relax them after seeing seed8351 fail.
QUALITY_GAP_MAX_NLL = base.QUALITY_GAP_MAX_NLL
MEMORY_SECOND_CHUNK_MIN_ADVANTAGE_NLL = base.MEMORY_SECOND_CHUNK_MIN_ADVANTAGE_NLL
MEMORY_OVERALL_MIN_ADVANTAGE_NLL = base.MEMORY_OVERALL_MIN_ADVANTAGE_NLL
WRITE_MEAN_MIN = base.WRITE_MEAN_MIN
WRITE_MEAN_MAX = base.WRITE_MEAN_MAX
WRITE_SPREAD_MIN = base.WRITE_SPREAD_MIN
OPTIONAL_STAGE_TARGET_MAE_MAX = base.OPTIONAL_STAGE_TARGET_MAE_MAX
OPTIONAL_STAGE_MIN_RUN_FRACTION = base.OPTIONAL_STAGE_MIN_RUN_FRACTION
TOTAL_STAGE_EXEC_MIN = base.TOTAL_STAGE_EXEC_MIN
TOTAL_STAGE_EXEC_MAX = base.TOTAL_STAGE_EXEC_MAX
BATCH8_MIN_SPEED_RATIO = base.BATCH8_MIN_SPEED_RATIO
BATCH64_MIN_SPEED_RATIO = base.BATCH64_MIN_SPEED_RATIO
SYSTEM_BATCH_SIZES = base.SYSTEM_BATCH_SIZES
MEMORY_EVAL_BATCHES = base.MEMORY_EVAL_BATCHES
MEMORY_EVAL_BATCH_SIZE = base.MEMORY_EVAL_BATCH_SIZE


def _install_v19_binding() -> None:
    """Point the frozen v18 harness at v19 and fresh development seeds."""
    base.v18 = v19
    base.SEED = SEED
    base.EVAL_SEED = EVAL_SEED
    base.MEMORY_EVAL_SEED = MEMORY_EVAL_SEED
    base.SYSTEMS_EVAL_SEED = SYSTEMS_EVAL_SEED


def _decorate_protocol(protocol: dict[str, Any]) -> dict[str, Any]:
    if protocol.get("development_seed") != SEED:
        raise RuntimeError("v19 base protocol did not pick up frozen seed")
    protocol["version"] = "aera-v19-tokenwise-fast-memory-development"
    protocol["gpu_authorization_scope"] = (
        "one guarded AERA-v19 development seed8371 L4 run only"
    )
    protocol["counts_toward_independent_replication"] = False
    protocol["architecture_delta_from_v18"] = {
        "fast_memory_read_addressing": "broadcast-first-token -> token-wise prior-state queries",
        "routing_changed": False,
        "memory_write_rule_changed": False,
        "memory_write_timing_changed": False,
        "memory_dimension_changed": False,
        "checkpoint_layout_changed": False,
        "stored_parameter_count_changed": False,
    }
    protocol["thresholds_inherited_unchanged_from_v18_seed8351"] = {
        "quality_gap_max_nll": QUALITY_GAP_MAX_NLL,
        "memory_second_chunk_min_advantage_nll": MEMORY_SECOND_CHUNK_MIN_ADVANTAGE_NLL,
        "memory_overall_min_advantage_nll": MEMORY_OVERALL_MIN_ADVANTAGE_NLL,
        "write_mean_range": [WRITE_MEAN_MIN, WRITE_MEAN_MAX],
        "write_spread_min": WRITE_SPREAD_MIN,
        "optional_stage_target_mae_max": OPTIONAL_STAGE_TARGET_MAE_MAX,
        "optional_stage_min_run_fraction": OPTIONAL_STAGE_MIN_RUN_FRACTION,
        "total_stage_execution_range": [TOTAL_STAGE_EXEC_MIN, TOTAL_STAGE_EXEC_MAX],
        "batch8_min_speed_ratio": BATCH8_MIN_SPEED_RATIO,
        "batch64_min_speed_ratio": BATCH64_MIN_SPEED_RATIO,
    }
    return protocol


def validate_protocol(data_dir: str) -> dict[str, Any]:
    _install_v19_binding()
    return _decorate_protocol(base.validate_protocol(data_dir))


def _remap_result(result: dict[str, Any], run_dir: str) -> dict[str, Any]:
    for old, new in (
        ("v18_memory_eval", "v19_memory_eval"),
        ("v18_heldout_adaptivity", "v19_heldout_adaptivity"),
        ("v18_systems_eval", "v19_systems_eval"),
        ("v18_development_checks", "v19_development_checks"),
        ("v18_development_pass", "v19_development_pass"),
    ):
        if old not in result:
            raise RuntimeError(f"v19 expected inherited result key {old!r}")
        result[new] = result.pop(old)

    protocol = result.get("protocol")
    if not isinstance(protocol, dict):
        raise RuntimeError("v19 inherited result missing protocol")
    result["protocol"] = _decorate_protocol(protocol)
    result["claims"] = {
        "development_seed_only": True,
        "counts_toward_independent_replication": False,
        "architecture_frozen_for_replication": False,
        "100m_authorized": False,
        "breakthrough_proven": False,
    }
    Path(run_dir, "result.json").write_text(json.dumps(result, indent=2))
    return result


def train_matched_pair(*, data_dir: str, run_dir: str, seed: int = SEED) -> dict[str, Any]:
    if seed != SEED:
        raise ValueError(f"v19 development run is frozen to fresh seed {SEED}")
    _install_v19_binding()

    # The inherited trainer installs base.validate_protocol into v11. Temporarily
    # decorate that function so the checkpoint/result embeds the v19 protocol.
    original_validate = base.validate_protocol

    def v19_validate(path: str) -> dict[str, Any]:
        _install_v19_binding()
        return _decorate_protocol(original_validate(path))

    base.validate_protocol = v19_validate
    try:
        result = base.train_matched_pair(data_dir=data_dir, run_dir=run_dir, seed=seed)
    finally:
        base.validate_protocol = original_validate
    return _remap_result(result, run_dir)


def frozen_protocol_summary() -> dict[str, Any]:
    return {
        "seed": SEED,
        "eval_seed": EVAL_SEED,
        "memory_eval_seed": MEMORY_EVAL_SEED,
        "systems_eval_seed": SYSTEMS_EVAL_SEED,
        "development_only": True,
        "architecture_change": "token-wise reads from fixed prior-chunk fast-memory state only",
        "thresholds_identical_to_v18": True,
        "quality_gap_max_nll": QUALITY_GAP_MAX_NLL,
        "memory_second_chunk_min_advantage_nll": MEMORY_SECOND_CHUNK_MIN_ADVANTAGE_NLL,
        "batch8_min_speed_ratio": BATCH8_MIN_SPEED_RATIO,
        "batch64_min_speed_ratio": BATCH64_MIN_SPEED_RATIO,
        "gpu_authorized_by_module": False,
    }
