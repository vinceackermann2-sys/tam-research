from __future__ import annotations

import json

from tam_research import aera_real_language_v18_gpu as v18g
from tam_research import aera_real_language_v19_gpu as v19g


def test_v19_development_seed_and_eval_seeds_are_fresh_and_frozen() -> None:
    assert v19g.SEED == 8371
    assert v19g.EVAL_SEED == 98_371
    assert v19g.MEMORY_EVAL_SEED == 108_371
    assert v19g.SYSTEMS_EVAL_SEED == 118_371
    assert v19g.SYSTEM_BATCH_SIZES == (8, 64)
    assert v19g.MEMORY_EVAL_BATCHES == 16
    assert v19g.MEMORY_EVAL_BATCH_SIZE == 8


def test_v19_inherits_every_v18_development_threshold_without_relaxation() -> None:
    names = (
        "QUALITY_GAP_MAX_NLL",
        "MEMORY_SECOND_CHUNK_MIN_ADVANTAGE_NLL",
        "MEMORY_OVERALL_MIN_ADVANTAGE_NLL",
        "WRITE_MEAN_MIN",
        "WRITE_MEAN_MAX",
        "WRITE_SPREAD_MIN",
        "OPTIONAL_STAGE_TARGET_MAE_MAX",
        "OPTIONAL_STAGE_MIN_RUN_FRACTION",
        "TOTAL_STAGE_EXEC_MIN",
        "TOTAL_STAGE_EXEC_MAX",
        "BATCH8_MIN_SPEED_RATIO",
        "BATCH64_MIN_SPEED_RATIO",
    )
    for name in names:
        assert getattr(v19g, name) == getattr(v18g, name), name
    assert v19g.MEMORY_SECOND_CHUNK_MIN_ADVANTAGE_NLL == 0.005
    assert v19g.BATCH8_MIN_SPEED_RATIO == 0.25
    assert v19g.BATCH64_MIN_SPEED_RATIO == 1.25


def test_v19_frozen_summary_authorizes_no_gpu_by_itself() -> None:
    summary = v19g.frozen_protocol_summary()
    assert summary["development_only"] is True
    assert summary["thresholds_identical_to_v18"] is True
    assert summary["gpu_authorized_by_module"] is False
    assert "token-wise" in summary["architecture_change"]


def test_v19_result_remap_preserves_failed_or_passed_checks_without_reinterpretation(tmp_path) -> None:
    inherited = {
        "protocol": {"development_seed": 8371},
        "v18_memory_eval": {"second_chunk": {"memory_advantage_nll": 0.006}},
        "v18_heldout_adaptivity": {"pass": True},
        "v18_systems_eval": {"rows": {"8": {}, "64": {}}},
        "v18_development_checks": {
            "memory_second_chunk_advantage_ge_0_005": True,
            "batch64_memory_enabled_speed_ge_1_25x": False,
        },
        "v18_development_pass": False,
        "claims": {"100m_authorized": True},
    }
    got = v19g._remap_result(inherited, str(tmp_path))
    assert "v18_memory_eval" not in got
    assert got["v19_memory_eval"]["second_chunk"]["memory_advantage_nll"] == 0.006
    assert got["v19_development_checks"]["batch64_memory_enabled_speed_ge_1_25x"] is False
    assert got["v19_development_pass"] is False
    assert got["claims"]["100m_authorized"] is False
    assert got["claims"]["counts_toward_independent_replication"] is False
    payload = json.loads((tmp_path / "result.json").read_text())
    assert payload["v19_development_pass"] is False
    assert payload["protocol"]["architecture_delta_from_v18"]["routing_changed"] is False
