from __future__ import annotations

from tam_research.aera_real_language import SEQ_LEN
from tam_research.aera_real_language_v14 import (
    CHUNK_SIZE,
    DENSE_WARMUP_STEPS,
    ROUTER_CALIBRATION_END,
    SPARSE_CALIBRATION_EVERY,
    cpu_preflight,
    router_policy_for_step,
)


def test_v14_preserves_two_256_token_chunk_geometry():
    assert CHUNK_SIZE == 256
    assert SEQ_LEN == 512
    assert SEQ_LEN // CHUNK_SIZE == 2


def test_v14_keeps_v13_persistent_calibration_schedule():
    warmup = router_policy_for_step(DENSE_WARMUP_STEPS - 1)
    first_calibration = router_policy_for_step(DENSE_WARMUP_STEPS)
    first_sparse_calibration = router_policy_for_step(ROUTER_CALIBRATION_END)
    first_hard = router_policy_for_step(ROUTER_CALIBRATION_END + 1)
    next_sparse_calibration = router_policy_for_step(
        ROUTER_CALIBRATION_END + SPARSE_CALIBRATION_EVERY
    )

    assert warmup["trainable"] is False
    assert first_calibration["supervised"] is True
    assert first_sparse_calibration["supervised"] is True
    assert first_hard["trainable"] is False
    assert next_sparse_calibration["supervised"] is True


def test_v14_cpu_preflight_is_gpu_blocked_and_records_gradient_contract():
    result = cpu_preflight()
    assert result["gpu_authorized"] is False
    assert result["version"] == "aera-v14-router-gradient-isolation"
    assert result["routing"]["target_mean_total_stage_execution"] == 0.50
    assert result["routing"]["calibration_primary_task_router_gradient"] == "detached"
    assert result["routing"]["calibration_explicit_router_supervision_gradient"] == "enabled"
    assert abs(result["parameter_accounting"]["stored_parameter_delta_fraction"]) <= 0.05
