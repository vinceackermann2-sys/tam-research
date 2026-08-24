from __future__ import annotations

from tam_research import aera_real_language_v17 as v17
from tam_research import aera_real_language_v18 as v18


def test_v18_keeps_v17_routing_schedule_and_enables_memory_only_for_pretraining() -> None:
    assert v18.CHUNK_SIZE == v17.CHUNK_SIZE
    assert v18.DENSE_WARMUP_STEPS == v17.DENSE_WARMUP_STEPS
    assert v18.ROUTER_CALIBRATION_END == v17.ROUTER_CALIBRATION_END
    assert v18.SPARSE_CALIBRATION_EVERY == v17.SPARSE_CALIBRATION_EVERY
    assert v18.STAGE_DIFFICULTY_WEIGHT == v17.STAGE_DIFFICULTY_WEIGHT
    assert v18.STAGE_BUDGET_WEIGHT == v17.STAGE_BUDGET_WEIGHT
    assert v18.STAGE_POLARIZATION_WEIGHT == v17.STAGE_POLARIZATION_WEIGHT


def test_v18_cpu_preflight_records_no_gpu_authorization_and_no_parameter_change() -> None:
    result = v18.cpu_preflight()
    assert result["gpu_authorized"] is False
    assert result["routing_schedule_changed"] is False
    assert result["routing_teacher_changed"] is False
    assert result["optional_stage_targets_changed"] is False
    assert result["hard_threshold_changed"] is False
    assert result["real_language_update_memory"] is True
    assert result["deployment_memory_pretraining_mode_default"] is False
    assert result["memory"]["deployment_local_update_detached"] is True
    assert result["memory"]["base_pretraining_update_differentiable"] is True
