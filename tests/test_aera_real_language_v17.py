from __future__ import annotations

import pytest

from tam_research import aera_real_language_v17 as v17


def test_v17_cpu_preflight_records_teacher_resolution_fix() -> None:
    result = v17.cpu_preflight()
    assert result["gpu_authorized"] is False
    assert result["micro_batch"] == 8
    assert result["legacy_binary_target_fractions_at_microbatch8"] == pytest.approx(
        [0.5, 0.375, 0.125]
    )
    assert result["exact_optional_stage_target_rates"] == pytest.approx(
        [0.5, 1.0 / 3.0, 1.0 / 6.0]
    )
    assert result["architecture_changed"] is False
    assert result["data_changed"] is False
    assert result["target_rates_changed"] is False
    assert result["inference_changed"] is False


def test_v17_policy_still_uses_v16_curriculum_boundaries() -> None:
    assert v17.CHUNK_SIZE == 256
    assert v17.DENSE_WARMUP_STEPS == 192
    assert v17.ROUTER_CALIBRATION_END == 384
    assert v17.SPARSE_CALIBRATION_EVERY == 4
