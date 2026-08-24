from __future__ import annotations

from tam_research import aera_real_language_v19 as v19


def test_v19_cpu_preflight_preserves_v18_protocol_except_read_addressing() -> None:
    result = v19.cpu_preflight()
    assert result["cpu_diagnostic_seed"] == 8361
    assert result["gpu_authorized"] is False
    assert result["routing_schedule_changed"] is False
    assert result["routing_teacher_changed"] is False
    assert result["optional_stage_targets_changed"] is False
    assert result["hard_threshold_changed"] is False
    assert result["checkpoint_layout_changed"] is False
    assert result["stored_parameter_count_changed"] is False
    assert result["memory"]["read_granularity"] == "token-wise"
    assert result["memory"]["memory_write_rule_changed"] is False
    assert result["memory"]["routing_changed_from_v17"] is False
