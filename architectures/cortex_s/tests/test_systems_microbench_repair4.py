from __future__ import annotations

import ast
import json
from pathlib import Path

from architectures.cortex_s.experiments.scale100m_2b import systems_microbench_v1_repair4 as repair4


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_repair4_protocol_is_fresh_and_engineering_only() -> None:
    protocol = repair4.validate_microbench_protocol()
    assert protocol["classification"] == "ENGINEERING_SYSTEMS_MICROBENCH_ONLY"
    assert protocol["engineering_seed"] == 2_026_090_904
    assert protocol["consumed_engineering_seeds"] == [910_001, 2_026_090_901, 2_026_090_902, 2_026_090_903]
    for seed in (8_100, 48_131, 48_132, 48_133):
        assert seed in protocol["forbidden_seeds"]
    assert protocol["parent_repair3_run_id"] == 34_335_827_718
    assert protocol["parent_repair3_job_id"] == 102_414_857_556
    assert protocol["full_training_authorized"] is False
    assert protocol["next_stage_authorized"] is False


def test_repair4_logical_padding_contract_and_cpu_semantics() -> None:
    probe = repair4.zero_gpu_repair4_contract_probe()
    assert probe["status"] == "PASS"
    assert probe["logical_hidden"] == 338
    assert probe["compute_hidden"] == 344
    assert probe["padding_channels"] == 6
    assert abs(probe["compute_overhead_ratio"] - 344 / 338) < 1e-12
    assert probe["w1_compute_shape"][-2:] == [512, 344]
    assert probe["w2_compute_shape"][-2:] == [344, 512]
    assert probe["w1_compute_stride"][-1] == 1
    assert probe["w1_compute_stride"][-2] % 8 == 0
    assert probe["w2_compute_stride"][-1] == 1
    assert probe["w2_compute_stride"][-2] % 8 == 0
    assert probe["no_slice_between_grouped_mm"] is True
    assert probe["parameter_count_changed"] is False
    assert probe["cpu_output_max_abs_delta"] <= 1e-5
    assert probe["cpu_input_grad_max_abs_delta"] <= 1e-5


def test_repair4_source_keeps_344_width_between_grouped_mms() -> None:
    path = REPO_ROOT / "architectures/cortex_s/experiments/scale100m_2b/systems_microbench_v1_repair4.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "LogicalPaddedBF16GroupedSparseMoE")
    forward = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "forward")
    forward_source = ast.get_source_segment(source, forward) or ""
    assert "self._w1_compute()" in forward_source
    assert "F.gelu(hidden)" in forward_source
    assert "self._w2_compute()" in forward_source
    assert "_aligned_row_view" not in forward_source
    assert "hidden[..., :" not in forward_source


def test_repair4_failed_timing_is_not_misclassified_as_speed_result() -> None:
    source = (REPO_ROOT / "architectures/cortex_s/experiments/scale100m_2b/systems_microbench_v1_repair4.py").read_text(encoding="utf-8")
    assert 'status = "ENGINEERING_RUNTIME_FAIL"' in source
    assert 'grouped.get("status") != "PASS"' in source
    assert 'status = "STOP_GROUPED_PATH"' in source


def test_repair4_launcher_uses_json_string_transport_and_has_no_2b_entrypoint() -> None:
    launcher_path = REPO_ROOT / "modal_cortex_s_100m_systems_microbench_v1_repair4.py"
    launcher = launcher_path.read_text(encoding="utf-8")
    ast.parse(launcher)
    assert 'ENGINEERING_SEED = 2_026_090_904' in launcher
    assert 'H100_TIMEOUT_SECONDS = 15 * 60' in launcher
    assert 'return payload' in launcher
    assert 'json.loads(result_json)' in launcher
    assert 'def full_2b(' not in launcher
    assert 'train_full_2b' not in launcher


def test_repair4_workflow_is_fresh_and_only_accepts_repair4_trigger() -> None:
    workflow = (REPO_ROOT / ".github/workflows/modal-cortex-s-100m-systems-microbench-v1-repair4.yml").read_text(encoding="utf-8")
    assert "[modal-cortex-s-100m-systems-microbench-v1-repair4]" in workflow
    assert "microbench-v1-repair4" in workflow
    assert "[modal-cortex-s-100m-systems-microbench-v1-repair3]" not in workflow
    assert "microbench-v1-repair3" not in workflow
    assert "modal run --detach" in workflow
