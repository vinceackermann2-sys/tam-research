from __future__ import annotations

import ast
from pathlib import Path

from architectures.cortex_s.experiments.scale100m_2b import systems_microbench_v1_repair3 as repair3


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_repair3_protocol_is_engineering_only_and_fresh() -> None:
    protocol = repair3.validate_microbench_protocol()
    assert protocol["classification"] == "ENGINEERING_SYSTEMS_MICROBENCH_ONLY"
    assert protocol["engineering_seed"] == 2_026_090_903
    assert protocol["consumed_engineering_seeds"] == [910_001, 2_026_090_901, 2_026_090_902]
    assert 8_100 in protocol["forbidden_seeds"]
    assert 48_131 in protocol["forbidden_seeds"]
    assert 48_132 in protocol["forbidden_seeds"]
    assert 48_133 in protocol["forbidden_seeds"]
    assert protocol["parent_repair2_run_id"] == 34_329_354_099
    assert protocol["parent_repair2_job_id"] == 102_394_032_871
    assert protocol["full_training_authorized"] is False
    assert protocol["next_stage_authorized"] is False


def test_repair3_scoping_probe_catches_consumed_failure_class() -> None:
    probe = repair3.zero_gpu_scoping_contract_probe()
    assert probe["status"] == "PASS"
    assert probe["torch_resolves_from_module_globals"] is True
    assert probe["dynamo_is_local"] is True
    assert "UnboundLocalError" in probe["parent_failure"]
    assert "torch" not in repair3._benchmark_full_model_variant.__code__.co_varnames
    assert "dynamo" in repair3._benchmark_full_model_variant.__code__.co_varnames


def test_repair3_function_has_no_function_local_torch_import() -> None:
    path = REPO_ROOT / "architectures/cortex_s/experiments/scale100m_2b/systems_microbench_v1_repair3.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    benchmark = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_benchmark_full_model_variant"
    )
    bad = []
    for node in ast.walk(benchmark):
        if isinstance(node, ast.Import):
            bad.extend(alias.name for alias in node.names if alias.name == "torch" or alias.name.startswith("torch."))
    assert bad == []


def test_repair3_reuses_repair2_stride_fix_without_parameter_change() -> None:
    probe = repair3.zero_gpu_repair3_contract_probe()
    assert probe["status"] == "PASS"
    layout = probe["layout_contract"]
    assert layout["old_repair1_w1_valid"] is False
    assert layout["repair2_w1_valid"] is True
    assert layout["repair2_post_gelu_valid"] is True
    assert layout["repair2_w2_valid"] is True
    assert layout["logical_hidden"] == 338
    assert layout["physical_hidden_row_stride"] == 344
    assert layout["parameter_count_changed"] is False


def test_repair3_launcher_and_workflow_contain_no_full_training_entrypoint() -> None:
    launcher = (REPO_ROOT / "modal_cortex_s_100m_systems_microbench_v1_repair3.py").read_text(encoding="utf-8")
    workflow = (REPO_ROOT / ".github/workflows/modal-cortex-s-100m-systems-microbench-v1-repair3.yml").read_text(encoding="utf-8")
    ast.parse(launcher)
    assert "def full_2b(" not in launcher
    assert "train_full_2b" not in launcher
    assert "ENGINEERING_SEED = 2_026_090_903" in launcher
    assert "H100_TIMEOUT_SECONDS = 15 * 60" in launcher
    assert "[modal-cortex-s-100m-systems-microbench-v1-repair3]" in workflow
    assert "microbench-v1-repair3" in workflow
