from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "tam_research" / "aera_v26_8_issue581_first_divergence_no_grad.py"
FROZEN_LOCALIZER = ROOT / "tam_research" / "aera_v26_8_issue578_first_divergence_localizer.py"
FROZEN_LAUNCHER = ROOT / "modal_aera_v26_8_issue578_first_divergence_localizer_app.py"
FROZEN_WORKFLOW = ROOT / ".github" / "workflows" / "aera-v26-8-issue578-first-divergence-localizer-l4.yml"
FROZEN_CPU_TEST = ROOT / "tests" / "test_aera_v26_8_issue578_first_divergence_localizer_cpu.py"

SOURCE_MAIN = "eba8d04ed262c9bf539a99d256af5e99eb6a87d1"
SOURCE_TREE = "e0275443a58a0dd8528e1329988713dbb2045bd0"
FROZEN_LOCALIZER_BLOB = "5933d1113b950348bb63e5f3eb8713ad36dbf580"
FROZEN_LAUNCHER_BLOB = "cd47e1252bed5617556998659eadfe6a61637d39"
FROZEN_WORKFLOW_BLOB = "b76282733903d220e7118ede283f789db0eb56ba"
FROZEN_CPU_TEST_BLOB = "6dd02b5a25514ad9987d7617e4a4b1ddbb1e6f0a"


def _blob(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def _literal_constant(path: Path, name: str):
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise AssertionError(f"missing literal constant {name}")


def test_issue581_freezes_source_and_historical_issue578_bytes() -> None:
    assert _literal_constant(WRAPPER, "RESEARCH_ISSUE") == 581
    assert _literal_constant(WRAPPER, "SOURCE_MAIN") == SOURCE_MAIN
    assert _literal_constant(WRAPPER, "SOURCE_TREE") == SOURCE_TREE
    assert _blob(FROZEN_LOCALIZER) == FROZEN_LOCALIZER_BLOB
    assert _blob(FROZEN_LAUNCHER) == FROZEN_LAUNCHER_BLOB
    assert _blob(FROZEN_WORKFLOW) == FROZEN_WORKFLOW_BLOB
    assert _blob(FROZEN_CPU_TEST) == FROZEN_CPU_TEST_BLOB


def test_issue581_freezes_consumed_issue580_pre_result_evidence() -> None:
    source = WRAPPER.read_text()
    assert _literal_constant(WRAPPER, "ISSUE580_TRIGGER") == 580
    assert _literal_constant(WRAPPER, "ISSUE580_RUN") == 33748196657
    assert _literal_constant(WRAPPER, "ISSUE580_JOB") == 100625461189
    assert _literal_constant(WRAPPER, "ISSUE580_ATTEMPT") == 1
    assert _literal_constant(WRAPPER, "ISSUE580_BOUND_MAIN") == SOURCE_MAIN
    assert _literal_constant(WRAPPER, "ISSUE580_L4_STARTED") is True
    assert _literal_constant(WRAPPER, "ISSUE580_AUTHORITATIVE_RESULT_EMITTED") is False
    assert _literal_constant(WRAPPER, "ISSUE580_FAILURE") == (
        "Inference tensors do not track version counter."
    )
    assert '"issue580_authoritative_result_emitted": ISSUE580_AUTHORITATIVE_RESULT_EMITTED' in source


def test_issue581_wrapper_is_only_no_grad_undecorated_delegation() -> None:
    source = WRAPPER.read_text()
    tree = ast.parse(source)
    functions = {
        node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    entry = functions["run_first_divergence_localization_issue581"]
    assert len(entry.decorator_list) == 1
    decorator = ast.unparse(entry.decorator_list[0])
    assert decorator == "torch.no_grad()"
    assert source.count('getattr(target, "__wrapped__", None)') == 2
    entry_source = ast.get_source_segment(source, entry)
    assert entry_source is not None
    assert entry_source.count("undecorated(run_dir=run_dir)") == 1
    assert "issue578.run_first_divergence_localization(" not in entry_source
    assert "torch.inference_mode(" not in entry_source
    assert "torch.is_inference_mode_enabled()" in entry_source
    assert "torch.is_grad_enabled()" in entry_source
    for copied_marker in (
        "for batch_size in BATCH_SIZES",
        "_capture_model(",
        "_compare_records(",
        "issue562.load_models_v26_8(",
        "parameter._version",
    ):
        assert copied_marker not in entry_source


def test_issue581_cpu_execution_context_reproduces_failure_and_repairs_it() -> None:
    @torch.inference_mode()
    def historical_shape():
        module = torch.nn.Linear(2, 2)
        return tuple(int(parameter._version) for parameter in module.parameters())

    with pytest.raises(RuntimeError, match="Inference tensors do not track version counter"):
        historical_shape()

    @torch.no_grad()
    def successor_shape():
        assert torch.is_grad_enabled() is False
        assert torch.is_inference_mode_enabled() is False
        module = torch.nn.Linear(2, 2)
        before = tuple(int(parameter._version) for parameter in module.parameters())
        with torch.no_grad():
            module.weight.add_(1.0)
        after = tuple(int(parameter._version) for parameter in module.parameters())
        return before, after

    before, after = successor_shape()
    assert len(before) == len(after) == 2
    assert after[0] == before[0] + 1
    assert after[1] == before[1]


def test_issue581_no_grad_does_not_hide_outer_inference_mode() -> None:
    @torch.no_grad()
    def guarded_shape():
        if torch.is_inference_mode_enabled():
            raise RuntimeError("outer inference mode refused")
        return True

    assert guarded_shape() is True
    with torch.inference_mode():
        with pytest.raises(RuntimeError, match="outer inference mode refused"):
            guarded_shape()


def test_issue581_frozen_body_retains_both_parameter_version_snapshots() -> None:
    source = FROZEN_LOCALIZER.read_text()
    assert "@torch.inference_mode()\ndef run_first_divergence_localization(" in source
    assert source.count("parameter._version") == 4
    assert "reference_versions = tuple(int(parameter._version)" in source
    assert "candidate_versions = tuple(int(parameter._version)" in source
    assert "reference_versions == tuple(int(parameter._version)" in source
    assert "candidate_versions == tuple(int(parameter._version)" in source


def test_issue581_contains_no_gpu_launcher_or_higher_authorization() -> None:
    source = WRAPPER.read_text()
    lowered = source.lower()
    assert 'gpu="l4"' not in lowered
    assert "modal." not in lowered
    assert "volume.commit" not in lowered
    assert "result_path" not in lowered
    assert "workflow_dispatch" not in lowered
    assert "optimizer.step(" not in source
    assert ".backward(" not in source
    for key in (
        "gpu_authorized_by_issue581",
        "localization_executed_by_issue581",
        "repair_authorized",
        "architecture_freeze_authorized",
        "s2_authorized",
        "fresh_scientific_seed_authorized",
        "independent_replication_credit",
        "100m_authorized",
        "breakthrough_proven",
    ):
        assert f'"{key}": False' in source
