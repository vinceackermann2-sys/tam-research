from __future__ import annotations

import ast
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INSPECTOR = ROOT / "modal_aera_v26_8_issue597_readonly_issue594_result_inspector.py"
WORKFLOW = ROOT / ".github" / "workflows" / "aera-v26-8-issue597-readonly-issue594-result-inspector.yml"
ISSUE594_LOCALIZER = ROOT / "tam_research" / "aera_v26_8_issue594_stage0_post_read_amplification_localizer.py"
ISSUE594_LAUNCHER = ROOT / "modal_aera_v26_8_issue594_stage0_post_read_amplification_localizer_app.py"
ISSUE594_WORKFLOW = ROOT / ".github" / "workflows" / "aera-v26-8-issue594-stage0-post-read-amplification-localizer-l4.yml"
ISSUE594_TEST = ROOT / "tests" / "test_aera_v26_8_issue594_stage0_post_read_amplification_localizer_cpu.py"

SOURCE_MAIN = "a6515564aef1738b6dc95cc50102a432751be24e"
SOURCE_TREE = "396d00f04132c14d987e9953015bb5bf76eecf0d"
SOURCE_SHA = "c950d8fa50e70a48ec64a87f860d70d854cf1a2b58e1acbdfbcb0052495e809e"
ISSUE594_LOCALIZER_BLOB = "2b72454ea74929ac7254cfc399bb2ab201dfc2cb"
ISSUE594_LAUNCHER_BLOB = "4b4d0d4640f6638e410d89fe4256e1aa868e8a4f"
ISSUE594_WORKFLOW_BLOB = "29895f8f60207507488f70ce9f2e1b9ce3c91510"
ISSUE594_TEST_BLOB = "80c3c427cc2e7f0eea12550276e71262d24fba51"


def _blob(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def _literal(path: Path, name: str):
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise AssertionError(f"missing literal {name}")


def test_issue597_freezes_authoritative_source_and_issue594_bytes() -> None:
    assert _literal(INSPECTOR, "RESEARCH_ISSUE") == 597
    assert _literal(INSPECTOR, "SOURCE_MAIN") == SOURCE_MAIN
    assert _literal(INSPECTOR, "SOURCE_TREE") == SOURCE_TREE
    assert _literal(INSPECTOR, "SOURCE_TRIGGER") == 596
    assert _literal(INSPECTOR, "SOURCE_RUN") == 33772104621
    assert _literal(INSPECTOR, "SOURCE_JOB") == 100704667286
    assert _literal(INSPECTOR, "SOURCE_ATTEMPT") == 1
    assert _literal(INSPECTOR, "SOURCE_RESULT_SHA256") == SOURCE_SHA
    assert _blob(ISSUE594_LOCALIZER) == ISSUE594_LOCALIZER_BLOB
    assert _blob(ISSUE594_LAUNCHER) == ISSUE594_LAUNCHER_BLOB
    assert _blob(ISSUE594_WORKFLOW) == ISSUE594_WORKFLOW_BLOB
    assert _blob(ISSUE594_TEST) == ISSUE594_TEST_BLOB


def test_issue597_inspector_is_exact_cpu_read_only_surface() -> None:
    source = INSPECTOR.read_text()
    assert _literal(INSPECTOR, "APP_NAME") == "aera-v26-8-issue597-readonly-issue594-result-inspector"
    assert _literal(INSPECTOR, "VOLUME_NAME") == "tam-research-data"
    assert _literal(INSPECTOR, "SOURCE_RESULT_PATH") == "/vol/aera-v26/issue594-stage0-post-read-amplification-localizer/result.json"
    assert _literal(INSPECTOR, "SOURCE_DECISION") == "COMPLETE_STAGE0_POST_READ_AMPLIFICATION_LOCALIZATION"
    assert "volume.reload()" in source
    assert "path.read_bytes()" in source
    assert "hashlib.sha256(raw).hexdigest()" in source
    assert "gpu=" not in source
    assert "volume.commit" not in source
    assert "import torch" not in source
    assert "tam_research" not in source
    for forbidden in (
        "write_text(",
        "write_bytes(",
        "unlink(",
        "rename(",
        "replace(",
        "optimizer.step(",
        ".backward(",
        "model.train(",
        "torch.load(",
    ):
        assert forbidden not in source


def test_issue597_preserves_complete_boundary_table_and_focus_surface() -> None:
    source = INSPECTOR.read_text()
    for required in (
        "boundaries_execution_order",
        "focus_boundaries",
        "run_selected_boundaries",
        "end_controller_and_returned_end_boundaries",
        "failures_execution_order",
        "failure_names_execution_order",
        "unavailable_fields",
        "first_bitwise_difference",
        "first_integrated_tolerance_or_metadata_failure",
        "first_discrete_decision_difference",
    ):
        assert required in source
    for required_name in (
        "chunk1.stage0.norm.output",
        "chunk1.stage0.tokenwise_context.context",
        "chunk1.stage0.post_context.attn_input",
        "chunk1.stage0.attn.output",
        "chunk1.stage0.post_attention.experts_input",
        "chunk1.stage0.experts.chosen_count",
        "chunk1.stage0.experts.run_selected_call_count",
        "chunk1.stage0.experts.output",
        "chunk1.stage0.end_summary",
        "chunk1.stage0.end_controller.proj_input",
        "chunk1.stage0.end_controller.raw",
    ):
        assert required_name in source
    assert "0.001953125" in source
    assert "0.03125" in source
    assert "0.031494140625" in source
    assert "unexpected first discrete difference" in source


def test_issue597_does_not_recompute_numerical_comparisons() -> None:
    source = INSPECTOR.read_text()
    for forbidden in (
        "allclose(",
        "isclose(",
        "einsum(",
        "matmul(",
        "topk(",
        "argmax(",
        "softmax(",
        "sigmoid(",
    ):
        assert forbidden not in source
    assert 'boundary.get("comparison")' in source
    assert 'comparison.get("boundaries")' in source


def test_issue597_workflow_is_canonical_cpu_only_one_shot() -> None:
    source = WORKFLOW.read_text()
    assert "[aera-v26-8-issue597-readonly-issue594-result-inspector]" in source
    assert 'test "${GITHUB_RUN_ATTEMPT}" = "1"' in source
    assert "${eligible[0]}" in source
    assert "Bind main:" in source
    assert "33772104621" in source and "100704667286" in source
    assert SOURCE_MAIN in source
    assert source.count("modal run modal_aera_v26_8_issue597_readonly_issue594_result_inspector.py") == 1
    lowered = source.lower()
    assert "workflow_dispatch" not in lowered
    assert "gh run rerun" not in lowered
    assert "rerun-failed" not in lowered
    assert "modal deploy" not in lowered
    assert "gpu:" not in lowered
    assert "cancel-in-progress: true" not in lowered


def test_issue597_higher_authorizations_remain_false() -> None:
    source = INSPECTOR.read_text()
    for key in (
        "production_repair_authorized",
        "evaluator_repair_authorized",
        "performance_tuning_authorized",
        "architecture_freeze_authorized",
        "s2_authorized",
        "fresh_scientific_seed_authorized",
        "independent_replication_credit",
        "100m_authorized",
        "breakthrough_proven",
    ):
        assert f'"{key}": False' in source
