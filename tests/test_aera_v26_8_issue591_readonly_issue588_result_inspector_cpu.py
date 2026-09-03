from __future__ import annotations

import ast
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INSPECTOR = ROOT / "modal_aera_v26_8_issue591_readonly_issue588_result_inspector.py"
WORKFLOW = ROOT / ".github" / "workflows" / "aera-v26-8-issue591-readonly-issue588-result-inspector.yml"
ISSUE588_LAUNCHER = ROOT / "modal_aera_v26_8_issue588_first_divergence_guard_repair1_app.py"
ISSUE588_WORKFLOW = ROOT / ".github" / "workflows" / "aera-v26-8-issue588-first-divergence-guard-repair1.yml"
ISSUE588_TEST = ROOT / "tests" / "test_aera_v26_8_issue588_first_divergence_guard_repair1_cpu.py"
ISSUE581_WRAPPER = ROOT / "tam_research" / "aera_v26_8_issue581_first_divergence_no_grad.py"
ISSUE578_LOCALIZER = ROOT / "tam_research" / "aera_v26_8_issue578_first_divergence_localizer.py"

SOURCE_MAIN = "e08e49e2e5d01010f42dc7119d7bcacc12dd1f83"
SOURCE_TREE = "24417dd9297e76d8899796ccccff8f5ff2462222"
SOURCE_SHA = "495c6f49210074580553aa4b55bf0970624a8abaee910f6d2bf7315e26d2a540"
ISSUE588_LAUNCHER_BLOB = "f5ea8a279578edd57266bfead57371e15faab853"
ISSUE588_WORKFLOW_BLOB = "33e721f457f518805ebfbb359e7a945947e48b31"
ISSUE588_TEST_BLOB = "703d5dc3f229d0fb5d5edcf981a65b6ac76369aa"
ISSUE581_WRAPPER_BLOB = "8800bb399e21b691e0d7703cc3eeaf486d3223b6"
ISSUE578_LOCALIZER_BLOB = "5933d1113b950348bb63e5f3eb8713ad36dbf580"


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


def test_issue591_freezes_authoritative_source_and_predecessor_bytes() -> None:
    assert _literal(INSPECTOR, "RESEARCH_ISSUE") == 591
    assert _literal(INSPECTOR, "SOURCE_MAIN") == SOURCE_MAIN
    assert _literal(INSPECTOR, "SOURCE_TREE") == SOURCE_TREE
    assert _literal(INSPECTOR, "SOURCE_TRIGGER") == 590
    assert _literal(INSPECTOR, "SOURCE_RUN") == 33753926605
    assert _literal(INSPECTOR, "SOURCE_JOB") == 100643674944
    assert _literal(INSPECTOR, "SOURCE_ATTEMPT") == 1
    assert _literal(INSPECTOR, "SOURCE_RESULT_SHA256") == SOURCE_SHA
    assert _blob(ISSUE588_LAUNCHER) == ISSUE588_LAUNCHER_BLOB
    assert _blob(ISSUE588_WORKFLOW) == ISSUE588_WORKFLOW_BLOB
    assert _blob(ISSUE588_TEST) == ISSUE588_TEST_BLOB
    assert _blob(ISSUE581_WRAPPER) == ISSUE581_WRAPPER_BLOB
    assert _blob(ISSUE578_LOCALIZER) == ISSUE578_LOCALIZER_BLOB


def test_issue591_inspector_is_exact_read_only_cpu_surface() -> None:
    source = INSPECTOR.read_text()
    assert _literal(INSPECTOR, "APP_NAME") == "aera-v26-8-issue591-readonly-issue588-result-inspector"
    assert _literal(INSPECTOR, "VOLUME_NAME") == "tam-research-data"
    assert _literal(INSPECTOR, "SOURCE_RESULT_PATH") == "/vol/aera-v26/issue588-first-divergence-guard-repair1/result.json"
    assert _literal(INSPECTOR, "SOURCE_DECISION") == "COMPLETE_FIRST_DIVERGENCE_LOCALIZATION"
    assert "volume.reload()" in source
    assert "path.read_bytes()" in source
    assert "hashlib.sha256(raw).hexdigest()" in source
    assert "gpu=" not in source
    assert "volume.commit" not in source
    for forbidden in (
        "write_text(", "write_bytes(", "unlink(", "rename(", "replace(",
        "optimizer.step(", ".backward(", "model.train(", "torch.load(",
    ):
        assert forbidden not in source


def test_issue591_preserves_boundary_order_and_unavailable_policy() -> None:
    source = INSPECTOR.read_text()
    for required in (
        "first_bitwise_difference",
        "first_integrated_tolerance_or_metadata_failure",
        "first_discrete_decision_difference",
        "boundary_chain_first_bitwise_through_first_integrated_failure",
        "boundary_chain_after_first_integrated_failure_through_first_discrete",
        "failures_execution_order",
        "threshold_margin_diagnostics",
        "requested_but_unavailable_tokens",
        "candidate_backend_internal_adjudication_decisions_available",
        "adjudication_replay_is_diagnostic_only",
        "batch64_route_context_around_first_discrete",
    ):
        assert required in source
    assert '"chunk1.stage0.read.recalled"' in source
    assert '"chunk1.stage0.end_controller.event"' in source
    assert '"chunk1.stage3.adjudication_replay.shadowed_incoming"' in source
    assert '"chunk1.stage3.route.gate"' in source
    assert "0.00048828125" in source
    assert "0.03125" in source
    assert "0.031494140625" in source


def test_issue591_never_upgrades_diagnostic_replay_to_kernel_evidence() -> None:
    source = INSPECTOR.read_text()
    assert 'comparison.get("candidate_backend_internal_adjudication_decisions_available") is not False' in source
    assert 'comparison.get("adjudication_replay_is_diagnostic_only") is not True' in source
    assert '"candidate_backend_internal_adjudication_decisions_available": False' in source
    assert '"adjudication_replay_is_diagnostic_only": True' in source


def test_issue591_workflow_is_canonical_cpu_only_one_shot() -> None:
    source = WORKFLOW.read_text()
    assert "[aera-v26-8-issue591-readonly-issue588-result-inspector]" in source
    assert 'test "${GITHUB_RUN_ATTEMPT}" = "1"' in source
    assert "${eligible[0]}" in source
    assert "Bind main:" in source
    assert "33753926605" in source and "100643674944" in source
    assert "e08e49e2e5d01010f42dc7119d7bcacc12dd1f83" in source
    assert source.count("modal run modal_aera_v26_8_issue591_readonly_issue588_result_inspector.py") == 1
    lowered = source.lower()
    assert "workflow_dispatch" not in lowered
    assert "gh run rerun" not in lowered
    assert "rerun-failed" not in lowered
    assert "modal deploy" not in lowered
    assert "gpu:" not in lowered
    assert "cancel-in-progress: true" not in lowered


def test_issue591_higher_authorizations_remain_false() -> None:
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
