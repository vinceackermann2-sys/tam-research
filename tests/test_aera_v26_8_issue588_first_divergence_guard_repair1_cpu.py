from __future__ import annotations

import ast
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "modal_aera_v26_8_issue588_first_divergence_guard_repair1_app.py"
WORKFLOW = ROOT / ".github" / "workflows" / "aera-v26-8-issue588-first-divergence-guard-repair1.yml"
ISSUE584_LAUNCHER = ROOT / "modal_aera_v26_8_issue584_first_divergence_no_grad_l4_app.py"
ISSUE584_WORKFLOW = ROOT / ".github" / "workflows" / "aera-v26-8-issue584-first-divergence-no-grad-l4.yml"
ISSUE584_TEST = ROOT / "tests" / "test_aera_v26_8_issue584_first_divergence_no_grad_l4_cpu.py"
ISSUE581_WRAPPER = ROOT / "tam_research" / "aera_v26_8_issue581_first_divergence_no_grad.py"
ISSUE578_LOCALIZER = ROOT / "tam_research" / "aera_v26_8_issue578_first_divergence_localizer.py"

SOURCE_MAIN = "ad3b56106ea80956dcd1e8b457a5ef3169507351"
SOURCE_TREE = "3c3940f4e910c5bc7d6155829183c03889f305bd"
ISSUE584_LAUNCHER_BLOB = "25842ba4c02546e4773764354919de683d5fd6f0"
ISSUE584_WORKFLOW_BLOB = "25d420e4f30fbbe8ac2fbb93419039db7b54bfa6"
ISSUE584_TEST_BLOB = "0e5995adb6bf6aa99e875115a9490783fb2287ff"
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


def test_issue588_freezes_source_and_consumed_issue584_bytes() -> None:
    assert _literal(LAUNCHER, "RESEARCH_ISSUE") == 588
    assert _literal(LAUNCHER, "SOURCE_MAIN") == SOURCE_MAIN
    assert _literal(LAUNCHER, "SOURCE_TREE") == SOURCE_TREE
    assert _blob(ISSUE584_LAUNCHER) == ISSUE584_LAUNCHER_BLOB
    assert _blob(ISSUE584_WORKFLOW) == ISSUE584_WORKFLOW_BLOB
    assert _blob(ISSUE584_TEST) == ISSUE584_TEST_BLOB
    assert _blob(ISSUE581_WRAPPER) == ISSUE581_WRAPPER_BLOB
    assert _blob(ISSUE578_LOCALIZER) == ISSUE578_LOCALIZER_BLOB


def test_issue588_uses_fresh_namespace_and_same_diagnostic_entry() -> None:
    source = LAUNCHER.read_text()
    assert _literal(LAUNCHER, "APP_NAME") == "aera-v26-8-issue588-first-divergence-guard-repair1"
    assert _literal(LAUNCHER, "RESULT_PATH") == "/vol/aera-v26/issue588-first-divergence-guard-repair1/result.json"
    assert _literal(LAUNCHER, "MAX_GPU_SECONDS") == 300
    assert "issue584-first-divergence-no-grad-l4/result.json" not in source
    assert source.count('gpu="L4"') == 1
    assert source.count("run_first_divergence_localization_issue581(") == 1
    assert "run_first_divergence_localization(" not in source
    assert "volume.reload()" in source
    assert "tmp.replace(path)" in source
    assert "volume.commit()" in source


def test_issue588_freezes_consumed_587_pre_gpu_boundary() -> None:
    assert _literal(LAUNCHER, "ISSUE587_TRIGGER") == 587
    assert _literal(LAUNCHER, "ISSUE587_RUN") == 33752190666
    assert _literal(LAUNCHER, "ISSUE587_JOB") == 100638055124
    assert _literal(LAUNCHER, "ISSUE587_ATTEMPT") == 1
    assert _literal(LAUNCHER, "ISSUE587_FAILURE") == "Resource not accessible by integration (HTTP 403)"
    assert _literal(LAUNCHER, "ISSUE587_MODAL_AUTH_SKIPPED") is True
    assert _literal(LAUNCHER, "ISSUE587_L4_SKIPPED") is True


def test_issue588_workflow_adds_only_missing_permission_and_guards_587() -> None:
    source = WORKFLOW.read_text()
    assert "permissions:\n  actions: read\n  contents: read\n  issues: write\n  pull-requests: read\n" in source
    assert "[aera-v26-8-issue588-first-divergence-guard-repair1-l4]" in source
    assert 'test "${GITHUB_RUN_ATTEMPT}" = "1"' in source
    assert "## #588 sole L4 diagnostic authorization" in source
    assert "33752190666" in source and "100638055124" in source
    assert 'select(.name=="Verify canonical authorized issue584 boundary") | .conclusion' in source
    assert 'select(.name=="Authenticate Modal") | .conclusion' in source
    assert 'select(.name=="Run sole issue584 L4 diagnostic") | .conclusion' in source
    assert '= "skipped"' in source
    assert "pulls/586" in source
    assert source.count("modal run modal_aera_v26_8_issue588_first_divergence_guard_repair1_app.py") == 1
    lowered = source.lower()
    assert "workflow_dispatch" not in lowered
    assert "gh run rerun" not in lowered
    assert "rerun-failed" not in lowered
    assert "modal deploy" not in lowered
    assert "cancel-in-progress: true" not in lowered


def test_issue588_launcher_is_orchestration_only_and_higher_auth_false() -> None:
    source = LAUNCHER.read_text()
    for forbidden in (
        "optimizer.step(",
        ".backward(",
        "model.train(",
        "DUPLICATE_THRESHOLD =",
        "INTEGRATED_ATOL =",
        "INTEGRATED_RTOL =",
    ):
        assert forbidden not in source
    for key in (
        "repair_authorized",
        "architecture_freeze_authorized",
        "s2_authorized",
        "fresh_scientific_seed_authorized",
        "independent_replication_credit",
        "100m_authorized",
        "breakthrough_proven",
    ):
        assert f'"{key}": False' in source or f'"{key}"] = False' in source


def test_issue588_preflight_before_gpu_and_result_authority_after_commit() -> None:
    source = LAUNCHER.read_text()
    main_at = source.index("def main()")
    preflight_at = source.index("contract = preflight.remote()", main_at)
    gpu_at = source.index("run_localization.remote()", main_at)
    assert preflight_at < gpu_at
    commit_at = source.index("volume.commit()")
    result_at = source.index("print(RESULT_MARKER")
    assert commit_at < result_at
