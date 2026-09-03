from __future__ import annotations

import ast
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "modal_aera_v26_8_issue584_first_divergence_no_grad_l4_app.py"
WORKFLOW = ROOT / ".github" / "workflows" / "aera-v26-8-issue584-first-divergence-no-grad-l4.yml"
ISSUE581_WRAPPER = ROOT / "tam_research" / "aera_v26_8_issue581_first_divergence_no_grad.py"
ISSUE581_CPU_TEST = ROOT / "tests" / "test_aera_v26_8_issue581_first_divergence_no_grad_cpu.py"
ISSUE578_LOCALIZER = ROOT / "tam_research" / "aera_v26_8_issue578_first_divergence_localizer.py"
ISSUE578_LAUNCHER = ROOT / "modal_aera_v26_8_issue578_first_divergence_localizer_app.py"
ISSUE578_WORKFLOW = ROOT / ".github" / "workflows" / "aera-v26-8-issue578-first-divergence-localizer-l4.yml"
ISSUE578_CPU_TEST = ROOT / "tests" / "test_aera_v26_8_issue578_first_divergence_localizer_cpu.py"

SOURCE_MAIN = "8a2ec1fcd9dd3ed2feb5c96147c1b824e669be33"
SOURCE_TREE = "9cb276a4f88497e980d058c84ad89e79a3647024"
ISSUE581_WRAPPER_BLOB = "8800bb399e21b691e0d7703cc3eeaf486d3223b6"
ISSUE581_CPU_TEST_BLOB = "6b5218bcc744dd8d9cbca65c9b5c7c0c1578f5e9"
ISSUE578_LOCALIZER_BLOB = "5933d1113b950348bb63e5f3eb8713ad36dbf580"
ISSUE578_LAUNCHER_BLOB = "cd47e1252bed5617556998659eadfe6a61637d39"
ISSUE578_WORKFLOW_BLOB = "b76282733903d220e7118ede283f789db0eb56ba"
ISSUE578_CPU_TEST_BLOB = "6dd02b5a25514ad9987d7617e4a4b1ddbb1e6f0a"
ISSUE571_RESULT_SHA256 = "afeeb62351cc4fb97d272c5b55c9621839e26f83753ae1fb237733d58a5ee472"

RESULT_PATH = "/vol/aera-v26/issue584-first-divergence-no-grad-l4/result.json"
APP_NAME = "aera-v26-8-issue584-first-divergence-no-grad-l4"
TRIGGER_PREFIX = "[aera-v26-8-issue584-first-divergence-no-grad-l4]"
MAX_GPU_SECONDS = 300

FROZEN_BLOBS = {
    ROOT / "tam_research" / "aera_v26_8_issue562_end_to_end_systems.py": "3534103eea21f7c4d9d31798ad34601fd47090d6",
    ROOT / "tam_research" / "aera_hardware_core_v26_8_ficem_read_mixed_strength_precision.py": "3575c58d1cd730be77649f087908c51dbf3e6088",
    ROOT / "tam_research" / "aera_hardware_core_v26_6_ficem_write_materialize_cast.py": "d45c262314a0b4691f26812a279937a225043ad9",
    ROOT / "tam_research" / "aera_hardware_core_v26.py": "268644ac4edee15a4cc4e29d3fed7f61eeb3caa7",
    ROOT / "tam_research" / "aera_hardware_core_v25_1_compact.py": "4e336b6e1a6238dac782fa320751d68281493ee1",
}


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


def test_issue584_freezes_exact_source_and_historical_bytes() -> None:
    assert _literal_constant(LAUNCHER, "RESEARCH_ISSUE") == 584
    assert _literal_constant(LAUNCHER, "SOURCE_MAIN") == SOURCE_MAIN
    assert _literal_constant(LAUNCHER, "SOURCE_TREE") == SOURCE_TREE
    assert _blob(ISSUE581_WRAPPER) == ISSUE581_WRAPPER_BLOB
    assert _blob(ISSUE581_CPU_TEST) == ISSUE581_CPU_TEST_BLOB
    assert _blob(ISSUE578_LOCALIZER) == ISSUE578_LOCALIZER_BLOB
    assert _blob(ISSUE578_LAUNCHER) == ISSUE578_LAUNCHER_BLOB
    assert _blob(ISSUE578_WORKFLOW) == ISSUE578_WORKFLOW_BLOB
    assert _blob(ISSUE578_CPU_TEST) == ISSUE578_CPU_TEST_BLOB
    for path, expected in FROZEN_BLOBS.items():
        assert _blob(path) == expected, path


def test_issue584_freezes_fresh_namespace_and_timeout() -> None:
    assert _literal_constant(LAUNCHER, "APP_NAME") == APP_NAME
    assert _literal_constant(LAUNCHER, "VOLUME_NAME") == "tam-research-data"
    assert _literal_constant(LAUNCHER, "RESULT_PATH") == RESULT_PATH
    assert _literal_constant(LAUNCHER, "MAX_GPU_SECONDS") == MAX_GPU_SECONDS

    source = LAUNCHER.read_text()
    assert source.count('gpu="L4"') == 1
    assert "timeout=MAX_GPU_SECONDS" in source
    assert source.count(RESULT_PATH) >= 1
    assert "/vol/aera-v26/issue578-first-divergence-localizer/result.json" not in source
    for marker in (
        "AERA_V26_8_ISSUE584_FIRST_DIVERGENCE_NO_GRAD_PRECHECK_JSON=",
        "AERA_V26_8_ISSUE584_FIRST_DIVERGENCE_NO_GRAD_PREFLIGHT_JSON=",
        "AERA_V26_8_ISSUE584_FIRST_DIVERGENCE_NO_GRAD_L4_START_JSON=",
        "AERA_V26_8_ISSUE584_FIRST_DIVERGENCE_NO_GRAD_RESULT_JSON=",
        "AERA_V26_8_ISSUE584_FIRST_DIVERGENCE_NO_GRAD_SUMMARY_JSON=",
    ):
        assert marker in source


def test_issue584_preflight_precedes_gpu_and_refuses_existing_result() -> None:
    source = LAUNCHER.read_text()
    main_start = source.index("def main() -> None:")
    main_source = source[main_start:]
    assert main_source.index("contract = preflight.remote()") < main_source.index(
        "run_localization.remote()"
    )

    assert source.count("volume.reload()") >= 2
    assert source.count("if result_path.exists():") == 1
    assert source.count("if path.exists():") == 1
    assert "issue581.cpu_contract_preflight_issue581()" in source
    assert '"gpu_authorized_by_preflight": False' in source
    assert '"localization_measurement_performed": False' in source


def test_issue584_calls_only_merged_issue581_successor_once() -> None:
    source = LAUNCHER.read_text()
    assert source.count("issue581.run_first_divergence_localization_issue581(") == 1
    assert "issue578.run_first_divergence_localization(" not in source
    assert "run_first_divergence_localization.__wrapped__" not in source
    assert ISSUE581_WRAPPER_BLOB in source
    assert ISSUE578_LOCALIZER_BLOB in source
    assert "torch.cuda.device_count() != 1" in source
    assert 'device_name != "NVIDIA L4"' in source


def test_issue584_preserves_immutability_and_commits_before_result_authority() -> None:
    source = LAUNCHER.read_text()
    assert "hashes_after != hashes_before" in source
    assert 'result["parameter_versions_unchanged"] is not True' in source
    assert 'result["checkpoint_hashes_unchanged"] is not True' in source
    assert "tmp.replace(path)" in source
    assert "volume.commit()" in source
    assert "RESULT_MARKER + json.dumps" in source
    assert source.index("volume.commit()") < source.index(
        "print(RESULT_MARKER + json.dumps"
    )
    assert ISSUE571_RESULT_SHA256 in source


def test_issue584_freezes_consumed_580_and_581_cpu_authority() -> None:
    source = LAUNCHER.read_text() + "\n" + WORKFLOW.read_text()
    for required in (
        "33748196657",
        "100625461189",
        "eba8d04ed262c9bf539a99d256af5e99eb6a87d1",
        "Inference tensors do not track version counter.",
        "33749720315",
        "100630230021",
        "fd5854e5a4fd11236998d478bb753829859e5e05",
        SOURCE_MAIN,
        ISSUE581_WRAPPER_BLOB,
        ISSUE581_CPU_TEST_BLOB,
    ):
        assert required in source


def test_issue584_workflow_is_canonical_owner_attempt1_and_later_authorized() -> None:
    source = WORKFLOW.read_text()
    lowered = source.lower()
    assert "issues:\n    types: [opened]" in source
    assert "github.event.issue.user.login == github.repository_owner" in source
    assert TRIGGER_PREFIX in source
    assert 'test "${GITHUB_RUN_ATTEMPT}" = "1"' in source
    assert "sort -n" in source
    assert 'test "${TRIGGER_ISSUE}" = "${eligible[0]}"' in source
    assert "## #584 sole L4 diagnostic authorization" in source
    assert "Authorize main:" in source
    assert "Recheck canonical trigger immediately before Modal" in source
    assert source.count(
        "modal run modal_aera_v26_8_issue584_first_divergence_no_grad_l4_app.py"
    ) == 1
    assert "workflow_dispatch" not in source
    assert "gh run rerun" not in lowered
    assert "rerun-failed" not in lowered
    assert "modal deploy" not in lowered
    assert "cancel-in-progress: true" not in lowered
    assert "\nconcurrency:" not in lowered


def test_issue584_workflow_hard_guards_frozen_blobs_and_consumed_failure() -> None:
    source = WORKFLOW.read_text()
    for blob in (
        ISSUE581_WRAPPER_BLOB,
        ISSUE581_CPU_TEST_BLOB,
        ISSUE578_LOCALIZER_BLOB,
        ISSUE578_LAUNCHER_BLOB,
        ISSUE578_WORKFLOW_BLOB,
        ISSUE578_CPU_TEST_BLOB,
        *FROZEN_BLOBS.values(),
    ):
        assert blob in source
    assert '33748196657" --jq \'.conclusion\')" = "failure"' in source
    assert '100625461189" --jq \'.conclusion\')" = "failure"' in source
    assert '33749720315" --jq \'.conclusion\')" = "success"' in source
    assert '100630230021" --jq \'.conclusion\')" = "success"' in source


def test_issue584_higher_authorizations_remain_false() -> None:
    source = LAUNCHER.read_text()
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
