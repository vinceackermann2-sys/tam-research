from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import tam_research.aera_v26_8_issue594_stage0_post_read_amplification_localizer as issue594

ROOT = Path(__file__).resolve().parents[1]
LOCALIZER = ROOT / "tam_research" / "aera_v26_8_issue594_stage0_post_read_amplification_localizer.py"
LAUNCHER = ROOT / "modal_aera_v26_8_issue594_stage0_post_read_amplification_localizer_app.py"
WORKFLOW = ROOT / ".github" / "workflows" / "aera-v26-8-issue594-stage0-post-read-amplification-localizer-l4.yml"
ISSUE578_LOCALIZER = ROOT / "tam_research" / "aera_v26_8_issue578_first_divergence_localizer.py"
ISSUE581_WRAPPER = ROOT / "tam_research" / "aera_v26_8_issue581_first_divergence_no_grad.py"
ISSUE562_ADAPTER = ROOT / "tam_research" / "aera_v26_8_issue562_end_to_end_systems.py"
V26_8_BACKEND = ROOT / "tam_research" / "aera_hardware_core_v26_8_ficem_read_mixed_strength_precision.py"
V26_INTERFACE = ROOT / "tam_research" / "aera_hardware_core_v26.py"
STABLE_REFERENCE = ROOT / "tam_research" / "aera_hardware_core_v25_1_compact.py"

SOURCE_MAIN = "9547bb0b4c340d793acb1e12c655dc3d22513234"
SOURCE_TREE = "6cd794ae3d893ab8f9f29e3d342c71aa081da324"
LOCALIZER_BLOB = "2b72454ea74929ac7254cfc399bb2ab201dfc2cb"
LAUNCHER_BLOB = "4b4d0d4640f6638e410d89fe4256e1aa868e8a4f"
ISSUE578_LOCALIZER_BLOB = "5933d1113b950348bb63e5f3eb8713ad36dbf580"
ISSUE581_WRAPPER_BLOB = "8800bb399e21b691e0d7703cc3eeaf486d3223b6"
ISSUE562_ADAPTER_BLOB = "3534103eea21f7c4d9d31798ad34601fd47090d6"
V26_8_BACKEND_BLOB = "3575c58d1cd730be77649f087908c51dbf3e6088"
V26_INTERFACE_BLOB = "268644ac4edee15a4cc4e29d3fed7f61eeb3caa7"
STABLE_REFERENCE_BLOB = "4e336b6e1a6238dac782fa320751d68281493ee1"
ISSUE588_RESULT_SHA256 = "495c6f49210074580553aa4b55bf0970624a8abaee910f6d2bf7315e26d2a540"


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


def test_issue594_freezes_source_and_predecessor_bytes() -> None:
    assert _literal(LOCALIZER, "RESEARCH_ISSUE") == 594
    assert _literal(LOCALIZER, "SOURCE_MAIN") == SOURCE_MAIN
    assert _literal(LOCALIZER, "SOURCE_TREE") == SOURCE_TREE
    assert _blob(LOCALIZER) == LOCALIZER_BLOB
    assert _blob(LAUNCHER) == LAUNCHER_BLOB
    assert _blob(ISSUE578_LOCALIZER) == ISSUE578_LOCALIZER_BLOB
    assert _blob(ISSUE581_WRAPPER) == ISSUE581_WRAPPER_BLOB
    assert _blob(ISSUE562_ADAPTER) == ISSUE562_ADAPTER_BLOB
    assert _blob(V26_8_BACKEND) == V26_8_BACKEND_BLOB
    assert _blob(V26_INTERFACE) == V26_INTERFACE_BLOB
    assert _blob(STABLE_REFERENCE) == STABLE_REFERENCE_BLOB


def test_issue594_cpu_preflight_authorizes_no_execution() -> None:
    contract = issue594.cpu_contract_preflight_issue594()
    assert contract["gpu_authorized_by_cpu_preflight"] is False
    assert contract["model_construction_performed"] is False
    assert contract["checkpoint_loaded"] is False
    assert contract["localization_measurement_performed"] is False
    assert contract["scientific_seed_consumed"] is False
    assert contract["protocol"]["target_chunk"] == 1
    assert contract["protocol"]["target_stage"] == 0
    assert contract["protocol"]["integrated_atol"] == 1e-2
    assert contract["protocol"]["integrated_rtol"] == 1e-2


def test_issue594_observes_direct_post_read_boundaries_without_reexecution() -> None:
    source = LOCALIZER.read_text()
    for required in (
        "chunk1.stage0.norm.output",
        "chunk1.stage0.tokenwise_context.context",
        "chunk1.stage0.tokenwise_context.memory_read",
        "chunk1.stage0.post_context.attn_input",
        "chunk1.stage0.attn.output",
        "chunk1.stage0.post_attention.experts_input",
        "chunk1.stage0.experts.output",
        "chunk1.stage0.end_summary",
        "chunk1.stage0.end_controller.stream",
        "chunk1.stage0.end_controller.proj_input",
        "chunk1.stage0.end_controller.raw",
    ):
        assert required in source
    assert "issue578._tensor_comparison" in source
    assert "torch.allclose(" not in source
    assert "register_forward_pre_hook" in source
    assert "register_forward_hook" in source
    assert "handle.remove()" in source
    assert "for obj, name, original in reversed(restorations)" in source
    assert "return original_run_selected(x, expert_ids)" in source
    assert "return output" in source
    assert "stage.attn(" not in source
    assert "stage.experts(" not in source
    assert "stage.controller(" not in source


def test_issue594_directly_observes_expert_dispatch_without_topk_replay() -> None:
    source = LOCALIZER.read_text()
    assert '"expert_ids": _cpu(expert_ids)' in source
    assert "_run_selected" in source
    assert "torch.topk(" not in source
    assert "argmax(" not in source
    assert "chosen_count" in source
    assert "expert second-call original batch indices are not exposed" in source


def test_issue594_no_grad_version_contract_and_higher_auth_false() -> None:
    source = LOCALIZER.read_text()
    assert source.count("@torch.no_grad()") == 1
    assert "@torch.inference_mode" not in source
    assert "torch.inference_mode(" not in source
    assert "parameter._version" in source
    for forbidden in (
        "optimizer.step(",
        ".backward(",
        "model.train(",
        "DUPLICATE_THRESHOLD =",
        "INTEGRATED_ATOL = 1",
        "INTEGRATED_RTOL = 1",
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
        assert f'"{key}": False' in source


def test_issue594_launcher_is_duplicate_safe_and_freezes_issue588_result() -> None:
    source = LAUNCHER.read_text()
    assert _literal(LAUNCHER, "APP_NAME") == "tam-research-aera-v26-8-issue594-stage0-post-read-amplification-localizer"
    assert _literal(LAUNCHER, "RESULT_PATH") == "/vol/aera-v26/issue594-stage0-post-read-amplification-localizer/result.json"
    assert _literal(LAUNCHER, "MAX_GPU_SECONDS") == 300
    assert _literal(LAUNCHER, "ISSUE588_RESULT_SHA256") == ISSUE588_RESULT_SHA256
    assert source.count('gpu="L4"') == 1
    assert source.count("run_stage0_post_read_amplification_localization(") == 1
    assert "volume.reload()" in source
    assert "predecessor_sha = _sha256(predecessor_path)" in source
    assert "tmp.replace(path)" in source
    assert "volume.commit()" in source
    assert "result already exists" in source


def test_issue594_preflight_before_gpu_and_result_authority_after_commit() -> None:
    source = LAUNCHER.read_text()
    main_at = source.index("def main()")
    preflight_at = source.index("contract = preflight.remote()", main_at)
    gpu_at = source.index("run_localization.remote()", main_at)
    assert preflight_at < gpu_at
    commit_at = source.index("volume.commit()")
    result_at = source.index("print(RESULT_MARKER")
    assert commit_at < result_at


def test_issue594_workflow_guards_authority_and_has_no_retry_path() -> None:
    source = WORKFLOW.read_text()
    assert "permissions:\n  actions: read\n  contents: read\n  issues: write\n  pull-requests: read\n" in source
    assert "[aera-v26-8-issue594-stage0-post-read-amplification-localizer-l4]" in source
    assert 'test "${GITHUB_RUN_ATTEMPT}" = "1"' in source
    assert "## #594 sole L4 diagnostic authorization" in source
    assert "pulls/592" in source
    assert "33763382230" in source and "100674987954" in source
    assert "33753926605" in source and "100643674944" in source
    assert "33764045085" in source and "100677235816" in source
    assert source.count("modal run modal_aera_v26_8_issue594_stage0_post_read_amplification_localizer_app.py") == 1
    lowered = source.lower()
    assert "workflow_dispatch" not in lowered
    assert "gh run rerun" not in lowered
    assert "rerun-failed" not in lowered
    assert "modal deploy" not in lowered
    assert "cancel-in-progress: true" not in lowered


def test_issue594_launcher_and_workflow_do_not_authorize_repair_or_higher_stage() -> None:
    combined = LOCALIZER.read_text() + LAUNCHER.read_text()
    for key in (
        "repair_authorized",
        "architecture_freeze_authorized",
        "s2_authorized",
        "fresh_scientific_seed_authorized",
        "independent_replication_credit",
        "100m_authorized",
        "breakthrough_proven",
    ):
        assert f'"{key}": False' in combined or f'"{key}"] = False' in combined
