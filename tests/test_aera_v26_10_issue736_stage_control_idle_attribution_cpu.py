from __future__ import annotations

import ast
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "modal_aera_v26_10_issue736_stage_control_idle_attribution.py"
WORKFLOW = ROOT / ".github/workflows/aera-v26-10-issue736-stage-control-idle-attribution.yml"

LAUNCHER_BLOB = "2a004bb09eed8f644a61353f762f3ce1bd7d14f2"
WORKFLOW_BLOB = "7452335e32c22e778f58519b85ed48e28a1aff36"
ORIGINAL_WORKFLOW_BLOB = "beaeaae0558a1208ae96fd555f7900f8e3813ee5"
ORIGINAL_CPU_TEST_BLOB = "0c4a903ff57b42ec349080427c65aa5c67fda708"
ISSUE710_BLOB = "b885250753ea169cd89dd7a978bb3647fd261fe8"
V26_10_IMPL_BLOB = "d8f691c198eed1fa96bcbb78a4e76cad82d18779"


def _git_blob(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def test_issue738_exact_launcher_corrected_workflow_and_frozen_dependencies():
    assert _git_blob(LAUNCHER) == LAUNCHER_BLOB
    assert _git_blob(WORKFLOW) == WORKFLOW_BLOB
    assert _git_blob(ROOT / "modal_aera_v26_10_issue710_memory_safe_harness.py") == ISSUE710_BLOB
    assert _git_blob(ROOT / "tam_research/aera_hardware_core_v26_10_latent_depth_sync_coalescing.py") == V26_10_IMPL_BLOB
    ast.parse(LAUNCHER.read_text())


def test_issue738_parent_lineage_and_measurement_contract_are_unchanged():
    text = LAUNCHER.read_text()
    for required in (
        'SOURCE_MAIN="d15d94f39e59d0c6cc8d1514f44a33452d279a74"',
        'SOURCE_TREE="fe5181034b3cb07076b17a43c7137829124d8cbb"',
        'PARENT_TRIGGER=735', 'PARENT_EVIDENCE_COMMENT=5574994790',
        'PARENT_RUN=34156438962', 'PARENT_JOB=101849137998', 'PARENT_ATTEMPT=1',
        'RESULT_PATH="/vol/aera-v26/issue736-stage-control-idle-attribution/result.json"',
        'PARENT_RESULT_PATH="/vol/aera-v26/issue732-post731-differential-kernel-time-attribution/result.json"',
        'BATCHES=(8,64)', 'WARMUP_CALLS=3', 'UNPROFILED_CALLS=3', 'PROFILE_CALLS=1',
        'MIN_REGION_MS=1.0', 'MIN_REGION_SHARE=0.10',
        '52.51891326904297', '89.11689758300781', '1716', '47.15760479756141',
        '101.74742126464844', '126.5898208618164', '1755', '58.12677116908063',
    ):
        assert required in text


def test_issue738_instrumentation_and_decision_rule_remain_unchanged():
    text = LAUNCHER.read_text()
    for required in (
        'torch_module.profiler.record_function(label)',
        'issue736.route_scope.', 'issue736.route_gate.', 'issue736.stage_scope.',
        'issue736.context_scope.', 'issue736.controller_start.', 'issue736.controller_end.',
        'issue736.expert_scope.', 'issue736.expert_child.',
        'issue736.reasoner_scope.', 'issue736.reasoner_cell.',
        'issue736.ficem_scope.', 'issue736.write_select',
        'exclusive_partition_no_double_count', 'children_subtracted',
        'exclusive_cpu_us', 'exclusive_device_active_union_us', 'exclusive_residual_us',
        'exclusive_inter_device_idle_total_us', 'exclusive_inter_device_idle_stats_us',
        'whole_inter_device_idle_total_us', 'whole_inter_device_idle_stats_us',
        'profiler_distortion_ratio', 'normalization_scale',
        'distortion_normalized_host_control_idle_ms', 'share_of_unprofiled_median',
        'durable_state_pack_output', 'first_vs_second_expert_subphases', 'kernel_to_aten_ownership',
        'unavailable_unattributed_cannot_authorize',
    ):
        assert required in text
    targets = (
        'stage_route_control', 'expert_dispatch_control', 'latent_reasoner_control',
        'ficem_read_write_control', 'write_select_update_control', 'chunk_or_stage_glue_control',
        'context_attention_control', 'norm_start_controller_control', 'end_controller_control',
        'reason_to_chunk_output_norm_control', 'recurrent_stream_update_control',
    )
    for target in targets:
        assert f'"{target}"' in text
    assert 'tab[k]["distortion_normalized_host_control_idle_ms"]>=MIN_REGION_MS' in text
    assert 'tab[k]["share_of_unprofiled_median"]>=MIN_REGION_SHARE' in text
    assert 'if top8==top64:target=top8' in text


def test_issue738_correctness_immutability_and_authority_ceiling_unchanged():
    text = LAUNCHER.read_text()
    for required in (
        '_schemas_and_weights_exact', '_capture_decisions', '_decision_equivalence', '_route_exact',
        '_chunked_logit_equivalence', '_state_equivalence', '_finite_output',
        'checkpoint_hashes_before', 'checkpoint_hashes_after', 'checkpoint_hashes_unchanged',
        'parameter_versions_unchanged', 'route_mode":"hard_sparse"', '"hard":True', '"update_memory":True',
        '"optimization_authorized":False', '"full_e2e_systems_gate_authorized":False',
        '"systems_pass_earned":False', '"architecture_freeze_authorized":False',
        '"s2_authorized":False', '"fresh_scientific_seed_authorized":False',
        '"independent_replication_credit":False', '"100m_authorized":False',
        '"breakthrough_proven":False', '"diagnostic_only":True',
    ):
        assert required in text
    for forbidden in (
        '.backward(', '.step(', '.zero_grad(', 'torch.optim.', 'optimizer=',
        'run_end_to_end_systems_v26_9(', 'run_end_to_end_systems_v26_8(',
    ):
        assert forbidden not in text


def test_issue738_workflow_uses_original_science_freeze_and_correction_lineage():
    text = WORKFLOW.read_text()
    assert 'issues:\n    types: [opened]' in text
    assert 'github.event.issue.user.login == github.repository_owner' in text
    assert 'workflow_dispatch:' not in text
    assert 'pull_request:' not in text
    assert '\npush:' not in text
    assert 'test "${GITHUB_RUN_ATTEMPT}" = "1"' in text
    assert 'issues/comments/5575217545' in text
    assert '## #736 pre-implementation stage/control-flow idle attribution freeze' in text
    assert 'issues/738/comments?per_page=100' in text
    assert '## #738 correction1 CPU-contract/immutable-guard freeze' in text
    assert 'CONSUMED_PR=737' in text
    assert 'CONSUMED_RUN=34159905706' in text
    assert 'CONSUMED_JOB=101859329645' in text
    assert 'CONSUMED_ATTEMPT=1' in text
    assert f'ORIGINAL_WORKFLOW_BLOB={ORIGINAL_WORKFLOW_BLOB}' in text
    assert f'ORIGINAL_CPU_TEST_BLOB={ORIGINAL_CPU_TEST_BLOB}' in text
    assert f'UNCHANGED_LAUNCHER_BLOB={LAUNCHER_BLOB}' in text
    assert 'CORRECTION_WORKFLOW_BLOB=' in text
    assert 'CORRECTION_CPU_TEST_BLOB=' in text
    assert 'FROZEN_TREE=' in text and 'FROZEN_COMMIT=' in text
    assert 'git merge-base --is-ancestor "$frozen_commit" "$bound_main"' in text
    assert 'test "$(git hash-object modal_aera_v26_10_issue736_stage_control_idle_attribution.py)" = "$launcher_blob"' in text
    assert 'test "$(git hash-object .github/workflows/aera-v26-10-issue736-stage-control-idle-attribution.yml)" = "$workflow_blob"' in text
    assert 'test "$(git hash-object tests/test_aera_v26_10_issue736_stage_control_idle_attribution_cpu.py)" = "$cpu_test_blob"' in text


def test_issue738_workflow_preserves_trigger_modal_and_evidence_contract():
    text = WORKFLOW.read_text()
    assert '[aera-v26-10-issue736-region-idle-preauth]' in text
    assert '[aera-v26-10-issue736-region-idle-l4]' in text
    assert '## #736 sole L4 stage/control-flow idle attribution authorization' in text
    assert 'Re-verify issue736 trigger immediately before Modal' in text
    assert text.count('test "${#matches[@]}" = "1"') >= 2
    assert text.count('test "$TRIGGER_ISSUE" = "${matches[0]}"') >= 2
    assert 'modal_aera_v26_10_issue736_stage_control_idle_attribution.py::preauth_main' in text
    assert 'modal_aera_v26_10_issue736_stage_control_idle_attribution.py::l4_main' in text
    assert '🔎 **AERA-v26.10 #736 stage/control idle preauthorization evidence**' in text
    assert '🔬 **AERA-v26.10 #736 stage/control-flow idle attribution evidence**' in text
    assert 'MIN_REGION_MS=1.0' in text and 'MIN_REGION_SHARE=0.10' in text


def test_issue738_cpu_contract_has_no_self_hash_requirement():
    tree = ast.parse(Path(__file__).read_text())
    assigned_names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assigned_names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            assigned_names.add(node.target.id)
    assert "CPU_TEST_BLOB" not in assigned_names

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "_git_blob" or len(node.args) != 1:
            continue
        arg = node.args[0]
        is_path_file = (
            isinstance(arg, ast.Call)
            and isinstance(arg.func, ast.Name)
            and arg.func.id == "Path"
            and len(arg.args) == 1
            and isinstance(arg.args[0], ast.Name)
            and arg.args[0].id == "__file__"
        )
        assert not is_path_file
