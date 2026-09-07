from __future__ import annotations

import ast
import hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
LAUNCHER=ROOT/"modal_aera_v26_9_issue702_cuda_timeline_dispatch_gap_attribution.py"
WORKFLOW=ROOT/".github/workflows/aera-v26-9-issue702-cuda-timeline-dispatch-gap-attribution.yml"
LAUNCHER_BLOB="bf38814a976d929119d17b9e907f1b89461a14a2"
WORKFLOW_BLOB="89ce600b3b0111379cf856404e3c7c4ec2d3a0e1"

FROZEN_FILES={
"modal_aera_v26_9_issue697_micro_op_attribution.py":"c82affc01d1ecb096371e8865dbd7b18e739f5e1",
"modal_aera_v26_9_issue687_stage_internal_throughput_attribution.py":"b9950c032686a496c6d944c30c441428499cb367",
"modal_aera_v26_9_issue665_frozen_throughput_component_attribution.py":"72f27391ff2f0a7bff8d4532f307ddc4869cf494",
"tam_research/aera_v26_9_issue643_bounded_memory_end_to_end_systems.py":"512572340cc09e2e7ad6729712258c12cb377ef2",
"tam_research/aera_v26_5_end_to_end_systems.py":"c9731cae7e386f09b2a190b045532591c4fa00be",
"tam_research/aera_hardware_core_v26.py":"268644ac4edee15a4cc4e29d3fed7f61eeb3caa7",
"tam_research/aera_hardware_core_v25_1.py":"1c3456d8040455b4cd1194db4c8586f77d0f3e43",
"tam_research/aera_hardware_core_v25_1_compact.py":"4e336b6e1a6238dac782fa320751d68281493ee1",
"tam_research/aera_hardware_core_v19.py":"98008bceb8c68af3bc346e5dfcc7a8218875661e",
"tam_research/aera_hardware_core.py":"ffe0341829fb905e40ef9ae3544f797471f1aa9c",
"tam_research/aera_hardware_core_v25_1_nohost.py":"237e5615cf32f644e8675808a6b3e9adaf04fb23",
"tam_research/aera_hardware_core_v26_9_ficem_read_identity_weight_visibility.py":"b81cc209f5d95abbe1fb8bd620c78e87c067bc19",
"tam_research/aera_v25_post8471_triage.py":"e5ce8cdda0777dce97816e2640f4492803a6b191",
}

def _git_blob(path:Path)->str:
    data=path.read_bytes(); return hashlib.sha1(f"blob {len(data)}\0".encode()+data).hexdigest()

def test_issue702_files_are_exact_and_python_parses():
    assert LAUNCHER.exists() and WORKFLOW.exists()
    assert _git_blob(LAUNCHER)==LAUNCHER_BLOB
    assert _git_blob(WORKFLOW)==WORKFLOW_BLOB
    ast.parse(LAUNCHER.read_text())
    for rel,expected in FROZEN_FILES.items():
        path=ROOT/rel; assert path.exists(),rel; assert _git_blob(path)==expected,rel

def test_issue702_restarts_governance_without_changing_science():
    text=LAUNCHER.read_text()
    for required in (
        'RESEARCH_ISSUE=702','SOURCE_MAIN="6fad37d0e4c611fcaff480899b25dc190e9e9127"',
        'SOURCE_TREE="87a22d981caf77263574503749757ba77290cc9a"','GOVERNANCE_PREDECESSOR_ISSUE=701',
        'GOVERNANCE_INCIDENT_COMMENT=5571988281','EXCLUDED_INERT_REF="__noop_probe_do_not_create"',
        'PARENT_TRIGGER=700','PARENT_EVIDENCE_COMMENT=5571118103','PARENT_RUN=34125640230','PARENT_JOB=101753558368','PARENT_ATTEMPT=1',
        'PARENT_NEXT_TARGET="dispatch_or_unattributed_gap"','"8":{"gap":9.2633691290416,"distortion":2.6684698738709085}',
        '"64":{"gap":12.370303117411762,"distortion":1.6960100325791625}',
        'RESULT_PATH="/vol/aera-v26/issue702-cuda-timeline-dispatch-gap-attribution/result.json"',
        'PARENT_RESULT_PATH="/vol/aera-v26/issue697-micro-op-attribution/result.json"',
        'BASELINE_CALLS=3','PROFILE_CALLS=1','SMALL_KERNEL_MAX_US=50.0','REPEATED_MIN_COUNT=2','MATERIAL_SHARE_MIN=0.15','MATERIAL_MS_MIN=1.0'):
        assert required in text
    assert 'issue701-cuda-timeline-dispatch-gap-attribution/result.json' not in text

def test_issue702_packages_root_import_graph_explicitly():
    text=LAUNCHER.read_text()
    for name in ('ISSUE665_LAUNCHER','ISSUE687_LAUNCHER','ISSUE697_LAUNCHER'):
        assert f'.add_local_file({name},f"/root/{{{name}}}")' in text
    assert 'app=modal.App(APP_NAME)' in text
    assert 'app=issue665.app' not in text

def test_issue702_timeline_contract_is_low_intrusion_and_route_grounded():
    text=LAUNCHER.read_text()
    for required in (
        'torch.profiler.ProfilerActivity.CPU','torch.profiler.ProfilerActivity.CUDA','record_shapes=False','profile_memory=False','with_stack=False',
        'torch.profiler.record_function(label)','prof.export_chrome_trace','chronological_device_activities','device_active_union_us','idle_gaps_us','idle_gap_total_us',
        'cuda_runtime','External id','aten_linked','kernel_external_id_attribution_complete','repeated_small_unlinked_kernel_names','top_large_kernel_names',
        'top_unlinked_large_kernel_names','unavailable_fields','unavailable_or_unattributed_residual_ms','route_rows','executed_fraction',
        'route_matches_all_unprofiled_calls','"route_mode":"hard_sparse"','"hard":True','"update_memory":True','cuda_bf16_autocast_via_frozen_model_call',
        'profiled_measurements_may_be_perturbed','comparative_systems_evidence":False'):
        assert required in text
    assert 'torch.cuda.Event(enable_timing=True)' in text
    assert 'torch.cuda.synchronize()' in text

def test_issue702_incomplete_correlation_disables_all_kernel_mechanisms():
    text=LAUNCHER.read_text()
    assert 'unlinked=[e for e in kernels if e["aten_linked"] is False] if complete else []' in text
    assert 'small=float(trace["repeated_small_unlinked_active_union_us"])/1000.0*correction if complete else 0.0' in text
    assert 'unlinked=float(trace["unlinked_kernel_active_union_us"])/1000.0*correction if complete else 0.0' in text
    assert 'if c["kernel_mechanism_classification_enabled"]:' in text
    assert '"one or more observed kernel events lacked External id; all kernel mechanism classification disabled"' in text
    assert '"launch_or_idle_reduction"' in text

def test_issue702_decision_and_authority_are_bounded():
    text=LAUNCHER.read_text()
    assert 'MATERIAL_SHARE_MIN=0.15' in text and 'MATERIAL_MS_MIN=1.0' in text
    assert 'target="explicit_device_kernel_optimization" if best.startswith' in text
    assert '"unavailable_residual_cannot_authorize":True' in text
    for required in ('"optimization_authorized":False','"systems_pass_earned":False','"architecture_freeze_authorized":False',
                     '"s2_authorized":False','"fresh_scientific_seed_authorized":False','"independent_replication_credit":False',
                     '"100m_authorized":False','"breakthrough_proven":False'):
        assert required in text
    for forbidden in ('.backward(','.step(','.zero_grad(','torch.optim.','optimizer=','run_end_to_end_systems_v26_9(','run_end_to_end_systems_v26_8('):
        assert forbidden not in text
    assert 'del reference,transformer' in text
    assert '"reference_model_executed":False' in text and '"transformer_model_executed":False' in text and '"scientific_seed_consumed":False' in text

def test_issue702_workflow_is_owner_only_issues_opened_and_no_manual_or_pr_trigger():
    text=WORKFLOW.read_text()
    assert 'issues:\n    types: [opened]' in text
    assert 'github.event.issue.user.login == github.repository_owner' in text
    assert 'workflow_dispatch:' not in text and 'pull_request:' not in text and 'push:' not in text
    assert 'test "${GITHUB_RUN_ATTEMPT}" = "1"' in text
    assert '[aera-v26-9-issue702-cuda-timeline-preauth]' in text and '[aera-v26-9-issue702-cuda-timeline-l4]' in text
    assert '## #702 sole L4 CUDA timeline attribution authorization' in text

def test_issue702_workflow_guards_incident_parent_and_exact_blobs():
    text=WORKFLOW.read_text()
    for required in (
        'issues/comments/5571988281','__noop_probe_do_not_create','fresh implementation-governance successor','issues/comments/5571118103',
        'SOURCE_MAIN=6fad37d0e4c611fcaff480899b25dc190e9e9127','SOURCE_TREE=87a22d981caf77263574503749757ba77290cc9a',
        'GOVERNANCE_PREDECESSOR_ISSUE=701','GOVERNANCE_INCIDENT_COMMENT=5571988281','EXCLUDED_INERT_REF=__noop_probe_do_not_create',
        'BRANCH=research/aera-v26-9-issue702-cuda-timeline-dispatch-gap-attribution','PARENT_JOB=101753558368',
        'LAUNCHER_BLOB=bf38814a976d929119d17b9e907f1b89461a14a2','WORKFLOW_BLOB=','CPU_TEST_BLOB=','FROZEN_TREE=','FROZEN_COMMIT=',
        'PROFILE_CALLS=1','BASELINE_CALLS=3','SMALL_KERNEL_MAX_US=50.0','REPEATED_MIN_COUNT=2'):
        assert required in text
    assert '101754071069' not in text and '5571178656' not in text
    for expected in FROZEN_FILES.values():assert expected in text or expected in LAUNCHER.read_text()

def test_issue702_preflight_and_l4_are_separately_gated():
    text=WORKFLOW.read_text()
    assert 'modal_aera_v26_9_issue702_cuda_timeline_dispatch_gap_attribution.py::preauth_main' in text
    assert 'modal_aera_v26_9_issue702_cuda_timeline_dispatch_gap_attribution.py::l4_main' in text
    assert "steps.guard.outputs.mode == 'preauth'" in text and "steps.guard.outputs.mode == 'l4'" in text
    assert 'result_absent' in text and 'gpu_used' in text and 'model_constructed' in text and 'new_measurement_performed' in text
    assert 'No L4 or optimization authority is created by this evidence.' in text
    assert 'No optimization, systems PASS, architecture freeze, S2, scaling, or breakthrough authority' in text
