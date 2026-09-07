from __future__ import annotations
import ast, hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
LAUNCHER=ROOT/'modal_aera_v26_10_issue732_differential_kernel_time_attribution.py'
WORKFLOW=ROOT/'.github/workflows/aera-v26-10-issue732-differential-kernel-time-attribution.yml'
LAUNCHER_BLOB='c0ed30de101c890f812d5b0fef0047056d81f9fb'
WORKFLOW_BLOB='193e03b429ae3829fb67fc2e1fe786c7ca9529f1'
ISSUE710_BLOB='b885250753ea169cd89dd7a978bb3647fd261fe8'
V26_10_BLOB='d8f691c198eed1fa96bcbb78a4e76cad82d18779'

def _blob(path:Path)->str:
    data=path.read_bytes(); return hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()

def test_issue732_exact_files_and_frozen_runtime_identity():
    assert _blob(LAUNCHER)==LAUNCHER_BLOB
    assert _blob(WORKFLOW)==WORKFLOW_BLOB
    assert _blob(ROOT/'modal_aera_v26_10_issue710_memory_safe_harness.py')==ISSUE710_BLOB
    assert _blob(ROOT/'tam_research/aera_hardware_core_v26_10_latent_depth_sync_coalescing.py')==V26_10_BLOB
    ast.parse(LAUNCHER.read_text()); ast.parse((ROOT/'modal_aera_v26_10_issue710_memory_safe_harness.py').read_text())

def test_issue732_parent_and_fixtures_are_exact():
    text=LAUNCHER.read_text()
    for token in (
      'SOURCE_MAIN="267f6d6018a0d355436869919ed73c7723315eab"','SOURCE_TREE="1739ccd80ad48636a4439cbc9d39ba1ab34df807"',
      'PARENT_TRIGGER=731','PARENT_EVIDENCE_COMMENT=5574730812','PARENT_RUN=34154255434','PARENT_JOB=101842659688','PARENT_ATTEMPT=1',
      'PARENT_DECISION="FAIL_V26_10_LATENT_DEPTH_SYNC_COALESCING_MICROBENCH"','BATCHES=(8,64)','WARMUP_CALLS=3','BASELINE_CALLS=3','PROFILE_CALLS=1',
      '33.98348808288574','76.74127960205078','33.72492790222168','76.23878479003906','PARENT_SYNC={8:(49,30),64:(51,30)}',
      'INTEGRATED_ATOL=frozen710.INTEGRATED_ATOL','INTEGRATED_RTOL=frozen710.INTEGRATED_RTOL'):
        assert token in text

def test_issue732_measurement_and_normalization_contract():
    text=LAUNCHER.read_text()
    for token in (
      'torch.profiler.ProfilerActivity.CPU','torch.profiler.ProfilerActivity.CUDA','record_shapes=False','profile_memory=False','with_stack=False',
      'torch.profiler.record_function(label)','prof.export_chrome_trace','torch.cuda.Event(enable_timing=True)','torch.cuda.synchronize()',
      'chronological_device_activities','device_active_union_us','idle_gap_total_us','kernel_duration_buckets','top_kernel_names_by_total','top_kernel_names_by_count',
      'cuda_runtime_counts','distortion_ratio','normalization_scale','min(float(us)/1000.0*scale,unprofiled_median)'):
        assert token in text
    assert 'kernel_aten_ownership":"not inferred; raw CUDA kernel names/device durations only"' in text

def test_issue732_preregistered_decision_rule_is_exact_and_common_batch():
    text=LAUNCHER.read_text()
    for token in (
      'REPEATED_MIN_CALLS=100','REPEATED_MIN_MS=1.0','REPEATED_MIN_SHARE=0.10',
      'WHOLE_STAGE_MIN_SMALL_CALLS=500','WHOLE_STAGE_MIN_SMALL_ACTIVE_SHARE=0.20','WHOLE_STAGE_MIN_IDLE_MS=1.0',
      'repeated_small_kernel_fusion::','explicit_device_kernel_optimization::','whole_stage_launch_coalescing',
      'common=set(qualifying) if common is None else common & set(qualifying)','if all(ws):return "whole_stage_launch_coalescing",evidence'):
        assert token in text

def test_issue732_correctness_precedes_performance_and_no_training():
    text=LAUNCHER.read_text()
    assert 'if not correct:raise RuntimeError(f"issue732 correctness drift batch {batch}")' in text
    for token in ('_decision_equivalence','_route_exact','_chunked_logit_equivalence','_state_equivalence','_finite_output','_schemas_and_weights_exact'):
        assert token in text
    for forbidden in ('.backward(','.step(','.zero_grad(','torch.optim.','optimizer='):
        assert forbidden not in text
    for flag in ('"optimization_authorized":False','"full_e2e_systems_gate_authorized":False','"systems_pass_earned":False','"architecture_freeze_authorized":False','"s2_authorized":False','"fresh_scientific_seed_authorized":False','"100m_authorized":False','"breakthrough_proven":False'):
        assert flag in text

def test_issue732_workflow_is_owner_only_fresh_two_stage_and_one_attempt():
    text=WORKFLOW.read_text()
    assert 'issues:\n    types: [opened]' in text
    assert 'github.event.issue.user.login == github.repository_owner' in text
    assert 'workflow_dispatch:' not in text and 'pull_request:' not in text and 'push:' not in text
    assert 'test "${GITHUB_RUN_ATTEMPT}" = "1"' in text
    for token in ('[aera-v26-10-issue732-differential-kernel-preauth]','[aera-v26-10-issue732-differential-kernel-l4]',
                  '## #732 sole L4 differential kernel-time attribution authorization','🔎 **AERA-v26.10 #732 differential-kernel preauthorization evidence**','🔬 **AERA-v26.10 #732 differential kernel-time attribution evidence**'):
        assert token in text

def test_issue732_workflow_freezes_parent_and_dynamic_file_hashes():
    text=WORKFLOW.read_text()
    for token in ('5574730812','34154255434','101842659688','b885250753ea169cd89dd7a978bb3647fd261fe8','d8f691c198eed1fa96bcbb78a4e76cad82d18779',
                  'launcher_blob="$(printf','workflow_blob="$(printf','cpu_test_blob="$(printf','frozen_tree="$(printf','frozen_commit="$(printf'):
        assert token in text
    assert LAUNCHER_BLOB not in text
    assert WORKFLOW_BLOB not in text
