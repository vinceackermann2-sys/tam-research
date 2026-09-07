from __future__ import annotations

import ast
import hashlib
from pathlib import Path

LAUNCHER=Path("modal_aera_v26_9_issue697_micro_op_attribution.py")
WORKFLOW=Path(".github/workflows/aera-v26-9-issue697-micro-op-attribution.yml")
TEST_FILE=Path("tests/test_aera_v26_9_issue697_micro_op_attribution_cpu.py")
ISSUE687=Path("modal_aera_v26_9_issue687_stage_internal_throughput_attribution.py")
ISSUE665=Path("modal_aera_v26_9_issue665_frozen_throughput_component_attribution.py")
BASE_CORE=Path("tam_research/aera_hardware_core.py")
EXPECTED_LAUNCHER_BLOB="c82affc01d1ecb096371e8865dbd7b18e739f5e1"
EXPECTED_WORKFLOW_BLOB="d99ee80ebd31bcd4a032f13c5869d624a0c13d07"
EXPECTED_ISSUE687_BLOB="b9950c032686a496c6d944c30c441428499cb367"
EXPECTED_ISSUE665_BLOB="72f27391ff2f0a7bff8d4532f307ddc4869cf494"
EXPECTED_BASE_CORE_BLOB="ffe0341829fb905e40ef9ae3544f797471f1aa9c"
PREAUTH_PREFIX="[aera-v26-9-issue697-micro-op-preauth]"
L4_PREFIX="[aera-v26-9-issue697-micro-op-l4]"
CONSUMED_PREFIXES=(
"[aera-v26-9-issue687-stage-internal-preauth]",
"[aera-v26-9-issue687-stage-internal-l4]",
"[aera-v26-9-issue690-stage-internal-preauth]",
"[aera-v26-9-issue690-stage-internal-l4]",
"[aera-v26-9-issue693-stage-internal-preauth]",
"[aera-v26-9-issue693-stage-internal-l4]",
)

def _blob(path:Path)->str:
    data=path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode()+data).hexdigest()

def _fn(tree:ast.Module,name:str)->ast.FunctionDef:
    xs=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name==name]
    assert len(xs)==1
    return xs[0]

def test_issue697_frozen_files_and_parent_blobs():
    assert LAUNCHER.is_file() and WORKFLOW.is_file() and TEST_FILE.is_file()
    assert _blob(LAUNCHER)==EXPECTED_LAUNCHER_BLOB
    assert _blob(WORKFLOW)==EXPECTED_WORKFLOW_BLOB
    assert _blob(ISSUE687)==EXPECTED_ISSUE687_BLOB
    assert _blob(ISSUE665)==EXPECTED_ISSUE665_BLOB
    assert _blob(BASE_CORE)==EXPECTED_BASE_CORE_BLOB

def test_issue697_remote_image_packages_both_root_dependencies():
    text=LAUNCHER.read_text()
    assert '.add_local_file(ISSUE665_LAUNCHER,f"/root/{ISSUE665_LAUNCHER}")' in text
    assert '.add_local_file(ISSUE687_LAUNCHER,f"/root/{ISSUE687_LAUNCHER}")' in text
    assert 'EXPECTED_BLOBS=' in text
    assert EXPECTED_ISSUE665_BLOB in text and EXPECTED_ISSUE687_BLOB in text and EXPECTED_BASE_CORE_BLOB in text

def test_issue697_preserves_frozen_model_forward_and_only_wraps_original_calls():
    text=LAUNCHER.read_text()
    assert "class StackedChunkExpertBank" not in text
    assert "class TokenwiseFastMemoryStage" not in text
    assert "torch.einsum(" not in text
    assert "F.gelu(" not in text
    assert "selected_probs, idx" not in text
    assert "return r.record" in text
    assert "record_function" in text
    assert "torch.profiler.profile" in text
    assert "self_device_time_total" in text
    assert "dispatch_or_unattributed_gap" in text

def test_issue697_has_required_micro_op_families_and_expert_split():
    text=LAUNCHER.read_text()
    for phrase in (
        "matmul_gemm","softmax_topk_sort","activation","index_gather_scatter",
        "transfer_cast_copy","shape_view_layout","reduction","allocation_mask",
        "elementwise","dispatch_or_unattributed_gap","projection_1","projection_2",
        "routing_selection","weight_gather_and_indexing","weighted_output_reduction",
        "telemetry_tail",
    ):
        assert phrase in text
    assert "MATERIAL_SHARE_MIN=0.15" in text
    assert "MATERIAL_MS_MIN=1.0" in text
    assert '"regime_specific_targets"' in text
    assert '"next_target"' in text

def test_issue697_preauth_is_zero_gpu_and_diagnostic_not_called():
    tree=ast.parse(LAUNCHER.read_text())
    body=ast.unparse(_fn(tree,"preauth_main"))
    assert body.count("preflight.remote()")==1
    assert "run_diagnostic.remote()" not in body
    assert "gpu=" not in body

def test_issue697_l4_entrypoint_calls_exactly_one_preflight_and_diagnostic():
    tree=ast.parse(LAUNCHER.read_text())
    body=ast.unparse(_fn(tree,"l4_main"))
    assert body.count("preflight.remote()")==1
    assert body.count("run_diagnostic.remote()")==1

def test_issue697_result_path_is_fresh_and_parent_is_read_only():
    text=LAUNCHER.read_text()
    assert 'RESULT_PATH="/vol/aera-v26/issue697-micro-op-attribution/result.json"' in text
    assert 'PARENT_RESULT_PATH="/vol/aera-v26/issue687-stage-internal-throughput-attribution/result.json"' in text
    assert 'path=Path(RESULT_PATH)' in text
    assert 'Path(PARENT_RESULT_PATH)' in text
    assert 'Path(PARENT_RESULT_PATH).write' not in text
    assert 'comparative_gate_rerun":False' in text
    assert '"reference_model_executed":False' in text
    assert '"transformer_model_executed":False' in text
    assert '"scientific_seed_consumed":False' in text

def test_issue697_workflow_fresh_prefixes_and_single_attempt_gate():
    text=WORKFLOW.read_text()
    assert PREAUTH_PREFIX in text and L4_PREFIX in text and PREAUTH_PREFIX!=L4_PREFIX
    for old in CONSUMED_PREFIXES:
        assert old not in text
    assert 'test "${GITHUB_RUN_ATTEMPT}" = "1"' in text
    assert "issues/697" in text
    assert "## #697 pre-implementation micro-op attribution freeze" in text
    assert "## #697 sole L4 micro-op attribution authorization" in text

def test_issue697_workflow_only_runs_successor_launcher_targets():
    lines=[x.strip() for x in WORKFLOW.read_text().splitlines() if x.strip().startswith("modal run ")]
    assert lines==[
      "modal run modal_aera_v26_9_issue697_micro_op_attribution.py::preauth_main 2>&1 | tee /tmp/issue697-preauth.log",
      "modal run modal_aera_v26_9_issue697_micro_op_attribution.py::l4_main 2>&1 | tee /tmp/issue697-l4.log",
    ]

def test_issue697_workflow_requires_preauth_and_separate_l4_authorization():
    text=WORKFLOW.read_text()
    assert "#697 micro-op preauthorization evidence" in text
    assert "preauth_count" in text and 'test "${preauth_count}" = "1"' in text
    assert "auth_count" in text and 'test "${auth_count}" = "1"' in text
    assert "steps.guard.outputs.mode == 'preauth'" in text
    assert "steps.guard.outputs.mode == 'l4'" in text

def test_issue697_workflow_binds_parent_and_thresholds():
    text=WORKFLOW.read_text()
    for phrase in (
      "PARENT_TRIGGER=696","PARENT_EVIDENCE_COMMENT=5569795957","PARENT_RUN=34115412471",
      "PARENT_JOB=101720942690","PARENT_ATTEMPT=1",
      "RESULT_PATH=/vol/aera-v26/issue697-micro-op-attribution/result.json",
      "PARENT_RESULT_PATH=/vol/aera-v26/issue687-stage-internal-throughput-attribution/result.json",
      "MATERIAL_SHARE_MIN=0.15","MATERIAL_MS_MIN=1.0",
      "ISSUE687_LAUNCHER_BLOB=b9950c032686a496c6d944c30c441428499cb367",
      "ISSUE665_LAUNCHER_BLOB=72f27391ff2f0a7bff8d4532f307ddc4869cf494",
      "BASE_CORE_BLOB=ffe0341829fb905e40ef9ae3544f797471f1aa9c",
    ):
        assert phrase in text

def test_issue697_no_alternate_dispatch_and_no_higher_stage_true_authority():
    combined=(LAUNCHER.read_text()+"\n"+WORKFLOW.read_text()).lower()
    assert "workflow_dispatch" not in combined
    assert "modal deploy" not in combined
    for phrase in (
      "optimization_authorized=true","systems_pass_earned=true","architecture_freeze_authorized=true",
      "s2_authorized=true","fresh_scientific_seed_authorized=true","independent_replication_credit=true",
      "100m_authorized=true","breakthrough_proven=true",
    ):
        assert phrase not in combined
