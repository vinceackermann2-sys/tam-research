from __future__ import annotations

import ast
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "modal_aera_v26_10_issue742_derived_common_subregion_attribution.py"
WORKFLOW = ROOT / ".github/workflows/aera-v26-10-issue742-derived-common-subregion-attribution.yml"
PARENT = ROOT / "modal_aera_v26_10_issue736_stage_control_idle_attribution.py"

LAUNCHER_BLOB = "f48a0cc353391b6b32a49d682f42656bb30ab62a"
WORKFLOW_BLOB = "449242e495182ad5e20143d8c21c9653eb6fc544"
PARENT_BLOB = "2a004bb09eed8f644a61353f762f3ce1bd7d14f2"


def _git_blob(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def _module_constants(tree: ast.Module) -> dict[str, object]:
    out: dict[str, object] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    return out


def test_issue742_exact_files_and_parent_blob():
    assert LAUNCHER.exists() and WORKFLOW.exists() and PARENT.exists()
    assert _git_blob(LAUNCHER) == LAUNCHER_BLOB
    assert _git_blob(WORKFLOW) == WORKFLOW_BLOB
    assert _git_blob(PARENT) == PARENT_BLOB
    ast.parse(LAUNCHER.read_text())


def test_issue742_launcher_is_cpu_only_derived_only():
    text = LAUNCHER.read_text()
    tree = ast.parse(text)
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
    assert "torch" not in imported_roots
    assert "numpy" not in imported_roots
    assert "subprocess" not in imported_roots
    assert "time" not in imported_roots

    app_function_decorators = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute) and dec.func.attr == "function":
                app_function_decorators.append(dec)
                assert all(keyword.arg != "gpu" for keyword in dec.keywords)
    assert len(app_function_decorators) == 1

    forbidden_calls = {"_model_call", "checkpoint_hashes", "load_models_v26_9", "_build_v26"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                assert node.func.id not in forbidden_calls
            elif isinstance(node.func, ast.Attribute):
                assert node.func.attr not in forbidden_calls
    assert "torch.profiler" not in text
    assert "torch.load" not in text
    assert "cudaEvent" not in text
    assert "gpu=\"L4\"" not in text


def test_issue742_frozen_constants_and_candidates():
    constants = _module_constants(ast.parse(LAUNCHER.read_text()))
    assert constants["SOURCE_MAIN"] == "24ae68c393e098ec7535276e58776ea8ef11e238"
    assert constants["SOURCE_TREE"] == "94b7f64fc4c98f459d6fd45ec2b31f9ac3b9218d"
    assert constants["RESEARCH_ISSUE"] == 742
    assert constants["PARENT_ISSUE"] == 736
    assert constants["PARENT_TRIGGER"] == 741
    assert constants["PARENT_EVIDENCE_COMMENT"] == 5581226071
    assert constants["PARENT_RUN"] == 34200593238
    assert constants["PARENT_JOB"] == 101978242814
    assert constants["PARENT_ATTEMPT"] == 1
    assert constants["PARENT_LAUNCHER_BLOB"] == PARENT_BLOB
    assert constants["PARENT_RESULT_PATH"] == "/vol/aera-v26/issue736-stage-control-idle-attribution/result.json"
    assert constants["RESULT_PATH"] == "/vol/aera-v26/issue742-derived-common-subregion-attribution/result.json"
    assert constants["TRIGGER_PREFIX"] == "[aera-v26-10-issue742-derived-subregion-cpu]"
    assert constants["MIN_SUBREGION_MS"] == 1.0
    assert constants["MIN_SUBREGION_SHARE"] == 0.10
    assert constants["CANDIDATES"] == (
        "ficem_read_control",
        "ficem_update_control",
        "ficem_parent_exclusive_control",
        "latent_reasoner_cell_control",
        "latent_reasoner_schedule_control",
        "stage_scope_exclusive_glue_control",
        "route_scope_exclusive_glue_control",
        "whole_call_outer_glue_control",
    )


def test_issue742_uses_only_persisted_child_subtracted_fields():
    text = LAUNCHER.read_text()
    required = (
        'trace.get("raw_invocations")',
        'row.get("children_subtracted") is not True',
        '"exclusive_cpu_us"',
        '"exclusive_device_active_union_us"',
        '"exclusive_residual_us"',
        '"exclusive_inter_device_idle_total_us"',
        '"cuda_runtime_counts_exclusive"',
        'trace["normalization_scale"]',
        'parent_hash_before = _sha256(parent_path)',
        'parent_hash_after = _sha256(parent_path)',
        'parent_result_unchanged',
        'same_largest_common_required',
    )
    for token in required:
        assert token in text
    assert "start_us" not in text
    assert "end_us" not in text
    assert "timestamp" not in text.lower()
    assert "_subtract(" not in text
    assert "_intersect(" not in text


def test_issue742_authority_ceiling_is_explicit():
    text = LAUNCHER.read_text()
    for token in (
        '"derived_only": True',
        '"cpu_only": True',
        '"gpu_used": False',
        '"model_constructed": False',
        '"checkpoint_loaded": False',
        '"new_measurement_performed": False',
        '"profiler_run": False',
        '"timing_loop_run": False',
        '"scientific_seed_consumed": False',
        '"optimization_authorized": False',
        '"full_e2e_systems_gate_authorized": False',
        '"systems_pass_earned": False',
        '"architecture_freeze_authorized": False',
        '"s2_authorized": False',
        '"fresh_scientific_seed_authorized": False',
        '"independent_replication_credit": False',
        '"100m_authorized": False',
        '"breakthrough_proven": False',
    ):
        assert token in text


def test_issue742_workflow_is_owner_only_one_shot_issue_opened():
    text = WORKFLOW.read_text()
    assert "issues:\n    types: [opened]" in text
    assert "github.event.issue.user.login == github.repository_owner" in text
    assert "[aera-v26-10-issue742-derived-subregion-cpu]" in text
    assert "workflow_dispatch:" not in text
    assert "pull_request:" not in text
    assert "push:" not in text
    assert 'test "${GITHUB_RUN_ATTEMPT}" = "1"' in text
    assert "modal_aera_v26_10_issue742_derived_common_subregion_attribution.py::main" in text
    assert "🧮 **AERA-v26.10 #742 CPU-derived common-subregion attribution evidence**" in text


def test_issue742_workflow_binds_freeze_parent_and_exact_launcher():
    text = WORKFLOW.read_text()
    for token in (
        "issues/comments/5581226071",
        "SOURCE_MAIN=24ae68c393e098ec7535276e58776ea8ef11e238",
        "SOURCE_TREE=94b7f64fc4c98f459d6fd45ec2b31f9ac3b9218d",
        "PARENT_TRIGGER=741",
        "PARENT_RUN=34200593238",
        "PARENT_JOB=101978242814",
        "PARENT_ATTEMPT=1",
        "PARENT_LAUNCHER_BLOB=2a004bb09eed8f644a61353f762f3ce1bd7d14f2",
        "MIN_SUBREGION_MS=1.0",
        "MIN_SUBREGION_SHARE=0.10",
        f"LAUNCHER_BLOB={LAUNCHER_BLOB}",
        "WORKFLOW_BLOB=",
        "CPU_TEST_BLOB=",
        "FROZEN_TREE=",
        "FROZEN_COMMIT=",
    ):
        assert token in text


def test_issue742_cpu_test_has_no_self_hash_contract():
    tree = ast.parse(Path(__file__).read_text())
    assigned = {
        node.targets[0].id
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    }
    assert "CPU_TEST_BLOB" not in assigned
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != "_git_blob":
            continue
        assert not (
            node.args
            and isinstance(node.args[0], ast.Call)
            and isinstance(node.args[0].func, ast.Name)
            and node.args[0].func.id == "Path"
            and node.args[0].args
            and isinstance(node.args[0].args[0], ast.Name)
            and node.args[0].args[0].id == "__file__"
        )
