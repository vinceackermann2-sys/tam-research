from __future__ import annotations

import ast
import hashlib
from pathlib import Path

WRAPPER = Path("modal_aera_v26_9_issue693_stage_internal_packaging_repair1.py")
WORKFLOW = Path(".github/workflows/aera-v26-9-issue693-stage-internal-packaging-repair1.yml")
TEST_FILE = Path("tests/test_aera_v26_9_issue693_stage_internal_packaging_repair1_cpu.py")
DIAGNOSTIC = Path("modal_aera_v26_9_issue687_stage_internal_throughput_attribution.py")
ISSUE665 = Path("modal_aera_v26_9_issue665_frozen_throughput_component_attribution.py")

EXPECTED_WRAPPER_BLOB = "eb059a857f60101c81b5e5c9518aff7bcf3aea97"
EXPECTED_WORKFLOW_BLOB = "2bde09812e3ae1ad435f755433945e48e8ded7b1"
EXPECTED_DIAGNOSTIC_BLOB = "b9950c032686a496c6d944c30c441428499cb367"
EXPECTED_ISSUE665_BLOB = "72f27391ff2f0a7bff8d4532f307ddc4869cf494"

PREAUTH_PREFIX = "[aera-v26-9-issue693-stage-internal-preauth]"
L4_PREFIX = "[aera-v26-9-issue693-stage-internal-l4]"
CONSUMED_PREAUTH_PREFIXES = (
    "[aera-v26-9-issue687-stage-internal-preauth]",
    "[aera-v26-9-issue690-stage-internal-preauth]",
)
CONSUMED_L4_PREFIXES = (
    "[aera-v26-9-issue687-stage-internal-l4]",
    "[aera-v26-9-issue690-stage-internal-l4]",
)


def _blob(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    matches = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    assert len(matches) == 1
    return matches[0]


def test_issue693_preserves_frozen_launchers_exactly() -> None:
    assert WRAPPER.is_file()
    assert WORKFLOW.is_file()
    assert TEST_FILE.is_file()
    assert DIAGNOSTIC.is_file()
    assert ISSUE665.is_file()
    assert _blob(WRAPPER) == EXPECTED_WRAPPER_BLOB
    assert _blob(WORKFLOW) == EXPECTED_WORKFLOW_BLOB
    assert _blob(DIAGNOSTIC) == EXPECTED_DIAGNOSTIC_BLOB
    assert _blob(ISSUE665) == EXPECTED_ISSUE665_BLOB


def test_issue693_mounts_exact_missing_root_dependency_before_importing_issue687() -> None:
    text = WRAPPER.read_text()
    issue665_import = "import modal_aera_v26_9_issue665_frozen_throughput_component_attribution as issue665"
    diagnostic_import = "import modal_aera_v26_9_issue687_stage_internal_throughput_attribution as diagnostic"
    mount_assignment = "issue665.image = issue665.image.add_local_file("
    assert issue665_import in text
    assert mount_assignment in text
    assert diagnostic_import in text
    assert text.index(issue665_import) < text.index(mount_assignment) < text.index(diagnostic_import)
    assert 'ISSUE665_LAUNCHER = "modal_aera_v26_9_issue665_frozen_throughput_component_attribution.py"' in text
    assert 'f"/root/{ISSUE665_LAUNCHER}"' in text
    assert "copy=True" not in text


def test_issue693_preauth_entrypoint_is_zero_gpu_and_calls_only_preflight() -> None:
    tree = ast.parse(WRAPPER.read_text())
    fn = _function(tree, "preauth_main")
    body = ast.unparse(fn)
    assert body.count("diagnostic.preflight.remote()") == 1
    assert "diagnostic.run_diagnostic.remote()" not in body
    assert "gpu=" not in body
    assert "volume.commit" not in body
    assert "torch" not in body


def test_issue693_l4_entrypoint_reuses_only_frozen_issue687_calls() -> None:
    tree = ast.parse(WRAPPER.read_text())
    fn = _function(tree, "l4_main")
    body = ast.unparse(fn)
    assert body.count("diagnostic.preflight.remote()") == 1
    assert body.count("diagnostic.run_diagnostic.remote()") == 1
    assert "diagnostic.main(" not in body
    assert "torch" not in body
    assert "volume.commit" not in body


def test_issue693_wrapper_registers_unique_entrypoints() -> None:
    text = WRAPPER.read_text()
    assert "def preauth_main()" in text
    assert "def l4_main()" in text
    assert "def main()" not in text
    assert text.count("@app.local_entrypoint()") == 2
    assert 'PREAUTH_MARKER = "AERA_V26_9_ISSUE693_STAGE_INTERNAL_PREAUTH_JSON="' in text


def test_issue693_workflow_uses_fresh_disjoint_prefixes() -> None:
    text = WORKFLOW.read_text()
    assert PREAUTH_PREFIX in text
    assert L4_PREFIX in text
    assert PREAUTH_PREFIX != L4_PREFIX
    for prefix in CONSUMED_PREAUTH_PREFIXES + CONSUMED_L4_PREFIXES:
        assert prefix not in text
    assert "issues/693" in text
    assert "## #693 pre-implementation remote-import packaging repair1 freeze" in text
    assert "## #693 sole L4 stage-internal attribution authorization" in text


def test_issue693_workflow_executes_only_successor_wrapper_targets() -> None:
    text = WORKFLOW.read_text()
    modal_lines = [line.strip() for line in text.splitlines() if line.strip().startswith("modal run ")]
    assert modal_lines == [
        "modal run modal_aera_v26_9_issue693_stage_internal_packaging_repair1.py::preauth_main 2>&1 | tee /tmp/issue693-preauth.log",
        "modal run modal_aera_v26_9_issue693_stage_internal_packaging_repair1.py::l4_main 2>&1 | tee /tmp/issue693-l4.log",
    ]


def test_issue693_workflow_requires_preauth_and_separate_l4_authorization() -> None:
    text = WORKFLOW.read_text()
    assert "#693/#687 preauthorization result-path absence evidence" in text
    assert "preauth_count" in text
    assert 'test "${preauth_count}" = "1"' in text
    assert "auth_count" in text
    assert 'test "${auth_count}" = "1"' in text
    assert "steps.guard.outputs.mode == 'preauth'" in text
    assert "steps.guard.outputs.mode == 'l4'" in text


def test_issue693_workflow_binds_consumed_failure_and_frozen_blobs() -> None:
    text = WORKFLOW.read_text()
    assert "CONSUMED_TRIGGER=692" in text
    assert "CONSUMED_RUN=34103828734" in text
    assert "CONSUMED_JOB=101684168757" in text
    assert "CONSUMED_ATTEMPT=1" in text
    assert "ISSUE687_LAUNCHER_BLOB=b9950c032686a496c6d944c30c441428499cb367" in text
    assert "ISSUE665_LAUNCHER_BLOB=72f27391ff2f0a7bff8d4532f307ddc4869cf494" in text
    assert "RESULT_PATH=/vol/aera-v26/issue687-stage-internal-throughput-attribution/result.json" in text
    assert "FAIL_FROZEN_E2E_SYSTEMS_GATE" in text


def test_issue693_workflow_yaml_block_shape_and_no_alternate_dispatch() -> None:
    text = WORKFLOW.read_text()
    allowed = ("name:", "on:", "permissions:", "concurrency:", "jobs:")
    unexpected = [
        (number, line)
        for number, line in enumerate(text.splitlines(), 1)
        if line and not line[0].isspace() and not line.startswith(allowed)
    ]
    assert unexpected == []
    lowered = text.lower()
    assert "workflow_dispatch" not in lowered
    assert "modal deploy" not in lowered
    assert "rerun" not in lowered
    assert "redispatch" not in lowered


def test_issue693_no_higher_stage_true_authority() -> None:
    combined = WRAPPER.read_text().lower() + "\n" + WORKFLOW.read_text().lower()
    for phrase in (
        "optimization_authorized=true",
        "systems_pass_earned=true",
        "architecture_freeze_authorized=true",
        "s2_authorized=true",
        "fresh_scientific_seed_authorized=true",
        "independent_replication_credit=true",
        "100m_authorized=true",
        "breakthrough_proven=true",
    ):
        assert phrase not in combined
