from __future__ import annotations

import hashlib
from pathlib import Path

WRAPPER = Path("modal_aera_v26_9_issue690_stage_internal_preflight_repair1.py")
WORKFLOW = Path(".github/workflows/aera-v26-9-issue690-stage-internal-preauth-repair1.yml")
TEST_FILE = Path("tests/test_aera_v26_9_issue690_stage_internal_preauth_repair1_cpu.py")
DIAGNOSTIC = Path("modal_aera_v26_9_issue687_stage_internal_throughput_attribution.py")

EXPECTED_WRAPPER_BLOB = "ef2ef06d74ed2e7e98cc510791c441aadd48fde2"
EXPECTED_WORKFLOW_BLOB = "ee3d2faa46b7ff89c2b1cbfcb1e39dac6b37c148"
EXPECTED_DIAGNOSTIC_BLOB = "b9950c032686a496c6d944c30c441428499cb367"

PREAUTH_PREFIX = "[aera-v26-9-issue690-stage-internal-preauth]"
L4_PREFIX = "[aera-v26-9-issue690-stage-internal-l4]"


def _blob(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def test_issue690_exact_runtime_blobs() -> None:
    assert WRAPPER.is_file()
    assert WORKFLOW.is_file()
    assert TEST_FILE.is_file()
    assert DIAGNOSTIC.is_file()
    assert _blob(WRAPPER) == EXPECTED_WRAPPER_BLOB
    assert _blob(WORKFLOW) == EXPECTED_WORKFLOW_BLOB
    assert _blob(DIAGNOSTIC) == EXPECTED_DIAGNOSTIC_BLOB


def test_issue690_wrapper_has_unique_local_entrypoint_and_is_zero_gpu() -> None:
    text = WRAPPER.read_text()
    assert "def preauth_main()" in text
    assert "def main()" not in text
    assert text.count("diagnostic.preflight.remote()") == 1
    assert "diagnostic.run_diagnostic.remote()" not in text
    assert "run_diagnostic.remote()" not in text
    assert "gpu=" not in text
    assert "volume.commit" not in text
    assert "torch" not in text
    assert 'PREAUTH_MARKER = "AERA_V26_9_ISSUE690_STAGE_INTERNAL_PREAUTH_JSON="' in text


def test_issue690_workflow_uses_fresh_disjoint_prefixes() -> None:
    text = WORKFLOW.read_text()
    assert PREAUTH_PREFIX in text
    assert L4_PREFIX in text
    assert PREAUTH_PREFIX != L4_PREFIX
    assert "[aera-v26-9-issue687-stage-internal-preauth]" not in text
    assert "[aera-v26-9-issue687-stage-internal-l4]" not in text
    assert "issues/690" in text
    assert "## #690 pre-implementation duplicate-entrypoint repair1 freeze" in text
    assert "## #690 sole L4 stage-internal attribution authorization" in text


def test_issue690_workflow_executes_only_expected_modal_targets() -> None:
    text = WORKFLOW.read_text()
    modal_lines = [line.strip() for line in text.splitlines() if line.strip().startswith("modal run ")]
    assert modal_lines == [
        "modal run modal_aera_v26_9_issue690_stage_internal_preflight_repair1.py::preauth_main 2>&1 | tee /tmp/issue690-preauth.log",
        "modal run modal_aera_v26_9_issue687_stage_internal_throughput_attribution.py::main 2>&1 | tee /tmp/issue690-l4.log",
    ]
    assert "::preauth_main" in modal_lines[0]
    assert "::main" in modal_lines[1]


def test_issue690_workflow_requires_preauth_and_separate_l4_authorization() -> None:
    text = WORKFLOW.read_text()
    assert "#690/#687 preauthorization result-path absence evidence" in text
    assert "preauth_count" in text
    assert 'test "${preauth_count}" = "1"' in text
    assert "auth_count" in text
    assert 'test "${auth_count}" = "1"' in text
    assert "steps.guard.outputs.mode == 'preauth'" in text
    assert "steps.guard.outputs.mode == 'l4'" in text


def test_issue690_workflow_preserves_scientific_launcher_and_result_path() -> None:
    text = WORKFLOW.read_text()
    assert "ISSUE687_LAUNCHER_BLOB=b9950c032686a496c6d944c30c441428499cb367" in text
    assert "modal_aera_v26_9_issue687_stage_internal_throughput_attribution.py" in text
    assert "RESULT_PATH=/vol/aera-v26/issue687-stage-internal-throughput-attribution/result.json" in text
    assert "FAIL_FROZEN_E2E_SYSTEMS_GATE" in text


def test_issue690_workflow_yaml_block_shape_and_no_alternate_dispatch() -> None:
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


def test_issue690_no_higher_stage_true_authority() -> None:
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
