from __future__ import annotations

from pathlib import Path

from tam_research.aera_v26_9_issue605_readonly_issue602_dtype_inspector import (
    ISSUE602_RESULT_SHA256,
    cpu_contract_preflight_issue605,
)

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/aera-v26-9-issue610-readonly-issue602-dtype-inspector-successor.yml"
LAUNCHER = ROOT / "modal_aera_v26_9_issue605_readonly_issue602_dtype_inspector_app.py"
INSPECTOR = ROOT / "tam_research/aera_v26_9_issue605_readonly_issue602_dtype_inspector.py"


def test_issue610_reuses_cpu_only_issue605_reader_contract() -> None:
    contract = cpu_contract_preflight_issue605()
    assert contract["gpu_authorized_by_cpu_preflight"] is False
    assert contract["model_constructed"] is False
    assert contract["checkpoint_loaded"] is False
    assert contract["corpus_accessed"] is False
    assert contract["volume_mutated"] is False
    assert contract["scientific_seed_consumed"] is False
    assert ISSUE602_RESULT_SHA256 == "5ab64b2aa9750babebec6e681c7be587f079436436b5a3cda86ac809018256fb"


def test_issue610_workflow_is_fresh_one_shot_cpu_read_only_successor() -> None:
    workflow = WORKFLOW.read_text()
    launcher = LAUNCHER.read_text()
    inspector = INSPECTOR.read_text()

    assert "[aera-v26-9-issue610-readonly-issue602-dtype-inspector]" in workflow
    assert "## #610 sole CPU inspector authorization" in workflow
    assert "33787400848" in workflow
    assert "100755389648" in workflow
    assert 'select(.name=="Verify canonical authorized issue605 boundary")' in workflow
    assert 'select(.name=="Authenticate Modal")' in workflow
    assert 'select(.name=="Run sole issue605 CPU inspection")' in workflow
    assert workflow.count("modal run modal_aera_v26_9_issue605_readonly_issue602_dtype_inspector_app.py") == 1
    assert "workflow_dispatch" not in workflow
    assert "rerun" not in workflow.lower()

    assert "gpu=" not in launcher
    assert "volume.commit(" not in launcher
    assert "Path(ISSUE602_RESULT_PATH)" in launcher
    assert "read_bytes()" in launcher
    assert "write_" not in launcher

    assert "import torch" not in inspector
    assert "dtype_inference_or_recomputation" in inspector
    assert '"gpu_authorized": False' in inspector
    assert '"repair_authorized": False' in inspector
    assert '"end_to_end_systems_authorized": False' in inspector
