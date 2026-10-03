from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = (
    ROOT
    / ".github"
    / "workflows"
    / "modal-cortex-attention8-v2-engineering-result-mirror-primary-v1.yml"
)


def _text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def test_engineering_result_mirror_is_storage_only_zero_gpu() -> None:
    workflow = _text()
    top = workflow.split("\njobs:", 1)[0]
    assert "workflow_dispatch:" not in top
    assert ("modal " + "run ") not in workflow
    assert ('gpu' + '="') not in workflow
    assert "modal volume get" in workflow
    assert "modal volume put" in workflow
    assert ("60" + "231") not in workflow
    assert ("60" + "232") not in workflow


def test_engineering_result_mirror_copies_only_terminal_result_and_manifest() -> None:
    workflow = _text()
    result_path = (
        "/cortex-s-v0/100m-200m/reduced-attention-v2-panel-v1/RESULT.json"
    )
    manifest_path = (
        "/cortex-s-v0/100m-200m/reduced-attention-v2-panel-v1/"
        "PRIMARY_MIRROR_MANIFEST.json"
    )
    assert result_path in workflow
    assert manifest_path in workflow
    assert "checkpoint" not in workflow.lower()
    assert "train.bin" not in workflow
    assert "val.bin" not in workflow


def test_engineering_result_mirror_verifies_exact_frozen_semantics() -> None:
    workflow = _text()
    assert '"ENGINEERING_PANEL_COMPLETE"' in workflow
    assert '"ENGINEERING_PANEL_CANDIDATE_SCREEN_PASS"' in workflow
    assert 'result["engineering_seed"]==2026092901' in workflow
    assert 'result["progression_candidates"]==["attention_8"]' in workflow
    assert 'a["combined_gate_pass"] is True' in workflow
    assert 'a["quality_gate_pass"] is True' in workflow
    assert 'a["throughput_gate_pass"] is True' in workflow
    assert "-0.01449833869934114" in workflow
    assert "1.0651255526571075" in workflow


def test_engineering_result_mirror_fail_closes_on_primary_preexistence() -> None:
    workflow = _text()
    assert "primary engineering RESULT.json already exists; fail closed" in workflow
    assert "primary mirror manifest already exists; fail closed" in workflow
    assert "--force" not in workflow


def test_engineering_result_mirror_rechecks_byte_parity_and_manifest_last() -> None:
    workflow = _text()
    assert "cmp mirror-src/RESULT.json mirror-verify/RESULT.json" in workflow
    upload_result = workflow.index(
        "Upload exact engineering result to primary"
    )
    parity = workflow.index("Re-download primary result and verify parity")
    build_manifest = workflow.index("Build primary mirror manifest after parity")
    upload_manifest = workflow.index("Upload manifest last and verify")
    assert upload_result < parity < build_manifest < upload_manifest


def test_engineering_result_mirror_preserves_no_new_authority() -> None:
    workflow = _text()
    assert '"new_seed_consumed":False' in workflow
    assert '"scientific_training_authorized":False' in workflow
    assert '"remote_function_allocated":False' in workflow
    assert '"gpu_allocated":False' in workflow
    assert '"automatic_retry_authorized":False' in workflow
