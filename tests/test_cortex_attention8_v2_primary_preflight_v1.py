from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "modal_cortex_attention8_v2_primary_preflight_v1.py"
WORKFLOW = (
    ROOT
    / ".github"
    / "workflows"
    / "modal-cortex-attention8-v2-primary-preflight-v1.yml"
)


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_primary_preflight_is_read_only_zero_gpu() -> None:
    runner = _text(RUNNER)
    workflow = _text(WORKFLOW)

    assert '"classification": "ZERO_GPU_ATTENTION8_V2_PRIMARY_PREFLIGHT_V1"' in runner
    assert '"gpu_allocated": False' in runner
    assert '"writes_performed": False' in runner
    assert '"seed_consumed": False' in runner
    assert '"scientific_training_authorized": False' in runner
    assert "retries=0" in runner

    top = workflow.split("\njobs:", 1)[0]
    assert "workflow_dispatch:" not in top
    assert ("modal volume " + "put") not in workflow
    assert ("modal volume " + "delete") not in workflow
    assert ('gpu' + '="') not in runner


def test_primary_preflight_verifies_frozen_corpus() -> None:
    runner = _text(RUNNER)
    assert 'TRAIN_SHA256 = "93e9cb0b7076a4ddd855fc696f657ea62592b8a03a05220be99c402f9043265b"' in runner
    assert 'VAL_SHA256 = "ae0bc5adf36d0aa8e55e5e3903401d3f114b93037f43221944b4e88c5d1a5760"' in runner
    assert 'META_SHA256 = "14bbbcf0ab0b8cba374074ef8ccb80a04ecead06c74780f35a1becd5aef1b8f3"' in runner
    assert 'sizes["train_bytes"] == 4_000_000_000' in runner
    assert 'sizes["val_bytes"] == 10_000_000' in runner
    assert 'meta.get("train_tokens") == 2_000_000_000' in runner
    assert 'meta.get("val_tokens") == 5_000_000' in runner


def test_primary_preflight_requires_exact_engineering_pass() -> None:
    runner = _text(RUNNER)
    assert 'ENGINEERING_RESULT = Path(' in runner
    assert '"ENGINEERING_PANEL_COMPLETE"' in runner
    assert '"ENGINEERING_PANEL_CANDIDATE_SCREEN_PASS"' in runner
    assert 'result.get("progression_candidates") == ["attention_8"]' in runner
    assert 'attention8.get("combined_gate_pass") is True' in runner
    assert 'attention8.get("quality_gate_pass") is True' in runner
    assert 'attention8.get("throughput_gate_pass") is True' in runner
    assert "EXPECTED_ENGINEERING_NLL_DELTA = -0.01449833869934114" in runner
    assert "EXPECTED_ENGINEERING_TPS_RATIO = 1.0651255526571075" in runner


def test_primary_preflight_requires_fresh_successor_namespace() -> None:
    runner = _text(RUNNER)
    assert (
        '"/vol/cortex-s-v0/100m-2b/reduced-attention-v2-attention8-pair1-successor-v1"'
        in runner
    )
    assert "successor_absent = not SUCCESSOR_RESULT_ROOT.exists()" in runner
    assert '"successor_result_root_absent": successor_absent' in runner


def test_primary_preflight_workflow_is_primary_only_single_attempt() -> None:
    workflow = _text(WORKFLOW)
    assert "github.event.issue.user.login == github.repository_owner" in workflow
    assert (
        "github.event.issue.title == '[modal-cortex-attention8-v2-primary-preflight-v1]'"
        in workflow
    )
    assert 'test "$RUN_ATTEMPT" = "1"' in workflow
    assert "MODAL_TOKEN_ID_SECONDARY" not in workflow
    assert "MODAL_TOKEN_ID_2" not in workflow
    assert "modal run --detach --timestamps modal_cortex_attention8_v2_primary_preflight_v1.py" in workflow
