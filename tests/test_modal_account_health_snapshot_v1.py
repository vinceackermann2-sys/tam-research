from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "modal_account_health_snapshot_v1.py"
WORKFLOW = ROOT / ".github" / "workflows" / "modal-account-health-snapshot-v1.yml"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_health_snapshot_is_read_only_zero_gpu() -> None:
    script = _text(SCRIPT)
    workflow = _text(WORKFLOW)

    assert '"classification": "ZERO_GPU_MODAL_ACCOUNT_HEALTH_SNAPSHOT_V1"' in script
    assert '"gpu_allocated": False' in script
    assert '"writes_performed": False' in script
    assert '"seed_consumed": False' in script

    forbidden = (
        "modal volume put",
        "modal volume delete",
        'gpu="',
        "h100",
        "PAIR1_DISPATCH_RESERVED",
        "ENGINEERING_PANEL_DISPATCH_RESERVED",
    )
    for needle in forbidden:
        assert needle not in workflow.lower() if needle == "h100" else needle not in workflow

    assert "workflow_dispatch:" not in workflow.split("\njobs:", 1)[0]
    assert "modal run " not in workflow


def test_health_snapshot_probes_both_accounts_and_required_volume() -> None:
    script = _text(SCRIPT)
    assert 'label="primary"' in script
    assert 'label="secondary"' in script
    assert 'REQUIRED_VOLUME = "tam-research-data"' in script
    assert 'RUNTIME_PROBE_PATH = "modal_runtime_admission_probe_1067_v1.py"' in script
    assert "resolve_secondary_credentials" in script
    assert "probe_account" in script


def test_health_snapshot_workflow_is_owner_issue_only_and_single_attempt() -> None:
    workflow = _text(WORKFLOW)
    assert "github.event.issue.user.login == github.repository_owner" in workflow
    assert "github.event.issue.title == '[modal-account-health-snapshot-v1]'" in workflow
    assert 'test "$RUN_ATTEMPT" = "1"' in workflow
    assert "actions/checkout@v6" in workflow
    assert "ref: main" in workflow


def test_health_snapshot_records_public_json_and_closes_issue() -> None:
    workflow = _text(WORKFLOW)
    assert "MODAL_ACCOUNT_HEALTH_SNAPSHOT_V1" in workflow
    assert "snapshot_json" in workflow
    assert "gh api --method POST" in workflow
    assert 'gh issue close "$ISSUE_NUMBER"' in workflow
