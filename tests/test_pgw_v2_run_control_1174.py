from __future__ import annotations

from pathlib import Path

from tam_research.pgw_v2.protocol import (
    EXPECTED_PARAMETERS,
    EXPECTED_SELECTED_FRACTION,
    REPLICATION_SEEDS,
    RESULT_ROOT,
    SMOKE_SEED,
    TRIGGER_TITLE,
)


def test_run_control_static_contract() -> None:
    runner = Path("modal_pgw_v2_app.py").read_text(encoding="utf-8")
    workflow = Path(
        ".github/workflows/modal-pgw-v2-1168.yml"
    ).read_text(encoding="utf-8")

    assert TRIGGER_TITLE == "[modal-pgw-v2-25m-1168]"
    assert SMOKE_SEED == 1_170_001
    assert REPLICATION_SEEDS == (1_170_101, 1_170_102, 1_170_103)
    assert RESULT_ROOT == "/vol/pgw-v2/token-conditioned-workspace-read"
    assert EXPECTED_PARAMETERS == 24_940_288
    assert EXPECTED_SELECTED_FRACTION == 0.0625

    assert runner.count('gpu="H100!"') == 1
    assert "retries=0" in runner
    assert "resume=False" not in runner
    assert '"retry_authorized": False' in runner
    assert '"resume_authorized": False' in runner
    assert "ATTEMPT.json" in runner
    assert "RESULT.json" in runner
    assert "FAILURE.json" in runner
    assert "PGW_V2_RESULT=" in runner
    assert "github-secret" not in runner
    assert "breakthrough_claim_allowed" in runner
    assert "train_pgw_v2_candidate" in runner

    top = workflow.split("\njobs:", 1)[0]
    assert "workflow_dispatch:" not in top
    assert "RUN_ATTEMPT" in workflow
    assert "test \"$RUN_ATTEMPT\" = \"1\"" in workflow
    assert "[modal-pgw-v2-25m-1168]" in workflow
    assert "09fb7b4cf7906db859a1797ad7f056b4cb751aaf" in workflow
    assert "git diff --exit-code" in workflow
    assert "tests/test_pgw_v2_1168.py" in workflow
    assert "tests/test_pgw_v2_run_control_1174.py" in workflow
    assert workflow.count("modal run --timestamps") == 1
    assert "--detach" not in workflow
    assert "modal_select_account_v3.py" in workflow
    assert "tam-research-data" in workflow
    assert "torch==2.10.0" in workflow
