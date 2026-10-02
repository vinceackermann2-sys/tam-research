from __future__ import annotations

from pathlib import Path

from tam_research.pgw_v2_mechanism.protocol import (
    ARMS,
    EFFECT_THRESHOLD_NLL,
    EXPECTED_PARAMETERS,
    EXPECTED_SELECTED_FRACTION,
    REPLICATION_SEEDS,
    RESULT_ROOT,
    SMOKE_SEED,
)


def test_mechanism_run_control_static_contract() -> None:
    runner = Path("modal_pgw_v2_mechanism_1181_app.py").read_text(
        encoding="utf-8"
    )
    workflow = Path(
        ".github/workflows/modal-pgw-v2-mechanism-1181-v1.yml"
    ).read_text(encoding="utf-8")

    assert ARMS == (
        "transformer",
        "mean_read_predictive",
        "token_read_predictive",
        "token_read_fixed_random",
        "token_read_recency",
        "no_workspace",
    )
    assert SMOKE_SEED == 1_180_001
    assert REPLICATION_SEEDS == (1_180_101, 1_180_102, 1_180_103)
    assert RESULT_ROOT == "/vol/pgw-v2/mechanism-panel-v1"
    assert EXPECTED_PARAMETERS == 24_940_288
    assert EXPECTED_SELECTED_FRACTION == 0.0625
    assert EFFECT_THRESHOLD_NLL == 0.005

    assert runner.count('gpu="H100!"') == 1
    assert "retries=0" in runner
    assert "ATTEMPT.json" in runner
    assert "RESULT.json" in runner
    assert "FAILURE.json" in runner
    assert '"retry_authorized": False' in runner
    assert '"resume_authorized": False' in runner
    assert '"breakthrough_claim_allowed": False' in runner
    assert '"scale_up_authorized": False' in runner
    assert "PGW_V2_MECHANISM_RESULT=" in runner
    assert "train_mechanism_arm" in runner
    assert "READ_ADDRESSING_SUPPORTED" in runner
    assert "PREDICTIVE_SALIENCE_SUPPORTED" in runner
    assert "WORKSPACE_SUPPORTED" in runner
    assert "PGW_V2_MECHANISM_STACK_SUPPORTED" in runner
    assert "github-secret" not in runner

    top = workflow.split("\njobs:", 1)[0]
    assert "workflow_dispatch:" not in top
    assert "[modal-pgw-v2-mechanism-1181-v1]" in workflow
    assert 'test "$RUN_ATTEMPT" = "1"' in workflow
    assert 'cfg["prereg_issue"] == 1181' in workflow
    assert 'cfg["run_control_issue"] == 1184' in workflow
    assert "0dba13099fdb323fa26b14621da7af726a1574cc" in workflow
    assert "f9b87bff04da0088839dd8dc32a838636c58714f" in workflow
    assert 'PANEL_BASE="0dba13099fdb323fa26b14621da7af726a1574cc"' in workflow
    assert 'PROTECTED="f9b87bff04da0088839dd8dc32a838636c58714f"' in workflow
    assert "tam_research/pgw_v1" in workflow
    assert "tam_research/pgw_v2" in workflow
    assert "tam_research/pgw_v2_mechanism" in workflow
    assert "tests/test_pgw_v2_mechanism_1181.py" in workflow
    assert "tests/test_pgw_v2_mechanism_run_control_1184.py" in workflow
    assert workflow.count("modal run --timestamps") == 1
    assert "--detach" not in workflow
    assert "modal_select_account_v3.py" in workflow
    assert "tam-research-data" in workflow
    assert "torch==2.10.0" in workflow
