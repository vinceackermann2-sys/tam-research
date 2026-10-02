from __future__ import annotations

from pathlib import Path

from tam_research.pgw_core_mechanism.protocol import (
    ARMS,
    EFFECT_THRESHOLD_NLL,
    EXPECTED_PARAMETERS,
    REPLICATION_SEEDS,
    RESULT_ROOT,
    SMOKE_SEED,
)


def test_core_mechanism_run_control_static_contract() -> None:
    runner = Path("modal_pgw_core_mechanism_1201_app.py").read_text(
        encoding="utf-8"
    )
    workflow = Path(
        ".github/workflows/modal-pgw-core-mechanism-1201-v1.yml"
    ).read_text(encoding="utf-8")

    assert ARMS == (
        "transformer",
        "chunk_local_256",
        "local224_predictor_carry",
        "local224_predictor_reset",
        "local224_only",
    )
    assert SMOKE_SEED == 1_200_001
    assert REPLICATION_SEEDS == (1_200_101, 1_200_102, 1_200_103)
    assert RESULT_ROOT == "/vol/pgw-v2/local-predictor-panel-v1"
    assert EXPECTED_PARAMETERS == 24_940_288
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
    assert "PGW_CORE_MECHANISM_RESULT=" in runner
    assert "train_core_mechanism_arm" in runner
    assert "PREDICTOR_SUPPORTED" in runner
    assert "CROSS_CHUNK_CARRY_SUPPORTED" in runner
    assert "LOCAL_PREDICTOR_CORE_SUPPORTED" in runner
    assert "github-secret" not in runner

    top = workflow.split("\njobs:", 1)[0]
    assert "workflow_dispatch:" not in top
    assert "[modal-pgw-core-mechanism-1201-v1]" in workflow
    assert 'test "$RUN_ATTEMPT" = "1"' in workflow
    assert 'cfg["prereg_issue"] == 1201' in workflow
    assert 'cfg["run_control_issue"] == 1204' in workflow
    assert "97432fc5fe36ec9a1ccbeef74c545f8713b9a475" in workflow
    assert "tam_research/pgw_core_mechanism" in workflow
    assert "tests/test_pgw_core_mechanism_1201.py" in workflow
    assert "tests/test_pgw_core_mechanism_run_control_1204.py" in workflow
    assert workflow.count("modal run --timestamps") == 1
    assert "--detach" not in workflow
    assert "modal_select_account_v3.py" in workflow
    assert "tam-research-data" in workflow
    assert "torch==2.10.0" in workflow
