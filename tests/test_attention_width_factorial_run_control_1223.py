from __future__ import annotations

from pathlib import Path

from tam_research.attention_width_factorial.protocol import (
    ARMS,
    EFFECT_THRESHOLD_NLL,
    EXPECTED_PARAMETERS,
    REPLICATION_SEEDS,
    RESULT_ROOT,
    SMOKE_SEED,
)


def test_attention_width_run_control_static_contract() -> None:
    runner = Path("modal_attention_width_factorial_1218_app.py").read_text(
        encoding="utf-8"
    )
    workflow = Path(
        ".github/workflows/modal-attention-width-factorial-1218-v1.yml"
    ).read_text(encoding="utf-8")

    assert ARMS == (
        "global256_ff1024",
        "local256_ff1024",
        "global224_ff1088",
        "local224_ff1088",
    )
    assert SMOKE_SEED == 1_220_001
    assert REPLICATION_SEEDS == (1_220_101, 1_220_102, 1_220_103)
    assert RESULT_ROOT == "/vol/attention-width-locality/factorial-25m-v1"
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
    assert "ATTENTION_WIDTH_FACTORIAL_RESULT=" in runner
    assert "train_attention_width_arm" in runner
    assert "GLOBAL_REDUCED_WIDTH_SUPPORTED" in runner
    assert "LOCAL_REDUCED_WIDTH_SUPPORTED" in runner
    assert "WIDTH_LOCALITY_INTERACTION_SUPPORTED" in runner
    assert "GLOBAL_WIDTH_EFFECT" in runner
    assert "LOCAL_WIDTH_EFFECT" in runner
    assert "WIDTH_LOCALITY_INTERACTION" in runner
    assert "github-secret" not in runner

    top = workflow.split("\njobs:", 1)[0]
    assert "workflow_dispatch:" not in top
    assert "[modal-attention-width-factorial-1218-v1]" in workflow
    assert 'test "$RUN_ATTEMPT" = "1"' in workflow
    assert 'cfg["prereg_issue"] == 1218' in workflow
    assert 'cfg["run_control_issue"] == 1223' in workflow
    assert "f337e4e05b21cecc5e7075f925cdde202a636a49" in workflow
    assert "tam_research/attention_width_factorial" in workflow
    assert "tam_research/pgw_core_mechanism" in workflow
    assert "tam_research/pgw_v1" in workflow
    assert "tests/test_attention_width_factorial_1218.py" in workflow
    assert "tests/test_attention_width_factorial_run_control_1223.py" in workflow
    assert workflow.count("modal run --timestamps") == 1
    assert "--detach" not in workflow
    assert "modal_select_account_v3.py" in workflow
    assert "tam-research-data" in workflow
    assert "torch==2.10.0" in workflow
