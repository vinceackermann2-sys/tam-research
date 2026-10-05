from __future__ import annotations

from pathlib import Path

from tam_research.attention_width_100m.protocol import (
    ARMS,
    EFFECT_THRESHOLD_NLL,
    EXPECTED_PARAMETERS,
    MIN_THROUGHPUT_RATIO,
    REPLICATION_SEEDS,
    RESULT_ROOT,
    SMOKE_SEED,
)


def test_attention_width_100m_run_control_static_contract() -> None:
    runner = Path("modal_attention_width_100m_1238_app.py").read_text(
        encoding="utf-8"
    )
    workflow = Path(
        ".github/workflows/modal-attention-width-100m-1238-v1.yml"
    ).read_text(encoding="utf-8")

    assert ARMS == (
        "global512_ff2048",
        "global448_ff2176",
    )
    assert SMOKE_SEED == 1_240_001
    assert REPLICATION_SEEDS == (1_240_101, 1_240_102, 1_240_103)
    assert RESULT_ROOT == "/vol/attention-width-global/100m-replication-v1"
    assert EXPECTED_PARAMETERS == 101_803_520
    assert EFFECT_THRESHOLD_NLL == 0.005
    assert MIN_THROUGHPUT_RATIO == 0.80

    assert runner.count('gpu="H100!"') == 1
    assert "retries=0" in runner
    assert "ATTEMPT.json" in runner
    assert "RESULT.json" in runner
    assert "FAILURE.json" in runner
    assert '"retry_authorized": False' in runner
    assert '"resume_authorized": False' in runner
    assert '"breakthrough_claim_allowed": False' in runner
    assert '"combination_authorized": False' in runner
    assert '"production_claim_allowed": False' in runner
    assert "ATTENTION_WIDTH_100M_RESULT=" in runner
    assert "train_attention_width_100m_arm" in runner
    assert "GLOBAL_REDUCED_WIDTH_100M_REPLICATED" in runner
    assert "GLOBAL_WIDTH_EFFECT_100M" in runner
    assert "github-secret" not in runner

    top = workflow.split("\njobs:", 1)[0]
    assert "workflow_dispatch:" not in top
    assert "[modal-attention-width-100m-1238-v1]" in workflow
    assert 'test "$RUN_ATTEMPT" = "1"' in workflow
    assert 'cfg["prereg_issue"] == 1238' in workflow
    assert 'cfg["run_control_issue"] == 1243' in workflow
    assert "5ec655e2efdf7a1cea97456007d21d4fe0ff2ace" in workflow
    assert "tam_research/attention_width_100m" in workflow
    assert "tam_research/attention_width_50m" in workflow
    assert "tam_research/attention_width_factorial" in workflow
    assert "tam_research/scales.py" in workflow
    assert "tests/test_attention_width_100m_1238.py" in workflow
    assert "tests/test_attention_width_100m_run_control_1243.py" in workflow
    assert workflow.count("modal run --timestamps") == 1
    assert "--detach" not in workflow
    assert "modal_select_account_v3.py" in workflow
    assert "tam-research-data" in workflow
    assert "torch==2.10.0" in workflow
