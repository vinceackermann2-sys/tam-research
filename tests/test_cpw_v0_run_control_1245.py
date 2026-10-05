from __future__ import annotations

from pathlib import Path

from tam_research.cpw_v0.protocol import (
    REPLICATION_SEEDS,
    REPLICATION_TOKENS,
    RESULT_ROOT,
    SMOKE_SEED,
    SMOKE_TOKENS,
)


def test_modal_launcher_is_one_shot_and_frozen() -> None:
    text = Path("modal_cpw_v0_app.py").read_text()
    assert 'gpu="H100!"' in text
    assert "retries=0" in text
    assert 'VOLUME_NAME = "tam-research-data"' in text
    assert "ATTEMPT.json" in text
    assert "RESULT.json" in text
    assert "FAILURE.json" in text
    assert "retry_authorized" in text
    assert "resume_authorized" in text
    assert "breakthrough_claim_allowed" in text
    assert "scale_up_authorized" in text
    assert "for seed in REPLICATION_SEEDS" in text


def test_protocol_identity_is_frozen() -> None:
    assert SMOKE_SEED == 1_260_001
    assert REPLICATION_SEEDS == (1_260_101, 1_260_102, 1_260_103)
    assert SMOKE_TOKENS == 1_000_000
    assert REPLICATION_TOKENS == 5_000_000
    assert RESULT_ROOT == "/vol/cpw-v0/multi-predictor-workspace"
