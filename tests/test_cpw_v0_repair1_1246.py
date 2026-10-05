from __future__ import annotations

from pathlib import Path

from tam_research.cpw_v0.protocol import REPLICATION_SEEDS, REPLICATION_TOKENS


def test_repair_is_replication_only_and_does_not_rerun_smoke() -> None:
    text = Path("modal_cpw_v0_repair1_app.py").read_text()
    assert "run_replication" in text
    assert "run_screen" not in text
    assert "SMOKE_SEED" not in text
    assert "SMOKE_TOKENS" not in text
    assert "consumed_smoke_seed_excluded" in text
    assert "1_260_001" in text
    assert "for seed in REPLICATION_SEEDS" in text


def test_repair_preserves_one_shot_controls() -> None:
    text = Path("modal_cpw_v0_repair1_app.py").read_text()
    assert 'gpu="H100!"' in text
    assert "retries=0" in text
    assert 'RESULT_ROOT = "/vol/cpw-v0/multi-predictor-workspace-repair1"' in text
    assert "ATTEMPT.json" in text
    assert "RESULT.json" in text
    assert "FAILURE.json" in text
    assert "retry_authorized" in text
    assert "resume_authorized" in text
    assert "breakthrough_claim_allowed" in text
    assert "scale_up_authorized" in text


def test_frozen_replication_identities_unchanged() -> None:
    assert REPLICATION_SEEDS == (1_260_101, 1_260_102, 1_260_103)
    assert REPLICATION_TOKENS == 5_000_000


def test_bf16_router_repair_uses_direct_active_fraction_not_exact_usage_sum() -> None:
    text = Path("modal_cpw_v0_repair1_app.py").read_text()
    assert "_router_ok_repaired" in text
    assert "active_fraction" in text
    assert "expected_active" in text
    assert "2e-3" in text
    assert "usage_sum" not in text
