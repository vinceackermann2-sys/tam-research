from pathlib import Path

from tam_research.cpw_v0_mechanism.protocol import (
    ARM_ORDER,
    REPLICATION_SEEDS,
    REPLICATION_TOKENS,
    SMOKE_SEED,
    SMOKE_TOKENS,
)


def test_mechanism_modal_runner_is_frozen_and_one_shot() -> None:
    text = Path("modal_cpw_v0_mechanism_app.py").read_text()
    assert 'gpu="H100!"' in text
    assert "retries=0" in text
    assert "ATTEMPT.json" in text
    assert "RESULT.json" in text
    assert "FAILURE.json" in text
    assert "for arm in ARM_ORDER" in text
    assert "for seed in REPLICATION_SEEDS" in text
    assert "breakthrough_claim_allowed" in text
    assert "scale_up_authorized" in text


def test_mechanism_identity_and_budget_frozen() -> None:
    assert ARM_ORDER == (
        "transformer",
        "full",
        "no_aux",
        "uniform_all",
        "no_memory",
        "no_sequence",
        "no_world",
        "no_attention",
    )
    assert SMOKE_SEED == 1_247_001
    assert REPLICATION_SEEDS == (1_247_101, 1_247_102, 1_247_103)
    assert SMOKE_TOKENS == 500_000
    assert REPLICATION_TOKENS == 3_000_000
