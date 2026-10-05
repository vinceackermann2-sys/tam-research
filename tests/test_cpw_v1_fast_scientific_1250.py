from pathlib import Path


def test_fast_scientific_runner_is_frozen_one_shot() -> None:
    text = Path("modal_cpw_v1_fast_scientific_1250.py").read_text()
    assert 'gpu="H100!"' in text
    assert "retries=0" in text
    assert 'RESULT_ROOT = "/vol/cpw-v1/fast-scientific-replication-v1"' in text
    assert "SMOKE_SEED = 1_250_001" in text
    assert "REPLICATION_SEEDS = (1_250_101, 1_250_102, 1_250_103)" in text
    assert "SMOKE_TOKENS = 1_000_000" in text
    assert "REPLICATION_TOKENS = 5_000_000" in text
    assert "ATTEMPT.json" in text
    assert "RESULT.json" in text
    assert "FAILURE.json" in text
    assert '"retry_authorized": False' in text
    assert '"breakthrough_claim_allowed": False' in text
    assert "CPW_V1_FAST_PARETO_SIGNAL" in text
