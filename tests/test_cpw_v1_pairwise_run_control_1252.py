from pathlib import Path


def test_pairwise_runner_is_one_shot_and_frozen() -> None:
    text = Path("modal_cpw_v1_pairwise_1252.py").read_text()
    assert 'gpu="H100!"' in text
    assert "retries=0" in text
    assert 'RESULT_ROOT' in text
    assert "SMOKE_SEED" in text
    assert "REPLICATION_SEEDS" in text
    assert "ATTEMPT.json" in text
    assert "RESULT.json" in text
    assert "FAILURE.json" in text
    assert '"retry_authorized": False' in text
    assert '"breakthrough_claim_allowed": False' in text
    assert "selected_successor" in text
