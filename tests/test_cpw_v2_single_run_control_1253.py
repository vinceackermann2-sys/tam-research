from pathlib import Path

def test_single_runner_is_frozen_one_shot() -> None:
    t=Path("modal_cpw_v2_single_1253.py").read_text()
    assert 'gpu="H100!"' in t
    assert "retries=0" in t
    assert "ATTEMPT.json" in t and "RESULT.json" in t and "FAILURE.json" in t
    assert "CPW_V2_SINGLE_RESULT" in t
    assert '"retry_authorized":False' in t or '"retry_authorized": False' in t
    assert "strong_pareto_3_of_3" in t
