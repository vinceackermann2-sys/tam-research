from pathlib import Path


def test_triton_preflight_is_one_shot_systems_only() -> None:
    text = Path("modal_cpw_v1_triton_preflight_1249.py").read_text()
    assert 'gpu="H100!"' in text
    assert "retries=0" in text
    assert 'RESULT_ROOT = "/vol/cpw-v1/triton-scan-preflight-v1"' in text
    assert "ATTEMPT.json" in text
    assert "RESULT.json" in text
    assert "FAILURE.json" in text
    assert '"scientific_claim_allowed": False' in text
    assert '"breakthrough_claim_allowed": False' in text
    assert "1249001" not in text.replace("1_249_001", "")
