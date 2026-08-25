from pathlib import Path


def test_v24_1_workflow_is_single_shot_cpu_only_and_bound_to_preregistration():
    text = Path(".github/workflows/aera-v24-1-causal-addressing-cpu.yml").read_text()
    assert "[aera-v24-1-causal-addressing-cpu]" in text
    assert "Refuse duplicate scientific trigger" in text
    assert "issues/355/comments" in text
    assert "issues/356" in text
    assert "seed" in text.lower()
    assert "8411/8412" in text
    assert "pip install -e '.[test]'" in text
    assert "aera_v24_1_causal_end_to_end_addressing_cpu.py" in text
    assert "automatic retry" in text
    assert "runs-on: ubuntu-latest" in text
    assert "modal" not in text.lower()
    assert "cuda" not in text.lower()
    assert "production corpus" in text


def test_v24_1_harness_has_no_address_contrastive_training_loss():
    text = Path("aera_v24_1_causal_end_to_end_addressing_cpu.py").read_text()
    assert "ADDRESS_CONTRASTIVE_WEIGHT = 0.0" in text
    assert "total = query_loss + PAYLOAD_TOKEN_WEIGHT * payload[\"payload_token_loss\"]" in text
    assert "multi_positive_contrastive_loss" not in text
    assert "SEED = 8411" in text
    assert "EVAL_SEED = 8412" in text
