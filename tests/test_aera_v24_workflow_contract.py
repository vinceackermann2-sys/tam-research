from pathlib import Path


def test_v24_workflow_is_cpu_only_single_shot_and_does_not_authorize_scale():
    text = Path('.github/workflows/aera-v24-vcem-memory-cpu.yml').read_text()
    assert '[aera-v24-vcem-memory-cpu]' in text
    assert 'Duplicate/rerun refused' in text
    assert "torch==2.10.0" in text
    assert 'index-url https://download.pytorch.org/whl/cpu' in text
    assert 'AERA_V24_VCEM_RESULT_JSON=' in text
    assert 'No GPU' in text
    assert 'real-language training' in text
    assert '100M' in text
    assert 'auto-retry' in text
    assert 'gpu:' not in text.lower()
