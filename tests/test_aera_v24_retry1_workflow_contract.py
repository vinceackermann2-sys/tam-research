from pathlib import Path


WORKFLOW = Path('.github/workflows/aera-v24-vcem-memory-cpu-retry1.yml')


def _text() -> str:
    return WORKFLOW.read_text()


def test_retry1_has_distinct_single_shot_prefix_and_no_gpu():
    text = _text()
    assert "[aera-v24-vcem-memory-cpu-retry1]" in text
    assert "aera-v24-vcem-memory-cpu-retry1" in text
    assert "modal" not in text.lower()
    assert "gpu" in text.lower()  # frozen guard explicitly says no GPU


def test_retry1_requires_349_pre_science_failure_and_no_result():
    text = _text()
    assert 'issues/349/comments' in text
    assert 'AERA-v24 VCEM controlled workflow implementation/runtime failure' in text
    assert 'AERA-v24 VCEM controlled result' in text
    assert 'result349' in text


def test_retry1_uses_repaired_test_install_before_pytest():
    text = _text()
    install = text.index("pip install -e '.[test]'")
    pytest = text.index('pytest -q tests/test_aera_hardware_core_v24.py')
    assert install < pytest
    assert 'Install package and test dependencies' in text


def test_retry1_executes_unchanged_controlled_harness_and_result_marker():
    text = _text()
    assert 'python aera_v24_vectorized_contextual_episodic_memory_cpu.py' in text
    assert 'AERA_V24_VCEM_RESULT_JSON=' in text
    assert 'train seed 8401' in text
    assert 'eval seed 8402' in text
    assert 'LR=4e-3' in text
    assert '500 steps' in text
    assert 'exact 2/5 controlled physical writes' in text


def test_retry1_cannot_authorize_scale_or_auto_retry():
    text = _text()
    assert 'No GPU' in text
    assert '100M' in text
    assert 'automatic retry2' in text
    assert 'separately preregistered no-training L4 systems benchmark' in text
