from pathlib import Path


WORKFLOW = Path('.github/workflows/aera-v24-vcem-memory-cpu.yml')


def test_v24_controlled_workflow_installs_test_dependencies_before_pytest():
    text = WORKFLOW.read_text()
    install = "pip install -e '.[test]'"
    targeted = "pytest -q tests/test_aera_hardware_core_v24.py tests/test_aera_v24_vectorized_contextual_episodic_cpu.py"
    assert install in text
    assert targeted in text
    assert text.index(install) < text.index(targeted)


def test_v24_controlled_workflow_captures_preflight_failures():
    text = WORKFLOW.read_text()
    assert 'tee /tmp/aera-v24-vcem-preflight.log' in text
    assert 'cat /tmp/aera-v24-vcem-preflight.log' in text
    assert 'cat /tmp/aera-v24-vcem.log' in text


def test_v24_workflow_repair_does_not_change_frozen_science_strings():
    text = WORKFLOW.read_text()
    required = (
        'seed ${SEED:-8401}',
        'LR=4e-3',
        '500 steps',
        'exact 2/5 controlled write budget',
        '48-slot causal contextual episodic KV state',
        'cosine>=0.95',
        'top-4 read at T=0.10',
        'No GPU, production corpus, real-language seed, architecture freeze, S2, independent-replication credit, 100M, auto-retry, or breakthrough claim.',
    )
    for item in required:
        assert item in text
