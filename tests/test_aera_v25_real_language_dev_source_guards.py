from __future__ import annotations

from pathlib import Path


def test_seed8471_workflow_and_launcher_are_single_shot() -> None:
    workflow = Path(".github/workflows/aera-v25-real-language-dev-8471.yml").read_text()
    launcher = Path("modal_aera_v25_real_language_dev_app.py").read_text()
    assert "[aera-v25-real-language-dev-8471]" in workflow
    assert 'GITHUB_RUN_ATTEMPT' in workflow
    assert 'duplicates' in workflow
    assert 'overall systems gate: **PASS**' in workflow
    assert 'No automatic retry' in workflow
    assert 'SEED = 8471' in launcher
    assert 'v25-dev-seed{SEED}' in launcher
    assert 'result_path.exists()' in launcher
    assert 'MAX_GPU_SECONDS = 1800' in launcher
    assert 'AERA_V25_DEV_L4_START_JSON=' in launcher
    assert 'AERA_V25_DEV_RESULT_JSON=' in launcher


def test_seed8471_workflow_does_not_authorize_scale() -> None:
    workflow = Path(".github/workflows/aera-v25-real-language-dev-8471.yml").read_text()
    assert "no freeze/S2/replication/100M/breakthrough authorization" in workflow
    assert "before any 100M/breakthrough authorization" in workflow
