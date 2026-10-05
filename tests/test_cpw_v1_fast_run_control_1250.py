from pathlib import Path


def test_fast_scientific_h100_workflow_is_pinned_and_one_shot() -> None:
    text = Path(".github/workflows/cpw-v1-fast-scientific-h100.yml").read_text()
    assert "CPW-v1-fast scientific H100 replication" in text
    assert "cpw/LAUNCH_V1_FAST_SCIENTIFIC.json" in text
    assert "37318151913" in text
    assert "github.run_attempt" in text
    assert "modal_cpw_v1_fast_scientific_1250.py" in text
    assert "tests/test_cpw_v1_fast_scientific_1250.py" in text
    assert "retries=0" in Path("modal_cpw_v1_fast_scientific_1250.py").read_text()
    assert "workflow_dispatch" not in text
