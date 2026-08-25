from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from tam_research import aera_v25_post8471_triage_repair1 as repair1
from tam_research import aera_v25_post8471_triage_repair2 as repair2


def _result(seed: int = 8471) -> dict:
    return {
        "protocol": {"development_seed": seed},
        "aera": {"seed": seed},
        "transformer": {"seed": seed},
    }


def test_validate_source_result_seed_uses_real_durable_schema() -> None:
    assert repair2.validate_source_result_seed(_result()) == {
        "protocol_development_seed": 8471,
        "aera_seed": 8471,
        "transformer_seed": 8471,
    }


@pytest.mark.parametrize(
    "path",
    ["protocol", "aera", "transformer"],
)
def test_validate_source_result_seed_rejects_each_mismatch(path: str) -> None:
    payload = _result()
    key = "development_seed" if path == "protocol" else "seed"
    payload[path][key] = 9999
    with pytest.raises(RuntimeError, match="seed schema mismatch"):
        repair2.validate_source_result_seed(payload)


def test_validate_source_result_seed_rejects_missing_summary() -> None:
    payload = _result()
    del payload["protocol"]
    with pytest.raises(RuntimeError, match="protocol"):
        repair2.validate_source_result_seed(payload)


def test_repair2_reuses_repair1_diagnostic_implementation_exactly() -> None:
    assert repair2.run_checkpoint_triage_repair1 is repair1.run_checkpoint_triage_repair1
    source = inspect.getsource(repair2)
    assert "def run_checkpoint_triage_repair1" not in source
    assert repair2.LOSS_TIME_SLICE == 32


def test_repair2_protocol_is_preflight_schema_only() -> None:
    p = repair2.repair2_protocol()
    assert p["research_issue"] == 369
    assert p["repair1_issue"] == 372
    assert p["repair2_issue"] == 377
    assert p["source_failed_trigger"] == 376
    assert p["source_checkpoint_seed"] == 8471
    assert p["diagnostic_sampling_seed"] == 138471
    assert p["diagnostic_implementation"] == "run_checkpoint_triage_repair1_unmodified"
    assert p["semantic_change"] == "source_result_seed_schema_preflight_only"
    assert p["scientific_protocol_changed"] is False
    assert p["training_performed"] is False


def test_repair2_workflow_requires_both_failed_predecessors_and_unique_trigger() -> None:
    workflow = Path(".github/workflows/aera-v25-post8471-triage-repair2.yml").read_text()
    assert "[aera-v25-post8471-triage-repair2]" in workflow
    assert "issue view 371" in workflow
    assert "issue view 376" in workflow
    assert "issue view 377" in workflow
    assert "github.run_attempt != 1" in workflow
    assert "AERA_V25_POST8471_TRIAGE_REPAIR2_RESULT_JSON=" in workflow


def test_repair2_launcher_checks_all_seed_paths_and_reuses_repair1() -> None:
    launcher = Path("modal_aera_v25_post8471_triage_repair2_app.py").read_text()
    assert 'a_payload.get("seed") != 8471' in launcher
    assert 't_payload.get("seed") != 8471' in launcher
    assert "validate_source_result_seed(source_result)" in launcher
    assert "run_checkpoint_triage_repair1(" in launcher
    assert "v25-post8471-issue369-repair2/result.json" in launcher
    assert "MAX_GPU_SECONDS = 900" in launcher
