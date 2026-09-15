from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

import tam_research.chm_v1_100m_stage_b as stage_b


def test_stage_b_static_freeze_and_seed_guards() -> None:
    p = stage_b.static_preflight()
    assert p["classification"] == "CHM_V1_100M_STAGE_B_STATIC_PREFLIGHT_PASS"
    assert p["systems_seed"] == 981_001
    assert p["gpu"] == "L4"
    assert p["cpu_cores"] == 4
    assert p["ram_gib"] == 16
    assert p["max_gpu_seconds"] == 900
    assert p["max_stage_b_cost_usd"] == 0.35
    assert p["batch_size"] == 2
    assert p["session_len"] == 1024
    assert p["chunk_len"] == 512
    assert p["warmup_steps"] == 1
    assert p["measured_steps"] == 20
    assert p["scientific_execution_authorized"] is False

    assert stage_b.validate_systems_seed(981_001, protocol=True) == 981_001
    assert stage_b.validate_systems_seed(981_099) == 981_099
    for seed in stage_b.BLOCKED_SEEDS:
        with pytest.raises(RuntimeError, match="blocked/scientific seed refused"):
            stage_b.validate_systems_seed(seed)
    with pytest.raises(RuntimeError, match="protocol accepts only"):
        stage_b.validate_systems_seed(981_099, protocol=True)


def _result(*, tps: float = 3000.0, vram: int = 10 * 1024**3, hours: float = 5.0, cost: float = 5.0, parity: bool = True):
    train = {
        "measured_steps": 20,
        "tokens_per_second": tps,
        "peak_allocated_vram_bytes": vram,
        "loss_finite": True,
        "gradients_finite": True,
        "parameters_finite": True,
    }
    return {
        "parameter_accounting": {
            "local_trainable_parameters": stage_b.EXPECTED_LOCAL_PARAMETERS,
            "eiem_trainable_parameters": stage_b.EXPECTED_EIEM_PARAMETERS,
            "delta_fraction": 0.0003269,
        },
        "training": {"local": dict(train), "eiem": dict(train)},
        "inference": {"logits_within_tolerance": parity},
        "projection": {
            "combined_projected_seconds": hours * 3600.0,
            "combined_projected_cost_usd": cost,
        },
    }


def test_stage_b_frozen_classification_logic() -> None:
    assert stage_b.classify_stage_b(_result()) == "CHM_V1_100M_STAGE_B_SYSTEMS_PASS"

    slow = _result(tps=1999.0)
    assert stage_b.classify_stage_b(slow) == "CHM_V1_100M_STAGE_B_SYSTEMS_STOP"

    expensive = _result(cost=12.01)
    assert stage_b.classify_stage_b(expensive) == "CHM_V1_100M_STAGE_B_SYSTEMS_STOP"

    bad = _result(parity=False)
    assert stage_b.classify_stage_b(bad) == "CHM_V1_100M_STAGE_B_CORRECTNESS_FAIL"


def test_stage_b_module_has_no_corpus_or_modal_path() -> None:
    source = inspect.getsource(stage_b)
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "modal" not in imported
    assert "TokenBin" not in source
    assert "FineWeb" not in source
    assert "train.bin" not in source
    assert "val.bin" not in source


def test_stage_b_launcher_and_workflow_are_one_shot_systems_only() -> None:
    launcher = Path("modal_chm_v1_100m_stage_b_981_v1.py").read_text(encoding="utf-8")
    workflow = Path(".github/workflows/modal-chm-v1-100m-stage-b-981-v1.yml").read_text(encoding="utf-8")

    assert "gpu=GPU_CLASS" in launcher
    assert "retries=0" in launcher
    assert "SYSTEMS_SEED_CONSUMED.json" in launcher
    assert "ATTEMPT_FAILURE.json" in launcher
    assert "TokenBin" not in launcher
    assert "FineWeb" not in launcher

    assert "github.run_attempt" in workflow
    assert "#981 refuses workflow reruns" in workflow
    assert "systems_seed\": 981001" in workflow
    assert "modal billing rates --json" in workflow
    assert "bound <= 0.35" in workflow
    assert "origin/main" in workflow
    assert "automatic_retry_authorized" in workflow
    assert "rerun/retry/rescue/redispatch" in workflow
