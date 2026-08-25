from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

from tam_research import aera_real_language_v25 as v25
from tam_research.aera_v25_post8471_triage import (
    ADAPTIVITY_BATCHES,
    BOOTSTRAP_RESAMPLES,
    DIAGNOSTIC_SEED,
    MEMORY_BATCHES,
    READ_ALPHAS,
    SOURCE_SEED,
    SYSTEM_ROUNDS,
    SYSTEM_TIMED_CALLS_PER_ROUND,
    SYSTEM_WARMUP_CALLS,
    adaptivity_summary,
    bootstrap_group_mean_difference,
    bootstrap_mean_ci,
    frozen_protocol,
    paired_difference_stats,
    parameter_versions,
    scaled_ficem_reads,
    systems_condition_names,
)


def test_frozen_issue369_protocol_constants() -> None:
    protocol = frozen_protocol()
    assert protocol["research_issue"] == 369
    assert protocol["source_result_issue"] == 368
    assert protocol["source_checkpoint_seed"] == SOURCE_SEED == 8471
    assert protocol["diagnostic_sampling_seed"] == DIAGNOSTIC_SEED == 138_471
    assert protocol["checkpoint_only"] is True
    assert protocol["training_performed"] is False
    assert MEMORY_BATCHES == 64
    assert ADAPTIVITY_BATCHES == 256
    assert READ_ALPHAS == (0.0, 0.5, 1.0, 1.5, 2.0)
    assert BOOTSTRAP_RESAMPLES == 2_000
    assert SYSTEM_WARMUP_CALLS >= 3
    assert SYSTEM_TIMED_CALLS_PER_ROUND == 20
    assert SYSTEM_ROUNDS == 5
    assert systems_condition_names() == (
        "transformer",
        "aera_core_memory_bypassed",
        "aera_read_path_empty_state",
        "aera_writes_only_reads_bypassed",
        "aera_full_memory",
    )
    assert protocol["claims"]["100m_authorized"] is False
    assert protocol["claims"]["breakthrough_proven"] is False


def test_bootstrap_and_paired_statistics_are_deterministic() -> None:
    values = torch.tensor([-0.10, -0.05, 0.0, 0.05, 0.10], dtype=torch.float64)
    first = bootstrap_mean_ci(values, seed=123)
    second = bootstrap_mean_ci(values, seed=123)
    assert first == second
    assert first["ci95_low"] <= first["mean"] <= first["ci95_high"]

    production = torch.tensor([1.0, 2.0, 3.0, 4.0])
    control = production + torch.tensor([0.2, 0.1, 0.3, 0.4])
    stats1 = paired_difference_stats(production, control, seed=456)
    stats2 = paired_difference_stats(production, control, seed=456)
    assert stats1 == stats2
    assert stats1["production_minus_control_mean"] == pytest.approx(-0.25)
    assert stats1["control_minus_production_advantage_mean"] == pytest.approx(0.25)


def test_adaptivity_deciles_and_tail_bootstrap_are_deterministic() -> None:
    difficulty = torch.arange(100, dtype=torch.float32)
    compute = torch.cat(
        (
            torch.zeros(25),
            torch.ones(25),
            torch.full((25,), 2.0),
            torch.full((25,), 1.0),
        )
    )
    positions = torch.arange(100) % 2
    stage_indicators = torch.stack(
        (
            (compute >= 1).float(),
            (compute >= 2).float(),
            (torch.arange(100) % 5 == 0).float(),
        ),
        dim=1,
    )
    first = adaptivity_summary(
        difficulty,
        compute,
        positions,
        stage_indicators,
        bootstrap_seed=777,
    )
    second = adaptivity_summary(
        difficulty,
        compute,
        positions,
        stage_indicators,
        bootstrap_seed=777,
    )
    assert first == second
    assert len(first["decile_compute_means_easy_to_hard"]) == 10
    assert len(first["optional_stage_run_fraction_by_difficulty_decile"]) == 10
    assert first["quartile_compute_means_easy_to_hard"] == pytest.approx(
        [0.0, 1.0, 2.0, 1.0]
    )
    assert first["quartile_monotonic_with_original_0_05_tolerance"] is False
    assert first["q4_minus_q3_compute"] == pytest.approx(-1.0)
    assert first["q4_minus_q3_bootstrap"]["ci95_high"] < 0.0

    tail = bootstrap_group_mean_difference(
        torch.full((20,), 2.0),
        torch.full((20,), 1.0),
        seed=991,
    )
    assert tail["right_minus_left_observed"] == pytest.approx(-1.0)
    assert tail["ci95_high"] < 0.0


def test_read_scaling_restores_exact_instance_method_state_and_parameter_versions() -> None:
    torch.manual_seed(12_345)
    model = v25.build_aera(torch.device("cpu")).eval()
    versions_before = parameter_versions(model)
    memories = [stage.memory for stage in model.stages]
    assert all("read" not in memory.__dict__ for memory in memories)
    with scaled_ficem_reads(model, (0.0, 0.5, 1.0, 2.0)):
        assert all("read" in memory.__dict__ for memory in memories)
        assert parameter_versions(model) == versions_before
    assert all("read" not in memory.__dict__ for memory in memories)
    assert parameter_versions(model) == versions_before

    with pytest.raises(ValueError):
        with scaled_ficem_reads(model, -1.0):
            pass


def test_checkpoint_triage_source_has_no_mutating_learning_api_calls() -> None:
    root = Path(__file__).resolve().parents[1]
    source_path = root / "tam_research" / "aera_v25_post8471_triage.py"
    source = source_path.read_text()
    tree = ast.parse(source)

    forbidden_call_attributes = {
        "backward",
        "step",
        "zero_grad",
        "train",
        "save",
        "save_checkpoint",
    }
    observed: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in forbidden_call_attributes:
            observed.append(func.attr)
        if isinstance(func, ast.Name) and func.id in {"Adam", "AdamW", "SGD"}:
            observed.append(func.id)
    assert observed == []
    assert "torch.optim" not in source
    assert "train_matched_pair" not in source
    assert "resume_from" not in source
    assert "write_text(" not in source


def test_issue369_modal_and_workflow_contract_are_checkpoint_only() -> None:
    root = Path(__file__).resolve().parents[1]
    modal_source = (root / "modal_aera_v25_post8471_triage_app.py").read_text()
    workflow = (
        root / ".github" / "workflows" / "aera-v25-post8471-triage.yml"
    ).read_text()

    assert "MAX_GPU_SECONDS = 900" in modal_source
    assert 'SOURCE_RUN_DIR = "/vol/aera-real-language/v25-dev-seed8471"' in modal_source
    assert 'RESULT_PATH = "/vol/aera-real-language/v25-post8471-issue369/result.json"' in modal_source
    assert "run_checkpoint_triage" in modal_source
    assert "torch.optim" not in modal_source
    assert ".backward(" not in modal_source
    assert "torch.save" not in modal_source

    assert "startsWith(github.event.issue.title, '[aera-v25-post8471-triage]')" in workflow
    assert "run-attempt" in workflow
    assert "github.run_attempt" in workflow
    assert "Refuse duplicate checkpoint triage" in workflow
    assert "Record issue369 hard guard" in workflow
    assert "AERA_V25_POST8471_TRIAGE_RESULT_JSON=" in workflow
    assert "No automatic retry" in workflow
