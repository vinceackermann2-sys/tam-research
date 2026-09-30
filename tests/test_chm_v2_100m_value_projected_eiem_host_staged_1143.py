from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tam_research.chm_v2_100m_value_projected_eiem_host_staged_1143 import (
    AUDIT_TITLE,
    CONTROL_ISSUE,
    PHASE,
    PREREG_ISSUE,
    RESULT_ROOT,
    SCIENTIFIC_SEED,
    TRIGGER_TITLE,
    assert_no_execution_authority,
    protocol_manifest,
    validate_contract,
    validate_seed_for_preparation,
)


ROOT = Path(__file__).resolve().parents[1]
OLD_RUNNER = ROOT / "modal_chm_v2_100m_value_projected_eiem_host_staged_1137_v1.py"
NEW_RUNNER = ROOT / "modal_chm_v2_100m_value_projected_eiem_host_staged_1143_v1.py"
OLD_CORE = ROOT / "tam_research" / "chm_v2_100m_value_projected_eiem_host_staged_rerun.py"
NEW_CORE = ROOT / "tam_research" / "chm_v2_100m_value_projected_eiem_host_staged_1143.py"

FRESH_SEED = 2_011_431
FRESH_ROOT = "/vol/chm-v2/100m-value-projected-eiem-host-staged/issue-1143/seed-2011431-v1"
FRESH_TRIGGER = "[modal-chm-v2-100m-value-projected-eiem-host-staged-1143-seed-2011431-v1]"
FRESH_AUDIT = "[modal-chm-v2-100m-value-projected-eiem-host-staged-1143-authority-audit-v1]"


def _function_source(source: str, name: str) -> str:
    tree = ast.parse(source)
    lines = source.splitlines()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            start = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
            end = node.end_lineno
            assert end is not None
            return "\n".join(lines[start:end])
    raise AssertionError(f"function {name!r} not found")


def _normalize_successor_source(source: str) -> str:
    return (
        source.replace("#1143", "#1137")
        .replace("2011431", "2011371")
        .replace("2_011_431", "2_011_371")
        .replace("issue-1143", "issue-1137")
        .replace(
            "chm_v2_100m_value_projected_eiem_host_staged_1143",
            "chm_v2_100m_value_projected_eiem_host_staged_rerun",
        )
    )


def test_successor_identity_is_fresh_and_has_no_execution_authority() -> None:
    manifest = validate_contract()
    assert CONTROL_ISSUE == 1143
    assert PREREG_ISSUE == 1137
    assert SCIENTIFIC_SEED == FRESH_SEED
    assert PHASE == "chm-v2-100m-value-projected-eiem-host-staged-1143-seed-2011431-v1"
    assert RESULT_ROOT == FRESH_ROOT
    assert TRIGGER_TITLE == FRESH_TRIGGER
    assert AUDIT_TITLE == FRESH_AUDIT
    assert manifest["scientific_seed_authorized"] is False
    assert manifest["trigger_authorized_by_module"] is False
    assert manifest["gpu_allocation_authorized_by_module"] is False
    assert manifest["scientific_seed_consumed_by_module"] is False
    assert manifest["stage_d_authorized"] is False
    assert manifest["scale_up_authorized"] is False
    assert manifest["multi_seed_replication_authorized"] is False
    assert protocol_manifest()["scientific_seed"] == FRESH_SEED


def test_consumed_scientific_seeds_are_explicitly_refused() -> None:
    assert validate_seed_for_preparation(FRESH_SEED, request_execution=False) == FRESH_SEED
    for consumed in (977_001, 2_011_121, 2_011_371):
        with pytest.raises(RuntimeError, match="consumed"):
            validate_seed_for_preparation(consumed, request_execution=False)
    with pytest.raises(RuntimeError, match="grants no scientific execution authority"):
        validate_seed_for_preparation(FRESH_SEED, request_execution=True)


def test_core_grants_no_gpu_training_trigger_or_seed_consumption_authority() -> None:
    assert_no_execution_authority()
    for kwargs in (
        {"gpu": True},
        {"training": True},
        {"trigger_creation": True},
        {"seed_consumption": True},
    ):
        with pytest.raises(RuntimeError):
            assert_no_execution_authority(**kwargs)


def test_new_files_are_syntax_valid_and_old_consumed_files_still_exist() -> None:
    old_runner = OLD_RUNNER.read_text(encoding="utf-8")
    new_runner = NEW_RUNNER.read_text(encoding="utf-8")
    old_core = OLD_CORE.read_text(encoding="utf-8")
    new_core = NEW_CORE.read_text(encoding="utf-8")
    ast.parse(old_runner)
    ast.parse(new_runner)
    ast.parse(old_core)
    ast.parse(new_core)

    assert "SCIENTIFIC_SEED = 2_011_371" in old_runner
    assert "SCIENTIFIC_SEED = 2_011_431" in new_runner
    assert "CONTROL_ISSUE = 1137" in old_core
    assert "CONTROL_ISSUE = 1143" in new_core


def test_training_function_is_semantically_identical_to_consumed_runner() -> None:
    old = _function_source(OLD_RUNNER.read_text(encoding="utf-8"), "_train_model")
    new = _function_source(NEW_RUNNER.read_text(encoding="utf-8"), "_train_model")
    assert _normalize_successor_source(new) == old


def test_validation_function_diff_is_exactly_the_missing_telemetry_return_field() -> None:
    old = _function_source(OLD_RUNNER.read_text(encoding="utf-8"), "_evaluate_language")
    new = _function_source(NEW_RUNNER.read_text(encoding="utf-8"), "_evaluate_language")
    required = '        "host_to_device_token_bytes_total": host_to_device_token_bytes_total,'
    assert required in new
    repaired_removed = new.replace(required + "\n", "")
    assert _normalize_successor_source(repaired_removed) == old

    tree = ast.parse(new)
    function = tree.body[0]
    assert isinstance(function, ast.FunctionDef)
    returns = [node for node in ast.walk(function) if isinstance(node, ast.Return)]
    assert len(returns) == 1
    payload = returns[0].value
    assert isinstance(payload, ast.Dict)
    keys = [key.value for key in payload.keys if isinstance(key, ast.Constant)]
    assert "host_to_device_token_bytes_total" in keys


def test_result_assembly_telemetry_keys_are_produced_by_validation_function() -> None:
    source = NEW_RUNNER.read_text(encoding="utf-8")
    validation = _function_source(source, "_evaluate_language")
    scientific = _function_source(source, "run_scientific")
    assert '"host_to_device_token_bytes_total": host_to_device_token_bytes_total' in validation
    for name in ("local_language", "raw_language", "vp_language"):
        assert f'{name}["host_to_device_token_bytes_total"]' in scientific


def test_scientific_function_has_no_nonidentity_semantic_drift() -> None:
    old = _function_source(OLD_RUNNER.read_text(encoding="utf-8"), "run_scientific")
    new = _function_source(NEW_RUNNER.read_text(encoding="utf-8"), "run_scientific")
    assert _normalize_successor_source(new) == old


def test_successor_runner_uses_fresh_core_and_retired_seed_only_as_provenance() -> None:
    source = NEW_RUNNER.read_text(encoding="utf-8")
    assert "tam_research.chm_v2_100m_value_projected_eiem_host_staged_1143" in source
    assert FRESH_ROOT in source
    assert FRESH_TRIGGER in source
    assert FRESH_AUDIT in source
    assert "SCIENTIFIC_SEED = 2_011_431" in source
    assert "seed-2011371-v1" not in source
    assert "[modal-chm-v2-100m-value-projected-eiem-host-staged-1137-seed-2011371-v1]" not in source


def test_repair_does_not_add_workflow_or_manual_execution_authority() -> None:
    # #1143's first implementation layer is intentionally runner/core/tests only.
    assert not (
        ROOT
        / ".github"
        / "workflows"
        / "modal-chm-v2-100m-value-projected-eiem-host-staged-1143-v1.yml"
    ).exists()
    assert not (
        ROOT
        / ".github"
        / "workflows"
        / "modal-chm-v2-100m-value-projected-eiem-host-staged-1143-authority-audit-v1.yml"
    ).exists()
