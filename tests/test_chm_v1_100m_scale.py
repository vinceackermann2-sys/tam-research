from __future__ import annotations

import ast
import inspect

import pytest

import tam_research.chm_v1_100m_scale as scale100


def test_100m_frozen_configuration_and_budgets() -> None:
    cfg = scale100.transformer_100m_config()
    assert cfg.vocab_size == 50_257
    assert cfg.d_model == 512
    assert cfg.n_layers == 24
    assert cfg.n_heads == 8
    assert cfg.max_seq_len == 1_024
    assert cfg.ff_mult == 4
    assert cfg.architecture == "transformer"
    assert scale100.HEAD_DIM == 64
    assert scale100.LOCAL_WINDOW == 512
    assert scale100.ADDRESS_DIM == 32
    assert scale100.FIRST_SCREEN_TOKEN_BUDGET == 4 * scale100.PRIOR_25M_TOKEN_BUDGET
    assert scale100.FIRST_SCREEN_TOKEN_BUDGET == 33_554_432
    assert scale100.CONFIRMATORY_CANDIDATE_TOKEN_BUDGET == 268_435_456


def test_100m_analytical_parameter_match_is_frozen() -> None:
    accounting = scale100.analytical_parameter_accounting()
    assert accounting["local_trainable_parameters"] == 101_803_520
    assert accounting["eiem_trainable_parameters"] == 101_836_800
    assert accounting["eiem_extra_parameters"] == 33_280
    assert accounting["within_point_one_percent"] is True
    assert float(accounting["delta_fraction"]) < 0.001


def test_100m_stage_a_seed_guards() -> None:
    assert scale100.validate_engineering_seed(scale100.ENGINEERING_SMOKE_SEED) == 977_099
    with pytest.raises(RuntimeError, match="reserved and unauthorized"):
        scale100.validate_engineering_seed(scale100.FUTURE_SCREEN_SEED)
    for seed in scale100.BLOCKED_PRIOR_SEEDS:
        with pytest.raises(RuntimeError, match="prior/blocked"):
            scale100.validate_engineering_seed(seed)
    with pytest.raises(RuntimeError, match="accepts only engineering smoke seed"):
        scale100.validate_engineering_seed(977_098)


def test_100m_stage_a_instantiated_pair_matches_backbone_and_parameter_gate() -> None:
    # This is the sole heavyweight Stage-A smoke: actual CPU instantiation verifies
    # that the analytical count matches the real PyTorch module graph and that
    # paired backbone initialization is bit-identical. It performs no training.
    result = scale100.stage_a_preflight()
    assert result["classification"] == "CHM_V1_100M_STAGE_A_ENGINEERING_PREFLIGHT_PASS"
    assert result["gpu_authorized"] is False
    assert result["training_authorized"] is False
    assert result["scientific_execution_authorized"] is False
    accounting = result["parameter_accounting"]
    assert accounting["local_trainable_parameters"] == scale100.EXPECTED_LOCAL_PARAMETERS
    assert accounting["eiem_trainable_parameters"] == scale100.EXPECTED_EIEM_PARAMETERS
    assert accounting["eiem_extra_parameters"] == scale100.EXPECTED_EIEM_EXTRA_PARAMETERS
    assert accounting["within_point_one_percent"] is True


def _dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def test_100m_stage_a_has_no_training_gpu_modal_or_corpus_execution_path() -> None:
    source = inspect.getsource(scale100)
    tree = ast.parse(source)
    imported_modules: set[str] = set()
    calls: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module)
        elif isinstance(node, ast.Call):
            name = _dotted_name(node.func)
            if name:
                calls.add(name)

    assert all("modal" not in name.lower() for name in imported_modules)
    assert "tam_research.data" not in imported_modules
    assert ".data" not in imported_modules
    assert "tam_research.train" not in imported_modules
    assert ".train" not in imported_modules
    assert all(not name.startswith("torch.cuda") for name in calls)
    assert all("optimizer" not in name.lower() for name in calls)
    assert all(not name.endswith(".backward") and name != "backward" for name in calls)
    assert all("train_one" not in name for name in calls)
