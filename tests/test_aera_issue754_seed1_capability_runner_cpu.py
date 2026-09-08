from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from tam_research import aera_issue748_memory_capability_seed1_harness as harness


RUNNER = Path("modal_aera_issue754_seed1_capability_runner.py")
WORKFLOW = Path(".github/workflows/aera-issue754-seed1-capability-runner.yml")
PROTOCOL = Path("docs/aera_issue748_seed1_harness_protocol.json")

SOURCE_MAIN = "bb4c01c71deed63c94b89d91c9a713fa2a0f3b34"
SOURCE_TREE = "3caa58461f9d0513c8564210b2e8179897abbfc4"
PREAUTH_PREFIX = "[aera-issue748-memory-capability-seed1-preauth]"
L4_PREFIX = "[aera-issue748-memory-capability-seed1-l4]"
RESULT_PATH = "/vol/aera-capability/issue748-memory-capability-seed1/result.json"
CHECKPOINT_DIR = "/vol/aera-capability/issue748-memory-capability-seed1/checkpoints"
MODEL_SEED = 17641


def _function_source(source: str, name: str) -> str:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            segment = ast.get_source_segment(source, node)
            assert segment is not None
            return segment
    raise AssertionError(f"missing function {name}")


def test_frozen_harness_protocol_and_scientific_seed_guard() -> None:
    protocol = json.loads(PROTOCOL.read_text())
    assert harness.harness_protocol_snapshot() == protocol
    future = protocol["future_execution"]
    assert future == {
        "checkpoint_dir": CHECKPOINT_DIR,
        "gpu_prefix": L4_PREFIX,
        "preauth_prefix": PREAUTH_PREFIX,
        "result_path": RESULT_PATH,
        "seed1_model_seed": MODEL_SEED,
    }
    assert tuple(harness.SCIENTIFIC_MODEL_SEEDS) == (17641, 27641, 37641)
    assert harness.SEED1_MODEL_SEED == MODEL_SEED
    assert all(value is False for value in protocol["authority"].values())

    with pytest.raises(
        PermissionError,
        match="scientific model seed requires separate explicit authorization",
    ):
        harness.build_model(
            "A_backbone",
            MODEL_SEED,
            scientific_seed_authorized=False,
            device="cpu",
        )


def test_runner_is_additive_frozen_source_orchestration_only() -> None:
    source = RUNNER.read_text()
    ast.parse(source)

    for literal in (
        'RUNNER_ISSUE = 754',
        'RESEARCH_ISSUE = 748',
        f'SOURCE_MAIN = "{SOURCE_MAIN}"',
        f'SOURCE_TREE = "{SOURCE_TREE}"',
        'HARNESS_WRAPPER_BLOB = "afc939a69633f68ded05eb585c95a599a8f5c981"',
        'HARNESS_BASE_BLOB = "40003b68987d026265b1d53fcb637f18277f9cac"',
        'PROTOCOL_BLOB = "8bd8d646a72c0d5a2b1cbe7bedddc25bff6b43bf"',
        'CPU_TEST_BLOB = "5ad56721532353a1b9d71fac4a470b4b1e1fa704"',
        f'PREAUTH_PREFIX = "{PREAUTH_PREFIX}"',
        f'L4_PREFIX = "{L4_PREFIX}"',
        f'RESULT_PATH = "{RESULT_PATH}"',
        f'CHECKPOINT_DIR = "{CHECKPOINT_DIR}"',
        "MODEL_SEED = 17641",
        'VOLUME_NAME = "tam-research-data"',
        "create_if_missing=False",
    ):
        assert literal in source

    assert "27641" not in source
    assert "37641" not in source
    assert source.count('gpu="L4"') == 1

    preflight = _function_source(source, "preflight")
    assert "volume.reload()" in preflight
    assert "_source_evidence()" in preflight
    assert "_checkpoint_absent_or_empty()" in preflight
    assert "gpu=" not in preflight
    assert "build_model" not in preflight
    assert "train_variant" not in preflight
    assert "evaluate_model" not in preflight
    assert "scientific_seed_authorized=True" not in preflight
    assert "volume.commit" not in preflight
    assert "_write_result" not in preflight

    scientific = _function_source(source, "run_seed1")
    assert "torch.cuda.is_available()" in scientific
    assert "harness.evaluation_cases()" in scientific
    assert "harness.train_variant(" in scientific
    assert "scientific_seed_authorized=True" in scientific
    assert "harness.evaluate_model(" in scientific
    assert "harness.select_equal_gpu_time_checkpoints(" in scientific
    assert "harness.load_model_checkpoint(" in scientific
    assert "harness.evaluate_simple_retrieval(" in scientific
    assert "harness.seed1_decision(" in scientific
    assert "harness.validate_seed1_result_schema(result)" in scientific
    assert "_write_result(result)" in scientific

    preauth_main = _function_source(source, "preauth_main")
    assert "preflight.remote()" in preauth_main
    assert "run_seed1.remote()" not in preauth_main

    l4_main = _function_source(source, "l4_main")
    assert "preflight.remote()" in l4_main
    assert "run_seed1.remote()" in l4_main


def test_workflow_requires_unique_attempt1_bound_preauth_and_separate_l4_authorization() -> None:
    source = WORKFLOW.read_text()

    assert "issues:" in source
    assert "types: [opened]" in source
    assert "workflow_dispatch" not in source
    assert "pull_request:" not in source
    assert 'test "${GITHUB_RUN_ATTEMPT}" = "1"' in source

    assert PREAUTH_PREFIX in source
    assert L4_PREFIX in source
    assert "Bind main:" in source
    assert "git checkout --detach" in source
    assert "git merge-base --is-ancestor" in source
    assert "mapfile -t matches" in source
    assert 'test "${#matches[@]}" = "1"' in source
    assert 'test "${TRIGGER_ISSUE}" = "${matches[0]}"' in source

    assert "## #754 pre-implementation seed1 Modal runner freeze" in source
    assert "## #754 sole seed1 L4 scientific authorization" in source
    assert 'contains("Authorize main: `" + $bound + "`")' in source
    assert 'contains("Authorize seed: `17641`")' in source
    assert "🔎 **AERA #754/#748 seed1 preauthorization evidence**" in source
    assert "🔬 **AERA #754/#748 seed1 scientific evidence**" in source

    assert source.count(
        "modal run modal_aera_issue754_seed1_capability_runner.py::preauth_main"
    ) == 1
    assert source.count(
        "modal run modal_aera_issue754_seed1_capability_runner.py::l4_main"
    ) == 1

    assert "result_absent" in source
    assert "checkpoint_absent_or_empty" in source
    assert "Seed `17641` remains unconsumed." in source
    assert "This evidence authorizes no L4 run" in source


def test_runner_never_auto_authorizes_followup_stages() -> None:
    source = RUNNER.read_text()
    for forbidden_true in (
        '"seeds_2_3_authorized": True',
        '"systems_optimization_authorized": True',
        '"architecture_freeze_authorized": True',
        '"s2_authorized": True',
        '"replication_authorized": True',
        '"scaling_authorized": True',
        '"breakthrough_proven": True',
    ):
        assert forbidden_true not in source

    assert '"seeds_2_3_authorized": False' in source
    assert '"systems_optimization_authorized": False' in source
    assert '"architecture_freeze_authorized": False' in source
    assert '"scaling_authorized": False' in source
    assert '"breakthrough_proven": False' in source
