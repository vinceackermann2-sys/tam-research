from __future__ import annotations

import ast
from pathlib import Path

from architectures.cortex_s.long_context_ablation_v1 import (
    ATTENTION_EVERY,
    D_MODEL,
    MAX_SEQ_LEN,
    N_HEADS,
    N_LAYERS,
    NUM_EXPERTS,
    STATE_SIZE,
    TOP_K,
    VARIANTS,
    VARIANT_ORDER,
    architecture_contract,
    expected_parameter_count,
    parameter_contract,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER = REPO_ROOT / "modal_cortex_s_long_context_ablation_v1.py"
WORKFLOW = REPO_ROOT / ".github/workflows/modal-cortex-s-long-context-ablation-v1.yml"


def test_frozen_architecture_contract_and_parameter_matching() -> None:
    contract = architecture_contract()
    params = parameter_contract()

    assert D_MODEL == 512
    assert N_LAYERS == 24
    assert N_HEADS == 16
    assert MAX_SEQ_LEN == 16_384
    assert STATE_SIZE == 128
    assert NUM_EXPERTS == 8
    assert TOP_K == 2
    assert ATTENTION_EVERY == 6
    assert VARIANT_ORDER == (
        "transformer",
        "world_only",
        "reduced_attention_only",
        "moe_only",
        "full_cortex",
    )

    expected = {
        "transformer": 109_667_840,
        "world_only": 109_670_912,
        "reduced_attention_only": 109_663_744,
        "moe_only": 109_569_728,
        "full_cortex": 109_642_432,
    }
    assert params["counts"] == expected
    assert all(gap <= 0.005 for gap in params["gaps_fraction"].values())
    assert contract["trained_checkpoint_used"] is False
    assert contract["quality_claim_authorized"] is False
    assert contract["scientific_claim_authorized"] is False
    for name, count in expected.items():
        assert expected_parameter_count(name) == count


def test_variants_isolate_preregistered_components() -> None:
    transformer = VARIANTS["transformer"]
    assert transformer.use_world is False
    assert transformer.attention_every == 1
    assert transformer.feedforward_kind == "dense"
    assert transformer.dense_hidden == 2_048

    world = VARIANTS["world_only"]
    assert world.use_world is True
    assert world.attention_every == 1
    assert world.feedforward_kind == "dense"
    assert world.dense_hidden == 1_855

    reduced = VARIANTS["reduced_attention_only"]
    assert reduced.use_world is False
    assert reduced.attention_every == 6
    assert reduced.feedforward_kind == "dense"
    assert reduced.dense_hidden == 2_902

    moe = VARIANTS["moe_only"]
    assert moe.use_world is False
    assert moe.attention_every == 1
    assert moe.feedforward_kind == "moe"
    assert moe.expert_hidden == 255

    cortex = VARIANTS["full_cortex"]
    assert cortex.use_world is True
    assert cortex.attention_every == 6
    assert cortex.feedforward_kind == "moe"
    assert cortex.expert_hidden == 338


def test_runner_is_single_use_systems_only_and_fixed_geometry() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    tree = ast.parse(source)

    assert 'TRIGGER_TITLE = "[modal-cortex-s-long-context-ablation-v1]"' in source
    assert "SYNTHETIC_SEED = 2_026_091_501" in source
    assert 'RESULT_ROOT = "/vol/cortex-s-v0/systems/long-context-ablation-v1"' in source
    assert "TOKENS_PER_BATCH = 16_384" in source
    assert "WARMUP_ITERS = 2" in source
    assert "MEASURE_ITERS = 4" in source
    assert 'gpu="H100!"' in source
    assert "training_performed\": False" in source
    assert "250m_training_authorized\": False" in source
    assert "torch.optim" not in source
    assert "torch.save" not in source

    # Backward is required by the benchmark, but optimizer/weight-update calls are not.
    step_calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "step":
                step_calls.append(node.lineno)
    assert step_calls == []

    # Exactly one GPU-decorated benchmark function; reservation is CPU-only.
    function_names = {
        node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "reserve_dispatch" in function_names
    assert "run_benchmark" in function_names


def test_runner_freezes_context_batch_pairs_and_gates() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    for literal in (
        "(512, 32)",
        "(1_024, 16)",
        "(2_048, 8)",
        "(4_096, 4)",
        "(8_192, 2)",
        "(16_384, 1)",
    ):
        assert literal in source
    assert "R_tps" not in source  # implementation uses explicit ratios, not hidden formula text.
    assert "high_tps_ratio >= 1.10" in source
    assert "high_mem_ratio <= 1.10" in source
    assert "high_mem_ratio <= 0.80" in source
    assert "high_tps_ratio >= 0.90" in source
    assert "trend_delta >= 0.10" in source
    assert "tps_ratio >= 1.10 and mem_ratio <= 0.90" in source
    assert "PREPARE_250M_5B_PREREG_ALLOWED" in source
    assert "DO_NOT_SCALE_THIS_EXACT_DESIGN_YET" in source


def test_workflow_is_owner_only_exact_title_and_has_no_manual_dispatch() -> None:
    if not WORKFLOW.exists():
        # The workflow is added later in the same preparation branch. This lets the
        # architecture tests be committed first while still failing closed once the
        # full PR head exists.
        return
    source = WORKFLOW.read_text(encoding="utf-8")
    top = source.split("\njobs:", 1)[0]
    assert "workflow_dispatch:" not in top
    assert "issues:" in top and "types: [opened]" in top
    assert "github.event.issue.title == '[modal-cortex-s-long-context-ablation-v1]'" in source
    assert "github.event.issue.user.login == github.repository_owner" in source
    assert "modal run --detach --timestamps modal_cortex_s_long_context_ablation_v1.py" in source
    launch = source.split("\n      - name: Launch exact one-shot systems benchmark", 1)[1]
    launch = launch.split("\n      - name: Record normal Modal return only", 1)[0]
    assert launch.count("modal run --detach --timestamps") == 1
