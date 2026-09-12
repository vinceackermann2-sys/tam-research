from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / "modal_cortex_s_v11_vs_transformer_pretrain_benchmark_v1.py"
WORKFLOW = ROOT / ".github/workflows/modal-cortex-s-v11-vs-transformer-pretrain-benchmark-v1.yml"


def _source() -> str:
    return RUNNER.read_text(encoding="utf-8")


def _tree() -> ast.Module:
    return ast.parse(_source())


def _call_name(call: ast.Call) -> str:
    parts: list[str] = []
    node = call.func
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _calls() -> list[str]:
    return [_call_name(node) for node in ast.walk(_tree()) if isinstance(node, ast.Call)]


def test_runner_is_syntax_valid_and_exact_eval_identity_is_frozen():
    source = _source()
    compile(source, str(RUNNER), "exec")
    assert 'TRIGGER_TITLE = "[modal-cortex-s-v11-vs-transformer-pretrain-benchmark-v1]"' in source
    assert "EVAL_SEED = 2_026_091_201" in source
    assert 'RESULT_ROOT = "/vol/cortex-s-v0/evals/cortex-s-v11-vs-transformer-pretrain-benchmark-v1"' in source
    assert 'CORTEX_CHECKPOINT = "/vol/cortex-s-v0/100m-2b/full-v11-compiled-explicit-fp32-ce/latest.pt"' in source
    assert 'TRANSFORMER_CHECKPOINT = "/vol/full100m-runs/transformer-pretrain/100m/ctx512-mb64-ga2/transformer-25m-compiled-seed8100/latest.pt"' in source
    assert "EXPECTED_CORTEX_PARAMS = 101_778_112" in source
    assert "EXPECTED_TRANSFORMER_PARAMS = 101_803_520" in source
    assert "EXPECTED_TOKENS = 2_000_000_000" in source
    assert "EXPECTED_STEPS = 30_518" in source


def test_evaluation_seed_is_not_any_training_or_reserved_scientific_seed():
    evaluation_seed = 2_026_091_201
    forbidden = {8_100, 48_131, 48_132, 48_133, 2_026_091_014}
    assert evaluation_seed not in forbidden


def test_runner_has_exactly_one_l4_allocation_and_bounded_timeout():
    gpu_decorators: list[tuple[object, object]] = []
    for node in ast.walk(_tree()):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            kwargs = {kw.arg: kw.value for kw in decorator.keywords if kw.arg}
            if "gpu" in kwargs:
                gpu_decorators.append((kwargs["gpu"], kwargs.get("timeout")))
    assert len(gpu_decorators) == 1
    gpu, timeout = gpu_decorators[0]
    assert isinstance(gpu, ast.Constant) and gpu.value == "L4"
    assert isinstance(timeout, ast.Constant) and timeout.value == 7200


def test_runner_has_no_training_backward_or_optimizer_execution_path():
    calls = _calls()
    forbidden_exact = {
        "train_language_model",
        "train_scaled_language_model",
        "train_full_2b",
        "optimizer.step",
    }
    assert not any(name in forbidden_exact for name in calls)
    assert not any(name.endswith(".backward") or name == "backward" for name in calls)
    assert not any(name.startswith("torch.optim") for name in calls)
    assert not any("train_sft" in name or "train_dpo" in name for name in calls)


def test_exact_checkpoint_builders_and_provenance_guards_are_used():
    source = _source()
    assert "build_memory_lean_grouped_triton_scan_cortex_100m" in source
    assert "load_checkpoint_model" in source
    assert 'cortex_ckpt.get("architecture") != "cortex_s"' in source
    assert 'transformer_ckpt.get("architecture") != "transformer"' in source
    assert "CORTEX_TRAINING_SEED" in source
    assert "TRANSFORMER_TRAINING_SEED" in source
    assert 'cortex.load_state_dict(cortex_ckpt["model"], strict=True)' in source


def test_dispatch_is_durable_before_gpu_remote_call_and_result_is_single_use():
    source = _source()
    reserve_call = source.index("reserve_evaluation_dispatch.remote(")
    gpu_call = source.index("run_benchmark.remote(")
    assert reserve_call < gpu_call
    assert '"status": "EVAL_DISPATCH_CONSUMED"' in source
    assert "volume.commit()" in source
    assert 'if result_path.exists():' in source
    assert '"retry_authorized": False' in source


def test_both_models_use_identical_samples_tokenizer_and_scoring_contract():
    source = _source()
    assert 'AutoTokenizer.from_pretrained("gpt2", use_fast=True)' in source
    assert "ds.shuffle(seed=EVAL_SEED)" in source
    assert "gsm.shuffle(seed=EVAL_SEED)" in source
    assert "mean continuation log-likelihood on raw pretraining prompt" in source
    assert '"same_sample_indices": True' in source
    assert '"same_evaluation_seed": True' in source
    assert '"weights_updated": False' in source
    assert '"optimizer_used": False' in source
    assert '"backward_used": False' in source


def test_workflow_is_owner_only_exact_title_source_bound_and_single_shot():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "workflow_dispatch" not in workflow
    assert "types: [opened]" in workflow
    assert "github.event.issue.user.login == github.repository_owner" in workflow
    assert "github.event.issue.title == '[modal-cortex-s-v11-vs-transformer-pretrain-benchmark-v1]'" in workflow
    assert 'test "$(git rev-parse origin/main)" = "$SOURCE_SHA"' in workflow
    assert 'test "$(git rev-parse HEAD^{tree})" = "$SOURCE_TREE"' in workflow
    assert 'git merge-base --is-ancestor "$HARNESS_SHA" "$SOURCE_SHA"' in workflow
    command = "modal run --timestamps modal_cortex_s_v11_vs_transformer_pretrain_benchmark_v1.py"
    assert workflow.count(command) == 1
    assert "--source-sha \"$SOURCE_SHA\"" in workflow
    assert "--source-tree \"$SOURCE_TREE\"" in workflow
    assert "--harness-sha \"$HARNESS_SHA\"" in workflow


def test_workflow_preflights_all_six_datasets_before_modal_run():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    preflight = workflow.index("Preflight benchmark datasets without GPU")
    modal_run = workflow.index("modal run --timestamps")
    assert preflight < modal_run
    for name in ("arc_easy", "arc_challenge", "piqa", "hellaswag", "openbookqa", "gsm8k"):
        assert name in workflow
