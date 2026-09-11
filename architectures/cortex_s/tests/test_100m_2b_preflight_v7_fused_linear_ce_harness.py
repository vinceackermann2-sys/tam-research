from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / "modal_cortex_s_100m_2b_v7_fused_linear_ce_preflight.py"
WORKFLOW = ROOT / ".github/workflows/modal-cortex-s-100m-2b-preflight-v7-fused-linear-ce.yml"
TRAIN_IMPL = ROOT / "architectures/cortex_s/experiments/scale100m_2b/train.py"
FUSED_IMPL = ROOT / "architectures/cortex_s/fused_linear_ce_v1.py"

EXPECTED_RUNNER_BLOB = "1816ea39d7c516826ebe47f27414a384d4d66ada"
EXPECTED_FUSED_BLOB = "3b1697b45e73c1a0e860c5ecdd4f8bc1ec1ae179"
EXPECTED_FUSED_TEST_BLOB = "8ba75d3404b78a383322473df4f3b622f4762e64"
EXPECTED_TRAIN_BLOB = "004b66b549d3d64f2dde7614ec84f22b9f37a7c6"
EXPECTED_TRAIN_SHA256 = "d0d91de09bba8b7631913b3e9c0407ab2bb0114f7bba133c123ee5598d490461"
EXPECTED_SCAN_INTEGRATION_BLOB = "c3d05f8b4d1e3e8ea7bcbad5fdeca5d7091651c7"
EXPECTED_TRITON_BLOB = "5c6bf7df95f49c306b4ffadd6a8a09b9f71344f9"
EXPECTED_GROUPED_MOE_BLOB = "7c738e6349536dd45f1c631a9ad5524ade1d5021"
EXPECTED_LIGER_WHEEL_SHA256 = "84c0a7bc9bf4d4cf8ea5ba89ff84d28686afc94215b220851d9f57dc87852741"


def _function_source(text: str, name: str, next_name: str | None = None) -> str:
    start = text.index(f"def {name}(")
    if next_name is None:
        return text[start:]
    return text[start:text.index(f"def {next_name}(", start + 1)]


def _function(tree: ast.AST, name: str) -> ast.FunctionDef:
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(matches) == 1, (name, len(matches))
    return matches[0]


def _call_name(call: ast.Call) -> str:
    return ast.unparse(call.func)


def _calls(node: ast.AST) -> list[ast.Call]:
    return [item for item in ast.walk(node) if isinstance(item, ast.Call)]


def _profiler_calls(tree: ast.AST) -> list[str]:
    return [
        _call_name(call)
        for call in _calls(tree)
        if _call_name(call).startswith("torch.profiler")
    ]


def _context_name(item: ast.withitem) -> str:
    expression = item.context_expr
    if isinstance(expression, ast.Call):
        return ast.unparse(expression.func)
    return ast.unparse(expression)


def test_v7_runner_is_syntax_valid_and_single_use_authority_is_frozen() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    tree = ast.parse(text)
    required = (
        'PHASE = "preflight-v7-fused-linear-ce"',
        'TRIGGER_TITLE = "[modal-cortex-s-100m-2b-preflight-v7-fused-linear-ce]"',
        "ENGINEERING_SEED = 2_026_091_009",
        'RESULT_ROOT = "/vol/cortex-s-v0/100m-2b/preflight-v7-fused-linear-ce"',
        'LIGER_KERNEL_VERSION = "0.8.2"',
        f'LIGER_KERNEL_WHEEL_SHA256 = "{EXPECTED_LIGER_WHEEL_SHA256}"',
        'FUSED_SYSTEMS_VARIANT = "fused_linear_ce_liger_v1"',
        f'TRAIN_IMPLEMENTATION_SHA256 = "{EXPECTED_TRAIN_SHA256}"',
        "EXPECTED_CORTEX_PARAMS = 101_778_112",
        "SEQ_LEN = 512",
        "MICRO_BATCH_SIZE = 64",
        "GRAD_ACCUM_STEPS = 2",
        "GLOBAL_BATCH = 128",
        "CALIBRATION_WARMUP_STEPS = 5",
        "MEASURED_STEPS = 40",
        "MEASURED_TOKENS = MEASURED_STEPS * TOKENS_PER_STEP",
        "TRAIN_TOKENS = 2_000_000_000",
        "PROJECTION_OVERHEAD_MULTIPLIER = 1.10",
        "MAX_PROJECTED_FULL_SECONDS = 8_500.0",
        "MAX_PREFLIGHT_VRAM_GIB = 70.0",
        "HARD_FULL_TIMEOUT_SECONDS = 10_000",
        '"liger-kernel==0.8.2"',
        "2_026_091_008",
        "FORBIDDEN_CONTROL_AND_SCIENTIFIC_SEEDS = (8_100, 48_131, 48_132, 48_133)",
    )
    for needle in required:
        assert needle in text

    gpu_sites: list[object] = []
    for call in _calls(tree):
        for keyword in call.keywords:
            if keyword.arg == "gpu" and isinstance(keyword.value, ast.Constant):
                gpu_sites.append(keyword.value.value)
    assert gpu_sites == ["H100!"]
    assert not any(_call_name(call).endswith("train_full_2b") for call in _calls(tree))
    assert _profiler_calls(tree) == []
    assert "ENGINEERING_SEED = 8_100" not in text
    assert "ENGINEERING_SEED = 48_131" not in text
    assert "ENGINEERING_SEED = 48_132" not in text
    assert "ENGINEERING_SEED = 48_133" not in text


def test_default_and_fused_training_steps_each_keep_one_bundled_host_readback() -> None:
    default_source = TRAIN_IMPL.read_text(encoding="utf-8")
    default_step = _function_source(default_source, "_one_optimizer_step", "evaluate")
    fused_source = FUSED_IMPL.read_text(encoding="utf-8")
    fused_step = _function_source(
        fused_source,
        "_one_optimizer_step_fused_linear_ce",
        "fused_linear_ce_training_builder",
    )

    for step in (default_step, fused_step):
        assert "running_loss: torch.Tensor | None = None" in step
        assert "all_finite: torch.Tensor | None = None" in step
        assert "float(loss.detach())" not in step
        assert "if not torch.isfinite(loss)" not in step
        assert "torch.logical_and(all_finite, torch.isfinite(grad_norm))" in step
        assert "host_report = torch.stack(" in step
        assert step.count('.to(device="cpu")') == 1
        assert "running_value, finite_value = host_report.tolist()" in step
        assert step.index("host_report = torch.stack(") < step.index("optimizer.step()")
        assert step.index("if not bool(finite_value):") < step.index("optimizer.step()")
    assert "runner(x, y) / training_module.GRAD_ACCUM_STEPS" in fused_step


def test_dispatch_marker_is_durable_before_h100_remote_call() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    tree = ast.parse(text)
    reserve = _function(tree, "reserve_h100_dispatch")
    h100 = _function(tree, "h100_preflight")
    main = _function(tree, "main")

    reserve_calls = [(_call_name(call), call.lineno) for call in _calls(reserve)]
    atomic_lines = [line for name, line in reserve_calls if name == "_atomic_write"]
    commit_lines = [line for name, line in reserve_calls if name == "volume.commit"]
    assert len(atomic_lines) == 1
    assert len(commit_lines) == 1
    assert atomic_lines[0] < commit_lines[0]

    main_calls = [(_call_name(call), call.lineno) for call in _calls(main)]
    reserve_remote = [line for name, line in main_calls if name == "reserve_h100_dispatch.remote"]
    h100_remote = [line for name, line in main_calls if name == "h100_preflight.remote"]
    assert len(reserve_remote) == 1 and len(h100_remote) == 1
    assert reserve_remote[0] < h100_remote[0]

    h100_text = _function_source(text, "h100_preflight", "main")
    assert 'marker_path = root / "H100_DISPATCH_CONSUMED.json"' in h100_text
    assert "if not marker_path.exists():" in h100_text
    assert 'marker["h100_allocation_started"] = True' in h100_text
    assert "_atomic_write(marker_path, marker)" in h100_text
    assert "volume.commit()" in h100_text


def test_h100_calibration_is_structurally_nested_under_scan_and_fused_builders() -> None:
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    h100 = _function(tree, "h100_preflight")
    scan_withs = [
        node
        for node in ast.walk(h100)
        if isinstance(node, ast.With)
        and any(_context_name(item) == "memory_lean_grouped_triton_scan_training_builder" for item in node.items)
    ]
    assert len(scan_withs) == 1
    scan_with = scan_withs[0]
    fused_withs = [
        node
        for node in ast.walk(scan_with)
        if isinstance(node, ast.With)
        and node is not scan_with
        and any(_context_name(item) == "fused_linear_ce_training_builder" for item in node.items)
    ]
    assert len(fused_withs) == 1
    fused_with = fused_withs[0]
    calibration_calls = [
        call
        for call in _calls(fused_with)
        if _call_name(call) == "training_module.run_h100_calibration"
    ]
    assert len(calibration_calls) == 1

    h100_text = _function_source(RUNNER.read_text(encoding="utf-8"), "h100_preflight", "main")
    assert "training_module.CALIBRATION_SEED = ENGINEERING_SEED" in h100_text
    assert "training_module.CALIBRATION_SEED = original_calibration_seed" in h100_text
    assert 'raw.get("execution") != "compiled"' in h100_text
    assert 'raw.get("measured_steps") != MEASURED_STEPS' in h100_text
    assert 'raw.get("measured_tokens") != MEASURED_TOKENS' in h100_text
    assert "TRAIN_TOKENS / tps * PROJECTION_OVERHEAD_MULTIPLIER + compile_seconds" in h100_text
    assert '"peak_vram_le_70_gib"' in h100_text
    assert '"projected_full_seconds_le_budget"' in h100_text
    assert '"profiling_inside_timed_calibration": False' in h100_text


def test_actual_cuda_dispatch_and_fused_loss_proofs_are_fail_closed() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    dispatch = _function_source(text, "_prove_triton_dispatch", "_prove_fused_loss_cuda")
    fused = _function_source(text, "_prove_fused_loss_cuda", "h100_preflight")
    h100 = _function_source(text, "h100_preflight", "main")

    assert "integration.affine_scan_triton_candidate = counted_candidate" in dispatch
    assert "integration.affine_scan = forbidden_fallback" in dispatch
    assert 'calls["candidate"] != EXPECTED_LAYERS' in dispatch
    assert 'calls["fallback"] != 0' in dispatch
    assert "legacy affine-scan fallback selected" in dispatch

    assert "FusedLinearCETrainingRunner(model)" in fused
    assert "not runner._use_liger or runner._liger_loss is None" in fused
    assert "model.lm_head.weight is not model.token_emb.weight" in fused
    assert "_prove_fused_loss_cuda(proof_model)" in h100
    assert 'importlib.metadata.version("liger-kernel") != LIGER_KERNEL_VERSION' in h100


def test_workflow_is_owner_only_source_bound_and_pins_every_frozen_blob() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    required = (
        "issues:",
        "types: [opened]",
        "github.event.issue.user.login == github.repository_owner",
        "github.event.issue.title == '[modal-cortex-s-100m-2b-preflight-v7-fused-linear-ce]'",
        "phase == 'preflight-v7-fused-linear-ce'",
        "re.fullmatch(r'[0-9a-f]{40}', value)",
        'test "$(git -C source rev-parse origin/main)" = "$SOURCE_SHA"',
        'git -C source merge-base --is-ancestor "$HARNESS_SHA" "$SOURCE_SHA"',
        EXPECTED_RUNNER_BLOB,
        EXPECTED_FUSED_BLOB,
        EXPECTED_FUSED_TEST_BLOB,
        EXPECTED_TRAIN_BLOB,
        EXPECTED_TRAIN_SHA256,
        EXPECTED_SCAN_INTEGRATION_BLOB,
        EXPECTED_TRITON_BLOB,
        EXPECTED_GROUPED_MOE_BLOB,
        EXPECTED_LIGER_WHEEL_SHA256,
        'LIGER_KERNEL_VERSION = "0.8.2"',
        'FUSED_SYSTEMS_VARIANT = "fused_linear_ce_liger_v1"',
        "ENGINEERING_SEED = 2_026_091_009",
        'RESULT_ROOT = "/vol/cortex-s-v0/100m-2b/preflight-v7-fused-linear-ce"',
        "tree = ast.parse(runner)",
        "gpu_sites == ['H100!']",
        "name.startswith('torch.profiler')",
        "modal run --detach --timestamps modal_cortex_s_100m_2b_v7_fused_linear_ce_preflight.py",
    )
    for needle in required:
        assert needle in text
    assert "workflow_dispatch" not in text
    executable_modal_lines = [
        line.strip() for line in text.splitlines() if line.strip().startswith("modal run ")
    ]
    assert len(executable_modal_lines) == 1
    assert executable_modal_lines[0].startswith(
        "modal run --detach --timestamps modal_cortex_s_100m_2b_v7_fused_linear_ce_preflight.py"
    )
    assert text.index("Fail closed on live source ancestry and frozen blobs") < text.index(
        "Install Modal only"
    ) < text.index("modal run --detach --timestamps")


def test_v7_harness_never_authorizes_full_training_or_scientific_claims() -> None:
    runner = RUNNER.read_text(encoding="utf-8")
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert '"full_training_authorized": False' in runner
    assert '"scientific_claim_authorized": False' in runner
    assert '"seed_8100_used": False' in runner
    assert '"reserved_scientific_seeds_used": False' in runner
    assert "train_full_2b(" not in runner
    assert "No full 2B or scientific/control seed is authorized" in workflow
