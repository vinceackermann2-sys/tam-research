from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / "modal_cortex_s_100m_2b_v6_host_sync_coalesced_preflight.py"
WORKFLOW = ROOT / ".github/workflows/modal-cortex-s-100m-2b-preflight-v6-host-sync-coalesced.yml"
TRAIN_IMPL = ROOT / "architectures/cortex_s/experiments/scale100m_2b/train.py"

EXPECTED_RUNNER_BLOB = "6593378b9ba790fb02d1fbb17dd830625c556978"
EXPECTED_TRAIN_BLOB = "004b66b549d3d64f2dde7614ec84f22b9f37a7c6"
EXPECTED_TRAIN_SHA256 = "d0d91de09bba8b7631913b3e9c0407ab2bb0114f7bba133c123ee5598d490461"


def _function_source(text: str, name: str, next_name: str | None = None) -> str:
    start = text.index(f"def {name}(")
    if next_name is None:
        return text[start:]
    return text[start:text.index(f"def {next_name}(", start + 1)]


def _profiler_calls(text: str) -> list[str]:
    tree = ast.parse(text)
    return [
        ast.unparse(node.func)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func).startswith("torch.profiler")
    ]


def test_v6_runner_is_syntax_valid_and_single_use_authority_is_frozen() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    ast.parse(text)
    required = (
        'PHASE = "preflight-v6-host-sync-coalesced"',
        'TRIGGER_TITLE = "[modal-cortex-s-100m-2b-preflight-v6-host-sync-coalesced]"',
        "ENGINEERING_SEED = 2_026_091_008",
        'RESULT_ROOT = "/vol/cortex-s-v0/100m-2b/preflight-v6-host-sync-coalesced"',
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
        "2_026_091_007",
        "FORBIDDEN_CONTROL_AND_SCIENTIFIC_SEEDS = (8_100, 48_131, 48_132, 48_133)",
    )
    for needle in required:
        assert needle in text
    assert text.count('gpu="H100!"') == 1
    assert "train_full_2b(" not in text
    assert _profiler_calls(text) == []
    assert 'if "torch.profiler" in step_source:' in text
    assert "ENGINEERING_SEED = 8_100" not in text
    assert "ENGINEERING_SEED = 48_131" not in text
    assert "ENGINEERING_SEED = 48_132" not in text
    assert "ENGINEERING_SEED = 48_133" not in text


def test_active_training_step_has_exactly_one_bundled_host_readback() -> None:
    source = TRAIN_IMPL.read_text(encoding="utf-8")
    step = _function_source(source, "_one_optimizer_step", "evaluate")
    assert ") -> float:" in step
    assert "running_loss: torch.Tensor | None = None" in step
    assert "all_finite: torch.Tensor | None = None" in step
    assert "float(loss.detach())" not in step
    assert "if not torch.isfinite(loss)" not in step
    assert "if not torch.isfinite(grad_norm)" not in step
    assert "torch.logical_and(all_finite, torch.isfinite(grad_norm))" in step
    assert "host_report = torch.stack(" in step
    assert step.count('.to(device="cpu")') == 1
    assert "running_value, finite_value = host_report.tolist()" in step
    assert step.index("host_report = torch.stack(") < step.index("optimizer.step()")
    assert step.index("if not bool(finite_value):") < step.index("optimizer.step()")


def test_dispatch_marker_is_durable_before_h100_invocation() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    reserve = _function_source(text, "reserve_h100_dispatch", "_prove_triton_dispatch")
    h100 = _function_source(text, "h100_preflight", "main")
    main = _function_source(text, "main")
    assert 'marker_path = root / "H100_DISPATCH_CONSUMED.json"' in reserve
    assert "_atomic_write(marker_path, marker)" in reserve
    assert "volume.commit()" in reserve
    assert 'marker_path = root / "H100_DISPATCH_CONSUMED.json"' in h100
    assert "if not marker_path.exists():" in h100
    assert main.index("reserve_h100_dispatch.remote") < main.index("h100_preflight.remote")
    assert 'marker["h100_allocation_started"] = True' in h100
    assert "_atomic_write(marker_path, marker)" in h100


def test_h100_path_reuses_exact_production_calibration_without_profiler() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    h100 = _function_source(text, "h100_preflight", "main")
    assert "with memory_lean_grouped_triton_scan_training_builder():" in h100
    assert "training_module.run_h100_calibration(" in h100
    assert "training_module.CALIBRATION_SEED = ENGINEERING_SEED" in h100
    assert "training_module.CALIBRATION_SEED = original_calibration_seed" in h100
    assert 'raw.get("measured_steps") != MEASURED_STEPS' in h100
    assert 'raw.get("measured_tokens") != MEASURED_TOKENS' in h100
    assert "TRAIN_TOKENS / tps * PROJECTION_OVERHEAD_MULTIPLIER + compile_seconds" in h100
    assert '"peak_vram_le_70_gib"' in h100
    assert '"projected_full_seconds_le_budget"' in h100
    assert '"profiling_inside_timed_calibration": False' in h100
    assert "torch.profiler" not in h100


def test_actual_cuda_dispatcher_proof_is_fail_closed() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    proof = _function_source(text, "_prove_triton_dispatch", "h100_preflight")
    assert "integration.affine_scan_triton_candidate = counted_candidate" in proof
    assert "integration.affine_scan = forbidden_fallback" in proof
    assert 'calls["candidate"] != EXPECTED_LAYERS' in proof
    assert 'calls["fallback"] != 0' in proof
    assert "legacy affine-scan fallback selected" in proof


def test_workflow_is_owner_only_source_bound_and_pins_active_blobs() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    required = (
        "issues:",
        "types: [opened]",
        "github.event.issue.user.login == github.repository_owner",
        "github.event.issue.title == '[modal-cortex-s-100m-2b-preflight-v6-host-sync-coalesced]'",
        "phase == 'preflight-v6-host-sync-coalesced'",
        'test "$(git -C source rev-parse origin/main)" = "$SOURCE_SHA"',
        'git -C source merge-base --is-ancestor "$HARNESS_SHA" "$SOURCE_SHA"',
        EXPECTED_RUNNER_BLOB,
        EXPECTED_TRAIN_BLOB,
        EXPECTED_TRAIN_SHA256,
        "c3d05f8b4d1e3e8ea7bcbad5fdeca5d7091651c7",
        "5c6bf7df95f49c306b4ffadd6a8a09b9f71344f9",
        "7c738e6349536dd45f1c631a9ad5524ade1d5021",
        "ENGINEERING_SEED = 2_026_091_008",
        'RESULT_ROOT = "/vol/cortex-s-v0/100m-2b/preflight-v6-host-sync-coalesced"',
        "runner_tree = ast.parse(runner)",
        "ast.walk(runner_tree)",
        "ast.unparse(node.func).startswith('torch.profiler')",
        "modal run --detach --timestamps modal_cortex_s_100m_2b_v6_host_sync_coalesced_preflight.py",
    )
    for needle in required:
        assert needle in text
    assert "assert 'torch.profiler' not in runner" not in text
    assert "workflow_dispatch" not in text
    executable_modal_lines = [
        line.strip() for line in text.splitlines() if line.strip().startswith("modal run ")
    ]
    assert len(executable_modal_lines) == 1
    assert executable_modal_lines[0].startswith(
        "modal run --detach --timestamps modal_cortex_s_100m_2b_v6_host_sync_coalesced_preflight.py"
    )


def test_v6_harness_never_authorizes_full_training_or_scientific_claims() -> None:
    runner = RUNNER.read_text(encoding="utf-8")
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert '"full_training_authorized": False' in runner
    assert '"scientific_claim_authorized": False' in runner
    assert '"seed_8100_used": False' in runner
    assert '"reserved_scientific_seeds_used": False' in runner
    assert "train_full_2b(" not in runner
    assert "full 2B or scientific/control seed is authorized" in workflow
