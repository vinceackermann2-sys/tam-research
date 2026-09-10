from __future__ import annotations

import ast
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
CORE = ROOT / "architectures/cortex_s/full_step_profile_v2.py"
RUNNER = ROOT / "modal_cortex_s_100m_full_step_profile_v2.py"
WORKFLOW = ROOT / ".github/workflows/modal-cortex-s-100m-full-step-profile-v2.yml"
TRAIN_IMPL = ROOT / "architectures/cortex_s/experiments/scale100m_2b/train.py"

EXPECTED_CORE_BLOB = "92963097dad00453c1fc1058e8a1f79f4f4ab51c"
EXPECTED_RUNNER_BLOB = "b7855246fda08bb6a2ede6b62ca4619f58b3fa27"
EXPECTED_TRAIN_IMPL_SHA256 = "47c9e327095f99eab921b8780bb968101296b11e62ef13dd22488f2974bfb10f"


def _function_source(text: str, name: str, next_name: str | None = None) -> str:
    start = text.index(f"def {name}(")
    if next_name is None:
        return text[start:]
    return text[start:text.index(f"def {next_name}(", start + 1)]


def test_profiler_core_is_exact_production_step_diagnostic() -> None:
    text = CORE.read_text(encoding="utf-8")
    ast.parse(text)
    required = (
        "STEADY_WARMUP_STEPS = 5",
        "STEADY_MEASURED_STEPS = 8",
        "PROFILED_STEPS = 2",
        "torch.profiler.ProfilerActivity.CPU",
        "torch.profiler.ProfilerActivity.CUDA",
        "training_module._one_optimizer_step(",
        "torch.cuda.synchronize(device)",
        '"training_tokens_per_second"',
        '"categories"',
        '"top_device_operators"',
        '"top_cpu_operators"',
        '"classification": "ENGINEERING_FULL_STEP_PROFILE_V2_ONLY"',
    )
    for needle in required:
        assert needle in text
    assert "train_full_2b(" not in text
    assert "_one_optimizer_step_no_amp" not in text
    assert "torch.compile(" not in text


def test_runner_freezes_fresh_single_use_authority() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    ast.parse(text)
    required = (
        'PHASE = "full-step-profile-v2"',
        'TRIGGER_TITLE = "[modal-cortex-s-100m-full-step-profile-v2]"',
        "ENGINEERING_SEED = 2_026_091_004",
        'RESULT_ROOT = "/vol/cortex-s-v0/100m-2b/full-step-profile-v2"',
        "EXPECTED_CORTEX_PARAMS = 101_778_112",
        "EXPECTED_SEQ_LEN = 512",
        "EXPECTED_MICRO_BATCH_SIZE = 64",
        "EXPECTED_GRAD_ACCUM_STEPS = 2",
        'EXPECTED_COMPILE_MODE = "max-autotune-no-cudagraphs"',
        f'TRAIN_IMPLEMENTATION_SHA256 = "{EXPECTED_TRAIN_IMPL_SHA256}"',
        "2_026_091_003",
        "FORBIDDEN_CONTROL_AND_SCIENTIFIC_SEEDS = (8_100, 48_131, 48_132, 48_133)",
        'gpu="H100!"',
        "run_full_step_profile(",
        "training_module._compile_model(model)",
        "training_module._make_optimizer(model)",
        "training_module._one_optimizer_step(",
        '"h100_allocation_started": False',
        '"full_training_authorized": False',
        '"scientific_claim_authorized": False',
    )
    for needle in required:
        assert needle in text
    assert text.count('gpu="H100!"') == 1
    assert "train_full_2b(" not in text
    assert "_one_optimizer_step_no_amp" not in text
    assert "ENGINEERING_SEED = 8_100" not in text
    assert "ENGINEERING_SEED = 48_131" not in text
    assert "ENGINEERING_SEED = 48_132" not in text
    assert "ENGINEERING_SEED = 48_133" not in text


def test_dispatch_marker_is_durable_before_h100_invocation() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    reserve = _function_source(text, "reserve_h100_dispatch", "_prove_triton_dispatch")
    h100 = _function_source(text, "h100_profile", "main")
    main = _function_source(text, "main")
    assert 'marker_path = root / "H100_DISPATCH_CONSUMED.json"' in reserve
    assert "_atomic_write(marker_path, marker)" in reserve
    assert "volume.commit()" in reserve
    assert '"h100_allocation_started": False' in reserve
    assert 'marker_path = root / "H100_DISPATCH_CONSUMED.json"' in h100
    assert "if not marker_path.exists():" in h100
    assert 'marker["h100_allocation_started"] = True' in h100
    assert "_atomic_write(marker_path, marker)" in h100
    assert main.index("verify_zero_gpu.remote") < main.index("reserve_h100_dispatch.remote")
    assert main.index("reserve_h100_dispatch.remote") < main.index("h100_profile.remote")


def test_h100_path_uses_frozen_builder_and_actual_triton_dispatch() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    proof = _function_source(text, "_prove_triton_dispatch", "h100_profile")
    h100 = _function_source(text, "h100_profile", "main")
    assert "original_candidate = integration.affine_scan_triton_candidate" in proof
    assert "original_fallback = integration.affine_scan" in proof
    assert "integration.affine_scan_triton_candidate = counted_candidate" in proof
    assert "integration.affine_scan = forbidden_fallback" in proof
    assert 'calls["candidate"] != EXPECTED_LAYERS' in proof
    assert 'calls["fallback"] != 0' in proof
    assert "model.train(was_training)" in proof
    assert "build_memory_lean_grouped_triton_scan_cortex_100m().to(device)" in h100
    assert "training_module.seed_all(ENGINEERING_SEED)" in h100
    assert "torch.Generator(device=\"cpu\").manual_seed(ENGINEERING_SEED + 10_000)" in h100
    assert "run_full_step_profile(" in h100
    assert '"status": "PROFILE_COMPLETE"' in h100
    assert '"seed_8100_used": False' in h100
    assert '"reserved_scientific_seeds_used": False' in h100


def test_workflow_is_owner_only_exact_title_source_bound_and_scoped() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    required = (
        "issues:",
        "types: [opened]",
        "github.event.issue.user.login == github.repository_owner",
        "github.event.issue.title == '[modal-cortex-s-100m-full-step-profile-v2]'",
        "phase == 'full-step-profile-v2'",
        'test "$(git -C source rev-parse origin/main)" = "$SOURCE_SHA"',
        'git -C source merge-base --is-ancestor "$HARNESS_SHA" "$SOURCE_SHA"',
        EXPECTED_CORE_BLOB,
        EXPECTED_RUNNER_BLOB,
        EXPECTED_TRAIN_IMPL_SHA256,
        "ENGINEERING_SEED = 2_026_091_004",
        'RESULT_ROOT = "/vol/cortex-s-v0/100m-2b/full-step-profile-v2"',
        "assert 'train_full_2b(' not in runner",
        "assert '_one_optimizer_step_no_amp' not in runner",
        "assert 'train_full_2b(' not in core",
        "assert '_one_optimizer_step_no_amp' not in core",
        "modal run --detach --timestamps modal_cortex_s_100m_full_step_profile_v2.py",
    )
    for needle in required:
        assert needle in text
    assert "workflow_dispatch" not in text
    executable_modal_lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip().startswith("modal run ")
    ]
    assert executable_modal_lines == [
        "modal run --detach --timestamps modal_cortex_s_100m_full_step_profile_v2.py \\",
    ]


def test_training_implementation_sha256_matches_preregistered_identity() -> None:
    assert hashlib.sha256(TRAIN_IMPL.read_bytes()).hexdigest() == EXPECTED_TRAIN_IMPL_SHA256
