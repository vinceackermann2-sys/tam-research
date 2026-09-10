from __future__ import annotations

import ast
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
CORE = ROOT / "architectures/cortex_s/full_step_profile_v1.py"
RUNNER = ROOT / "modal_cortex_s_100m_full_step_profile_v1.py"
WORKFLOW = ROOT / ".github/workflows/modal-cortex-s-100m-full-step-profile-v1.yml"
TRAIN_IMPL = ROOT / "architectures/cortex_s/experiments/scale100m_2b/train.py"

EXPECTED_CORE_BLOB = "ac9d9530b4ce318efbe71ee07fe0afe5095d1d7a"
EXPECTED_RUNNER_BLOB = "c8952b26034a1ad650b8df07395c264127d8a1f0"
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
    )
    for needle in required:
        assert needle in text
    assert "train_full_2b(" not in text
    assert "_one_optimizer_step_no_amp" not in text
    assert "optimizer =" not in text
    assert "torch.compile(" not in text


def test_runner_freezes_fresh_single_use_authority() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    ast.parse(text)
    required = (
        'PHASE = "full-step-profile-v1"',
        'TRIGGER_TITLE = "[modal-cortex-s-100m-full-step-profile-v1]"',
        "ENGINEERING_SEED = 2_026_091_003",
        'RESULT_ROOT = "/vol/cortex-s-v0/100m-2b/full-step-profile-v1"',
        "EXPECTED_CORTEX_PARAMS = 101_778_112",
        "EXPECTED_SEQ_LEN = 512",
        "EXPECTED_MICRO_BATCH_SIZE = 64",
        "EXPECTED_GRAD_ACCUM_STEPS = 2",
        'EXPECTED_COMPILE_MODE = "max-autotune-no-cudagraphs"',
        f'TRAIN_IMPLEMENTATION_SHA256 = "{EXPECTED_TRAIN_IMPL_SHA256}"',
        "FORBIDDEN_CONTROL_AND_SCIENTIFIC_SEEDS = (8_100, 48_131, 48_132, 48_133)",
        "2_026_091_002",
        'gpu="H100!"',
        "run_full_step_profile(",
        "training_module._compile_model(model)",
        "training_module._make_optimizer(model)",
        "training_module._one_optimizer_step(",
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
    reserve = _function_source(text, "reserve_h100_dispatch", "_prove_triton_selection")
    h100 = _function_source(text, "h100_profile", "main")
    main = _function_source(text, "main")
    assert 'marker_path = root / "H100_DISPATCH_CONSUMED.json"' in reserve
    assert "_atomic_write(marker_path, marker)" in reserve
    assert "volume.commit()" in reserve
    assert 'marker_path = root / "H100_DISPATCH_CONSUMED.json"' in h100
    assert "if not marker_path.exists():" in h100
    assert "_atomic_write(marker_path" not in h100
    assert main.index("verify_zero_gpu.remote") < main.index("reserve_h100_dispatch.remote")
    assert main.index("reserve_h100_dispatch.remote") < main.index("h100_profile.remote")


def test_h100_path_uses_frozen_builder_and_actual_triton_selection() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    proof = _function_source(text, "_prove_triton_selection", "h100_profile")
    h100 = _function_source(text, "h100_profile", "main")
    assert "integration.triton_available()" in proof
    assert "integration._should_use_triton(probe)" in proof
    assert "worlds != EXPECTED_LAYERS" in proof
    assert "build_memory_lean_grouped_triton_scan_cortex_100m().to(device)" in h100
    assert "training_module.seed_all(ENGINEERING_SEED)" in h100
    assert "torch.Generator(device=\"cpu\").manual_seed(ENGINEERING_SEED + 10_000)" in h100
    assert "run_full_step_profile(" in h100
    assert '"status": "PROFILE_COMPLETE"' in h100
    assert '"seed_8100_used": False' in h100
    assert '"reserved_scientific_seeds_used": False' in h100


def test_workflow_is_owner_only_exact_title_source_bound_and_single_use() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    required = (
        "issues:",
        "types: [opened]",
        "github.event.issue.user.login == github.repository_owner",
        "github.event.issue.title == '[modal-cortex-s-100m-full-step-profile-v1]'",
        "phase == 'full-step-profile-v1'",
        'test "$(git -C source rev-parse origin/main)" = "$SOURCE_SHA"',
        'git -C source merge-base --is-ancestor "$HARNESS_SHA" "$SOURCE_SHA"',
        EXPECTED_CORE_BLOB,
        EXPECTED_RUNNER_BLOB,
        EXPECTED_TRAIN_IMPL_SHA256,
        "ENGINEERING_SEED = 2_026_091_003",
        'RESULT_ROOT = "/vol/cortex-s-v0/100m-2b/full-step-profile-v1"',
        "modal run --detach --timestamps modal_cortex_s_100m_full_step_profile_v1.py",
    )
    for needle in required:
        assert needle in text
    assert "workflow_dispatch" not in text
    assert "train_full_2b(" not in text
    assert "2026091002" not in text
    assert "seed 8100" in text.lower()


def test_training_implementation_sha256_matches_preregistered_identity() -> None:
    assert hashlib.sha256(TRAIN_IMPL.read_bytes()).hexdigest() == EXPECTED_TRAIN_IMPL_SHA256
