from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / "modal_cortex_s_100m_2b_v8_fused_linear_ce_packaging_preflight.py"
WORKFLOW = ROOT / ".github/workflows/modal-cortex-s-100m-2b-preflight-v8-fused-linear-ce-packaging.yml"
CANDIDATE = ROOT / "architectures/cortex_s/fused_linear_ce_v1.py"
TRAIN = ROOT / "architectures/cortex_s/experiments/scale100m_2b/train.py"

EXPECTED_RUNNER_BLOB = "70ee2cdc6bd26a7f818be2771e18de6daee7aaf2"
EXPECTED_WORKFLOW_BLOB = "a91c4c76af4560d937c4e1174ba49796272007d3"
EXPECTED_CANDIDATE_BLOB = "3b1697b45e73c1a0e860c5ecdd4f8bc1ec1ae179"
EXPECTED_TRAIN_BLOB = "004b66b549d3d64f2dde7614ec84f22b9f37a7c6"
EXPECTED_TRAIN_SHA256 = "d0d91de09bba8b7631913b3e9c0407ab2bb0114f7bba133c123ee5598d490461"


def _function_source(text: str, name: str, next_marker: str | None = None) -> str:
    start = text.index(f"def {name}(")
    if next_marker is None:
        return text[start:]
    return text[start:text.index(next_marker, start + 1)]


def test_v8_runner_identity_geometry_and_fresh_authority_are_frozen() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    tree = ast.parse(text)
    required = (
        'PHASE = "preflight-v8-fused-linear-ce-packaging"',
        'TRIGGER_TITLE = "[modal-cortex-s-100m-2b-preflight-v8-fused-linear-ce-packaging]"',
        "ENGINEERING_SEED = 2_026_091_010",
        'RESULT_ROOT = "/vol/cortex-s-v0/100m-2b/preflight-v8-fused-linear-ce-packaging"',
        'PACKAGING_VERSION = "26.3"',
        '"packaging==26.3"',
        'LIGER_KERNEL_VERSION = "0.8.2"',
        "EXPECTED_CORTEX_PARAMS = 101_778_112",
        "CALIBRATION_WARMUP_STEPS = 5",
        "MEASURED_STEPS = 40",
        "MEASURED_TOKENS = MEASURED_STEPS * TOKENS_PER_STEP",
        "MAX_PROJECTED_FULL_SECONDS = 8_500.0",
        "MAX_PREFLIGHT_VRAM_GIB = 70.0",
        "2_026_091_009",
        "FORBIDDEN_CONTROL_AND_SCIENTIFIC_SEEDS = (8_100, 48_131, 48_132, 48_133)",
    )
    for needle in required:
        assert needle in text

    gpu_sites: list[str] = []
    calls: list[str] = []
    profiler_calls: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = ast.unparse(node.func)
        calls.append(name)
        if name.startswith("torch.profiler"):
            profiler_calls.append(name)
        for kw in node.keywords:
            if kw.arg == "gpu" and isinstance(kw.value, ast.Constant):
                gpu_sites.append(str(kw.value.value))
    assert gpu_sites == ["H100!"]
    assert not any(name.endswith("train_full_2b") for name in calls)
    assert not profiler_calls


def test_dependency_smoke_is_zero_gpu_and_precedes_dispatch_reservation() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    smoke = _function_source(text, "_dependency_smoke", "\ndef _training_module_and_path")
    zero = _function_source(text, "verify_zero_gpu", "\n\n@app.function(image=base_image, cpu=1")
    reserve = _function_source(text, "reserve_h100_dispatch", "\ndef _prove_triton_dispatch")
    main = _function_source(text, "main")

    assert 'importlib.metadata.version("packaging")' in smoke
    assert 'importlib.metadata.version("liger-kernel")' in smoke
    assert "from packaging.version import Version" in smoke
    assert "from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss" in smoke
    assert "_dependency_smoke()" in zero
    assert '"dependency_smoke": dependency_smoke' in zero
    assert 'zero.get("dependency_smoke", {}).get("packaging_version") != PACKAGING_VERSION' in reserve
    assert 'zero.get("dependency_smoke", {}).get("liger_fused_linear_ce_import_ok")' in reserve
    assert main.index("verify_zero_gpu.remote") < main.index("reserve_h100_dispatch.remote") < main.index("h100_preflight.remote")


def test_v8_preserves_v7_fused_and_scan_calibration_contract() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    h100 = _function_source(text, "h100_preflight", "\n\n@app.local_entrypoint()")
    assert "_dependency_smoke()" in h100
    assert "_prove_triton_dispatch(proof_model" in h100
    assert "_prove_fused_loss_cuda(proof_model)" in h100
    assert "with memory_lean_grouped_triton_scan_training_builder():" in h100
    assert "with fused_linear_ce_training_builder():" in h100
    assert "training_module.run_h100_calibration(" in h100
    assert 'raw.get("execution") != "compiled"' in h100
    assert "TRAIN_TOKENS / tps * PROJECTION_OVERHEAD_MULTIPLIER + compile_seconds" in h100
    assert '"full_training_authorized": False' in h100
    assert '"scientific_claim_authorized": False' in h100
    assert '"seed_8100_used": False' in h100
    assert '"reserved_scientific_seeds_used": False' in h100


def test_dispatch_marker_is_durable_before_h100_remote_call() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    reserve = _function_source(text, "reserve_h100_dispatch", "\ndef _prove_triton_dispatch")
    main = _function_source(text, "main")
    assert '_atomic_write(marker_path, marker)' in reserve
    assert "volume.commit()" in reserve
    assert '"h100_allocation_started": False' in reserve
    assert main.index("reserve_h100_dispatch.remote") < main.index("h100_preflight.remote")


def test_workflow_is_owner_only_exact_title_source_bound_and_blob_pinned() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    required = (
        "issues:",
        "types: [opened]",
        "github.event.issue.user.login == github.repository_owner",
        "github.event.issue.title == '[modal-cortex-s-100m-2b-preflight-v8-fused-linear-ce-packaging]'",
        "phase == 'preflight-v8-fused-linear-ce-packaging'",
        'test "$(git -C source rev-parse origin/main)" = "$SOURCE_SHA"',
        EXPECTED_RUNNER_BLOB,
        EXPECTED_CANDIDATE_BLOB,
        EXPECTED_TRAIN_BLOB,
        EXPECTED_TRAIN_SHA256,
        'PACKAGING_VERSION = "26.3"',
        '"packaging==26.3"',
        "modal run --detach --timestamps modal_cortex_s_100m_2b_v8_fused_linear_ce_packaging_preflight.py",
    )
    for needle in required:
        assert needle in text
    assert "workflow_dispatch" not in text
    assert text.count("modal run --detach --timestamps") == 1


def test_v8_harness_historically_pins_candidate_and_production_contract() -> None:
    runner = RUNNER.read_text(encoding="utf-8")
    workflow = WORKFLOW.read_text(encoding="utf-8")
    train = TRAIN.read_text(encoding="utf-8")

    # v8's historical runner/workflow remain immutable and prove exactly which
    # candidate it executed, while allowing later preregistered candidate versions
    # to exist on current main.
    assert 'FUSED_SYSTEMS_VARIANT = "fused_linear_ce_liger_v1"' in runner
    assert EXPECTED_CANDIDATE_BLOB in workflow
    assert 'architectures/cortex_s/fused_linear_ce_v1.py' in workflow
    assert 'LIGER_KERNEL_VERSION = "0.8.2"' in runner
    assert "host_report = torch.stack(" in train
    assert train.count('.to(device="cpu")') >= 1
    assert "train_full_2b(" not in runner
