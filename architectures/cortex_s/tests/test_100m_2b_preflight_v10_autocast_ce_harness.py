from __future__ import annotations

import ast
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / "modal_cortex_s_100m_2b_v10_autocast_ce_no_explicit_fp32_preflight.py"
WORKFLOW = ROOT / ".github/workflows/modal-cortex-s-100m-2b-preflight-v10-autocast-ce-no-explicit-fp32.yml"
CANDIDATE = ROOT / "architectures/cortex_s/autocast_ce_no_explicit_fp32_v1.py"
CANDIDATE_TEST = ROOT / "architectures/cortex_s/tests/test_autocast_ce_no_explicit_fp32_v1.py"
TRAINER = ROOT / "architectures/cortex_s/experiments/scale100m_2b/train.py"

EXPECTED_TRAINER_SHA256 = "d0d91de09bba8b7631913b3e9c0407ab2bb0114f7bba133c123ee5598d490461"
EXPECTED_RUNNER_BLOB = "6a7c8d122ed25a222bbca0f3aeaf4efcb1d9b723"
EXPECTED_CANDIDATE_BLOB = "34d637a164a9318e6a1f3c7e1750529c793e49fa"
EXPECTED_CANDIDATE_TEST_BLOB = "c15530422d3e0f4f270476b9ed716acce276091d"
EXPECTED_TRAINER_BLOB = "004b66b549d3d64f2dde7614ec84f22b9f37a7c6"
EXPECTED_SCAN_INTEGRATION_BLOB = "c3d05f8b4d1e3e8ea7bcbad5fdeca5d7091651c7"
EXPECTED_TRITON_CANDIDATE_BLOB = "5c6bf7df95f49c306b4ffadd6a8a09b9f71344f9"
EXPECTED_GROUPED_MOE_BLOB = "7c738e6349536dd45f1c631a9ad5524ade1d5021"


def test_v10_runner_freezes_single_use_paid_boundary_and_budget() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    required = (
        'PHASE = "preflight-v10-autocast-ce-no-explicit-fp32"',
        'TRIGGER_TITLE = "[modal-cortex-s-100m-2b-preflight-v10-autocast-ce-no-explicit-fp32]"',
        'ENGINEERING_SEED = 2_026_091_012',
        'RESULT_ROOT = "/vol/cortex-s-v0/100m-2b/preflight-v10-autocast-ce-no-explicit-fp32"',
        'AUTocast_SYSTEMS_VARIANT = "autocast_ce_no_explicit_fp32_v1"',
        'EXPECTED_CORTEX_PARAMS = 101_778_112',
        'CALIBRATION_WARMUP_STEPS = 5',
        'MEASURED_STEPS = 40',
        'MEASURED_TOKENS = MEASURED_STEPS * TOKENS_PER_STEP',
        'PROJECTION_OVERHEAD_MULTIPLIER = 1.10',
        'MAX_PROJECTED_FULL_SECONDS = 8_500.0',
        'MAX_PREFLIGHT_VRAM_GIB = 70.0',
        'with memory_lean_grouped_triton_scan_training_builder():',
        'with autocast_ce_no_explicit_fp32_training_builder():',
        'training_module.run_h100_calibration(',
        'training_module.CALIBRATION_SEED = ENGINEERING_SEED',
        '"profiling_inside_timed_calibration": False',
        '"full_training_authorized": False',
        '"scientific_claim_authorized": False',
    )
    for needle in required:
        assert needle in source, needle

    tree = ast.parse(source)
    gpu_sites: list[str] = []
    call_names: list[str] = []
    profiler_calls: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = ast.unparse(node.func)
        call_names.append(name)
        if name.startswith("torch.profiler"):
            profiler_calls.append(name)
        for kw in node.keywords:
            if kw.arg == "gpu" and isinstance(kw.value, ast.Constant):
                gpu_sites.append(kw.value.value)
    assert gpu_sites == ["H100!"]
    assert not any(name.endswith("train_full_2b") for name in call_names)
    assert not profiler_calls
    assert "2_026_091_011" in source
    assert "8_100, 48_131, 48_132, 48_133" in source


def test_dispatch_marker_is_committed_before_h100_and_result_is_durable() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    reserve = source[source.index("def reserve_h100_dispatch("):source.index("def _prove_triton_dispatch(")]
    h100 = source[source.index("def h100_preflight("):source.index("@app.local_entrypoint()")]
    main = source[source.index("def main("):]
    assert '_atomic_write(marker_path, marker)' in reserve
    assert "volume.commit()" in reserve
    assert main.index("reserve_h100_dispatch.remote") < main.index("h100_preflight.remote")
    assert 'if result_path.exists():' in h100
    assert 'marker["h100_allocation_started"] = True' in h100
    assert h100.index('with memory_lean_grouped_triton_scan_training_builder():') < h100.index('with autocast_ce_no_explicit_fp32_training_builder():') < h100.index('training_module.run_h100_calibration(')
    assert '_atomic_write(result_path, result)' in h100
    assert '_atomic_write(result_path, failure)' in h100
    assert h100.count("volume.commit()") >= 3


def test_v10_candidate_and_production_trainer_semantics_are_frozen() -> None:
    candidate = CANDIDATE.read_text(encoding="utf-8")
    trainer = TRAINER.read_text(encoding="utf-8")
    assert 'SYSTEMS_VARIANT = "autocast_ce_no_explicit_fp32_v1"' in candidate
    assert "with training_module._autocast(device):" in candidate
    assert "F.cross_entropy(" in candidate
    assert "logits.float()" not in candidate
    assert candidate.count('.to(device="cpu")') == 1
    assert "logits.float().reshape(-1, logits.size(-1))" in trainer
    assert hashlib.sha256(TRAINER.read_bytes()).hexdigest() == EXPECTED_TRAINER_SHA256


def test_workflow_is_owner_only_exact_title_source_bound_and_one_shot() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "types: [opened]" in workflow
    assert "github.event.issue.user.login == github.repository_owner" in workflow
    assert "github.event.issue.title == '[modal-cortex-s-100m-2b-preflight-v10-autocast-ce-no-explicit-fp32]'" in workflow
    assert "workflow_dispatch" not in workflow
    assert "git -C source rev-parse origin/main" in workflow
    assert "git -C source merge-base --is-ancestor \"$HARNESS_SHA\" \"$SOURCE_SHA\"" in workflow
    for blob in (
        EXPECTED_RUNNER_BLOB,
        EXPECTED_CANDIDATE_BLOB,
        EXPECTED_CANDIDATE_TEST_BLOB,
        EXPECTED_TRAINER_BLOB,
        EXPECTED_SCAN_INTEGRATION_BLOB,
        EXPECTED_TRITON_CANDIDATE_BLOB,
        EXPECTED_GROUPED_MOE_BLOB,
    ):
        assert blob in workflow
    assert "modal run --detach --timestamps modal_cortex_s_100m_2b_v10_autocast_ce_no_explicit_fp32_preflight.py" in workflow
    assert workflow.count("modal run --detach --timestamps") == 1
    assert "--phase \"preflight-v10-autocast-ce-no-explicit-fp32\"" in workflow


def test_harness_has_no_full_training_or_scientific_authority() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert '"full_training_authorized": True' not in source
    assert '"scientific_claim_authorized": True' not in source
    assert "PAIRED_SEED" not in source
    assert "48131" not in source
    assert "48132" not in source
    assert "48133" not in source
    assert CANDIDATE_TEST.exists()
