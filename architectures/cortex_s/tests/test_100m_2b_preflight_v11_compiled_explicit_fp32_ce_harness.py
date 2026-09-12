from __future__ import annotations

import ast
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / "modal_cortex_s_100m_2b_v11_compiled_explicit_fp32_ce_preflight.py"
WORKFLOW = ROOT / ".github/workflows/modal-cortex-s-100m-2b-preflight-v11-compiled-explicit-fp32-ce.yml"
CANDIDATE = ROOT / "architectures/cortex_s/compiled_explicit_fp32_ce_v1.py"
CANDIDATE_TEST = ROOT / "architectures/cortex_s/tests/test_compiled_explicit_fp32_ce_v1.py"
TRAINER = ROOT / "architectures/cortex_s/experiments/scale100m_2b/train.py"

EXPECTED_TRAINER_SHA256 = "d0d91de09bba8b7631913b3e9c0407ab2bb0114f7bba133c123ee5598d490461"
EXPECTED_RUNNER_BLOB = "10d2e8b071a89298f9bd71b9d34f1ece3b8fbc52"
EXPECTED_CANDIDATE_BLOB = "970a0dcf130348662f4c9a3ed11d8e6ad250781a"
EXPECTED_CANDIDATE_TEST_BLOB = "f61abfefbc1959502cdb3a9f9e3330774db2914d"
EXPECTED_TRAINER_BLOB = "004b66b549d3d64f2dde7614ec84f22b9f37a7c6"
EXPECTED_SCAN_INTEGRATION_BLOB = "c3d05f8b4d1e3e8ea7bcbad5fdeca5d7091651c7"
EXPECTED_TRITON_CANDIDATE_BLOB = "5c6bf7df95f49c306b4ffadd6a8a09b9f71344f9"
EXPECTED_GROUPED_MOE_BLOB = "7c738e6349536dd45f1c631a9ad5524ade1d5021"


def test_v11_runner_freezes_single_use_paid_boundary_and_budget() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    required = (
        'PHASE = "preflight-v11-compiled-explicit-fp32-ce"',
        'TRIGGER_TITLE = "[modal-cortex-s-100m-2b-preflight-v11-compiled-explicit-fp32-ce]"',
        'ENGINEERING_SEED = 2_026_091_013',
        'RESULT_ROOT = "/vol/cortex-s-v0/100m-2b/preflight-v11-compiled-explicit-fp32-ce"',
        'SYSTEMS_VARIANT = "compiled_explicit_fp32_ce_v1"',
        'LOSS_COMPILE_MODE = "max-autotune-no-cudagraphs"',
        'LOSS_FULLGRAPH = True',
        'EXPECTED_CORTEX_PARAMS = 101_778_112',
        'TOKENS_PER_STEP = MICRO_BATCH_SIZE * SEQ_LEN * GRAD_ACCUM_STEPS',
        'CALIBRATION_WARMUP_STEPS = 5',
        'MEASURED_STEPS = 40',
        'MEASURED_TOKENS = MEASURED_STEPS * TOKENS_PER_STEP',
        'PROJECTION_OVERHEAD_MULTIPLIER = 1.10',
        'MAX_PROJECTED_FULL_SECONDS = 8_500.0',
        'MAX_PREFLIGHT_VRAM_GIB = 70.0',
        'with memory_lean_grouped_triton_scan_training_builder():',
        'with compiled_explicit_fp32_ce_training_builder():',
        'training_module.run_h100_calibration(',
        'training_module.CALIBRATION_SEED = ENGINEERING_SEED',
        '"semantic_proof_after_timed_calibration": True',
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
    assert "2_026_091_012" in source
    assert "8_100, 48_131, 48_132, 48_133" in source
    assert 'if "liger" in module_source.lower():' not in source
    for marker in ("liger_kernel", "ligerfused", "from liger", "import liger"):
        assert marker in source
    assert '"liger_used": False' in source


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
    assert '_atomic_write(result_path, result)' in h100
    assert '_atomic_write(result_path, failure)' in h100
    assert h100.count("volume.commit()") >= 3
    assert '"partial_calibration": _raw_calibration_summary(raw)' in h100


def test_v11_candidate_and_production_semantics_are_frozen() -> None:
    candidate = CANDIDATE.read_text(encoding="utf-8")
    trainer = TRAINER.read_text(encoding="utf-8")

    assert 'SYSTEMS_VARIANT = "compiled_explicit_fp32_ce_v1"' in candidate
    assert 'LOSS_COMPILE_MODE = "max-autotune-no-cudagraphs"' in candidate
    assert "LOSS_FULLGRAPH = True" in candidate
    assert "logits.float().reshape(-1, logits.size(-1))" in candidate
    assert "targets.reshape(-1)" in candidate
    assert "compiled_explicit_fp32_cross_entropy = torch.compile(" in candidate
    assert "mode=LOSS_COMPILE_MODE" in candidate
    assert "fullgraph=LOSS_FULLGRAPH" in candidate
    assert "compiled_explicit_fp32_cross_entropy(logits, y)" in candidate
    assert candidate.count('.to(device="cpu")') == 1
    lowered = candidate.lower()
    assert "liger_kernel" not in lowered
    assert "ligerfused" not in lowered
    assert "from liger" not in lowered
    assert "import liger" not in lowered
    assert '"liger_used": false' in lowered

    assert "return torch.compile(model, mode=COMPILE_MODE, fullgraph=False)" in trainer
    assert "logits.float().reshape(-1, logits.size(-1))" in trainer
    assert hashlib.sha256(TRAINER.read_bytes()).hexdigest() == EXPECTED_TRAINER_SHA256


def test_compile_timer_includes_first_use_of_model_and_loss_compilers() -> None:
    trainer = TRAINER.read_text(encoding="utf-8")
    calibration = trainer[trainer.index("def run_h100_calibration("):trainer.index("def _save_checkpoint(")]

    assert calibration.index("compile_start = time.perf_counter()") < calibration.index("runner = _compile_model(model)")
    assert calibration.index("runner = _compile_model(model)") < calibration.index("_one_optimizer_step(")
    assert calibration.index("_one_optimizer_step(") < calibration.index("compile_seconds = max(time.perf_counter() - compile_start, 0.0)")

    runner = RUNNER.read_text(encoding="utf-8")
    h100 = runner[runner.index("def h100_preflight("):runner.index("@app.local_entrypoint()")]
    assert h100.index("with memory_lean_grouped_triton_scan_training_builder():") < h100.index(
        "with compiled_explicit_fp32_ce_training_builder():"
    ) < h100.index("training_module.run_h100_calibration(")


def test_cuda_semantic_proof_is_actual_compiled_loss_and_after_timing() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    proof = source[source.index("def _prove_compiled_explicit_fp32_ce_cuda("):source.index("def _raw_calibration_summary(")]
    h100 = source[source.index("def h100_preflight("):source.index("@app.local_entrypoint()")]

    assert "candidate.compiled_explicit_fp32_cross_entropy(candidate_logits, targets)" in proof
    assert "reference_logits.float().reshape(-1, reference_logits.size(-1))" in proof
    assert "reference_loss.backward()" in proof
    assert "candidate_loss.backward()" in proof
    assert "torch.allclose(candidate_loss, reference_loss, rtol=1e-6, atol=1e-6)" in proof
    assert "torch.allclose(candidate_logits.grad, reference_logits.grad, rtol=2e-3, atol=2e-3)" in proof
    assert h100.index("training_module.run_h100_calibration(") < h100.index(
        "_prove_compiled_explicit_fp32_ce_cuda(device)"
    )


def test_dispatcher_proof_is_24_triton_worlds_and_zero_fallback() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    proof = source[source.index("def _prove_triton_dispatch("):source.index("def _prove_compiled_explicit_fp32_ce_cuda(")]

    assert "integration.affine_scan_triton_candidate = counted_candidate" in proof
    assert "integration.affine_scan = forbidden_fallback" in proof
    assert 'calls["candidate"] != EXPECTED_LAYERS' in proof
    assert 'calls["fallback"] != 0' in proof
    assert '"selected_triton_for_all_worlds": True' in proof


def test_workflow_is_owner_only_exact_title_source_bound_and_one_shot() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "types: [opened]" in workflow
    assert "github.event.issue.user.login == github.repository_owner" in workflow
    assert "github.event.issue.title == '[modal-cortex-s-100m-2b-preflight-v11-compiled-explicit-fp32-ce]'" in workflow
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
    assert "modal run --detach --timestamps modal_cortex_s_100m_2b_v11_compiled_explicit_fp32_ce_preflight.py" in workflow
    assert workflow.count("modal run --detach --timestamps") == 1
    assert "--phase \"preflight-v11-compiled-explicit-fp32-ce\"" in workflow


def test_harness_has_no_full_training_or_scientific_authority() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert '"full_training_authorized": True' not in source
    assert '"scientific_claim_authorized": True' not in source
    assert '"seed_8100_used": True' not in source
    assert '"reserved_scientific_seeds_used": True' not in source
    assert "PAIRED_SEED" not in source
    assert "train_full_2b(" not in source
    assert "FORBIDDEN_CONTROL_AND_SCIENTIFIC_SEEDS = (8_100, 48_131, 48_132, 48_133)" in source
    assert "workflow_dispatch" not in workflow
    assert CANDIDATE_TEST.exists()