from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / "modal_cortex_s_100m_2b_v9_fused_linear_ce_compile_boundary_preflight.py"
WORKFLOW = ROOT / ".github/workflows/modal-cortex-s-100m-2b-preflight-v9-fused-linear-ce-compile-boundary.yml"
CANDIDATE = ROOT / "architectures/cortex_s/fused_linear_ce_v1.py"
TRAINER = ROOT / "architectures/cortex_s/experiments/scale100m_2b/train.py"


def _runner() -> str:
    return RUNNER.read_text(encoding="utf-8")


def _workflow() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def _candidate() -> str:
    return CANDIDATE.read_text(encoding="utf-8")


def test_v9_identity_is_fresh_and_v8_is_retired() -> None:
    source = _runner()
    assert 'PHASE = "preflight-v9-fused-linear-ce-compile-boundary"' in source
    assert 'TRIGGER_TITLE = "[modal-cortex-s-100m-2b-preflight-v9-fused-linear-ce-compile-boundary]"' in source
    assert 'ENGINEERING_SEED = 2_026_091_011' in source
    assert 'RESULT_ROOT = "/vol/cortex-s-v0/100m-2b/preflight-v9-fused-linear-ce-compile-boundary"' in source
    assert '2_026_091_010' in source
    assert '2_026_091_009' in source
    assert 'FORBIDDEN_CONTROL_AND_SCIENTIFIC_SEEDS = (8_100, 48_131, 48_132, 48_133)' in source


def test_candidate_keeps_trunk_compiled_and_only_liger_loss_is_disabled() -> None:
    source = _candidate()
    boundary_start = source.index("@torch.compiler.disable")
    boundary_end = source.index("    def _loss(", boundary_start)
    boundary = source[boundary_start:boundary_end]
    features = source[source.index("def forward_training_features("):source.index("def chunked_linear_cross_entropy_reference(")]
    compiler = source[source.index("def _compile_fused_linear_ce_runner("):source.index("def _normalize_fused_training_runner(")]

    assert 'SYSTEMS_VARIANT = "fused_linear_ce_liger_v2_compile_boundary"' in source
    assert 'COMPILE_BOUNDARY = "liger_loss_eager_dynamo_boundary"' in source
    assert "self._liger_loss(weight, flat_hidden, flat_targets)" in boundary
    assert "forward_training_features" not in boundary
    assert "torch.compiler.disable" not in features
    assert "torch.compile(runner, mode=training_module.COMPILE_MODE, fullgraph=False)" in compiler
    assert "accum_dtype=torch.float32" in source
    assert "return self._liger_loss_eager(weight, flat_hidden, flat_targets)" in source


def test_candidate_normalizes_generic_eager_fallback_without_changing_default_trainer() -> None:
    source = _candidate()
    normalizer = source[source.index("def _normalize_fused_training_runner("):source.index("def _one_optimizer_step_fused_linear_ce(")]
    step = source[source.index("def _one_optimizer_step_fused_linear_ce("):source.index("@contextmanager")]
    trainer = TRAINER.read_text(encoding="utf-8")

    assert "if runner is model:" in normalizer
    assert "return FusedLinearCETrainingRunner(model)" in normalizer
    assert "runner = _normalize_fused_training_runner(model, runner)" in step
    assert "runner(x, y) / training_module.GRAD_ACCUM_STEPS" in step
    assert step.count('.to(device="cpu")') == 1
    assert "runner = model" in trainer
    assert "CortexSLM.forward" not in source


def test_runner_has_one_h100_site_and_no_full_training_or_profiler() -> None:
    source = _runner()
    tree = ast.parse(source)
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
        for keyword in node.keywords:
            if keyword.arg == "gpu" and isinstance(keyword.value, ast.Constant):
                gpu_sites.append(keyword.value.value)
    assert gpu_sites == ["H100!"]
    assert not any(name.endswith("train_full_2b") for name in calls)
    assert profiler_calls == []
    assert '"full_training_authorized": False' in source
    assert '"scientific_claim_authorized": False' in source
    assert '"seed_8100_used": False' in source
    assert '"reserved_scientific_seeds_used": False' in source


def test_runner_preserves_exact_frozen_calibration_contract() -> None:
    source = _runner()
    required = (
        'EXPECTED_CORTEX_PARAMS = 101_778_112',
        'SEQ_LEN = 512',
        'MICRO_BATCH_SIZE = 64',
        'GRAD_ACCUM_STEPS = 2',
        'GLOBAL_BATCH = 128',
        'CALIBRATION_WARMUP_STEPS = 5',
        'MEASURED_STEPS = 40',
        'MEASURED_TOKENS = MEASURED_STEPS * TOKENS_PER_STEP',
        'PROJECTION_OVERHEAD_MULTIPLIER = 1.10',
        'MAX_PROJECTED_FULL_SECONDS = 8_500.0',
        'MAX_PREFLIGHT_VRAM_GIB = 70.0',
        '"projection_formula": "2_000_000_000 / measured_tps * 1.10 + compile_seconds"',
        'raw.get("execution") != "compiled"',
        'training_module.CALIBRATION_SEED = ENGINEERING_SEED',
        '"profiling_inside_timed_calibration": False',
    )
    for needle in required:
        assert needle in source, needle


def test_zero_gpu_dependency_and_candidate_contract_precede_dispatch() -> None:
    source = _runner()
    zero = source[source.index("def verify_zero_gpu("):source.index("def reserve_h100_dispatch(")]
    reserve = source[source.index("def reserve_h100_dispatch("):source.index("def _prove_triton_dispatch(")]
    main = source[source.index("def main("):]

    assert "dependency_smoke = _dependency_smoke()" in zero
    assert "frozen_inputs = _verify_frozen_inputs()" in zero
    assert '"packaging_version_import_ok": True' in source
    assert '"liger_fused_linear_ce_import_ok": True' in source
    assert 'smoke = zero.get("dependency_smoke", {})' in reserve
    assert "dependency import smoke did not pass before dispatch" in reserve
    assert '_atomic_write(marker_path, marker)' in reserve
    assert 'volume.commit()' in reserve
    assert main.index("verify_zero_gpu.remote") < main.index("reserve_h100_dispatch.remote") < main.index("h100_preflight.remote")


def test_h100_path_proves_triton_and_fused_boundary_before_calibration() -> None:
    source = _runner()
    h100 = source[source.index("def h100_preflight("):source.index("@app.local_entrypoint()")]
    assert "dispatcher_proof = _prove_triton_dispatch(proof_model, train_data, device)" in h100
    assert "fused_cuda_proof = _prove_fused_loss_cuda(proof_model)" in h100
    assert h100.index("dispatcher_proof = _prove_triton_dispatch") < h100.index("training_module.run_h100_calibration(")
    assert h100.index("fused_cuda_proof = _prove_fused_loss_cuda") < h100.index("training_module.run_h100_calibration(")
    assert h100.index("with memory_lean_grouped_triton_scan_training_builder():") < h100.index("with fused_linear_ce_training_builder():") < h100.index("training_module.run_h100_calibration(")


def test_workflow_is_owner_only_exact_title_and_pins_repaired_blobs() -> None:
    workflow = _workflow()
    assert "github.event.issue.user.login == github.repository_owner" in workflow
    assert "github.event.issue.title == '[modal-cortex-s-100m-2b-preflight-v9-fused-linear-ce-compile-boundary]'" in workflow
    assert '211e4a2f40338bb454bbd223db6fc072da660d25' in workflow
    assert '24ec7090a680ed1d25096f95ac24a6730518826e' in workflow
    assert '487b5ce53f1a4691ad68b528571c003014050b86' in workflow
    assert '004b66b549d3d64f2dde7614ec84f22b9f37a7c6' in workflow
    assert 'c3d05f8b4d1e3e8ea7bcbad5fdeca5d7091651c7' in workflow
    assert '5c6bf7df95f49c306b4ffadd6a8a09b9f71344f9' in workflow
    assert '7c738e6349536dd45f1c631a9ad5524ade1d5021' in workflow
    assert 'd0d91de09bba8b7631913b3e9c0407ab2bb0114f7bba133c123ee5598d490461' in workflow
    assert "modal run --detach --timestamps modal_cortex_s_100m_2b_v9_fused_linear_ce_compile_boundary_preflight.py" in workflow
