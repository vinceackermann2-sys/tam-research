from __future__ import annotations

import ast
import hashlib
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / "modal_cortex_s_100m_2b_v11_compiled_explicit_fp32_ce_full.py"
WORKFLOW = ROOT / ".github/workflows/modal-cortex-s-100m-2b-full-v11-compiled-explicit-fp32-ce.yml"
TRAINER = ROOT / "architectures/cortex_s/experiments/scale100m_2b/train.py"
PROTOCOL = ROOT / "architectures/cortex_s/experiments/scale100m_2b/protocol.py"
CANDIDATE = ROOT / "architectures/cortex_s/compiled_explicit_fp32_ce_v1.py"
SCAN = ROOT / "architectures/cortex_s/production_scan_integration_v1.py"

EXPECTED_RUNNER_BLOB = "f7ff64368787401f97c6c0df7e553461c368bc8d"
EXPECTED_CANDIDATE_BLOB = "970a0dcf130348662f4c9a3ed11d8e6ad250781a"
EXPECTED_CANDIDATE_TEST_BLOB = "f61abfefbc1959502cdb3a9f9e3330774db2914d"
EXPECTED_TRAINER_BLOB = "004b66b549d3d64f2dde7614ec84f22b9f37a7c6"
EXPECTED_SCAN_BLOB = "c3d05f8b4d1e3e8ea7bcbad5fdeca5d7091651c7"
EXPECTED_TRITON_BLOB = "5c6bf7df95f49c306b4ffadd6a8a09b9f71344f9"
EXPECTED_GROUPED_MOE_BLOB = "7c738e6349536dd45f1c631a9ad5524ade1d5021"
EXPECTED_TRAINER_SHA256 = "d0d91de09bba8b7631913b3e9c0407ab2bb0114f7bba133c123ee5598d490461"


def test_full_v11_runner_freezes_one_shot_identity_and_authoritative_preflight() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    required = (
        'PHASE = "full-v11-compiled-explicit-fp32-ce"',
        'TRIGGER_TITLE = "[modal-cortex-s-100m-2b-full-v11-compiled-explicit-fp32-ce]"',
        'FULL_TRAINING_SEED = 2_026_091_014',
        'RESULT_ROOT = "/vol/cortex-s-v0/100m-2b/full-v11-compiled-explicit-fp32-ce"',
        'V11_PREFLIGHT_ROOT = "/vol/cortex-s-v0/100m-2b/preflight-v11-compiled-explicit-fp32-ce"',
        'V11_PREFLIGHT_SOURCE_SHA = "be5c0c19c7ebf3339e0c8be80158eb52b1a3dae2"',
        'V11_PREFLIGHT_SOURCE_TREE = "2820a74ad1ea4e694cbf0a08ce42011e9e483d6e"',
        'V11_PREFLIGHT_HARNESS_SHA = "dcbec42960139757bf0cc528a099edea433edd66"',
        'V11_PREFLIGHT_SEED = 2_026_091_013',
        'V11_PREFLIGHT_STATUS = "SYSTEMS_PREFLIGHT_PASS"',
        'V11_PREFLIGHT_CLASSIFICATION = "ENGINEERING_SYSTEMS_PREFLIGHT_V11_COMPILED_EXPLICIT_FP32_CE_ONLY"',
        'V11_EXPECTED_TPS = 287_002.5286255846',
        'V11_EXPECTED_COMPILE_SECONDS = 371.187105083',
        'V11_EXPECTED_PROJECTED_FULL_SECONDS = 8_036.624794902304',
        'V11_EXPECTED_PEAK_VRAM_GIB = 37.06346607208252',
    )
    for needle in required:
        assert needle in source, needle
    assert 'Path(V11_PREFLIGHT_ROOT) / "RESULT.json"' in source
    assert '_atomic_write(Path(V11_PREFLIGHT_ROOT)' not in source
    assert '"full_training_authorized": False' in source
    assert '"seed_8100_used": False' in source
    assert '"reserved_scientific_seeds_used": False' in source
    assert 'dispatcher.get("candidate_calls") != 24' in source
    assert 'dispatcher.get("legacy_fallback_calls") != 0' in source


def test_full_training_geometry_matches_frozen_protocol() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    protocol = PROTOCOL.read_text(encoding="utf-8")
    required = (
        'EXPECTED_CORTEX_PARAMS = 101_778_112',
        'TRAIN_TOKENS = 2_000_000_000',
        'SEQ_LEN = 512',
        'MICRO_BATCH_SIZE = 64',
        'GRAD_ACCUM_STEPS = 2',
        'TOKENS_PER_STEP = MICRO_BATCH_SIZE * SEQ_LEN * GRAD_ACCUM_STEPS',
        'TOTAL_OPTIMIZER_STEPS = 30_518',
        'FULL_BATCH_TOKEN_EXPOSURES = 2_000_027_648',
        'EVAL_EVERY_TOKENS = 200_000_000',
        'CHECKPOINT_EVERY_TOKENS = 200_000_000',
        'LEARNING_RATE = 3e-4',
        'WEIGHT_DECAY = 0.1',
        'WARMUP_RATIO = 0.02',
        'MODEL_COMPILE_MODE = "max-autotune-no-cudagraphs"',
        'LOSS_COMPILE_MODE = "max-autotune-no-cudagraphs"',
        'LOSS_FULLGRAPH = True',
        'HARD_FULL_TIMEOUT_SECONDS = 10_000',
    )
    for needle in required:
        assert needle in source, needle
    assert 30_518 == math.ceil(2_000_000_000 / 65_536)
    assert 30_518 * 65_536 == 2_000_027_648
    assert "TOTAL_OPTIMIZER_STEPS = math.ceil(TRAIN_TOKENS / TOKENS_PER_OPTIMIZER_STEP)" in protocol
    assert "EVAL_EVERY_TOKENS = 200_000_000" in protocol
    assert "CHECKPOINT_EVERY_TOKENS = 200_000_000" in protocol


def test_full_loop_reuses_frozen_training_primitives_and_v11_builders() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    body = source[source.index("def _run_full_training("):source.index("@app.function(\n    image=base_image,\n    gpu=\"H100!\"")]
    assert "with memory_lean_grouped_triton_scan_training_builder():" in body
    assert "with compiled_explicit_fp32_ce_training_builder():" in body
    assert "training_module.build_cortex_100m().to(device)" in body
    assert "training_module._make_optimizer(model)" in body
    assert "training_module._compile_model(model)" in body
    assert "training_module._one_optimizer_step(" in body
    assert "training_module.evaluate(" in body
    assert "cosine_lr(step, TOTAL_OPTIMIZER_STEPS, warmup_steps, LEARNING_RATE)" in body
    assert "torch.nn.utils.clip_grad_norm_" not in body  # clipping stays inside the frozen v11 optimizer step
    assert "train_full_2b(" not in source
    assert "PAIRED_SEED" not in source

    candidate = CANDIDATE.read_text(encoding="utf-8")
    assert "logits.float().reshape(-1, logits.size(-1))" in candidate
    assert "compiled_explicit_fp32_cross_entropy = torch.compile(" in candidate
    assert "mode=LOSS_COMPILE_MODE" in candidate
    assert "fullgraph=LOSS_FULLGRAPH" in candidate
    assert candidate.count('.to(device="cpu")') == 1

    scan = SCAN.read_text(encoding="utf-8")
    assert "def memory_lean_grouped_triton_scan_training_builder()" in scan
    assert "build_memory_lean_grouped_triton_scan_cortex_100m" in scan


def test_warmup_does_not_advance_actual_training_generator() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    body = source[source.index("def _run_full_training("):source.index("@app.function(\n    image=base_image,\n    gpu=\"H100!\"")]
    assert 'warm_generator = torch.Generator(device="cpu").manual_seed(FULL_TRAINING_SEED + 99_999)' in body
    assert "training_module.seed_all(FULL_TRAINING_SEED)" in body
    assert body.count("training_module.build_cortex_100m().to(device)") == 2
    assert 'generator = torch.Generator(device="cpu").manual_seed(FULL_TRAINING_SEED + 10_000)' in body
    assert body.index("warm_generator =") < body.index("training_module.seed_all(FULL_TRAINING_SEED)", body.index("warm_generator ="))
    assert body.index('generator = torch.Generator(device="cpu").manual_seed(FULL_TRAINING_SEED + 10_000)') > body.index(
        "training_module.seed_all(FULL_TRAINING_SEED)", body.index("warm_generator =")
    )


def test_fresh_seed_is_only_training_seed_and_forbidden_seeds_cannot_run() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "FORBIDDEN_CONTROL_AND_SCIENTIFIC_SEEDS = (8_100, 48_131, 48_132, 48_133)" in source
    assert "2_026_091_013" in source
    assert "FULL_TRAINING_SEED in forbidden" in source
    assert "seed_all(8_100)" not in source
    assert "manual_seed(8_100" not in source
    assert "seed=8_100" not in source
    assert "seed_all(48_131)" not in source
    assert "seed_all(48_132)" not in source
    assert "seed_all(48_133)" not in source
    assert "training_module.seed_all(FULL_TRAINING_SEED)" in source
    assert '"seed": FULL_TRAINING_SEED' in source


def test_checkpoint_and_failure_semantics_are_one_shot_no_resume() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    reserve = source[source.index("def reserve_full_dispatch("):source.index("def _save_v11_checkpoint(")]
    training = source[source.index("def _run_full_training("):source.index("@app.function(\n    image=base_image,\n    gpu=\"H100!\"")]
    h100 = source[source.index("def h100_full_training("):source.index("@app.local_entrypoint()")]
    main = source[source.index("def main("):]

    assert '_atomic_write(marker_path, marker)' in reserve
    assert "volume.commit()" in reserve
    assert main.index("reserve_full_dispatch.remote") < main.index("h100_full_training.remote")
    assert 'marker["h100_allocation_started"] = True' in h100
    assert h100.index('_atomic_write(marker_path, marker)') < h100.index("_run_full_training(source)")
    assert 'started_path = run_dir / "ATTEMPT_STARTED.json"' in training
    assert 'if started_path.exists() or result_path.exists() or success_path.exists():' in training
    assert '"resume_authorized": False' in source
    assert "resume_from" not in source
    assert "load_state_dict" not in source
    assert "latest.pt" in source
    assert "PROGRESS.json" in source
    assert training.count("volume.commit()") >= 2
    assert '_atomic_write(result_path, result)' in h100
    assert '_atomic_write(result_path, failure)' in h100
    assert h100.count("volume.commit()") >= 3


def test_runner_has_exactly_one_h100_site_and_no_profiler() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    gpu_sites: list[str] = []
    profiler_calls: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = ast.unparse(node.func)
        if name.startswith("torch.profiler"):
            profiler_calls.append(name)
        for kw in node.keywords:
            if kw.arg == "gpu" and isinstance(kw.value, ast.Constant):
                gpu_sites.append(kw.value.value)
    assert gpu_sites == ["H100!"]
    assert not profiler_calls
    assert "timeout=HARD_FULL_TIMEOUT_SECONDS" in source
    assert "HARD_FULL_TIMEOUT_SECONDS = 10_000" in source


def test_production_trainer_is_byte_frozen() -> None:
    trainer = TRAINER.read_text(encoding="utf-8")
    assert hashlib.sha256(TRAINER.read_bytes()).hexdigest() == EXPECTED_TRAINER_SHA256
    assert "return torch.compile(model, mode=COMPILE_MODE, fullgraph=False)" in trainer
    assert "fused=True" in trainer
    assert "betas=(0.9, 0.95)" in trainer
    assert "torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)" in trainer


def test_workflow_is_owner_only_source_bound_and_one_shot() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "types: [opened]" in workflow
    assert "github.event.issue.user.login == github.repository_owner" in workflow
    assert "github.event.issue.title == '[modal-cortex-s-100m-2b-full-v11-compiled-explicit-fp32-ce]'" in workflow
    assert "workflow_dispatch" not in workflow
    assert "git -C source rev-parse origin/main" in workflow
    assert "git -C source merge-base --is-ancestor \"$HARNESS_SHA\" \"$SOURCE_SHA\"" in workflow
    for blob in (
        EXPECTED_RUNNER_BLOB,
        EXPECTED_CANDIDATE_BLOB,
        EXPECTED_CANDIDATE_TEST_BLOB,
        EXPECTED_TRAINER_BLOB,
        EXPECTED_SCAN_BLOB,
        EXPECTED_TRITON_BLOB,
        EXPECTED_GROUPED_MOE_BLOB,
    ):
        assert blob in workflow
    assert "modal run --detach --timestamps modal_cortex_s_100m_2b_v11_compiled_explicit_fp32_ce_full.py" in workflow
    assert workflow.count("modal run --detach --timestamps") == 1
    assert '--phase "full-v11-compiled-explicit-fp32-ce"' in workflow
    assert "rerun" not in workflow.lower()
    assert "retry" not in workflow.lower()


def test_no_scientific_or_breakthrough_authority_is_embedded() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert '"scientific_claim_authorized": True' not in source
    assert '"breakthrough_claim_allowed": True' not in source
    assert '"resume_authorized": True' not in source
    assert '"same_seed": False' in source
    assert '"classification": "ENGINEERING_FULL_2B_V11_TRAINING_ONLY"' in source
    assert '"classification": "ENGINEERING_FULL_2B_V11_COMPLETE_NO_SCIENTIFIC_CLAIM"' in source
