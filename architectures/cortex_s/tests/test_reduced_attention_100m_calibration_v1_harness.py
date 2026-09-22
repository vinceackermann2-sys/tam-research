from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER = REPO_ROOT / "modal_cortex_reduced_attention_100m_calibration_v1.py"
WORKFLOW = (
    REPO_ROOT
    / ".github/workflows/modal-cortex-reduced-attention-100m-calibration-v1.yml"
)


def test_runner_freezes_engineering_only_identity_and_geometry() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    tree = ast.parse(source)

    assert (
        'TRIGGER_TITLE = "[modal-cortex-reduced-attention-100m-calibration-v1]"'
        in source
    )
    assert "ENGINEERING_SEED = 2_026_092_001" in source
    assert (
        'RESULT_ROOT = "/vol/cortex-s-v0/100m-2b/reduced-attention-calibration-v1"'
        in source
    )
    assert "SEQ_LEN = 512" in source
    assert "MICRO_BATCH_SIZE = 64" in source
    assert "GRAD_ACCUM_STEPS = 2" in source
    assert "TOKENS_PER_STEP = SEQ_LEN * MICRO_BATCH_SIZE * GRAD_ACCUM_STEPS" in source
    assert "CALIBRATION_WARMUP_STEPS = 5" in source
    assert "MEASURED_STEPS = 40" in source
    assert 'COMPILE_MODE = "max-autotune-no-cudagraphs"' in source
    assert "HARD_FULL_TIMEOUT_SECONDS = 10_000.0" in source
    assert "MAX_PREFLIGHT_VRAM_GIB = 70.0" in source
    assert source.count('gpu="H100!"') == 1
    assert "torch.save" not in source
    assert "latest.pt" not in source
    assert "resume" not in source.lower()

    function_names = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert {
        "verify_zero_gpu",
        "reserve_dispatch",
        "h100_calibration",
        "_one_optimizer_step",
        "_calibrate_one",
    } <= function_names


def test_runner_uses_same_training_step_for_both_parameter_matched_models() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "EXPECTED_TRANSFORMER_PARAMS = 101_803_520" in source
    assert "EXPECTED_REDUCED_ATTENTION_PARAMS = 101_799_424" in source
    assert 'name="transformer"' in source
    assert 'name="reduced_attention"' in source
    assert "builder=_build_transformer" in source
    assert "builder=_build_reduced_attention" in source
    assert "torch.optim.AdamW(" in source
    assert "betas=ADAMW_BETAS" in source
    assert "weight_decay=WEIGHT_DECAY" in source
    assert "fused=True" in source
    assert "torch.compile(" in source
    assert "mode=COMPILE_MODE" in source
    assert "fullgraph=False" in source
    assert 'torch.autocast(device_type="cuda", dtype=torch.bfloat16)' in source
    assert "logits.float().reshape(-1, logits.size(-1))" in source
    assert "torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)" in source
    assert "measured_generator = torch.Generator(device="cpu").manual_seed(" in source
    assert "ENGINEERING_SEED + 10_000" in source


def test_runner_does_not_use_reserved_scientific_seeds_or_write_checkpoints() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "SCIENTIFIC_PAIR_SEEDS" not in source
    for forbidden_call in (
        "manual_seed(58_231",
        "manual_seed(58_232",
        "manual_seed(58_233",
    ):
        assert forbidden_call not in source
    assert "scientific_seed_used" in source
    assert '"scientific_seeds_consumed": False' in source
    assert '"checkpoint_written": False' in source
    assert '"training_performed": False' in source
    assert '"250m_5b_authorized": False' in source


def test_runner_orders_zero_gpu_then_reservation_then_h100() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    main_source = source.split("@app.local_entrypoint()", 1)[1]
    zero_index = main_source.index("verify_zero_gpu.remote(")
    reserve_index = main_source.index("reserve_dispatch.remote(")
    h100_index = main_source.index("h100_calibration.remote(")
    assert zero_index < reserve_index < h100_index
    assert '"h100_allocation_started": False' in source
    assert 'marker["h100_allocation_started"] = True' in source
    assert "_atomic_write(marker_path, marker)" in source
    assert "volume.commit()" in source


def test_workflow_is_owner_only_exact_title_source_bound_and_single_launch() -> None:
    if not WORKFLOW.exists():
        return
    source = WORKFLOW.read_text(encoding="utf-8")
    top = source.split("\njobs:", 1)[0]
    assert "workflow_dispatch:" not in top
    assert "issues:" in top and "types: [opened]" in top
    assert (
        "github.event.issue.title == "
        "'[modal-cortex-reduced-attention-100m-calibration-v1]'"
        in source
    )
    assert "github.event.issue.user.login == github.repository_owner" in source
    assert "set(body) != expected" in source
    assert "git merge-base --is-ancestor" in source
    assert "git diff --exit-code" in source

    launch = source.split(
        "\n      - name: Launch exact one-shot reduced-attention calibration", 1
    )[1]
    launch = launch.split("\n      - name: Record normal Modal return only", 1)[0]
    assert launch.count("modal run --detach --timestamps") == 1
    assert "modal_cortex_reduced_attention_100m_calibration_v1.py" in launch
