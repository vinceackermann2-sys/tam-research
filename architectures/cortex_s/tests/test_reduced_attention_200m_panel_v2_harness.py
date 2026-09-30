from pathlib import Path

from architectures.cortex_s.reduced_attention_200m_panel_v2 import (
    ATTENTION_8,
    EARLY_4,
    V1_REPLICATE,
)
from architectures.cortex_s.reduced_attention_200m_panel_v2_protocol import (
    ENGINEERING_SEED,
    FULL_BATCH_TOKEN_EXPOSURES,
    MAX_200M_NLL_DELTA,
    MIN_TRAINING_TPS_RATIO,
    TOTAL_OPTIMIZER_STEPS,
)


ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / "modal_cortex_reduced_attention_200m_panel_v2.py"
WORKFLOW = (
    ROOT
    / ".github"
    / "workflows"
    / "modal-cortex-reduced-attention-200m-panel-v2.yml"
)


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_engineering_panel_runner_freezes_identity_geometry_and_order() -> None:
    runner = _text(RUNNER)
    assert 'PHASE = "reduced-attention-v2-100m-200m-engineering-panel"' in runner
    assert (
        'TRIGGER_TITLE = "[modal-cortex-reduced-attention-200m-engineering-panel-v2]"'
        in runner
    )
    assert "ENGINEERING_SEED = 2_026_092_901" in runner
    assert "TOTAL_OPTIMIZER_STEPS = 3_052" in runner
    assert "FULL_BATCH_TOKEN_EXPOSURES = 200_015_872" in runner
    assert "MAX_200M_NLL_DELTA = 0.025" in runner
    assert "MIN_TRAINING_TPS_RATIO = 1.05" in runner
    assert 'MODEL_ORDER = (' in runner
    for name in (
        '"fresh_transformer"',
        '"v1_replicate"',
        '"early_4"',
        '"attention_8"',
    ):
        assert name in runner

    assert ENGINEERING_SEED == 2_026_092_901
    assert TOTAL_OPTIMIZER_STEPS == 3_052
    assert FULL_BATCH_TOKEN_EXPOSURES == 200_015_872
    assert MAX_200M_NLL_DELTA == 0.025
    assert MIN_TRAINING_TPS_RATIO == 1.05


def test_engineering_panel_runner_is_single_attempt_and_non_scientific() -> None:
    runner = _text(RUNNER)
    assert runner.count('gpu="H100!"') == 1
    assert "retries=0" in runner
    assert "torch.load(" not in runner
    assert '"automatic_retry_authorized": False' in runner
    assert '"resume_authorized": False' in runner
    assert '"scientific_training_authorized": False' in runner
    assert '"scientific_seed_consumed": False' in runner
    assert '"replication_authorized": False' in runner
    assert '"250m_5b_authorized": False' in runner
    assert '"breakthrough_claim_allowed": False' in runner
    assert "SCIENTIFIC_PAIR_SEEDS" not in runner
    assert "58_232" not in runner
    assert "58_233" not in runner
    assert "59_231" not in runner


def test_engineering_panel_reservation_consumes_only_engineering_seed() -> None:
    runner = _text(RUNNER)
    reservation = runner.split("def reserve_panel_dispatch(", 1)[1].split(
        "def _one_optimizer_step(", 1
    )[0]
    assert '"status": "ENGINEERING_PANEL_DISPATCH_RESERVED"' in reservation
    assert '"engineering_seed_consumed": True' in reservation
    assert '"scientific_seed_consumed": False' in reservation
    assert '"h100_allocation_started": False' in reservation
    assert "volume.commit()" in reservation


def test_engineering_panel_fail_closes_after_any_model_failure() -> None:
    runner = _text(RUNNER)
    main = runner.split("@app.local_entrypoint()", 1)[1]
    assert "prior_complete = True" in main
    assert "if not prior_complete:" in main
    assert "prior model did not complete; fail closed to avoid " in main
    assert "additional engineering spend" in main
    assert 'result.get("status") == "COMPLETE"' in main


def test_engineering_panel_finalizer_has_seed_sensitivity_guard() -> None:
    runner = _text(RUNNER)
    finalizer = runner.split("def finalize_panel(", 1)[1].split(
        "@app.local_entrypoint()", 1
    )[0]
    assert 'comparisons["v1_replicate"]["quality_gate_pass"]' in finalizer
    assert (
        '"ENGINEERING_PANEL_SEED_SENSITIVITY_DIAGNOSTIC_ONLY"'
        in finalizer
    )
    assert '"ENGINEERING_PANEL_CANDIDATE_SCREEN_PASS"' in finalizer
    assert '"ENGINEERING_PANEL_NO_CANDIDATE_PASSES"' in finalizer
    assert '"progression_candidates": progression_candidates' in finalizer
    assert '"scientific_training_authorized": False' in finalizer


def test_engineering_panel_workflow_is_one_shot_source_bound_and_secondary_only() -> None:
    workflow = _text(WORKFLOW)
    top = workflow.split("\njobs:", 1)[0]
    assert "workflow_dispatch:" not in top
    assert (
        "[modal-cortex-reduced-attention-200m-engineering-panel-v2]"
        in workflow
    )
    assert 'test "$RUN_ATTEMPT" = "1"' in workflow
    assert 'git merge-base --is-ancestor "$SOURCE_SHA" "$LIVE_MAIN"' in workflow
    assert (
        'test "$(git rev-parse "$SOURCE_SHA:$path")" = '
        '"$(git rev-parse "$LIVE_MAIN:$path")"'
        in workflow
    )
    assert "--force-account secondary" in workflow
    assert 'test "$SELECTED_ACCOUNT" = "secondary"' in workflow
    assert workflow.count("modal run --detach --timestamps") == 2
    launch = workflow.split(
        "\n      - name: Launch exact one-shot 200M engineering panel", 1
    )[1].split(
        "\n      - name: Record normal engineering-panel Modal return only", 1
    )[0]
    assert launch.count("modal run --detach --timestamps") == 1


def test_engineering_panel_workflow_protects_exact_execution_files() -> None:
    workflow = _text(WORKFLOW)
    protected = (
        "architectures/cortex_s/reduced_attention_200m_panel_v2.py",
        "architectures/cortex_s/reduced_attention_200m_panel_v2_protocol.py",
        "architectures/cortex_s/tests/test_reduced_attention_200m_panel_v2_harness.py",
        "tam_research/models.py",
        "tam_research/train.py",
        "tam_research/data.py",
        "tam_research/modal_dual_account_v3.py",
        "scripts/modal_select_account_v3.py",
        "modal_runtime_admission_probe_1067_v1.py",
        "modal_cortex_reduced_attention_200m_panel_v2.py",
        ".github/workflows/modal-cortex-reduced-attention-200m-panel-v2.yml",
    )
    loop = workflow.split("for path in", 1)[1].split("; do", 1)[0]
    for path in protected:
        assert path in loop, path


def test_engineering_panel_architecture_schedules_remain_frozen() -> None:
    assert V1_REPLICATE.attention_layers_one_based == (6, 12, 18, 24)
    assert EARLY_4.attention_layers_one_based == (1, 8, 16, 24)
    assert ATTENTION_8.attention_layers_one_based == (
        3,
        6,
        9,
        12,
        15,
        18,
        21,
        24,
    )
