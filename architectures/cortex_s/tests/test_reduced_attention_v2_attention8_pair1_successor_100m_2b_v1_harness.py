from pathlib import Path

from architectures.cortex_s.reduced_attention_100m_v2_attention8 import (
    ATTENTION_LAYERS_ONE_BASED,
    EXPECTED_PARAMETERS,
)
from architectures.cortex_s.reduced_attention_100m_v2_attention8_successor_protocol import (
    MIN_TRAINING_TPS_RATIO_FOR_PROGRESSION,
    PAIR_SCREEN_MAX_NLL_DELTA,
    PAIR_SEED,
)


ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / "modal_cortex_reduced_attention_v2_attention8_pair1_successor_100m_2b_v1.py"
WORKFLOW = (
    ROOT
    / ".github"
    / "workflows"
    / "modal-cortex-reduced-attention-v2-attention8-pair1-successor-100m-2b-v1.yml"
)


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_attention8_v2_pair1_successor_runner_freezes_identity_and_geometry() -> None:
    runner = _text(RUNNER)
    assert 'PHASE = "reduced-attention-v2-attention8-pair1-successor-100m-2b-v1"' in runner
    assert (
        'TRIGGER_TITLE = "[modal-cortex-reduced-attention-v2-attention8-pair1-successor-100m-2b-v1]"'
        in runner
    )
    assert "PAIR_SEED = 60_232" in runner
    assert 'REQUIRED_MODAL_ACCOUNT = "primary"' in runner
    assert "prior_pair_seed_consumed" in runner
    assert "60_231" in runner
    assert (
        "/vol/cortex-s-v0/100m-2b/"
        "reduced-attention-v2-attention8-pair1-successor-v1"
        in runner
    )
    assert "TOTAL_OPTIMIZER_STEPS = 30_518" in runner
    assert "FULL_BATCH_TOKEN_EXPOSURES = 2_000_027_648" in runner
    assert "PAIR_SCREEN_MAX_NLL_DELTA = 0.015" in runner
    assert "MIN_TRAINING_TPS_RATIO_FOR_PROGRESSION = 1.03" in runner
    assert PAIR_SEED == 60_232
    assert PAIR_SCREEN_MAX_NLL_DELTA == 0.015
    assert MIN_TRAINING_TPS_RATIO_FOR_PROGRESSION == 1.03


def test_attention8_v2_pair1_successor_candidate_schedule_and_parameters_remain_frozen() -> None:
    assert ATTENTION_LAYERS_ONE_BASED == (
        3,
        6,
        9,
        12,
        15,
        18,
        21,
        24,
    )
    assert EXPECTED_PARAMETERS == 101_795_328


def test_attention8_v2_pair1_successor_requires_durable_engineering_pass() -> None:
    runner = _text(RUNNER)
    verifier = runner.split("def _verify_engineering_prerequisite()", 1)[1].split(
        "@app.function(", 1
    )[0]
    assert 'ENGINEERING_ROOT = "/vol/cortex-s-v0/100m-200m/reduced-attention-v2-panel-v1"' in runner
    assert '"status": "ENGINEERING_PANEL_COMPLETE"' in verifier
    assert '"classification": "ENGINEERING_PANEL_CANDIDATE_SCREEN_PASS"' in verifier
    assert 'result.get("progression_candidates") != ["attention_8"]' in verifier
    assert 'attention8.get("combined_gate_pass") is not True' in verifier
    assert 'attention8.get("quality_gate_pass") is not True' in verifier
    assert 'attention8.get("throughput_gate_pass") is not True' in verifier


def test_attention8_v2_pair1_successor_runner_is_single_attempt_and_non_escalating() -> None:
    runner = _text(RUNNER)
    assert runner.count('gpu="H100!"') == 1
    assert "retries=0" in runner
    assert "torch.load(" not in runner
    assert '"automatic_retry_authorized": False' in runner
    assert '"resume_authorized": False' in runner
    assert '"replication_authorized": False' in runner
    assert '"replication_seeds_authorized": False' in runner
    assert '"250m_5b_authorized": False' in runner
    assert '"breakthrough_claim_allowed": False' in runner


def test_attention8_v2_pair1_successor_reservation_consumes_seed_before_h100() -> None:
    runner = _text(RUNNER)
    reservation = runner.split("def reserve_pair_dispatch(", 1)[1].split(
        "def _one_optimizer_step(", 1
    )[0]
    assert '"status": "PAIR1_DISPATCH_RESERVED"' in reservation
    assert '"pair_seed_consumed": True' in reservation
    assert '"h100_allocation_started": False' in reservation
    assert "volume.commit()" in reservation


def test_attention8_v2_pair1_successor_fail_closes_after_prior_model_failure() -> None:
    runner = _text(RUNNER)
    main = runner.split("@app.local_entrypoint()", 1)[1]
    assert "prior_complete = True" in main
    assert "if not prior_complete:" in main
    assert "prior model did not complete; fail closed to avoid " in main
    assert "additional scientific spend" in main
    assert 'result.get("status") == "COMPLETE"' in main


def test_attention8_v2_pair1_successor_finalizer_has_quality_and_systems_gate() -> None:
    runner = _text(RUNNER)
    finalizer = runner.split("def finalize_pair(", 1)[1].split(
        "@app.local_entrypoint()", 1
    )[0]
    assert "quality_pass = finite and delta <= PAIR_SCREEN_MAX_NLL_DELTA" in finalizer
    assert (
        "systems_pass = finite and ratio >= MIN_TRAINING_TPS_RATIO_FOR_PROGRESSION"
        in finalizer
    )
    assert '"SCIENTIFIC_ATTENTION8_V2_PAIR1_SCREEN_PASS"' in finalizer
    assert '"SCIENTIFIC_ATTENTION8_V2_PAIR1_QUALITY_PASS_SYSTEMS_FAIL"' in finalizer
    assert '"SCIENTIFIC_ATTENTION8_V2_PAIR1_SCREEN_FAIL"' in finalizer
    assert '"replication_authorized": False' in finalizer
    assert '"250m_5b_authorized": False' in finalizer
    assert '"breakthrough_claim_allowed": False' in finalizer


def test_attention8_v2_pair1_successor_workflow_is_one_shot_source_bound_primary_only() -> None:
    workflow = _text(WORKFLOW)
    top = workflow.split("\njobs:", 1)[0]
    assert "workflow_dispatch:" not in top
    assert (
        "[modal-cortex-reduced-attention-v2-attention8-pair1-successor-100m-2b-v1]"
        in workflow
    )
    assert 'test "$RUN_ATTEMPT" = "1"' in workflow
    assert 'git merge-base --is-ancestor "$SOURCE_SHA" "$LIVE_MAIN"' in workflow
    assert (
        'test "$(git rev-parse "$SOURCE_SHA:$path")" = '
        '"$(git rev-parse "$LIVE_MAIN:$path")"'
        in workflow
    )
    assert "--force-account primary" in workflow
    assert 'test "$SELECTED_ACCOUNT" = "primary"' in workflow
    for forbidden in (
        "MODAL_TOKEN_ID_SECONDARY",
        "MODAL_TOKEN_SECRET_SECONDARY",
        "MODAL_TOKEN_ID_2",
        "MODAL_TOKEN_SECRET_2",
        "MODAL_TOKEN_ID_B",
        "MODAL_TOKEN_SECRET_B",
        "--force-account secondary",
    ):
        assert forbidden not in workflow
    launch = workflow.split(
        "\n      - name: Launch exact one-shot attention-8 v2 Pair-1", 1
    )[1].split(
        "\n      - name: Record normal scientific Pair-1 Modal return only", 1
    )[0]
    assert launch.count("modal run --detach --timestamps") == 1


def test_attention8_v2_pair1_successor_workflow_protects_exact_execution_files() -> None:
    workflow = _text(WORKFLOW)
    protected = (
        "architectures/cortex_s/reduced_attention_100m_v2_attention8.py",
        "architectures/cortex_s/reduced_attention_100m_v2_attention8_successor_protocol.py",
        "architectures/cortex_s/tests/test_reduced_attention_v2_attention8_pair1_successor_100m_2b_v1_harness.py",
        "tam_research/models.py",
        "tam_research/train.py",
        "tam_research/data.py",
        "tam_research/modal_dual_account_v3.py",
        "scripts/modal_select_account_v3.py",
        "modal_runtime_admission_probe_1067_v1.py",
        "modal_cortex_reduced_attention_v2_attention8_pair1_successor_100m_2b_v1.py",
        ".github/workflows/modal-cortex-reduced-attention-v2-attention8-pair1-successor-100m-2b-v1.yml",
    )
    loop = workflow.split("for path in", 1)[1].split("; do", 1)[0]
    for path in protected:
        assert path in loop, path
