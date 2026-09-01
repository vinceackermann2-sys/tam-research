from __future__ import annotations

from pathlib import Path

from tam_research import aera_v26_3_bf16_correctness_localize as localize
from tam_research import aera_v26_3_ficem_read_probe as probe

ROOT = Path(__file__).resolve().parents[1]
LOCALIZE = ROOT / "tam_research" / "aera_v26_3_bf16_correctness_localize.py"
LAUNCHER = ROOT / "modal_aera_v26_3_bf16_correctness_localize_app.py"
WORKFLOW = ROOT / ".github" / "workflows" / "aera-v26-3-bf16-correctness-localize-l4.yml"
BACKEND = ROOT / "tam_research" / "aera_hardware_core_v26_3_ficem_read_triton.py"
PROBE = ROOT / "tam_research" / "aera_v26_3_ficem_read_probe.py"


def test_issue423_contract_is_localization_only() -> None:
    contract = localize.cpu_contract_preflight()
    assert contract["research_issue"] == 423
    assert contract["source_main"] == "7a10d91258f7daa0247369554533e3b2f6445bad"
    assert contract["source_failed_issue"] == 418
    assert contract["source_failed_trigger"] == 422
    assert contract["source_failed_actions_run"] == 33499743719
    assert contract["target_row"] == "bfloat16_batch8_mixed"
    assert contract["target_ordinal"] == 5
    assert contract["design_seed"] == 408411
    assert contract["design_seed_is_scientific_seed"] is False
    assert contract["bf16_atol"] == 1e-2
    assert contract["bf16_rtol"] == 1e-2
    assert contract["original_global_case_order_preserved"] is True
    assert contract["resampling"] is False
    assert contract["rejection_sampling"] is False
    assert contract["fixture_nudging"] is False
    assert contract["timing_authorized"] is False
    assert contract["profiling_authorized"] is False
    assert contract["performance_decision_authorized"] is False
    assert contract["scientific_seed_consumed"] is False


def test_issue423_reuses_frozen_probe_order_and_tolerance() -> None:
    assert probe.DESIGN_SEED == 408411
    assert probe.DTYPE_NAMES == ("float32", "bfloat16")
    assert probe.BATCH_SIZES == (8, 64)
    assert probe.VALIDITY_KINDS == ("mixed", "full")
    assert (probe.BF16_ATOL, probe.BF16_RTOL) == (1e-2, 1e-2)
    rows = [
        probe._row_key(dtype_name, batch_size, validity_kind)
        for dtype_name in probe.DTYPE_NAMES
        for batch_size in probe.BATCH_SIZES
        for validity_kind in probe.VALIDITY_KINDS
    ]
    assert rows == [
        "float32_batch8_mixed",
        "float32_batch8_full",
        "float32_batch64_mixed",
        "float32_batch64_full",
        "bfloat16_batch8_mixed",
        "bfloat16_batch8_full",
        "bfloat16_batch64_mixed",
        "bfloat16_batch64_full",
    ]


def test_issue423_source_has_no_timing_training_or_alternate_fixture_path() -> None:
    source = LOCALIZE.read_text()
    assert "torch.Generator().manual_seed(probe.DESIGN_SEED)" in source
    assert "probe.make_case(" in source
    assert "TARGET_ORDINAL = 5" in source
    assert "TARGET_ROW = \"bfloat16_batch8_mixed\"" in source
    assert "probe.BF16_ATOL, probe.BF16_RTOL" in source
    assert "probe._tie_aware_top4_equivalence(" in source
    assert "probe.fused_ficem_read_tail(" in source
    for forbidden in (
        "torch.cuda.Event",
        "torch.profiler",
        "_timed_round_us",
        "_timed_summaries",
        "_cuda_profile",
        "latency_ratio",
        "MAX_GEOMEAN_LATENCY_RATIO",
        "MAX_ROW_LATENCY_RATIO",
        "MAX_FULL_EVENT_RATIO",
        "torch.load",
        "load_state_dict",
        "torch.optim",
        ".backward(",
        ".step(",
        "SOURCE_RUN_DIR",
        "aera.pt",
        "transformer.pt",
        "seed8471",
        "manual_seed(probe.DESIGN_SEED +",
        "while ",
    ):
        assert forbidden not in source


def test_issue423_backend_and_probe_are_guarded_as_frozen() -> None:
    workflow = WORKFLOW.read_text()
    assert "e8ab9778fe3f3b853e6b18327cbea2c73250624c" in workflow
    assert "16aa99b9f6f0a1d11bd7bf5f36b2f0b1fb97047b" in workflow
    assert "git hash-object tam_research/aera_hardware_core_v26_3_ficem_read_triton.py" in workflow
    assert "git hash-object tam_research/aera_v26_3_ficem_read_probe.py" in workflow
    assert BACKEND.exists()
    assert PROBE.exists()


def test_issue423_launcher_is_duplicate_safe_and_one_l4_only() -> None:
    source = LAUNCHER.read_text()
    assert "issue423-bf16-correctness-localize/result.json" in source
    assert "AERA_V26_ISSUE423_BF16_CORRECTNESS_LOCALIZE_RESULT_JSON=" in source
    assert 'gpu="L4"' in source
    assert source.count('gpu="L4"') == 1
    assert "MAX_GPU_SECONDS = 180" in source
    assert "if result_path.exists():" in source
    assert "run_localization" in source
    for forbidden in (
        "torch.load",
        "load_state_dict",
        "torch.optim",
        ".backward(",
        ".step(",
        "modal deploy",
    ):
        assert forbidden not in source


def test_issue423_workflow_is_attempt1_issue_open_only() -> None:
    source = WORKFLOW.read_text()
    assert "types: [opened]" in source
    assert "[aera-v26-3-bf16-correctness-localize-l4]" in source
    assert '${GITHUB_RUN_ATTEMPT}' in source
    assert '!= "1"' in source
    assert "33499743719" in source
    assert "7a10d91258f7daa0247369554533e3b2f6445bad" in source
    assert "Expected exactly one issue423 localization trigger" in source
    assert "AERA_V26_ISSUE423_BF16_CORRECTNESS_LOCALIZE_RESULT_JSON=" in source
    assert "No rerun, automatic retry, redispatch, or timeout increase is authorized" in source
    assert "workflow_dispatch" not in source
    assert "gh run rerun" not in source
    assert "modal deploy" not in source
    assert "rerun_workflow" not in source
