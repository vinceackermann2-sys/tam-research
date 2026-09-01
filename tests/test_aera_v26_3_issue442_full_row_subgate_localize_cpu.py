from __future__ import annotations

from pathlib import Path
import subprocess

from tam_research import aera_v26_3_ficem_read_probe as probe
from tam_research import aera_v26_3_repair3_full_row_subgate_localize as localize
from tam_research.aera_hardware_core_v26_3_ficem_read_triton import (
    fused_ficem_read_v26_3_protocol,
)

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "tam_research" / "aera_v26_3_repair3_full_row_subgate_localize.py"
LAUNCHER = ROOT / "modal_aera_v26_3_repair3_full_row_subgate_localize_app.py"
WORKFLOW = ROOT / ".github" / "workflows" / "aera-v26-3-repair3-full-row-subgate-localize-l4.yml"
BACKEND = ROOT / "tam_research" / "aera_hardware_core_v26_3_ficem_read_triton.py"
PROBE = ROOT / "tam_research" / "aera_v26_3_ficem_read_probe.py"
HISTORICAL_REPAIR3_BACKEND_BLOB = "b6b37f0379b280eea4e5c2b16f349951dadc4df9"
FROZEN_PROBE_BLOB = "16aa99b9f6f0a1d11bd7bf5f36b2f0b1fb97047b"


def _git_blob(path: Path) -> str:
    return subprocess.check_output(
        ["git", "hash-object", str(path.relative_to(ROOT))], cwd=ROOT, text=True
    ).strip()


def _assert_historical_repair3_or_explicit_repair4_successor() -> None:
    protocol = fused_ficem_read_v26_3_protocol()
    if protocol.get("bf16_product_rounding_repair4") is not True:
        assert _git_blob(BACKEND) == HISTORICAL_REPAIR3_BACKEND_BLOB
        return
    assert protocol["bf16_reference_rounding_repair3"] is True
    assert protocol["float32_path_changed_by_repair3"] is False
    assert protocol["bf16_product_rounding_repair4"] is True
    assert protocol["float32_path_changed_by_repair4"] is False
    assert protocol["capacity"] == 48
    assert protocol["memory_dim"] == 50
    assert protocol["read_top_k"] == 4
    assert protocol["read_temperature"] == 0.10
    assert protocol["min_strength"] == 1e-4
    assert protocol["read_tail_triton_launches_target"] == 1
    for key in (
        "address_projection_changed",
        "key_normalization_changed",
        "similarity_einsum_changed",
        "learned_out_projection_changed",
        "write_backend_changed",
        "training_backend_changed",
        "persistent_state_changed",
        "gpu_authorized_by_module",
        "scientific_training_authorized",
        "end_to_end_systems_authorized",
        "architecture_freeze_authorized",
        "s2_authorized",
        "fresh_scientific_seed_authorized",
        "100m_authorized",
        "breakthrough_proven",
    ):
        assert protocol[key] is False


def test_issue442_contract_is_full_row_localization_only() -> None:
    contract = localize.cpu_contract_preflight()
    assert contract["research_issue"] == 442
    assert contract["source_main"] == "58e7a2d15b7bc935eecb3ffce9097111adc8bcd7"
    assert contract["source_failed_trigger"] == 438
    assert contract["source_failed_actions_run"] == 33510242472
    assert contract["source_localization_issue"] == 439
    assert contract["source_localization_trigger"] == 441
    assert contract["source_localization_actions_run"] == 33512923203
    assert contract["target_row"] == "bfloat16_batch8_mixed"
    assert contract["target_ordinal"] == 5
    assert contract["design_seed"] == 408411
    assert contract["design_seed_is_scientific_seed"] is False
    assert contract["bf16_atol"] == 1e-2
    assert contract["bf16_rtol"] == 1e-2
    assert contract["primary_evidence_is_frozen_correctness_row"] is True
    assert contract["original_global_case_order_preserved"] is True
    assert contract["resampling"] is False
    assert contract["rejection_sampling"] is False
    assert contract["fixture_nudging"] is False
    assert contract["alternate_seed"] is False
    assert contract["localization_only"] is True
    assert contract["timing_authorized"] is False
    assert contract["profiling_authorized"] is False
    assert contract["performance_decision_authorized"] is False
    assert contract["scientific_seed_consumed"] is False
    assert contract["100m_authorized"] is False
    assert contract["breakthrough_proven"] is False


def test_issue442_frozen_probe_order_and_gate_are_unchanged() -> None:
    assert probe.DESIGN_SEED == 408411
    assert probe.DTYPE_NAMES == ("float32", "bfloat16")
    assert probe.BATCH_SIZES == (8, 64)
    assert probe.VALIDITY_KINDS == ("mixed", "full")
    assert (probe.BF16_ATOL, probe.BF16_RTOL) == (1e-2, 1e-2)
    assert localize._ordinary_rows()[4] == "bfloat16_batch8_mixed"
    assert localize.PRIMARY_SUBGATES == (
        "selection_semantically_equivalent",
        "pre_out_recalled_close",
        "final_out_close",
        "query_and_normalized_keys_bit_exact",
        "source_unchanged",
        "finite",
        "dtype_device_shape_exact",
    )


def test_issue442_production_backend_and_probe_blobs_are_frozen() -> None:
    _assert_historical_repair3_or_explicit_repair4_successor()
    assert _git_blob(PROBE) == FROZEN_PROBE_BLOB


def test_issue442_source_invokes_frozen_correctness_row_and_has_no_benchmark_path() -> None:
    source = MODULE.read_text()
    assert "torch.Generator().manual_seed(probe.DESIGN_SEED)" in source
    assert "TARGET_ORDINAL = 5" in source
    assert 'TARGET_ROW = "bfloat16_batch8_mixed"' in source
    assert "primary = probe.correctness_row(memory, case, reference, candidate)" in source
    assert "correctness = probe.correctness_row(memory, case, reference, candidate)" in source
    assert "_case_fingerprints" in source
    assert "sequence_target_fixture_matches_primary" in source
    for forbidden in (
        "torch.cuda.Event",
        "torch.profiler",
        "_timed_round_us",
        "_timed_summaries",
        "_cuda_profile(",
        "_peak_vram(",
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


def test_issue442_launcher_is_duplicate_safe_and_one_l4_only() -> None:
    source = LAUNCHER.read_text()
    assert "issue442-repair3-full-row-subgate-localize/result.json" in source
    assert "AERA_V26_ISSUE442_REPAIR3_FULL_ROW_SUBGATE_LOCALIZE_RESULT_JSON=" in source
    assert 'gpu="L4"' in source
    assert source.count('gpu="L4"') == 1
    assert "MAX_GPU_SECONDS = 180" in source
    assert "if result_path.exists():" in source
    assert "run_localization" in source
    assert FROZEN_PROBE_BLOB in source
    assert HISTORICAL_REPAIR3_BACKEND_BLOB in source
    for forbidden in (
        "torch.load",
        "load_state_dict",
        "torch.optim",
        ".backward(",
        ".step(",
        "modal deploy",
    ):
        assert forbidden not in source


def test_issue442_workflow_is_attempt1_issue_open_only_and_narrowly_scoped() -> None:
    source = WORKFLOW.read_text()
    assert "types: [opened]" in source
    assert "[aera-v26-3-repair3-full-row-subgate-localize-l4]" in source
    assert '${GITHUB_RUN_ATTEMPT}' in source
    assert '!= "1"' in source
    assert "33512923203" in source
    assert "99872764955" in source
    assert "33510242472" in source
    assert "58e7a2d15b7bc935eecb3ffce9097111adc8bcd7" in source
    assert "Expected exactly one issue442 full-row localization trigger" in source
    assert "AERA_V26_ISSUE442_REPAIR3_FULL_ROW_SUBGATE_LOCALIZE_RESULT_JSON=" in source
    assert "actions: read" in source
    assert "contents: read" in source
    assert "issues: write" in source
    assert "pull-requests:" not in source
    assert "workflow_dispatch" not in source
    assert "gh run rerun" not in source
    assert "rerun_workflow" not in source
    assert "modal deploy" not in source
    assert "timeout increase" in source