from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from tam_research import aera_v26_3_issue460_reference_autocast_reportfix1 as reportfix
from tam_research import aera_v26_3_repair4_reference_autocast_localize as frozen_localize
from tam_research.aera_hardware_core_v26_3_ficem_read_triton import fused_ficem_read_v26_3_protocol

ROOT = Path(__file__).resolve().parents[1]
SHIM = ROOT / "tam_research" / "aera_v26_3_issue460_reference_autocast_reportfix1.py"
FROZEN_LOCALIZATION = (
    ROOT / "tam_research" / "aera_v26_3_repair4_reference_autocast_localize.py"
)
BACKEND = ROOT / "tam_research" / "aera_hardware_core_v26_3_ficem_read_triton.py"
PROBE = ROOT / "tam_research" / "aera_v26_3_ficem_read_probe.py"
LAUNCHER = ROOT / "modal_aera_v26_3_issue460_reference_autocast_reportfix1_app.py"
WORKFLOW = (
    ROOT
    / ".github"
    / "workflows"
    / "aera-v26-3-issue460-reference-autocast-reportfix1-l4.yml"
)

SOURCE_MAIN = "d620b2a349ebb8e494b397bc534430abaeba394f"
FROZEN_LOCALIZATION_BLOB = "8ed7de14a0f29f3ac66d6228a71892fbf97e150f"
FROZEN_PROBE_BLOB = "16aa99b9f6f0a1d11bd7bf5f36b2f0b1fb97047b"
FROZEN_REPAIR4_BACKEND_BLOB = "a3a603c8a2d4b20ebcccd7663970978f4288a760"


def _git_blob(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def _assert_repair5_successor_protocol(protocol: dict[str, object]) -> None:
    assert protocol["bf16_reference_rounding_repair3"] is True
    assert protocol["bf16_product_rounding_repair4"] is True
    assert protocol["bf16_actual_autocast_tail_repair5"] is True
    assert protocol["bf16_strength_bias_fp32_repair5"] is True
    assert protocol["bf16_logits_fp32_repair5"] is True
    assert protocol["bf16_final_weights_fp32_repair5"] is True
    assert protocol["bf16_recalled_fp32_repair5"] is True
    assert protocol["bf16_product_rounding_active_after_repair5"] is False
    assert protocol["float32_path_changed_by_repair5"] is False
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


def test_issue460_contract_is_reporting_only_and_source_attempt_is_exhausted() -> None:
    contract = reportfix.cpu_contract_preflight()
    assert contract["research_issue"] == 460
    assert contract["source_diagnostic_issue"] == 456
    assert contract["source_trigger"] == 459
    assert contract["source_run"] == 33546452532
    assert contract["source_job"] == 99985042556
    assert contract["source_main"] == SOURCE_MAIN
    assert contract["source_failure"] == "torch.isclose Float did not match BFloat16"
    assert contract["source_authoritative_result_marker_emitted"] is False
    assert contract["source_attempt_exhausted"] is True
    assert contract["target_row"] == "bfloat16_batch8_mixed"
    assert contract["target_ordinal"] == 5
    assert contract["design_seed"] == 408411
    assert contract["design_seed_is_scientific_seed"] is False
    assert contract["original_global_case_order_preserved"] is True
    assert contract["resampling"] is False
    assert contract["rejection_sampling"] is False
    assert contract["fixture_nudging"] is False
    assert contract["alternate_seed"] is False
    assert contract["frozen_correctness_row_invocation_unchanged"] is True
    assert contract["dtype_safe_reporting_only"] is True
    assert contract["numeric_reporting_dtype"] == "torch.float32"
    assert contract["mixed_dtype_bit_equality_not_applicable"] is True
    assert contract["native_execution_tensors_unchanged"] is True
    assert contract["outside_projection_control_unchanged"] is True
    assert contract["localization_only"] is True
    assert contract["timing_authorized"] is False
    assert contract["profiling_authorized"] is False
    assert contract["performance_decision_authorized"] is False
    assert contract["production_backend_modified"] is False
    assert contract["production_probe_modified"] is False
    assert contract["source_localization_modified"] is False
    assert contract["scientific_seed_consumed"] is False
    assert contract["end_to_end_systems_authorized"] is False
    assert contract["architecture_freeze_authorized"] is False
    assert contract["s2_authorized"] is False
    assert contract["fresh_scientific_seed_authorized"] is False
    assert contract["100m_authorized"] is False
    assert contract["breakthrough_proven"] is False


def test_issue460_freezes_localization_backend_and_probe_byte_for_byte() -> None:
    assert _git_blob(FROZEN_LOCALIZATION) == FROZEN_LOCALIZATION_BLOB
    assert _git_blob(PROBE) == FROZEN_PROBE_BLOB
    assert reportfix.FROZEN_LOCALIZATION_GIT_BLOB == FROZEN_LOCALIZATION_BLOB
    assert reportfix.FROZEN_REPAIR4_BACKEND_GIT_BLOB == FROZEN_REPAIR4_BACKEND_BLOB
    assert reportfix.FROZEN_PROBE_GIT_BLOB == FROZEN_PROBE_BLOB
    protocol = fused_ficem_read_v26_3_protocol()
    if protocol.get("bf16_actual_autocast_tail_repair5") is True:
        _assert_repair5_successor_protocol(protocol)
        return
    assert _git_blob(BACKEND) == FROZEN_REPAIR4_BACKEND_BLOB


def test_issue460_preserves_frozen_target_order_seed_and_primary_call() -> None:
    source = FROZEN_LOCALIZATION.read_text()
    assert frozen_localize.TARGET_ROW == "bfloat16_batch8_mixed"
    assert frozen_localize.TARGET_ORDINAL == 5
    assert frozen_localize.probe.DESIGN_SEED == 408411
    assert frozen_localize._ordinary_rows()[4] == "bfloat16_batch8_mixed"
    assert source.count("probe.correctness_row(memory, case, reference, candidate)") == 1
    assert "torch.Generator().manual_seed(probe.DESIGN_SEED)" in source
    assert "manual_seed(probe.DESIGN_SEED +" not in source
    assert "rejection_sampling" in source
    assert "while " not in source
    assert "correctness_row(" not in SHIM.read_text()


def test_issue460_reproduces_the_old_float_vs_bfloat16_isclose_hazard() -> None:
    reference = torch.tensor([1.0, 2.0], dtype=torch.float32)
    candidate = reference.to(torch.bfloat16)
    with pytest.raises(RuntimeError):
        torch.isclose(reference, candidate, atol=0.0, rtol=0.0)


def test_issue460_mixed_dtype_stats_are_safe_and_bit_fields_are_not_applicable() -> None:
    reference = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    candidate = reference.to(torch.bfloat16)
    reference_before = reference.clone()
    candidate_before = candidate.clone()

    stats = reportfix.dtype_safe_stats(reference, candidate, atol=0.0, rtol=0.0)

    assert stats["shape_equal"] is True
    assert stats["dtype_equal"] is False
    assert stats["device_equal"] is True
    assert stats["reference_dtype"] == "torch.float32"
    assert stats["candidate_dtype"] == "torch.bfloat16"
    assert stats["numeric_comparison_dtype"] == "torch.float32"
    assert stats["numeric_reporting_copies_only"] is True
    assert stats["numeric_comparison_performed"] is True
    assert stats["bit_equal"] is None
    assert stats["bit_mismatch_count"] is None
    assert stats["bit_mismatch_fraction"] is None
    assert stats["allclose"] is True
    assert stats["max_abs_error"] == 0.0
    assert stats["first_mismatch"] is None
    assert torch.equal(reference, reference_before)
    assert torch.equal(candidate, candidate_before)
    assert reference.dtype == torch.float32
    assert candidate.dtype == torch.bfloat16


def test_issue460_mixed_dtype_first_mismatch_is_numeric_and_deterministic() -> None:
    reference = torch.tensor([1.0, 1.001, 3.0], dtype=torch.float32)
    candidate = torch.tensor([1.0, 1.0, 3.0], dtype=torch.bfloat16)
    stats = reportfix.dtype_safe_stats(reference, candidate, atol=0.0, rtol=0.0)
    assert stats["dtype_equal"] is False
    assert stats["allclose"] is False
    assert stats["first_mismatch"]["kind"] == "numeric_fp32"
    assert stats["first_mismatch"]["coordinate"] == [1]
    assert stats["first_mismatch"]["reference_dtype"] == "torch.float32"
    assert stats["first_mismatch"]["candidate_dtype"] == "torch.bfloat16"


def test_issue460_same_dtype_exact_semantics_are_preserved() -> None:
    reference = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    identical = reference.clone()
    changed = reference.clone()
    changed[1] = 2.25

    exact = reportfix.dtype_safe_stats(reference, identical)
    assert exact["dtype_equal"] is True
    assert exact["bit_equal"] is True
    assert exact["bit_mismatch_count"] == 0
    assert exact["bit_mismatch_fraction"] == 0.0
    assert exact["allclose"] is True
    assert exact["first_mismatch"] is None

    mismatch = reportfix.dtype_safe_stats(reference, changed)
    assert mismatch["dtype_equal"] is True
    assert mismatch["bit_equal"] is False
    assert mismatch["bit_mismatch_count"] == 1
    assert mismatch["first_mismatch"]["kind"] == "native_exact"
    assert mismatch["first_mismatch"]["coordinate"] == [1]


def test_issue460_shape_mismatch_is_reported_without_numeric_comparison() -> None:
    reference = torch.zeros(2, dtype=torch.float32)
    candidate = torch.zeros(3, dtype=torch.bfloat16)
    stats = reportfix.dtype_safe_stats(reference, candidate)
    assert stats["shape_equal"] is False
    assert stats["dtype_equal"] is False
    assert stats["bit_equal"] is None
    assert stats["bit_mismatch_count"] is None
    assert stats["first_mismatch"]["kind"] == "shape"


def test_issue460_shim_substitutes_only_stats_and_restores_after_success(monkeypatch) -> None:
    sentinel = frozen_localize._stats
    seen: dict[str, object] = {}

    def fake_run() -> dict:
        seen["stats"] = frozen_localize._stats
        return {
            "device": "cpu-test",
            "target_row": "bfloat16_batch8_mixed",
            "primary_pass": False,
            "primary_false_subgates": ["final_out_close"],
        }

    monkeypatch.setattr(frozen_localize, "run_localization", fake_run)
    result = reportfix.run_localization_reportfix1()
    assert seen["stats"] is reportfix.dtype_safe_stats
    assert frozen_localize._stats is sentinel
    assert result["reportfix1_applied"] is True
    assert result["source_diagnostic_issue"] == 456
    assert result["source_failed_trigger"] == 459
    assert result["native_execution_tensors_changed"] is False


def test_issue460_shim_restores_stats_even_if_frozen_run_raises(monkeypatch) -> None:
    sentinel = frozen_localize._stats

    def fake_run() -> dict:
        assert frozen_localize._stats is reportfix.dtype_safe_stats
        raise RuntimeError("synthetic reporting test failure")

    monkeypatch.setattr(frozen_localize, "run_localization", fake_run)
    with pytest.raises(RuntimeError, match="synthetic reporting test failure"):
        reportfix.run_localization_reportfix1()
    assert frozen_localize._stats is sentinel


def test_issue460_shim_contains_no_execution_timing_training_or_acceptance_changes() -> None:
    source = SHIM.read_text()
    assert "frozen_localize._stats = dtype_safe_stats" in source
    assert "frozen_localize._stats = original_stats" in source
    assert "finally:" in source
    assert ".detach().to(dtype=torch.float32).clone()" in source
    assert "torch.isclose(reference_report, candidate_report" in source
    for forbidden in (
        "torch.cuda.Event",
        "torch.profiler",
        "latency_ratio",
        "MAX_GEOMEAN_LATENCY_RATIO",
        "MAX_ROW_LATENCY_RATIO",
        "torch.load(",
        "load_state_dict",
        "torch.optim",
        ".backward(",
        ".step(",
        "memory.out(",
        "fused_ficem_read_tail(",
        "TritonFICEMReadBackend(",
        "TorchFICEMReferenceBackend(",
        "make_case(",
        "manual_seed(",
        "set_float32_matmul_precision(",
        "allow_tf32 =",
        "allow_bf16_reduced_precision_reduction =",
    ):
        assert forbidden not in source


def test_issue460_launcher_is_unique_duplicate_safe_one_l4_and_frozen() -> None:
    source = LAUNCHER.read_text()
    assert (
        'RESULT_PATH = "/vol/aera-v26/issue460-reference-autocast-localize-reportfix1/result.json"'
        in source
    )
    assert (
        'EXHAUSTED_ISSUE456_RESULT_PATH = "/vol/aera-v26/issue456-repair4-reference-autocast-localize/result.json"'
        in source
    )
    assert "AERA_V26_ISSUE460_REFERENCE_AUTOCAST_LOCALIZE_REPORTFIX1_RESULT_JSON=" in source
    assert "AERA_V26_ISSUE460_REFERENCE_AUTOCAST_LOCALIZE_REPORTFIX1_SUMMARY_JSON=" in source
    assert "MAX_GPU_SECONDS = 180" in source
    assert source.count('gpu="L4"') == 1
    assert source.count("result_path.exists()") >= 2
    assert FROZEN_LOCALIZATION_BLOB in source
    assert FROZEN_PROBE_BLOB in source
    assert FROZEN_REPAIR4_BACKEND_BLOB in source
    assert "run_localization_reportfix1" in source
    for forbidden in (
        "torch.load(",
        "load_state_dict",
        "torch.optim",
        ".backward(",
        "modal deploy",
        "seed8471",
    ):
        assert forbidden not in source


def test_issue460_workflow_is_attempt1_bound_and_grounded_in_459_failure() -> None:
    source = WORKFLOW.read_text()
    assert "types: [opened]" in source
    assert "[aera-v26-3-repair4-reference-autocast-localize-reportfix1-l4]" in source
    assert '${GITHUB_RUN_ATTEMPT}' in source
    assert '!= "1"' in source
    assert "33546452532" in source
    assert "99985042556" in source
    assert 'test "${source_run_conclusion}" = "failure"' in source
    assert 'test "${source_run_attempt}" = "1"' in source
    assert 'test "${source_guard_step}" = "success"' in source
    assert 'test "${source_auth_step}" = "success"' in source
    assert 'test "${source_l4_step}" = "failure"' in source
    assert "did not produce the authoritative result marker" in source
    assert "Expected exactly one issue460 reportfix1 trigger" in source
    assert SOURCE_MAIN in source
    assert FROZEN_LOCALIZATION_BLOB in source
    assert FROZEN_PROBE_BLOB in source
    assert FROZEN_REPAIR4_BACKEND_BLOB in source
    assert "Bind main:" in source
    assert 'test "$(git rev-parse HEAD)" = "${bound_main}"' in source
    assert "workflow_dispatch" not in source
    assert "gh run rerun" not in source
    assert "rerun_workflow" not in source
    assert "modal deploy" not in source
    assert "cancel-in-progress: false" in source


def test_issue460_workflow_permissions_and_modal_call_are_narrow() -> None:
    source = WORKFLOW.read_text()
    permissions = source.split("permissions:\n", 1)[1].split("\n\nconcurrency:", 1)[0]
    assert "actions: read" in permissions
    assert "contents: read" in permissions
    assert "issues: write" in permissions
    assert "pull-requests:" not in permissions
    assert "actions: write" not in permissions
    assert "contents: write" not in permissions
    assert (
        source.count(
            "modal run modal_aera_v26_3_issue460_reference_autocast_reportfix1_app.py"
        )
        == 1
    )
    assert "AERA_V26_ISSUE460_REFERENCE_AUTOCAST_LOCALIZE_REPORTFIX1_RESULT_JSON=" in source
    assert "No rerun, retry, redispatch, alternate trigger, or timeout increase is authorized" in source
