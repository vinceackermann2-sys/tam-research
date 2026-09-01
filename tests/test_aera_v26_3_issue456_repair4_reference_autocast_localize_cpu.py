from __future__ import annotations

import hashlib
from pathlib import Path

from tam_research import aera_v26_3_ficem_read_probe as probe
from tam_research import aera_v26_3_repair4_reference_autocast_localize as localize
from tam_research.aera_hardware_core_v26_3_ficem_read_triton import fused_ficem_read_v26_3_protocol

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "tam_research" / "aera_v26_3_repair4_reference_autocast_localize.py"
LAUNCHER = ROOT / "modal_aera_v26_3_repair4_reference_autocast_localize_app.py"
WORKFLOW = ROOT / ".github" / "workflows" / "aera-v26-3-repair4-reference-autocast-localize-l4.yml"
BACKEND = ROOT / "tam_research" / "aera_hardware_core_v26_3_ficem_read_triton.py"
PROBE = ROOT / "tam_research" / "aera_v26_3_ficem_read_probe.py"
FROZEN_PROBE_BLOB = "16aa99b9f6f0a1d11bd7bf5f36b2f0b1fb97047b"
FROZEN_REPAIR4_BACKEND_BLOB = "a3a603c8a2d4b20ebcccd7663970978f4288a760"
SOURCE_MAIN = "a04e43af14b64205ee84472768cf2be850a88e75"


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


def test_issue456_contract_is_localization_only_and_frozen() -> None:
    contract = localize.cpu_contract_preflight()
    assert contract["research_issue"] == 456
    assert contract["source_main"] == SOURCE_MAIN
    assert contract["source_localization_issue"] == 452
    assert contract["source_localization_trigger"] == 454
    assert contract["source_localization_run"] == 33539885620
    assert contract["source_localization_job"] == 99963230232
    assert contract["source_duplicate_trigger"] == 455
    assert contract["source_duplicate_run"] == 33539909378
    assert contract["source_duplicate_job"] == 99963512537
    assert contract["target_row"] == "bfloat16_batch8_mixed"
    assert contract["target_ordinal"] == 5
    assert contract["design_seed"] == 408411
    assert contract["design_seed_is_scientific_seed"] is False
    assert contract["frozen_probe_git_blob"] == FROZEN_PROBE_BLOB
    assert contract["frozen_repair4_backend_git_blob"] == FROZEN_REPAIR4_BACKEND_BLOB
    assert contract["original_global_case_order_preserved"] is True
    assert contract["resampling"] is False
    assert contract["rejection_sampling"] is False
    assert contract["fixture_nudging"] is False
    assert contract["alternate_seed"] is False
    assert contract["primary_evidence_is_frozen_correctness_row"] is True
    assert contract["inside_autocast_reference_path"] is True
    assert contract["outside_autocast_reference_path"] is True
    assert contract["same_similarity_tail_control"] is True
    assert contract["outside_projection_failure_is_reported_not_repaired"] is True
    assert contract["localization_only"] is True
    assert contract["timing_authorized"] is False
    assert contract["profiling_authorized"] is False
    assert contract["performance_decision_authorized"] is False
    assert contract["production_backend_modified"] is False
    assert contract["production_probe_modified"] is False
    assert contract["scientific_seed_consumed"] is False
    assert contract["end_to_end_systems_authorized"] is False
    assert contract["architecture_freeze_authorized"] is False
    assert contract["s2_authorized"] is False
    assert contract["fresh_scientific_seed_authorized"] is False
    assert contract["100m_authorized"] is False
    assert contract["breakthrough_proven"] is False


def test_issue456_frozen_backend_probe_and_fixture_contract_are_exact() -> None:
    protocol = fused_ficem_read_v26_3_protocol()
    if protocol.get("bf16_actual_autocast_tail_repair5") is True:
        _assert_repair5_successor_protocol(protocol)
        assert localize.cpu_contract_preflight()["frozen_repair4_backend_git_blob"] == FROZEN_REPAIR4_BACKEND_BLOB
    else:
        assert _git_blob(BACKEND) == FROZEN_REPAIR4_BACKEND_BLOB
    assert _git_blob(PROBE) == FROZEN_PROBE_BLOB
    assert probe.DESIGN_SEED == 408411
    assert probe.DTYPE_NAMES == ("float32", "bfloat16")
    assert probe.BATCH_SIZES == (8, 64)
    assert probe.VALIDITY_KINDS == ("mixed", "full")
    assert (probe.D_MODEL, probe.TIME, probe.CAPACITY, probe.MEMORY_DIM) == (200, 256, 48, 50)
    assert (probe.BF16_ATOL, probe.BF16_RTOL) == (1e-2, 1e-2)
    assert localize._ordinary_rows()[4] == "bfloat16_batch8_mixed"


def test_issue456_invokes_primary_once_and_preserves_original_fixture_order() -> None:
    source = MODULE.read_text()
    assert source.count("probe.correctness_row(memory, case, reference, candidate)") == 1
    assert "torch.Generator().manual_seed(probe.DESIGN_SEED)" in source
    assert "probe.make_case(" in source
    assert 'TARGET_ROW = "bfloat16_batch8_mixed"' in source
    assert "TARGET_ORDINAL = 5" in source
    assert "manual_seed(probe.DESIGN_SEED +" not in source
    assert "manual_seed(probe.DESIGN_SEED -" not in source
    assert "candidate_ordinal" not in source
    assert "eligible_case" not in source
    assert "random.choice" not in source
    assert "while " not in source


def test_issue456_has_explicit_inside_outside_autocast_reference_controls() -> None:
    source = MODULE.read_text()
    assert 'torch.autocast(device_type="cuda", dtype=torch.bfloat16)' in source
    assert "outside = _tail_math(memory, similarity, case.state, autocast_enabled=False)" in source
    assert "autocast_enabled=True" in source
    assert "probe._reference_tail(" in source
    assert "same_similarity_inside" in source
    assert "actual_reference = _full_reference_equations(memory, case)" in source
    assert "production = _production_capture(memory, case)" in source
    assert '"first_differing_checkpoint_actual_reference_vs_outside"' in source
    assert '"outside_tail_matches_frozen_reference_recalled"' in source
    assert '"outside_tail_matches_frozen_reference_indices"' in source


def test_issue456_outside_projection_is_literal_fail_safe_not_a_repair() -> None:
    source = MODULE.read_text()
    assert "def _attempt_memory_out(memory, recalled: torch.Tensor)" in source
    assert "output = memory.out(recalled)" in source
    assert "except RuntimeError as exc:" in source
    assert '"supported": False' in source
    assert '"input_dtype": str(recalled.dtype)' in source
    assert '"weight_dtype": str(memory.out.weight.dtype)' in source
    for forbidden in (
        ".to(memory.out.weight.dtype)",
        ".to(dtype=memory.out.weight.dtype)",
        "memory.out.weight.data",
        "memory.out.weight.copy_(",
        "memory.out.weight =",
        "memory.out.to(",
        "set_float32_matmul_precision(",
        "allow_tf32 =",
        "allow_bf16_reduced_precision_reduction =",
    ):
        assert forbidden not in source


def test_issue456_diagnostic_has_no_timing_training_model_or_acceptance_path() -> None:
    source = MODULE.read_text()
    for forbidden in (
        "torch.cuda.Event",
        "torch.profiler",
        "_timed_round",
        "_timed_summaries",
        "latency_ratio",
        "MAX_GEOMEAN_LATENCY_RATIO",
        "MAX_ROW_LATENCY_RATIO",
        "MAX_FULL_EVENT_RATIO",
        "torch.load(",
        "load_state_dict",
        "torch.optim",
        ".backward(",
        ".step(",
        "SOURCE_RUN_DIR",
        "aera.pt",
        "transformer.pt",
        "seed8471",
    ):
        assert forbidden not in source
    assert '"timing_performed": False' in source
    assert '"profiling_performed": False' in source
    assert '"performance_decision": None' in source
    assert '"candidate_acceptance_changed": False' in source


def test_issue456_projection_weight_and_training_state_are_checked_unchanged() -> None:
    source = MODULE.read_text()
    assert "weight_before = memory.out.weight.detach().clone()" in source
    assert "memory_training_before = bool(memory.training)" in source
    assert '"projection_weight_unchanged"' in source
    assert "torch.equal(memory.out.weight.detach(), weight_before)" in source
    assert '"memory_training_state_unchanged"' in source


def test_issue456_launcher_is_duplicate_safe_one_l4_and_durable() -> None:
    source = LAUNCHER.read_text()
    assert 'RESULT_PATH = "/vol/aera-v26/issue456-repair4-reference-autocast-localize/result.json"' in source
    assert "AERA_V26_ISSUE456_REPAIR4_REFERENCE_AUTOCAST_LOCALIZE_RESULT_JSON=" in source
    assert "AERA_V26_ISSUE456_REPAIR4_REFERENCE_AUTOCAST_LOCALIZE_SUMMARY_JSON=" in source
    assert "MAX_GPU_SECONDS = 180" in source
    assert source.count('gpu="L4"') == 1
    assert source.count("result_path.exists()") >= 2
    assert FROZEN_PROBE_BLOB in source
    assert FROZEN_REPAIR4_BACKEND_BLOB in source
    for forbidden in (
        "torch.load(",
        "load_state_dict",
        "torch.optim",
        ".backward(",
        "modal deploy",
        "seed8471",
    ):
        assert forbidden not in source


def test_issue456_workflow_is_attempt1_bound_and_source_evidence_grounded() -> None:
    source = WORKFLOW.read_text()
    assert "types: [opened]" in source
    assert "[aera-v26-3-repair4-reference-autocast-localize-l4]" in source
    assert '${GITHUB_RUN_ATTEMPT}' in source
    assert '!= "1"' in source
    assert "33539885620" in source
    assert "99963230232" in source
    assert "Run sole issue452 repair4 product reduction localization" in source
    assert 'test "${source_l4_step}" = "success"' in source
    assert "33539909378" in source
    assert "99963512537" in source
    assert 'test "${duplicate_guard_step}" = "failure"' in source
    assert 'test "${duplicate_auth_step}" = "skipped"' in source
    assert 'test "${duplicate_l4_step}" = "skipped"' in source
    assert SOURCE_MAIN in source
    assert "Expected exactly one issue456 repair4 reference-autocast localization trigger" in source
    assert FROZEN_PROBE_BLOB in source
    assert FROZEN_REPAIR4_BACKEND_BLOB in source
    assert "Bind main:" in source
    assert 'test "$(git rev-parse HEAD)" = "${bound_main}"' in source
    assert "workflow_dispatch" not in source
    assert "gh run rerun" not in source
    assert "rerun_workflow" not in source
    assert "modal deploy" not in source
    assert "cancel-in-progress: false" in source


def test_issue456_workflow_permissions_marker_and_modal_call_are_narrow() -> None:
    source = WORKFLOW.read_text()
    permissions = source.split("permissions:\n", 1)[1].split("\n\nconcurrency:", 1)[0]
    assert "actions: read" in permissions
    assert "contents: read" in permissions
    assert "issues: write" in permissions
    assert "pull-requests:" not in permissions
    assert "actions: write" not in permissions
    assert "contents: write" not in permissions
    assert source.count("modal run modal_aera_v26_3_repair4_reference_autocast_localize_app.py") == 1
    assert "AERA_V26_ISSUE456_REPAIR4_REFERENCE_AUTOCAST_LOCALIZE_RESULT_JSON=" in source
    assert "No rerun, retry, redispatch, alternate trigger, or timeout increase is authorized" in source
