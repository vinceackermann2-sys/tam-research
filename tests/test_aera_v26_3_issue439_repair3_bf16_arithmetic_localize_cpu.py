from __future__ import annotations

import hashlib
from pathlib import Path

from tam_research import aera_v26_3_ficem_read_probe as probe
from tam_research import aera_v26_3_repair3_bf16_arithmetic_localize as localize
from tam_research.aera_hardware_core_v26_3_ficem_read_triton import (
    fused_ficem_read_v26_3_protocol,
)

ROOT = Path(__file__).resolve().parents[1]
LOCALIZE = ROOT / "tam_research" / "aera_v26_3_repair3_bf16_arithmetic_localize.py"
LAUNCHER = ROOT / "modal_aera_v26_3_repair3_bf16_arithmetic_localize_app.py"
WORKFLOW = (
    ROOT
    / ".github"
    / "workflows"
    / "aera-v26-3-repair3-bf16-arithmetic-localize-l4.yml"
)
BACKEND = ROOT / "tam_research" / "aera_hardware_core_v26_3_ficem_read_triton.py"
PROBE = ROOT / "tam_research" / "aera_v26_3_ficem_read_probe.py"

FROZEN_PROBE_BLOB = "16aa99b9f6f0a1d11bd7bf5f36b2f0b1fb97047b"
FROZEN_BACKEND_BLOB = "b6b37f0379b280eea4e5c2b16f349951dadc4df9"


def _git_blob(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def _assert_historical_repair3_or_explicit_repair4_successor() -> None:
    protocol = fused_ficem_read_v26_3_protocol()
    if protocol.get("bf16_product_rounding_repair4") is not True:
        assert _git_blob(BACKEND) == FROZEN_BACKEND_BLOB
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


def test_issue439_contract_is_localization_only_and_frozen() -> None:
    contract = localize.cpu_contract_preflight()
    assert contract["research_issue"] == 439
    assert contract["source_main"] == "1ec7229f976b360440171b979bde63dedd8e9697"
    assert contract["source_failed_trigger"] == 438
    assert contract["source_failed_actions_run"] == 33510242472
    assert contract["source_failed_job"] == 99863885932
    assert contract["target_row"] == "bfloat16_batch8_mixed"
    assert contract["target_ordinal"] == 5
    assert contract["design_seed"] == 408411
    assert contract["design_seed_is_scientific_seed"] is False
    assert (contract["bf16_atol"], contract["bf16_rtol"]) == (1e-2, 1e-2)
    assert contract["frozen_probe_git_blob"] == FROZEN_PROBE_BLOB
    assert contract["frozen_repair3_backend_git_blob"] == FROZEN_BACKEND_BLOB
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


def test_issue439_frozen_probe_order_geometry_and_blobs_are_exact() -> None:
    assert probe.DESIGN_SEED == 408411
    assert probe.DTYPE_NAMES == ("float32", "bfloat16")
    assert probe.BATCH_SIZES == (8, 64)
    assert probe.VALIDITY_KINDS == ("mixed", "full")
    assert (probe.D_MODEL, probe.TIME, probe.CAPACITY, probe.MEMORY_DIM) == (
        200,
        256,
        48,
        50,
    )
    assert (probe.BF16_ATOL, probe.BF16_RTOL) == (1e-2, 1e-2)
    rows = [
        probe._row_key(dtype_name, batch_size, validity_kind)
        for dtype_name in probe.DTYPE_NAMES
        for batch_size in probe.BATCH_SIZES
        for validity_kind in probe.VALIDITY_KINDS
    ]
    assert rows[4] == "bfloat16_batch8_mixed"
    assert _git_blob(PROBE) == FROZEN_PROBE_BLOB
    _assert_historical_repair3_or_explicit_repair4_successor()


def test_issue439_diagnostic_has_one_isolated_triton_mirror_and_checkpoint_path() -> None:
    source = LOCALIZE.read_text()
    assert source.count("@triton.jit") == 1
    assert "def _repair3_checkpoint_kernel(" in source
    assert "probe.fused_ficem_read_tail(" in source
    assert '"diagnostic_mirror_valid"' in source
    assert '"first_reference_vs_repair3_divergence"' in source
    for checkpoint in (
        '"similarity_visible"',
        '"clamped_visible"',
        '"strength_bias"',
        '"logits"',
        '"masked_logits"',
        '"top_indices"',
        '"safe_logits_fp32"',
        '"softmax_fp32"',
        '"weight_bf16"',
        '"weight_valid_bf16"',
        '"weight_sum_bf16"',
        '"denominator_bf16"',
        '"final_weight_bf16"',
        '"selected_value_bf16"',
        '"product_fp32"',
        '"recalled_fp32"',
        '"recalled_bf16"',
    ):
        assert checkpoint in source

    module_token = "aera_v26_3_repair3_bf16_arithmetic_localize"
    for path in (ROOT / "tam_research").glob("*.py"):
        if path == LOCALIZE:
            continue
        assert module_token not in path.read_text()


def test_issue439_replays_only_original_target_without_timing_training_or_fixture_escape() -> None:
    source = LOCALIZE.read_text()
    assert "torch.Generator().manual_seed(probe.DESIGN_SEED)" in source
    assert "probe.make_case(" in source
    assert 'TARGET_ROW = "bfloat16_batch8_mixed"' in source
    assert "TARGET_ORDINAL = 5" in source
    assert "probe._diagnostic_tail_inputs(memory, case)" in source
    assert "probe._tie_aware_top4_equivalence(" in source
    assert "state.strengths.clamp(probe.MIN_STRENGTH, 1.0)" in source
    assert "torch.log(clamped)" in source
    assert "torch.softmax(safe_logits.float(), dim=-1)" in source
    assert "products = final_weight.unsqueeze(-1) * selected_values" in source
    assert "recalled = products.sum(dim=2)" in source

    for forbidden in (
        "torch.cuda.Event",
        "torch.profiler",
        "_timed_round",
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
        "\n    while ",
    ):
        assert forbidden not in source


def test_issue439_launcher_is_duplicate_safe_one_l4_and_durable() -> None:
    source = LAUNCHER.read_text()
    assert "issue439-repair3-bf16-arithmetic-localize/result.json" in source
    assert (
        "AERA_V26_ISSUE439_REPAIR3_BF16_ARITHMETIC_LOCALIZE_RESULT_JSON="
        in source
    )
    assert 'gpu="L4"' in source
    assert source.count('gpu="L4"') == 1
    assert "MAX_GPU_SECONDS = 180" in source
    assert "if result_path.exists():" in source
    assert FROZEN_PROBE_BLOB in source
    assert FROZEN_BACKEND_BLOB in source
    for forbidden in (
        "torch.load",
        "load_state_dict",
        "torch.optim",
        ".backward(",
        ".step(",
        "modal deploy",
    ):
        assert forbidden not in source


def test_issue439_workflow_is_attempt1_bound_and_checks_source_l4_failure() -> None:
    source = WORKFLOW.read_text()
    assert "types: [opened]" in source
    assert "[aera-v26-3-repair3-bf16-arithmetic-localize-l4]" in source
    assert '${GITHUB_RUN_ATTEMPT}' in source
    assert '!= "1"' in source
    assert "33510242472" in source
    assert "99863885932" in source
    assert "Run sole issue436 repaired synthetic L4 probe" in source
    assert 'test "${source_l4_step}" = "failure"' in source
    assert "1ec7229f976b360440171b979bde63dedd8e9697" in source
    assert FROZEN_PROBE_BLOB in source
    assert FROZEN_BACKEND_BLOB in source
    assert "Expected exactly one issue439 arithmetic-localization trigger" in source
    assert (
        "AERA_V26_ISSUE439_REPAIR3_BF16_ARITHMETIC_LOCALIZE_RESULT_JSON="
        in source
    )
    assert "actions: read" in source
    assert "contents: read" in source
    assert "issues: write" in source
    assert "pull-requests:" not in source
    assert "workflow_dispatch" not in source
    assert "gh run rerun" not in source
    assert "modal deploy" not in source
    assert "rerun_workflow" not in source
    assert (
        "No rerun, automatic retry, redispatch, alternate trigger, or timeout increase is authorized"
        in source
    )