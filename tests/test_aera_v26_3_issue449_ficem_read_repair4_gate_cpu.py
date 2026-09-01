from __future__ import annotations

import hashlib
from pathlib import Path

from tam_research.aera_hardware_core_v26_3_ficem_read_triton import fused_ficem_read_v26_3_protocol
from tam_research import aera_v26_3_ficem_read_probe as probe

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "tam_research" / "aera_v26_3_ficem_read_probe.py"
BACKEND = ROOT / "tam_research" / "aera_hardware_core_v26_3_ficem_read_triton.py"
LAUNCHER = ROOT / "modal_aera_v26_3_ficem_read_repair4_app.py"
WORKFLOW = ROOT / ".github" / "workflows" / "aera-v26-3-ficem-read-l4-repair4.yml"
FROZEN_PROBE_BLOB = "16aa99b9f6f0a1d11bd7bf5f36b2f0b1fb97047b"
REPAIR4_BACKEND_BLOB = "a3a603c8a2d4b20ebcccd7663970978f4288a760"
LAUNCHER_BLOB = "ea45263722edc02753221a885b7948770a5054e0"


def _blob(path: Path) -> str:
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


def test_issue449_frozen_probe_and_repair4_backend_are_exact() -> None:
    assert _blob(PROBE) == FROZEN_PROBE_BLOB
    assert _blob(LAUNCHER) == LAUNCHER_BLOB
    protocol = fused_ficem_read_v26_3_protocol()
    if protocol.get("bf16_actual_autocast_tail_repair5") is True:
        _assert_repair5_successor_protocol(protocol)
        assert REPAIR4_BACKEND_BLOB == "a3a603c8a2d4b20ebcccd7663970978f4288a760"
        return
    assert _blob(BACKEND) == REPAIR4_BACKEND_BLOB


def test_issue449_protocol_is_explicit_repair4_and_still_non_authorizing() -> None:
    protocol = fused_ficem_read_v26_3_protocol()
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


def test_issue449_reuses_original_issue418_fixture_timing_and_thresholds() -> None:
    assert probe.DESIGN_SEED == 408_411
    assert (probe.D_MODEL, probe.TIME, probe.CAPACITY, probe.MEMORY_DIM) == (200, 256, 48, 50)
    assert probe.BATCH_SIZES == (8, 64)
    assert probe.DTYPE_NAMES == ("float32", "bfloat16")
    assert probe.VALIDITY_KINDS == ("mixed", "full")
    assert (probe.WARMUP_CALLS, probe.TIMED_ROUNDS, probe.CALLS_PER_ROUND) == (10, 5, 100)
    assert (probe.FP32_ATOL, probe.FP32_RTOL) == (1e-5, 1e-5)
    assert (probe.BF16_ATOL, probe.BF16_RTOL) == (1e-2, 1e-2)
    assert probe.MAX_GEOMEAN_LATENCY_RATIO == 0.90
    assert probe.MAX_ROW_LATENCY_RATIO == 1.05
    assert probe.MAX_FULL_EVENT_RATIO == 0.75
    source = PROBE.read_text()
    assert "generator = torch.Generator().manual_seed(DESIGN_SEED)" in source
    assert "candidate_ordinal" not in source
    assert "eligible_case" not in source
    assert "resampl" not in source.lower()
    assert "nudge" not in source.lower()


def test_issue449_launcher_is_duplicate_safe_synthetic_only_and_one_l4() -> None:
    source = LAUNCHER.read_text()
    assert 'APP_NAME = "tam-research-aera-v26-3-issue449-ficem-read-repair4"' in source
    assert 'RESULT_PATH = "/vol/aera-v26/issue449-ficem-read-repair4/result.json"' in source
    assert "MAX_GPU_SECONDS = 300" in source
    assert source.count('gpu="L4"') == 1
    assert source.count("def run_probe()") == 1
    assert source.count("result_path.exists()") >= 2
    assert "refusing duplicate issue449 FICEM read repair4 run" in source
    assert FROZEN_PROBE_BLOB in source
    assert REPAIR4_BACKEND_BLOB in source
    assert "run_ficem_read_probe" in source
    assert "AERA_V26_ISSUE449_FICEM_READ_REPAIR4_RESULT_JSON=" in source
    assert "AERA_V26_ISSUE449_FICEM_READ_REPAIR4_SUMMARY_JSON=" in source
    assert 'result["rows"]["bfloat16_batch8_mixed"]["correctness"]' in source
    for key in (
        "selection_semantically_equivalent",
        "distinct_selected_set_exact",
        "pre_out_recalled_close",
        "final_out_close",
        "query_and_normalized_keys_bit_exact",
        "source_unchanged",
        "finite",
        "dtype_device_shape_exact",
        "pre_out_max_abs_diff",
    ):
        assert f'"{key}"' in source
    for forbidden in (
        "torch.load(",
        "load_state_dict",
        "torch.optim",
        ".backward(",
        "seed8471",
        "aera.pt",
        "transformer.pt",
    ):
        assert forbidden not in source


def test_issue449_workflow_is_attempt1_bound_and_has_no_retry_path() -> None:
    source = WORKFLOW.read_text()
    assert "issues:\n    types: [opened]" in source
    assert "workflow_dispatch" not in source
    assert "[aera-v26-3-ficem-read-l4-repair4]" in source
    assert 'test "${GITHUB_RUN_ATTEMPT}" = "1"' in source
    assert "Expected exactly one" not in source  # exact count is enforced directly below.
    assert 'test "${trigger_count}" = "1"' in source
    assert "Bind main:" in source
    assert 'test "$(git rev-parse HEAD)" = "${bound_main}"' in source
    assert "33ec518f978cc95afe5165fc3b3b054151a2475c" in source
    assert "4081eff33c754113b4df7efd39bf1a6f04a9556a" in source
    assert "33535610189" in source
    assert FROZEN_PROBE_BLOB in source
    assert REPAIR4_BACKEND_BLOB in source
    assert LAUNCHER_BLOB in source
    assert source.count("modal run modal_aera_v26_3_ficem_read_repair4_app.py") == 1
    assert "AERA_V26_ISSUE449_FICEM_READ_REPAIR4_RESULT_JSON=" in source
    assert "gh run rerun" not in source
    assert "modal deploy" not in source
    assert "cancel-in-progress: false" in source
    assert "automatic retry" in source
    assert "redispatch" in source
    assert "alternate trigger" in source
    assert "timeout increase" in source


def test_issue449_workflow_permissions_are_narrow() -> None:
    source = WORKFLOW.read_text()
    permissions = source.split("permissions:\n", 1)[1].split("\n\nconcurrency:", 1)[0]
    assert "actions: read" in permissions
    assert "contents: read" in permissions
    assert "issues: write" in permissions
    assert "pull-requests: read" in permissions
    assert "actions: write" not in permissions
    assert "contents: write" not in permissions
    assert "pull-requests: write" not in permissions
