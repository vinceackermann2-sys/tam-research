from __future__ import annotations

import hashlib
from pathlib import Path

from tam_research import aera_v26_3_ficem_read_probe as probe
from tam_research.aera_hardware_core_v26_3_ficem_read_triton import (
    fused_ficem_read_v26_3_protocol,
)

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "tam_research" / "aera_v26_3_ficem_read_probe.py"
BACKEND = ROOT / "tam_research" / "aera_hardware_core_v26_3_ficem_read_triton.py"
LAUNCHER = ROOT / "modal_aera_v26_3_ficem_read_repair5_app.py"
WORKFLOW = ROOT / ".github" / "workflows" / "aera-v26-3-ficem-read-l4-repair5.yml"

SOURCE_MAIN = "34be2d4f1311fb00acdc5acf14b4914fb80c6bd5"
FROZEN_PROBE_BLOB = "16aa99b9f6f0a1d11bd7bf5f36b2f0b1fb97047b"
REPAIR5_BACKEND_BLOB = "263f68eb1186a8ac14a08fc4b4df1fc5b292c711"
LAUNCHER_BLOB = "581e330b50c5631bde9e17a1f03748f980fab83f"
CPU_PR = 473
CPU_RUN = 33606059884
CPU_JOB = 100170186089
CPU_HEAD = "55a8262d144a4644a392ea2ab81eda99124518ca"


def _blob(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def test_issue474_freezes_exact_repair5_candidate_and_original_probe() -> None:
    assert _blob(PROBE) == FROZEN_PROBE_BLOB
    assert _blob(BACKEND) == REPAIR5_BACKEND_BLOB
    assert _blob(LAUNCHER) == LAUNCHER_BLOB

    protocol = fused_ficem_read_v26_3_protocol()
    expected_true = (
        "bf16_reference_rounding_repair3",
        "bf16_product_rounding_repair4",
        "bf16_actual_autocast_tail_repair5",
        "bf16_strength_bias_fp32_repair5",
        "bf16_logits_fp32_repair5",
        "bf16_final_weights_fp32_repair5",
        "bf16_recalled_fp32_repair5",
    )
    for key in expected_true:
        assert protocol[key] is True
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


def test_issue474_reuses_original_fixture_order_tolerances_and_thresholds() -> None:
    assert probe.DESIGN_SEED == 408_411
    assert (probe.D_MODEL, probe.TIME, probe.CAPACITY, probe.MEMORY_DIM) == (
        200,
        256,
        48,
        50,
    )
    assert probe.BATCH_SIZES == (8, 64)
    assert probe.DTYPE_NAMES == ("float32", "bfloat16")
    assert probe.VALIDITY_KINDS == ("mixed", "full")
    assert (probe.WARMUP_CALLS, probe.TIMED_ROUNDS, probe.CALLS_PER_ROUND) == (
        10,
        5,
        100,
    )
    assert (probe.FP32_ATOL, probe.FP32_RTOL) == (1e-5, 1e-5)
    assert (probe.BF16_ATOL, probe.BF16_RTOL) == (1e-2, 1e-2)
    assert probe.MAX_GEOMEAN_LATENCY_RATIO == 0.90
    assert probe.MAX_ROW_LATENCY_RATIO == 1.05
    assert probe.MAX_FULL_EVENT_RATIO == 0.75

    source = PROBE.read_text()
    assert "generator = torch.Generator().manual_seed(DESIGN_SEED)" in source
    assert 'for dtype_name in DTYPE_NAMES:' in source
    assert 'for batch_size in BATCH_SIZES:' in source
    assert 'for validity_kind in VALIDITY_KINDS:' in source
    assert 'raise RuntimeError(f"issue411 correctness failed for {key}")' in source
    assert "candidate_ordinal" not in source
    assert "eligible_case" not in source
    assert "resampl" not in source.lower()
    assert "nudge" not in source.lower()


def test_issue474_frozen_probe_keeps_tie_aware_and_full_tensor_correctness() -> None:
    source = PROBE.read_text()
    assert "selection_semantically_equivalent" in source
    assert "distinct_selected_set_exact" in source
    assert "tied_selection_semantically_valid" in source
    assert "pre_out_recalled_close" in source
    assert "final_out_close" in source
    assert "query_and_normalized_keys_bit_exact" in source
    assert "torch.allclose(\n        reference_recalled, candidate_recalled" in source
    assert "torch.allclose(\n        reference_result.recalled,\n        candidate_result.recalled" in source
    assert "torch.where(distinct, raw_set_equal, tied_semantically_valid)" in source


def test_issue474_launcher_is_duplicate_safe_one_l4_and_exact_probe_once() -> None:
    source = LAUNCHER.read_text()
    assert 'APP_NAME = "tam-research-aera-v26-3-issue474-ficem-read-repair5"' in source
    assert 'VOLUME_NAME = "tam-research-data"' in source
    assert 'RESULT_PATH = "/vol/aera-v26/issue474-ficem-read-repair5/result.json"' in source
    assert "MAX_GPU_SECONDS = 300" in source
    assert source.count('gpu="L4"') == 1
    assert source.count("def run_probe()") == 1
    assert source.count("result_path.exists()") >= 2
    assert "refusing duplicate issue474 FICEM read repair5 run" in source
    assert FROZEN_PROBE_BLOB in source
    assert REPAIR5_BACKEND_BLOB in source
    assert SOURCE_MAIN in source
    assert f"SOURCE_CPU_PR = {CPU_PR}" in source
    assert f"SOURCE_CPU_RUN = {CPU_RUN}" in source
    assert f"SOURCE_CPU_JOB = {CPU_JOB}" in source
    assert CPU_HEAD in source
    assert source.count("result = run_ficem_read_probe()") == 1
    assert "run_ficem_read_probe(" not in source.replace("result = run_ficem_read_probe()", "")


def test_issue474_launcher_persists_before_compact_authoritative_marker() -> None:
    source = LAUNCHER.read_text()
    write = source.index("result_path.write_text(durable_json)")
    commit = source.index("volume.commit()")
    digest = source.index("result_sha256 = hashlib.sha256")
    marker = source.index("AERA_V26_ISSUE474_FICEM_READ_REPAIR5_RESULT_JSON=")
    summary = source.index("AERA_V26_ISSUE474_FICEM_READ_REPAIR5_SUMMARY_JSON=")
    assert write < commit < digest < marker < summary
    assert '"result_sha256": result_sha256' in source
    assert '"result_path": RESULT_PATH' in source
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
        "final_out_max_abs_diff",
    ):
        assert f'"{key}"' in source


def test_issue474_launcher_has_no_model_training_seed_or_alternate_acceptance_path() -> None:
    source = LAUNCHER.read_text()
    for forbidden in (
        "torch.load(",
        "load_state_dict",
        "torch.optim",
        ".backward(",
        "seed8471",
        "aera.pt",
        "transformer.pt",
        "candidate_ordinal",
        "eligible_case",
        "resample",
        "nudge",
    ):
        assert forbidden not in source
    assert '"synthetic_only": True' in source
    assert '"scientific_seed_consumed": False' in source
    assert '"end_to_end_systems_authorized": False' in source
    assert '"architecture_freeze_authorized": False' in source
    assert '"100m_authorized": False' in source
    assert '"breakthrough_proven": False' in source


def test_issue474_workflow_is_attempt1_exact_bound_and_single_invocation() -> None:
    source = WORKFLOW.read_text()
    assert "issues:\n    types: [opened]" in source
    assert "workflow_dispatch" not in source
    assert "[aera-v26-3-ficem-read-l4-repair5]" in source
    assert 'test "${GITHUB_RUN_ATTEMPT}" = "1"' in source
    assert 'test "${trigger_count}" = "1"' in source
    assert 'test "${marker_count}" = "0"' in source
    assert "Bind main:" in source
    assert 'test "$(git rev-parse HEAD)" = "${bound_main}"' in source
    assert SOURCE_MAIN in source
    assert CPU_HEAD in source
    assert str(CPU_RUN) in source
    assert str(CPU_JOB) in source
    assert FROZEN_PROBE_BLOB in source
    assert REPAIR5_BACKEND_BLOB in source
    assert LAUNCHER_BLOB in source
    assert source.count("modal run modal_aera_v26_3_ficem_read_repair5_app.py") == 1
    assert "AERA_V26_ISSUE474_FICEM_READ_REPAIR5_RESULT_JSON=" in source
    assert "AERA_V26_ISSUE474_FICEM_READ_REPAIR5_SUMMARY_JSON=" in source
    assert "gh run rerun" not in source
    assert "modal deploy" not in source
    assert "cancel-in-progress: false" in source
    assert "automatic retry" in source
    assert "redispatch" in source
    assert "alternate trigger" in source
    assert "timeout increase" in source


def test_issue474_workflow_permissions_are_narrow_and_reporting_is_best_effort() -> None:
    source = WORKFLOW.read_text()
    permissions = source.split("permissions:\n", 1)[1].split("\n\nconcurrency:", 1)[0]
    assert "actions: read" in permissions
    assert "contents: read" in permissions
    assert "issues: write" in permissions
    assert "pull-requests: read" in permissions
    assert "actions: write" not in permissions
    assert "contents: write" not in permissions
    assert "pull-requests: write" not in permissions
    assert source.count("continue-on-error: true") == 3
    assert "Durable Modal result + authoritative marker are the experiment record" in source
