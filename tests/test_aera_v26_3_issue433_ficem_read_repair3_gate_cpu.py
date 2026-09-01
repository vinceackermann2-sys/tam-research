from __future__ import annotations

import hashlib
from pathlib import Path

from tam_research.aera_hardware_core_v26_3_ficem_read_triton import (
    fused_ficem_read_v26_3_protocol,
)
from tam_research.aera_v26_3_ficem_read_probe import (
    BATCH_SIZES,
    BF16_ATOL,
    BF16_RTOL,
    CALLS_PER_ROUND,
    CAPACITY,
    DESIGN_SEED,
    DTYPE_NAMES,
    D_MODEL,
    FP32_ATOL,
    FP32_RTOL,
    MAX_FULL_EVENT_RATIO,
    MAX_GEOMEAN_LATENCY_RATIO,
    MAX_ROW_LATENCY_RATIO,
    MEMORY_DIM,
    TIME,
    TIMED_ROUNDS,
    VALIDITY_KINDS,
    WARMUP_CALLS,
)

ROOT = Path(__file__).resolve().parents[1]
PROBE_PATH = ROOT / "tam_research" / "aera_v26_3_ficem_read_probe.py"
BACKEND_PATH = ROOT / "tam_research" / "aera_hardware_core_v26_3_ficem_read_triton.py"
LAUNCHER_PATH = ROOT / "modal_aera_v26_3_ficem_read_repair3_app.py"
WORKFLOW_PATH = ROOT / ".github" / "workflows" / "aera-v26-3-ficem-read-l4-repair3.yml"
FROZEN_PROBE_GIT_BLOB = "16aa99b9f6f0a1d11bd7bf5f36b2f0b1fb97047b"
MERGED_REPAIR3_BACKEND_GIT_BLOB = "b6b37f0379b280eea4e5c2b16f349951dadc4df9"
SOURCE_MAIN = "7e1346709d3c1eb158c9ec7d621cafdf498da315"
SOURCE_CPU_HEAD = "0ad68f50c5ca937a0de9e4bd1c5464e1c0aeab24"
SOURCE_CPU_RUN = 33503499118
SOURCE_CPU_JOB = 99842027190


def _git_blob_sha(path: Path) -> str:
    data = path.read_bytes()
    header = f"blob {len(data)}\0".encode()
    return hashlib.sha1(header + data).hexdigest()


def _assert_historical_repair3_or_explicit_repair4_successor() -> None:
    protocol = fused_ficem_read_v26_3_protocol()
    if protocol.get("bf16_product_rounding_repair4") is not True:
        assert _git_blob_sha(BACKEND_PATH) == MERGED_REPAIR3_BACKEND_GIT_BLOB
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


def test_issue433_probe_and_merged_repair3_backend_are_byte_frozen():
    assert _git_blob_sha(PROBE_PATH) == FROZEN_PROBE_GIT_BLOB
    _assert_historical_repair3_or_explicit_repair4_successor()


def test_issue433_reuses_exact_issue418_geometry_fixture_timing_and_thresholds():
    assert DESIGN_SEED == 408_411
    assert (D_MODEL, TIME, CAPACITY, MEMORY_DIM) == (200, 256, 48, 50)
    assert BATCH_SIZES == (8, 64)
    assert DTYPE_NAMES == ("float32", "bfloat16")
    assert VALIDITY_KINDS == ("mixed", "full")
    assert (WARMUP_CALLS, TIMED_ROUNDS, CALLS_PER_ROUND) == (10, 5, 100)
    assert (FP32_ATOL, FP32_RTOL) == (1e-5, 1e-5)
    assert (BF16_ATOL, BF16_RTOL) == (1e-2, 1e-2)
    assert MAX_GEOMEAN_LATENCY_RATIO == 0.90
    assert MAX_ROW_LATENCY_RATIO == 1.05
    assert MAX_FULL_EVENT_RATIO == 0.75

    probe = PROBE_PATH.read_text()
    assert "generator = torch.Generator().manual_seed(DESIGN_SEED)" in probe
    assert "case = make_case(" in probe
    assert "while " not in probe.split("def run_ficem_read_probe()", 1)[1]
    assert "candidate_ordinal" not in probe
    assert "eligible_case" not in probe
    assert "resampl" not in probe.lower()
    assert "nudge" not in probe.lower()


def test_issue433_repair3_protocol_is_the_cpu_proven_successor_only():
    protocol = fused_ficem_read_v26_3_protocol()
    assert protocol["bf16_reference_rounding_repair3"] is True
    assert protocol["float32_path_changed_by_repair3"] is False
    assert protocol["capacity"] == 48
    assert protocol["memory_dim"] == 50
    assert protocol["read_top_k"] == 4
    assert protocol["read_temperature"] == 0.10
    assert protocol["min_strength"] == 1e-4
    assert protocol["read_tail_triton_launches_target"] == 1
    assert protocol["write_backend_changed"] is False
    assert protocol["training_backend_changed"] is False
    assert protocol["persistent_state_changed"] is False
    assert protocol["persistent_cache"] is False
    assert protocol["persistent_packed_state"] is False
    assert protocol["gpu_authorized_by_module"] is False
    assert protocol["scientific_training_authorized"] is False
    assert protocol["end_to_end_systems_authorized"] is False
    assert protocol["architecture_freeze_authorized"] is False
    assert protocol["s2_authorized"] is False
    assert protocol["fresh_scientific_seed_authorized"] is False
    assert protocol["100m_authorized"] is False
    assert protocol["breakthrough_proven"] is False


def test_issue433_launcher_is_unique_duplicate_safe_synthetic_only_and_one_l4():
    source = LAUNCHER_PATH.read_text()
    assert 'APP_NAME = "tam-research-aera-v26-3-issue433-ficem-read-repair3"' in source
    assert 'RESULT_PATH = "/vol/aera-v26/issue433-ficem-read-repair3/result.json"' in source
    assert "MAX_GPU_SECONDS = 300" in source
    assert source.count('gpu="L4"') == 1
    assert source.count("def run_probe()") == 1
    assert source.count("result_path.exists()") >= 2
    assert "refusing duplicate issue433 FICEM read repair3 run" in source
    assert FROZEN_PROBE_GIT_BLOB in source
    assert MERGED_REPAIR3_BACKEND_GIT_BLOB in source
    assert "run_ficem_read_probe" in source
    assert "AERA_V26_ISSUE433_FICEM_READ_REPAIR3_RESULT_JSON=" in source
    assert "AERA_V26_ISSUE433_FICEM_READ_REPAIR3_SUMMARY_JSON=" in source
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
    ):
        assert f'"{key}"' in source
    assert "torch.load(" not in source
    assert ".backward(" not in source
    assert "torch.optim" not in source
    assert "seed8471" not in source.lower()


def test_issue433_workflow_is_attempt1_exact_bound_and_has_no_retry_path():
    source = WORKFLOW_PATH.read_text()
    assert "issues:\n    types: [opened]" in source
    assert "workflow_dispatch" not in source
    assert "[aera-v26-3-ficem-read-l4-repair3]" in source
    assert 'if [ "${GITHUB_RUN_ATTEMPT}" != "1" ]; then' in source
    assert "Expected exactly one issue433 FICEM read repair3 trigger" in source
    assert "Bind main:" in source
    assert 'test "$(git rev-parse HEAD)" = "${bound_main}"' in source
    assert f"git merge-base --is-ancestor {SOURCE_MAIN} HEAD" in source
    assert FROZEN_PROBE_GIT_BLOB in source
    assert MERGED_REPAIR3_BACKEND_GIT_BLOB in source
    assert SOURCE_CPU_HEAD in source
    assert str(SOURCE_CPU_RUN) in source
    assert source.count("modal run modal_aera_v26_3_ficem_read_repair3_app.py") == 1
    assert "AERA_V26_ISSUE433_FICEM_READ_REPAIR3_RESULT_JSON=" in source
    assert "gh run rerun" not in source
    assert "modal deploy" not in source
    assert "cancel-in-progress: false" in source
    assert "timeout increase" in source
    assert "redispatch" in source
    assert "alternate trigger" in source


def test_issue433_workflow_permissions_are_minimal_and_reporting_non_authoritative():
    source = WORKFLOW_PATH.read_text()
    permissions = source.split("permissions:\n", 1)[1].split("\n\nconcurrency:", 1)[0]
    assert "actions: read" in permissions
    assert "contents: read" in permissions
    assert "issues: write" in permissions
    assert "contents: write" not in permissions
    assert "actions: write" not in permissions
    assert "pull-requests: write" not in permissions
    assert "Record issue433 result (best effort)" in source
    assert "continue-on-error: true" in source
    assert "Durable Modal result + authoritative result marker" in source


def test_issue433_source_evidence_constants_are_frozen_in_launcher_and_workflow():
    launcher = LAUNCHER_PATH.read_text()
    workflow = WORKFLOW_PATH.read_text()
    assert SOURCE_MAIN in launcher
    assert SOURCE_MAIN in workflow
    assert SOURCE_CPU_HEAD in launcher
    assert SOURCE_CPU_HEAD in workflow
    assert str(SOURCE_CPU_RUN) in launcher
    assert str(SOURCE_CPU_RUN) in workflow
    assert str(SOURCE_CPU_JOB) in launcher