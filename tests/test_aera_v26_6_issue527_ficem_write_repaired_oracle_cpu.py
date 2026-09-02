from __future__ import annotations

import hashlib
from pathlib import Path

from tam_research import aera_v26_6_issue519_ficem_write_materialize_cast_probe as issue519
from tam_research import aera_v26_6_issue527_ficem_write_repaired_oracle_probe as issue527

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "tam_research" / "aera_hardware_core_v26_6_ficem_write_materialize_cast.py"
ISSUE519_PROBE = ROOT / "tam_research" / "aera_v26_6_issue519_ficem_write_materialize_cast_probe.py"
ORACLE = ROOT / "tam_research" / "aera_v26_6_issue525_mixed_dtype_write_oracle.py"
ORACLE_TEST = ROOT / "tests" / "test_aera_v26_6_issue525_mixed_dtype_write_oracle_cpu.py"
PROBE = ROOT / "tam_research" / "aera_v26_6_issue527_ficem_write_repaired_oracle_probe.py"
LAUNCHER = ROOT / "modal_aera_v26_6_issue527_ficem_write_repaired_oracle_app.py"
WORKFLOW = ROOT / ".github" / "workflows" / "aera-v26-6-issue527-ficem-write-repaired-oracle-l4.yml"

SOURCE_MAIN = "e18aa12f1ddd96ba30f1b3f5e2be67d5f0922116"
CANDIDATE_BLOB = "d45c262314a0b4691f26812a279937a225043ad9"
ISSUE519_PROBE_BLOB = "ec22807434192f58e292bffc3de9828be2b44272"
ORACLE_BLOB = "8f472451af4024bb3faacb56d814f7d6bdb25cc9"
ORACLE_TEST_BLOB = "de3ae08b9db04803359d216f601d5c68dac3a542"
PROBE_BLOB = "bcfeb6a93ed062b7d00359603dc9fbc7aca5767f"
LAUNCHER_BLOB = "f893c4bf7d07d8adcf6a20bb43fe7d8e941e4868"
WORKFLOW_BLOB = "6556ce5675b8a844aa04ec283ecacbe899706b64"
ISSUE519_RESULT_SHA = "b9fba0fca96644ef8db9bc46faf2c73d0c0cc1f1aaac6a321abe2411d3703cd5"
TRIGGER_PREFIX = "[aera-v26-6-issue527-ficem-write-repaired-oracle-l4]"
RESULT_PATH = "/vol/aera-v26/issue527-ficem-write-repaired-oracle/result.json"


def _blob(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def test_issue527_freezes_candidate_oracle_and_gate_blobs() -> None:
    assert _blob(CANDIDATE) == CANDIDATE_BLOB
    assert _blob(ISSUE519_PROBE) == ISSUE519_PROBE_BLOB
    assert _blob(ORACLE) == ORACLE_BLOB
    assert _blob(ORACLE_TEST) == ORACLE_TEST_BLOB
    assert _blob(PROBE) == PROBE_BLOB
    assert _blob(LAUNCHER) == LAUNCHER_BLOB
    assert _blob(WORKFLOW) == WORKFLOW_BLOB
    assert issue527.SOURCE_MAIN == SOURCE_MAIN
    assert issue527.CANDIDATE_BLOB == CANDIDATE_BLOB
    assert issue527.FROZEN_ISSUE519_RESULT_SHA256 == ISSUE519_RESULT_SHA
    assert issue527.ORACLE_BLOB == ORACLE_BLOB


def test_issue527_decision_surface_is_exactly_aliased_to_issue519() -> None:
    assert issue527.DESIGN_SEED == issue519.DESIGN_SEED == 408_514
    assert issue527.D_MODEL == issue519.D_MODEL == 200
    assert issue527.WRITE_COUNT == issue519.WRITE_COUNT == 16
    assert issue527.CAPACITY == issue519.CAPACITY == 48
    assert issue527.MEMORY_DIM == issue519.MEMORY_DIM == 50
    assert issue527.DUPLICATE_THRESHOLD == issue519.DUPLICATE_THRESHOLD == 0.95
    assert issue527.BATCH_SIZES is issue519.BATCH_SIZES
    assert issue527.MATRIX_MASKS is issue519.MATRIX_MASKS
    assert issue527.MATRIX_MASKS == tuple(range(256))
    assert issue527.FLOAT_FIELD_NAMES is issue519.FLOAT_FIELD_NAMES
    assert issue527.EDGE_FIXTURES is issue519.EDGE_FIXTURES
    assert issue527.EDGE_LAYOUTS is issue519.EDGE_LAYOUTS
    assert issue527.PUBLIC_LAYOUTS is issue519.PUBLIC_LAYOUTS
    assert issue527.FP32_ATOL == issue519.FP32_ATOL == 1e-5
    assert issue527.FP32_RTOL == issue519.FP32_RTOL == 1e-5
    assert issue527.BF16_ATOL == issue519.BF16_ATOL == 1e-2
    assert issue527.BF16_RTOL == issue519.BF16_RTOL == 1e-2
    assert (
        issue527.EXPECTED_DIRECT_CASES,
        issue527.EXPECTED_EDGE_CASES,
        issue527.EXPECTED_PUBLIC_ROWS,
        issue527.EXPECTED_TOPOLOGY_ROWS,
    ) == (256, 32, 6, 4)


def test_issue527_only_replaces_direct_edge_reference_and_reuses_candidate_public_topology() -> None:
    source = PROBE.read_text()
    assert "reference = durable_mixed_dtype_reference_tail(memory, inputs)" in source
    assert "candidate = frozen519._candidate_tail(inputs)" in source
    assert "row = frozen519._run_public_case(" in source
    assert "row = frozen519._profile_candidate_tail(representative_inputs[layout])" in source
    assert "frozen519.frozen._base_matrix_inputs(device)" in source
    assert "frozen519.frozen._mask_dtypes(mask)" in source
    assert "frozen519.frozen._layout_dtypes(layout)" in source
    assert "frozen519.frozen.historical.make_edge_fixture(" in source
    assert "for mask in MATRIX_MASKS:" in source
    assert "for fixture_name in EDGE_FIXTURES:" in source
    assert "for layout in EDGE_LAYOUTS:" in source
    assert "for layout in PUBLIC_LAYOUTS:" in source
    assert "for batch_size in BATCH_SIZES:" in source
    assert "MATRIX_MASKS = tuple(" not in source
    assert "DESIGN_SEED = 408" not in source


def test_issue527_protocol_and_cpu_preflight_keep_all_higher_authorizations_false() -> None:
    contract = issue527.cpu_contract_preflight()
    assert contract["gpu_authorized_by_cpu_preflight"] is False
    protocol = contract["protocol"]
    assert protocol["gate_research_issue"] == 527
    assert protocol["source_main"] == SOURCE_MAIN
    assert protocol["decision_surface_reused_from_issue519"] is True
    assert protocol["direct_edge_oracle"] == "issue525_durable_mixed_dtype_reference_tail"
    assert protocol["public_reference_reused_from_issue519"] is True
    assert protocol["topology_contract_reused_from_issue519"] is True
    assert protocol["matrix_case_count"] == 256
    assert protocol["edge_case_count"] == 32
    assert protocol["public_row_count"] == 6
    assert protocol["topology_row_count"] == 4
    assert protocol["performance_threshold_added"] is False
    assert protocol["scientific_seed_consumed"] is False
    for key in (
        "end_to_end_systems_authorized",
        "architecture_freeze_authorized",
        "fresh_scientific_seed_authorized",
        "independent_replication_credit",
        "100m_authorized",
        "breakthrough_proven",
    ):
        assert protocol[key] is False


def test_issue527_launcher_is_one_l4_300s_durable_first_and_refuses_overwrite() -> None:
    source = LAUNCHER.read_text()
    assert f'RESULT_PATH = "{RESULT_PATH}"' in source
    assert 'gpu="L4"' in source
    assert source.count('gpu="L4"') == 1
    assert "MAX_GPU_SECONDS = 300" in source
    assert "timeout=MAX_GPU_SECONDS" in source
    assert "create_if_missing=False" in source
    assert "if result_path.exists():" in source
    assert "frozen_path.read_bytes()" in source
    assert ISSUE519_RESULT_SHA in source
    assert source.count("run_repaired_oracle_write_probe()") == 1
    write_index = source.index("result_path.write_text(durable_json)")
    commit_index = source.index("volume.commit()")
    result_marker_index = source.index("print(RESULT_MARKER", commit_index)
    summary_marker_index = source.index("print(SUMMARY_MARKER", result_marker_index)
    assert write_index < commit_index < result_marker_index < summary_marker_index
    assert '"successor_systems_preregistration_permitted"' not in source


def test_issue527_workflow_is_unique_issue_open_attempt1_and_no_retry_route() -> None:
    source = WORKFLOW.read_text()
    lowered = source.lower()
    assert "issues:\n    types: [opened]" in source
    assert TRIGGER_PREFIX in source
    assert 'test "${GITHUB_RUN_ATTEMPT}" = "1"' in source
    assert 'test "${trigger_count}" = "1"' in source
    assert 'test "$(git rev-parse HEAD)" = "${bound_main}"' in source
    assert f"git merge-base --is-ancestor {SOURCE_MAIN} HEAD" in source
    assert "33672232063" in source and "100388368044" in source
    assert "33675476637" in source and "100398984660" in source
    assert "33676365160" in source and "100401938039" in source
    assert CANDIDATE_BLOB in source
    assert ORACLE_BLOB in source
    assert PROBE_BLOB in source
    assert LAUNCHER_BLOB in source
    assert source.count("modal run modal_aera_v26_6_issue527_ficem_write_repaired_oracle_app.py") == 1
    assert "workflow_dispatch" not in lowered
    assert "gh run rerun" not in lowered
    assert "rerun-failed" not in lowered
    assert "re-run-failed" not in lowered
    assert "modal deploy" not in lowered


def test_issue527_result_claim_boundary_requires_full_surface_pass() -> None:
    source = PROBE.read_text()
    assert "overall_pass = bool(direct_pass and edge_pass and public_pass and topology_pass)" in source
    assert '"decision": "PASS" if overall_pass else "FAIL"' in source
    assert '"repaired_oracle_write_primitive_passed": overall_pass' in source
    assert '"successor_systems_preregistration_permitted": overall_pass' in source
    assert '"end_to_end_systems_authorized": False' in source
    assert '"architecture_freeze_authorized": False' in source
    assert '"fresh_scientific_seed_authorized": False' in source
    assert '"independent_replication_credit": False' in source
    assert '"100m_authorized": False' in source
    assert '"breakthrough_proven": False' in source
