from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from tam_research import aera_v26_7_issue553_ficem_read_mixed_dtype_probe as probe


ROOT = Path(__file__).resolve().parents[1]
V27 = ROOT / "tam_research/aera_hardware_core_v26_7_ficem_read_mixed_dtype.py"
REPAIR5 = ROOT / "tam_research/aera_hardware_core_v26_3_ficem_read_triton.py"
WRITE_V26_6 = ROOT / "tam_research/aera_hardware_core_v26_6_ficem_write_materialize_cast.py"
HISTORICAL_PROBE = ROOT / "tam_research/aera_v26_3_ficem_read_probe.py"
REPAIR5_PROBE = ROOT / "tam_research/aera_v26_3_ficem_read_probe_repair5.py"
SYSTEMS_530 = ROOT / "tam_research/aera_v26_6_issue530_end_to_end_systems.py"
V26_INTERFACE = ROOT / "tam_research/aera_hardware_core_v26.py"
STABLE_REFERENCE = ROOT / "tam_research/aera_hardware_core_v25_1_compact.py"
ISSUE553_PROBE = ROOT / "tam_research/aera_v26_7_issue553_ficem_read_mixed_dtype_probe.py"
LAUNCHER = ROOT / "modal_aera_v26_7_issue553_ficem_read_mixed_dtype_l4_app.py"
WORKFLOW = ROOT / ".github/workflows/aera-v26-7-issue553-ficem-read-mixed-dtype-l4.yml"

SOURCE_MAIN = "89ef42e447fd797146a45cf4ea869e3012542761"
SOURCE_TREE = "c9890b3985976265e68d785ed1ba9854b89eb8a1"
EXPECTED_BLOBS = {
    V27: "d8133c6b204b1ee5f23955255fb2fb09d09bd723",
    REPAIR5: "263f68eb1186a8ac14a08fc4b4df1fc5b292c711",
    WRITE_V26_6: "d45c262314a0b4691f26812a279937a225043ad9",
    HISTORICAL_PROBE: "16aa99b9f6f0a1d11bd7bf5f36b2f0b1fb97047b",
    REPAIR5_PROBE: "6fd6518e10ed1ef4115863f98ac591ffd77ce903",
    SYSTEMS_530: "9d5a3c31f4a3862f96b957540baa2e0ec6a84c6b",
    V26_INTERFACE: "268644ac4edee15a4cc4e29d3fed7f61eeb3caa7",
    STABLE_REFERENCE: "4e336b6e1a6238dac782fa320751d68281493ee1",
    ISSUE553_PROBE: "ff9a47f510be07e8adeff018f327338147163cdb",
    LAUNCHER: "b3630e065c56c93a1b7f6f164416f068ccb2ecac",
    WORKFLOW: "eef7826f1a76a853d9cf745243612dd457d79a10",
}


def _git_blob(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def test_issue553_frozen_blobs_and_source_boundary() -> None:
    assert all(path.exists() for path in EXPECTED_BLOBS)
    assert {_git_blob(path) for path in EXPECTED_BLOBS} == set(EXPECTED_BLOBS.values())
    for path, expected in EXPECTED_BLOBS.items():
        assert _git_blob(path) == expected
    assert probe.SOURCE_MAIN == SOURCE_MAIN
    assert probe.V26_7_BACKEND_BLOB == EXPECTED_BLOBS[V27]
    assert probe.REPAIR5_BACKEND_BLOB == EXPECTED_BLOBS[REPAIR5]
    assert probe.V26_6_WRITE_BACKEND_BLOB == EXPECTED_BLOBS[WRITE_V26_6]
    assert probe.HISTORICAL_PROBE_BLOB == EXPECTED_BLOBS[HISTORICAL_PROBE]
    assert probe.REPAIR5_PROBE_BLOB == EXPECTED_BLOBS[REPAIR5_PROBE]
    assert probe.ISSUE530_SYSTEMS_BLOB == EXPECTED_BLOBS[SYSTEMS_530]
    assert probe.V26_INTERFACE_BLOB == EXPECTED_BLOBS[V26_INTERFACE]
    assert probe.STABLE_REFERENCE_BLOB == EXPECTED_BLOBS[STABLE_REFERENCE]


def test_issue553_cpu_contract_preserves_historical_surface_and_thresholds() -> None:
    contract = probe.cpu_contract_preflight_issue553()
    protocol = contract["protocol"]
    assert contract["gpu_authorized_by_cpu_preflight"] is False
    assert contract["synthetic_only"] is True
    assert contract["scientific_seed_consumed"] is False
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
    assert (
        probe.FP32_ATOL,
        probe.FP32_RTOL,
        probe.BF16_ATOL,
        probe.BF16_RTOL,
    ) == (1e-5, 1e-5, 1e-2, 1e-2)
    assert probe.MAX_GEOMEAN_LATENCY_RATIO == 0.90
    assert probe.MAX_ROW_LATENCY_RATIO == 1.05
    assert probe.MAX_FULL_EVENT_RATIO == 0.75
    assert protocol["historical_surface_preserved"] is True
    assert protocol["historical_surface_candidate_is_v26_7"] is True
    assert protocol["historical_timing_decision_bearing"] is True
    assert protocol["mixed_timing_decision_bearing"] is False


def test_issue553_mixed_layout_contract_is_exact_and_narrow() -> None:
    assert probe.MIXED_LAYOUTS == (
        ("bfloat16", "float32"),
        ("float32", "bfloat16"),
    )
    protocol = probe.issue553_protocol()
    assert protocol["mixed_layouts"] == [
        ["bfloat16", "float32"],
        ["float32", "bfloat16"],
    ]
    assert protocol["mixed_regular_generator_continues_historical_stream"] is True
    assert (
        protocol["integration_bf16_compute_fp32_durable_full_backend_required"]
        is True
    )
    assert (
        protocol["complementary_fp32_compute_bf16_durable_full_backend_required"]
        is False
    )
    v27_source = V27.read_text()
    assert "@triton.jit" not in v27_source
    assert "_ficem_read_tail_kernel" in v27_source
    assert '"read_new_triton_kernels": 0' in v27_source
    assert '"read_tail_triton_launches_target": 1' in v27_source
    assert '"read_strengths_values_dtype_equality_required": True' in v27_source
    assert '"read_global_cross_field_dtype_equality_required": False' in v27_source
    assert '"read_arbitrary_strengths_values_mixing_authorized": False' in v27_source
    assert ".to(" not in v27_source
    assert "_to_copy" not in v27_source
    assert "copy_(" not in v27_source


def test_issue553_probe_preserves_order_and_uses_one_continued_regular_rng() -> None:
    source = ISSUE553_PROBE.read_text()
    assert "generator = torch.Generator().manual_seed(DESIGN_SEED)" in source
    historical_call = source.index("historical = _historical_surface(")
    mixed_rows = source.index("mixed_rows: dict[str, dict[str, Any]] = {}")
    assert historical_call < mixed_rows
    make_case = source[
        source.index("def _make_mixed_case("):
        source.index("def _direct_similarity_from_compute_sources(")
    ]
    ordered_tokens = [
        "identity = frozen._cpu_randn(",
        "context = frozen._cpu_randn(",
        "keys = frozen._cpu_randn(",
        "values = frozen._cpu_randn(",
        "strengths = (",
        'if validity_kind == "full":',
    ]
    positions = [make_case.index(token) for token in ordered_tokens]
    assert positions == sorted(positions)
    run_source = source[source.index("def run_ficem_read_mixed_dtype_probe_v26_7()"):]
    assert "for compute_dtype_name, durable_dtype_name in MIXED_LAYOUTS:" in run_source
    assert "for batch_size in BATCH_SIZES:" in run_source
    assert "for validity_kind in VALIDITY_KINDS:" in run_source
    assert "generator=generator" in run_source
    assert '== ("bfloat16", "float32")' in source
    assert "return _direct_similarity_from_compute_sources(case), None, None" in source
    assert '"latency_decision_bearing": False' in source


def test_issue553_historical_rows_reuse_frozen_helpers_but_execute_v26_7() -> None:
    source = ISSUE553_PROBE.read_text()
    historical = source[
        source.index("def _historical_surface("):
        source.index("def run_ficem_read_mixed_dtype_probe_v26_7()")
    ]
    assert "frozen.make_case(" in historical
    assert "frozen.known_empty_case(" in historical
    assert "frozen._timed_summaries(calls)" in historical
    assert "frozen._cuda_profile(call)" in historical
    assert "MAX_ROW_LATENCY_RATIO" in historical
    assert "MAX_FULL_EVENT_RATIO" in historical
    assert "MAX_GEOMEAN_LATENCY_RATIO" in historical
    assert "fused_ficem_read_tail_mixed_dtype(" in historical
    assert "candidate: MixedDtypeTritonFICEMReadWriteBackend" in historical
    assert "candidate_no_reference_tail_ops_pass" in historical
    assert "single_tail_kernel_pass" in historical


def test_issue553_launcher_is_duplicate_safe_one_l4_and_durable_before_marker() -> None:
    source = LAUNCHER.read_text()
    assert f'SOURCE_MAIN = "{SOURCE_MAIN}"' in source
    assert f'SOURCE_TREE = "{SOURCE_TREE}"' in source
    assert 'APP_NAME = "tam-research-aera-v26-7-issue553-ficem-read-mixed-dtype-l4"' in source
    assert 'VOLUME_NAME = "tam-research-data"' in source
    assert 'RESULT_PATH = "/vol/aera-v26/issue553-ficem-read-mixed-dtype/result.json"' in source
    assert "MAX_GPU_SECONDS = 300" in source
    assert source.count('gpu="L4"') == 1
    assert source.count("run_ficem_read_mixed_dtype_probe_v26_7()") == 1
    assert "if result_path.exists():" in source
    assert "refusing duplicate issue553 run because durable result exists" in source
    write_index = source.index("result_path.write_text(durable_json)")
    commit_index = source.index("volume.commit()", write_index)
    result_marker_index = source.index("print(RESULT_MARKER", commit_index)
    assert write_index < commit_index < result_marker_index
    preflight_index = source.index("check = preflight.remote()")
    gate_index = source.index("result = run_gate.remote()")
    assert preflight_index < gate_index
    for marker in (
        "AERA_V26_7_ISSUE553_FICEM_READ_MIXED_DTYPE_PRECHECK_JSON=",
        "AERA_V26_7_ISSUE553_FICEM_READ_MIXED_DTYPE_PREFLIGHT_JSON=",
        "AERA_V26_7_ISSUE553_FICEM_READ_MIXED_DTYPE_L4_START_JSON=",
        "AERA_V26_7_ISSUE553_FICEM_READ_MIXED_DTYPE_RESULT_JSON=",
        "AERA_V26_7_ISSUE553_FICEM_READ_MIXED_DTYPE_SUMMARY_JSON=",
    ):
        assert marker in source


def test_issue553_workflow_is_canonical_lowest_attempt1_without_concurrency() -> None:
    source = WORKFLOW.read_text()
    assert "issues:\n    types: [opened]" in source
    assert "workflow_dispatch" not in source
    assert "\nconcurrency:" not in source
    assert "startsWith(github.event.issue.title, '[aera-v26-7-issue553-ficem-read-mixed-dtype-l4]')" in source
    assert 'test "${GITHUB_RUN_ATTEMPT}" = "1"' in source
    assert 'test "${#matching_triggers[@]}" -ge 1' in source
    assert 'canonical_trigger="${matching_triggers[0]}"' in source
    assert 'test "${TRIGGER_ISSUE}" = "${canonical_trigger}"' in source
    assert 'test "${TRIGGER_ISSUE}" = "${matching_triggers[0]}"' in source
    assert 'test "${#matching_triggers[@]}" = "1"' not in source
    assert "Bind main:" in source
    assert "git rev-parse HEAD" in source
    assert "Re-verify issue553 canonical trigger immediately before Modal" in source
    assert source.index("Re-verify issue553 canonical trigger immediately before Modal") < source.index("Authenticate Modal")
    assert source.count(
        "modal run modal_aera_v26_7_issue553_ficem_read_mixed_dtype_l4_app.py"
    ) == 1
    assert "modal deploy" not in source
    assert "actions/runs/33686037672" in source
    assert "actions/jobs/100433658768" in source
    assert "actions/runs/33722918693" in source
    assert "actions/jobs/100545652710" in source


def test_issue553_consumed_and_higher_authorization_flags_remain_false() -> None:
    assert probe.ISSUE479_TRIGGER == 484
    assert probe.ISSUE479_RUN == 33618950619
    assert probe.ISSUE479_JOB == 100211244996
    assert probe.ISSUE545_TRIGGER == 550
    assert probe.ISSUE545_RUN == 33686037672
    assert probe.ISSUE545_JOB == 100433658768
    assert probe.ISSUE545_FAILURE == "FICEM read-tail floating dtypes must match"
    protocol = probe.issue553_protocol()
    for key in (
        "gpu_authorized_by_probe_module",
        "end_to_end_systems_authorized",
        "architecture_freeze_authorized",
        "s2_authorized",
        "fresh_scientific_seed_authorized",
        "independent_replication_credit",
        "100m_authorized",
        "breakthrough_proven",
    ):
        assert protocol[key] is False
    assert torch.float32 is not torch.bfloat16
