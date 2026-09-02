from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import torch

from tam_research import aera_hardware_core_v26_4_ficem_write_triton as predecessor
from tam_research import aera_hardware_core_v26_5_ficem_write_mixed_dtype as candidate
from tam_research import aera_v26_4_ficem_write_probe as historical
from tam_research import aera_v26_5_issue514_ficem_write_mixed_dtype_probe as probe

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "tam_research" / "aera_hardware_core_v26_5_ficem_write_mixed_dtype.py"
PREDECESSOR = ROOT / "tam_research" / "aera_hardware_core_v26_4_ficem_write_triton.py"
READ_BACKEND = ROOT / "tam_research" / "aera_hardware_core_v26_3_ficem_read_triton.py"
V26 = ROOT / "tam_research" / "aera_hardware_core_v26.py"
STABLE_REFERENCE = ROOT / "tam_research" / "aera_hardware_core_v25_1_compact.py"
HISTORICAL_PROBE = ROOT / "tam_research" / "aera_v26_4_ficem_write_probe.py"
PROBE = ROOT / "tam_research" / "aera_v26_5_issue514_ficem_write_mixed_dtype_probe.py"
LAUNCHER = ROOT / "modal_aera_v26_5_issue514_ficem_write_mixed_dtype_app.py"
WORKFLOW = ROOT / ".github" / "workflows" / "aera-v26-5-issue514-ficem-write-mixed-dtype-l4.yml"


def _blob(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def _function_source(path: Path, name: str) -> str:
    source = path.read_text()
    start = source.index(f"def {name}(")
    next_def = source.find("\ndef ", start + 4)
    next_class = source.find("\nclass ", start + 4)
    ends = [position for position in (next_def, next_class) if position >= 0]
    return source[start : min(ends) if ends else len(source)]


def test_issue514_freezes_candidate_dependencies_probe_and_harness_blobs() -> None:
    assert _blob(CANDIDATE) == "dab24c733eff7aa08e5f818614f7504eaac48dc3"
    assert _blob(PREDECESSOR) == "e54570292489bd17570038dca7518419ac00418c"
    assert _blob(READ_BACKEND) == "263f68eb1186a8ac14a08fc4b4df1fc5b292c711"
    assert _blob(V26) == "268644ac4edee15a4cc4e29d3fed7f61eeb3caa7"
    assert _blob(STABLE_REFERENCE) == "4e336b6e1a6238dac782fa320751d68281493ee1"
    assert _blob(HISTORICAL_PROBE) == "7d8c2c4990beb4c7b4a719d02d009ffefe94671f"
    assert _blob(PROBE) == "e48dc308bef3b4ef9b6590ab0324db4a50c0f7af"
    assert _blob(LAUNCHER) == "1ab50f7b184feda61a6f6e1c7553296bed8863a6"
    assert _blob(WORKFLOW) == "5871b0a12e6168f16b59a1e7f1895feea6e8426c"


def test_issue514_freezes_exact_matrix_geometry_layouts_and_tolerances() -> None:
    assert probe.RESEARCH_ISSUE == 514
    assert probe.SOURCE_MAIN == "d9bda2bd3143308407c0d11e640d984385eb095a"
    assert probe.DESIGN_SEED == 408_514
    assert (probe.D_MODEL, probe.WRITE_COUNT, probe.CAPACITY, probe.MEMORY_DIM) == (200, 16, 48, 50)
    assert probe.DUPLICATE_THRESHOLD == 0.95
    assert probe.BATCH_SIZES == (8, 64)
    assert probe.MATRIX_MASKS == tuple(range(256))
    assert probe.FLOAT_FIELD_NAMES == (
        "incoming_similarity",
        "old_similarity",
        "new_keys",
        "new_values",
        "new_strengths",
        "state_keys",
        "state_values",
        "state_strengths",
    )
    assert probe.EDGE_LAYOUTS == (
        "all_float32",
        "all_bfloat16",
        "compute_bfloat16_durable_float32",
        "compute_float32_durable_bfloat16",
    )
    assert probe.PUBLIC_LAYOUTS == (
        "float32_inputs_float32_state",
        "bfloat16_inputs_bfloat16_state",
        "bfloat16_autocast_compute_float32_state",
    )
    assert (probe.FP32_ATOL, probe.FP32_RTOL) == (1e-5, 1e-5)
    assert (probe.BF16_ATOL, probe.BF16_RTOL) == (1e-2, 1e-2)
    assert (probe.EXPECTED_DIRECT_CASES, probe.EXPECTED_EDGE_CASES, probe.EXPECTED_PUBLIC_ROWS, probe.EXPECTED_TOPOLOGY_ROWS) == (256, 32, 6, 4)


def test_issue514_reuses_exact_historical_edge_semantics_and_two_kernels() -> None:
    assert probe.EDGE_FIXTURES == historical.EDGE_FIXTURES
    assert candidate._write_adjudicate_map_kernel is predecessor._write_adjudicate_map_kernel
    assert candidate._write_materialize_kernel is predecessor._write_materialize_kernel
    candidate_source = CANDIDATE.read_text()
    assert "@triton.jit" not in candidate_source
    assert "out_keys = torch.empty_like(state.keys)" in candidate_source
    assert "out_values = torch.empty_like(state.values)" in candidate_source
    assert "out_strengths = torch.empty_like(state.strengths)" in candidate_source
    assert ".to(" not in inspect.getsource(candidate.MixedDtypeTritonFICEMReadWriteBackend._inference_update_from_projected)
    assert ".copy_(" not in inspect.getsource(candidate.MixedDtypeTritonFICEMReadWriteBackend._inference_update_from_projected)


def test_issue514_exhaustive_matrix_is_fixed_binary_mask_order_and_not_candidate_admitted() -> None:
    mask_source = _function_source(PROBE, "_mask_dtypes")
    assert "mask < 0 or mask > 255" in mask_source
    assert "tuple(_dtype_from_bit((mask >> index) & 1) for index in range(8))" in mask_source

    run_source = _function_source(PROBE, "run_mixed_dtype_write_probe")
    assert "for mask in MATRIX_MASKS:" in run_source
    assert 'direct_results[f"mask_{mask:03d}"] = row' in run_source
    assert "for fixture_name in EDGE_FIXTURES:" in run_source
    assert "for layout in EDGE_LAYOUTS:" in run_source
    assert "for layout in PUBLIC_LAYOUTS:" in run_source
    assert "for batch_size in BATCH_SIZES:" in run_source
    for forbidden in ("resample", "rejection", "candidate_admission", "while True", "random.choice"):
        assert forbidden not in run_source


def test_issue514_reference_decisions_precede_only_final_durable_field_conversion() -> None:
    source = _function_source(PROBE, "_reference_tail_durable")
    assert "reference = historical._reference_tail(memory, inputs)" in source
    assert source.index("historical._reference_tail") < source.index("reference.keys.to")
    assert source.index("historical._reference_tail") < source.index("reference.values.to")
    assert source.index("historical._reference_tail") < source.index("reference.strengths.to")
    assert "dtype=inputs.state.keys.dtype" in source
    assert "dtype=inputs.state.values.dtype" in source
    assert "dtype=inputs.state.strengths.dtype" in source

    candidate_tail = _function_source(PROBE, "_candidate_tail")
    assert "fused_ficem_write_tail_mixed_dtype(" in candidate_tail


def test_issue514_public_integration_layout_is_fixed_and_reference_cast_is_final_only() -> None:
    make_source = _function_source(PROBE, "_make_public_case")
    integration_start = make_source.index('layout == "bfloat16_autocast_compute_float32_state"')
    integration_tail = make_source[integration_start:]
    assert "key_dtype = payload_dtype = torch.bfloat16" in integration_tail
    assert "strength_dtype = torch.float32" in integration_tail
    assert "state_dtype = torch.float32" in integration_tail
    assert "use_autocast = True" in integration_tail

    run_source = _function_source(PROBE, "_run_public_case")
    ref_call = run_source.index("reference_backend.update_from_projected")
    cand_call = run_source.index("candidate_backend.update_from_projected")
    final_cast = run_source.index("reference_raw.keys.to")
    assert ref_call < final_cast
    assert cand_call < final_cast
    assert "keys=reference_raw.keys.to(dtype=state.keys.dtype)" in run_source
    assert "values=reference_raw.values.to(dtype=state.values.dtype)" in run_source
    assert "strengths=reference_raw.strengths.to(dtype=state.strengths.dtype)" in run_source


def test_issue514_topology_profile_requires_only_two_historical_cuda_kernels() -> None:
    source = _function_source(PROBE, "_profile_candidate_tail")
    assert "TOPOLOGY_WARMUP_CALLS" in source
    assert '"_write_adjudicate_map_kernel"' in source
    assert '"_write_materialize_kernel"' in source
    assert "len(adjudicate) == 1" in source
    assert "len(materialize) == 1" in source
    assert "len(device_events) == 2" in source
    for forbidden in (
        '"aten::to"',
        '"aten::_to_copy"',
        '"aten::copy_"',
        '"aten::cat"',
        '"aten::stack"',
        '"aten::cumsum"',
        '"aten::scatter_add"',
        '"aten::topk"',
    ):
        assert forbidden in source


def test_issue514_protocol_adds_no_performance_threshold_or_higher_authorization() -> None:
    contract = probe.cpu_contract_preflight()
    assert contract["gpu_authorized_by_cpu_preflight"] is False
    protocol = contract["protocol"]
    assert protocol["matrix_case_count"] == 256
    assert protocol["edge_case_count"] == 32
    assert protocol["public_row_count"] == 6
    assert protocol["topology_row_count"] == 4
    assert protocol["performance_threshold_added"] is False
    assert protocol["synthetic_only"] is True
    for key in (
        "end_to_end_systems_authorized",
        "architecture_freeze_authorized",
        "s2_authorized",
        "fresh_scientific_seed_authorized",
        "independent_replication_credit",
        "100m_authorized",
        "breakthrough_proven",
    ):
        assert protocol[key] is False

    source = PROBE.read_text().lower()
    for forbidden in (
        "/vol/aera-real-language",
        "torch.optim",
        ".backward(",
        "optimizer.step",
        "torch.save(",
        "workflow_dispatch",
        "modal.",
    ):
        assert forbidden not in source


def test_issue514_launcher_is_duplicate_safe_one_l4_300_seconds_and_durable_before_marker() -> None:
    source = LAUNCHER.read_text()
    assert 'APP_NAME = "tam-research-aera-v26-5-issue514-ficem-write-mixed-dtype"' in source
    assert 'VOLUME_NAME = "tam-research-data"' in source
    assert 'RESULT_PATH = "/vol/aera-v26/issue514-ficem-write-mixed-dtype/result.json"' in source
    assert "MAX_GPU_SECONDS = 300" in source
    assert source.count('gpu="L4"') == 1
    assert "timeout=MAX_GPU_SECONDS" in source
    assert source.count("run_mixed_dtype_write_probe()") == 1
    assert source.count("if result_path.exists():") == 2
    assert "result_path.write_text(durable_json)" in source
    assert "volume.commit()" in source
    assert source.index("result_path.write_text(durable_json)") < source.index("volume.commit()")
    assert source.index("volume.commit()") < source.index(
        '"AERA_V26_5_ISSUE514_FICEM_WRITE_MIXED_DTYPE_RESULT_JSON="'
    )
    assert 'PROBE_BLOB = "e48dc308bef3b4ef9b6590ab0324db4a50c0f7af"' in source
    assert "REPAIR_CPU_RUN = 33662720255" in source
    assert "EXHAUSTED_508_RUN = 33661498305" in source


def test_issue514_workflow_is_issue_open_attempt1_exact_bound_main_and_no_retry_path() -> None:
    source = WORKFLOW.read_text()
    assert "issues:\n    types: [opened]" in source
    assert "workflow_dispatch" not in source
    assert "cancel-in-progress: false" in source
    assert "github.event.issue.user.login == github.repository_owner" in source
    assert "startsWith(github.event.issue.title, '[aera-v26-5-issue514-ficem-write-mixed-dtype-l4]')" in source
    assert 'test "${GITHUB_RUN_ATTEMPT}" = "1"' in source
    assert 'test "${trigger_count}" = "1"' in source
    assert 'startswith("🔬 **AERA-v26.5 #514 mixed-dtype FICEM WRITE result**")' in source
    assert 'test "${report_count}" = "0"' in source
    assert 'test "$(git rev-parse HEAD)" = "${bound_main}"' in source
    assert "git merge-base --is-ancestor d9bda2bd3143308407c0d11e640d984385eb095a HEAD" in source
    assert source.count("modal run modal_aera_v26_5_issue514_ficem_write_mixed_dtype_app.py") == 1
    assert "modal deploy" not in source
    assert "continue-on-error: true" in source


def test_issue514_workflow_freezes_repair_cpu_success_exhausted_run_and_harness_blobs() -> None:
    source = WORKFLOW.read_text()
    for required in (
        'pulls/513" --jq \'.merged\')" = "true"',
        'pulls/513" --jq \'.head.sha\')" = "2268dd022a3bfcb0eda8ab65a9a6b78874231650"',
        'pulls/513" --jq \'.merge_commit_sha\')" = "d9bda2bd3143308407c0d11e640d984385eb095a"',
        'actions/runs/33662720255" --jq \'.conclusion\')" = "success"',
        'actions/jobs/100356904904" --jq \'.conclusion\')" = "success"',
        'actions/runs/33661498305" --jq \'.conclusion\')" = "failure"',
        'actions/jobs/100352870198" --jq \'.conclusion\')" = "failure"',
        "dab24c733eff7aa08e5f818614f7504eaac48dc3",
        "e54570292489bd17570038dca7518419ac00418c",
        "263f68eb1186a8ac14a08fc4b4df1fc5b292c711",
        "268644ac4edee15a4cc4e29d3fed7f61eeb3caa7",
        "4e336b6e1a6238dac782fa320751d68281493ee1",
        "7d8c2c4990beb4c7b4a719d02d009ffefe94671f",
        "e48dc308bef3b4ef9b6590ab0324db4a50c0f7af",
        "1ab50f7b184feda61a6f6e1c7553296bed8863a6",
    ):
        assert required in source
