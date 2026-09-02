from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

from tam_research import aera_hardware_core_v26_4_ficem_write_triton as predecessor
from tam_research import aera_hardware_core_v26_6_ficem_write_materialize_cast as candidate
from tam_research import aera_v26_5_issue514_ficem_write_mixed_dtype_probe as frozen
from tam_research import aera_v26_6_issue519_ficem_write_materialize_cast_probe as probe

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "tam_research" / "aera_hardware_core_v26_6_ficem_write_materialize_cast.py"
FAILED_V26_5 = ROOT / "tam_research" / "aera_hardware_core_v26_5_ficem_write_mixed_dtype.py"
PREDECESSOR = ROOT / "tam_research" / "aera_hardware_core_v26_4_ficem_write_triton.py"
READ_BACKEND = ROOT / "tam_research" / "aera_hardware_core_v26_3_ficem_read_triton.py"
V26 = ROOT / "tam_research" / "aera_hardware_core_v26.py"
STABLE_REFERENCE = ROOT / "tam_research" / "aera_hardware_core_v25_1_compact.py"
HISTORICAL_PROBE = ROOT / "tam_research" / "aera_v26_4_ficem_write_probe.py"
FROZEN_ISSUE514_PROBE = ROOT / "tam_research" / "aera_v26_5_issue514_ficem_write_mixed_dtype_probe.py"
FROZEN_ISSUE514_LAUNCHER = ROOT / "modal_aera_v26_5_issue514_ficem_write_mixed_dtype_app.py"
FROZEN_ISSUE514_WORKFLOW = ROOT / ".github" / "workflows" / "aera-v26-5-issue514-ficem-write-mixed-dtype-l4.yml"
PROBE = ROOT / "tam_research" / "aera_v26_6_issue519_ficem_write_materialize_cast_probe.py"
LAUNCHER = ROOT / "modal_aera_v26_6_issue519_ficem_write_materialize_cast_app.py"
WORKFLOW = ROOT / ".github" / "workflows" / "aera-v26-6-issue519-ficem-write-materialize-cast-l4.yml"


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


def test_issue519_freezes_candidate_dependencies_consumed_514_and_new_harness_blobs() -> None:
    assert _blob(CANDIDATE) == "d45c262314a0b4691f26812a279937a225043ad9"
    assert _blob(FAILED_V26_5) == "dab24c733eff7aa08e5f818614f7504eaac48dc3"
    assert _blob(PREDECESSOR) == "e54570292489bd17570038dca7518419ac00418c"
    assert _blob(READ_BACKEND) == "263f68eb1186a8ac14a08fc4b4df1fc5b292c711"
    assert _blob(V26) == "268644ac4edee15a4cc4e29d3fed7f61eeb3caa7"
    assert _blob(STABLE_REFERENCE) == "4e336b6e1a6238dac782fa320751d68281493ee1"
    assert _blob(HISTORICAL_PROBE) == "7d8c2c4990beb4c7b4a719d02d009ffefe94671f"
    assert _blob(FROZEN_ISSUE514_PROBE) == "e48dc308bef3b4ef9b6590ab0324db4a50c0f7af"
    assert _blob(FROZEN_ISSUE514_LAUNCHER) == "1ab50f7b184feda61a6f6e1c7553296bed8863a6"
    assert _blob(FROZEN_ISSUE514_WORKFLOW) == "5871b0a12e6168f16b59a1e7f1895feea6e8426c"
    assert _blob(PROBE) == "ec22807434192f58e292bffc3de9828be2b44272"
    assert _blob(LAUNCHER) == "fcb4a98f62f512bca33b7baa600c7d445da767b6"
    assert _blob(WORKFLOW) == "cbda1f9e8a7970d36a4cd33435d46580e4253b8a"


def test_issue519_reuses_full_frozen_514_decision_surface_without_narrowing() -> None:
    assert probe.RESEARCH_ISSUE == 519
    assert probe.SOURCE_MAIN == "cc9f401d7d3b5ed5c75dc8905ffc8f12df32616b"
    assert probe.CANDIDATE_BLOB == "d45c262314a0b4691f26812a279937a225043ad9"
    assert probe.DESIGN_SEED == frozen.DESIGN_SEED == 408_514
    assert probe.MATRIX_MASKS is frozen.MATRIX_MASKS
    assert probe.MATRIX_MASKS == tuple(range(256))
    assert probe.FLOAT_FIELD_NAMES is frozen.FLOAT_FIELD_NAMES
    assert probe.EDGE_FIXTURES is frozen.EDGE_FIXTURES
    assert probe.EDGE_LAYOUTS is frozen.EDGE_LAYOUTS
    assert probe.PUBLIC_LAYOUTS is frozen.PUBLIC_LAYOUTS
    assert probe.BATCH_SIZES is frozen.BATCH_SIZES
    assert (probe.D_MODEL, probe.WRITE_COUNT, probe.CAPACITY, probe.MEMORY_DIM) == (200, 16, 48, 50)
    assert probe.DUPLICATE_THRESHOLD == 0.95
    assert (probe.FP32_ATOL, probe.FP32_RTOL) == (1e-5, 1e-5)
    assert (probe.BF16_ATOL, probe.BF16_RTOL) == (1e-2, 1e-2)
    assert (
        probe.EXPECTED_DIRECT_CASES,
        probe.EXPECTED_EDGE_CASES,
        probe.EXPECTED_PUBLIC_ROWS,
        probe.EXPECTED_TOPOLOGY_ROWS,
    ) == (256, 32, 6, 4)


def test_issue519_candidate_reuses_adjudication_and_versions_only_materialization() -> None:
    assert candidate._write_adjudicate_map_kernel is predecessor._write_adjudicate_map_kernel
    source = CANDIDATE.read_text()
    assert source.count("@triton.jit") == 1
    assert "def _write_materialize_cast_kernel(" in source
    assert "def _write_materialize_kernel(" not in source
    assert source.count(".to(out_keys_ptr.dtype.element_ty)") == 2
    assert source.count(".to(out_values_ptr.dtype.element_ty)") == 2
    assert source.count(".to(out_strengths_ptr.dtype.element_ty)") == 2
    assert "bitcast=True" not in source
    assert "out_keys = torch.empty_like(state.keys)" in source
    assert "out_values = torch.empty_like(state.values)" in source
    assert "out_strengths = torch.empty_like(state.strengths)" in source
    tail = inspect.getsource(candidate.fused_ficem_write_tail_materialize_cast)
    assert tail.count("_write_adjudicate_map_kernel[(batch,)](") == 1
    assert tail.count("_write_materialize_cast_kernel[(batch * WRITE_CAPACITY,)](") == 1
    assert ".to(" not in tail
    assert ".copy_(" not in tail


def test_issue519_probe_changes_only_candidate_and_expected_materializer_identity() -> None:
    source = PROBE.read_text()
    assert "from . import aera_v26_5_issue514_ficem_write_mixed_dtype_probe as frozen" in source
    candidate_tail = _function_source(PROBE, "_candidate_tail")
    assert "fused_ficem_write_tail_materialize_cast(" in candidate_tail
    assert "fused_ficem_write_tail_mixed_dtype(" not in candidate_tail

    run_source = _function_source(PROBE, "run_materialize_cast_write_probe")
    assert "base = frozen._base_matrix_inputs(device)" in run_source
    assert "for mask in MATRIX_MASKS:" in run_source
    assert "dtypes = frozen._mask_dtypes(mask)" in run_source
    assert "for fixture_name in EDGE_FIXTURES:" in run_source
    assert "for layout in EDGE_LAYOUTS:" in run_source
    assert "for layout in PUBLIC_LAYOUTS:" in run_source
    assert "for batch_size in BATCH_SIZES:" in run_source
    assert 'direct_results[f"mask_{mask:03d}"] = row' in run_source
    for forbidden in ("resample", "rejection", "candidate_admission", "while True", "random.choice"):
        assert forbidden not in run_source

    profile = _function_source(PROBE, "_profile_candidate_tail")
    assert '"_write_adjudicate_map_kernel"' in profile
    assert '"_write_materialize_cast_kernel"' in profile
    assert '"_write_materialize_kernel"' not in profile
    assert "len(adjudicate) == 1" in profile
    assert "len(materialize) == 1" in profile
    assert "len(device_events) == 2" in profile
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
        assert forbidden in profile


def test_issue519_reference_and_public_fixture_semantics_are_imported_from_consumed_514() -> None:
    source = PROBE.read_text()
    assert "reference = frozen._reference_tail_durable(memory, inputs)" in source
    assert "projected, normalized_old, payload, strength, state, use_autocast = frozen._make_public_case(" in source
    assert "with torch.no_grad(), frozen._public_context(use_autocast):" in source
    assert "keys=reference_raw.keys.to(dtype=state.keys.dtype)" in source
    assert "values=reference_raw.values.to(dtype=state.values.dtype)" in source
    assert "strengths=reference_raw.strengths.to(dtype=state.strengths.dtype)" in source
    frozen_reference = _function_source(FROZEN_ISSUE514_PROBE, "_reference_tail_durable")
    assert "reference = historical._reference_tail(memory, inputs)" in frozen_reference
    assert frozen_reference.index("historical._reference_tail") < frozen_reference.index("reference.keys.to")


def test_issue519_protocol_keeps_gpu_cpu_preflight_and_all_higher_authorizations_false() -> None:
    contract = probe.cpu_contract_preflight()
    assert contract["gpu_authorized_by_cpu_preflight"] is False
    protocol = contract["protocol"]
    assert protocol["gate_research_issue"] == 519
    assert protocol["decision_surface_reused_from_issue514"] is True
    assert protocol["matrix_case_count"] == 256
    assert protocol["edge_case_count"] == 32
    assert protocol["public_row_count"] == 6
    assert protocol["topology_row_count"] == 4
    assert protocol["expected_adjudicate_kernel"] == "_write_adjudicate_map_kernel"
    assert protocol["expected_materialize_kernel"] == "_write_materialize_cast_kernel"
    assert protocol["performance_threshold_added"] is False
    assert protocol["synthetic_only"] is True
    assert protocol["scientific_seed_consumed"] is False
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


def test_issue519_launcher_is_duplicate_safe_one_l4_300_seconds_and_durable_before_marker() -> None:
    source = LAUNCHER.read_text()
    assert 'APP_NAME = "tam-research-aera-v26-6-issue519-ficem-write-materialize-cast"' in source
    assert 'VOLUME_NAME = "tam-research-data"' in source
    assert 'RESULT_PATH = "/vol/aera-v26/issue519-ficem-write-materialize-cast/result.json"' in source
    assert "MAX_GPU_SECONDS = 300" in source
    assert source.count('gpu="L4"') == 1
    assert "timeout=MAX_GPU_SECONDS" in source
    assert source.count("run_materialize_cast_write_probe()") == 1
    assert source.count("if result_path.exists():") == 2
    assert 'PROBE_BLOB = "ec22807434192f58e292bffc3de9828be2b44272"' in source
    assert 'FROZEN_ISSUE514_RESULT_SHA256 = "c1a8936458c57e975787a27288d3caf494e360ec8ae8acb8d0f5742aef6bf505"' in source
    assert "REPAIR_CPU_RUN = 33668780903" in source
    assert "REPAIR_CPU_JOB = 100376942271" in source
    assert "FROZEN_ISSUE514_RUN = 33664645415" in source
    assert "result_path.write_text(durable_json)" in source
    assert "volume.commit()" in source
    write_pos = source.index("result_path.write_text(durable_json)")
    commit_pos = source.index("volume.commit()", write_pos)
    marker_pos = source.index("print(RESULT_MARKER", commit_pos)
    assert write_pos < commit_pos < marker_pos


def test_issue519_workflow_is_issue_open_attempt1_single_trigger_exact_bound_and_no_manual_dispatch() -> None:
    source = WORKFLOW.read_text()
    assert "issues:\n    types: [opened]" in source
    assert "workflow_dispatch" not in source
    assert "cancel-in-progress: false" in source
    assert "github.event.issue.user.login == github.repository_owner" in source
    assert "startsWith(github.event.issue.title, '[aera-v26-6-issue519-ficem-write-materialize-cast-l4]')" in source
    assert 'test "${GITHUB_RUN_ATTEMPT}" = "1"' in source
    assert 'test "${trigger_count}" = "1"' in source
    assert 'startswith("🔬 **AERA-v26.6 #519 materialize-cast FICEM WRITE result**")' in source
    assert 'test "${report_count}" = "0"' in source
    assert 'test "$(git rev-parse HEAD)" = "${bound_main}"' in source
    assert "git merge-base --is-ancestor cc9f401d7d3b5ed5c75dc8905ffc8f12df32616b HEAD" in source
    assert source.count("modal run modal_aera_v26_6_issue519_ficem_write_materialize_cast_app.py") == 1
    assert "modal deploy" not in source
    assert "continue-on-error: true" in source
    assert "AERA_V26_6_ISSUE519_FICEM_WRITE_MATERIALIZE_CAST_RESULT_JSON=" in source


def test_issue519_workflow_freezes_repair_514_evidence_and_harness_blobs() -> None:
    source = WORKFLOW.read_text()
    for required in (
        "pulls/518",
        "c2edcfeb28efebe3818a92c5e00d53ea72689c75",
        "cc9f401d7d3b5ed5c75dc8905ffc8f12df32616b",
        "actions/runs/33668780903",
        "actions/jobs/100376942271",
        "issues/516",
        "actions/runs/33664645415",
        "actions/jobs/100363263710",
        "d45c262314a0b4691f26812a279937a225043ad9",
        "dab24c733eff7aa08e5f818614f7504eaac48dc3",
        "e54570292489bd17570038dca7518419ac00418c",
        "263f68eb1186a8ac14a08fc4b4df1fc5b292c711",
        "268644ac4edee15a4cc4e29d3fed7f61eeb3caa7",
        "4e336b6e1a6238dac782fa320751d68281493ee1",
        "7d8c2c4990beb4c7b4a719d02d009ffefe94671f",
        "e48dc308bef3b4ef9b6590ab0324db4a50c0f7af",
        "ec22807434192f58e292bffc3de9828be2b44272",
        "fcb4a98f62f512bca33b7baa600c7d445da767b6",
    ):
        assert required in source
