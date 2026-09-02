from __future__ import annotations

import ast
import hashlib
import inspect
from pathlib import Path

import torch

from tam_research import aera_v26_5_end_to_end_systems as base
from tam_research import aera_v26_5_end_to_end_systems_repair1 as repair1

ROOT = Path(__file__).resolve().parents[1]
BASE_MODULE = ROOT / "tam_research" / "aera_v26_5_end_to_end_systems.py"
REPAIR_MODULE = ROOT / "tam_research" / "aera_v26_5_end_to_end_systems_repair1.py"
V26 = ROOT / "tam_research" / "aera_hardware_core_v26.py"
READ_BACKEND = ROOT / "tam_research" / "aera_hardware_core_v26_3_ficem_read_triton.py"
WRITE_BACKEND = ROOT / "tam_research" / "aera_hardware_core_v26_4_ficem_write_triton.py"
STABLE_REFERENCE = ROOT / "tam_research" / "aera_hardware_core_v25_1_compact.py"


def _blob(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def _qualified_call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        parts = [func.attr]
        value = func.value
        while isinstance(value, ast.Attribute):
            parts.append(value.attr)
            value = value.value
        if isinstance(value, ast.Name):
            parts.append(value.id)
        return ".".join(reversed(parts))
    return ""


def test_issue503_freezes_historical_issue501_and_dependencies_byte_exact() -> None:
    assert repair1.REPAIR_ISSUE == 503
    assert repair1.SOURCE_MAIN == "bd7fe8aab50af30006b7cb8a5f790699736379e0"
    assert repair1.PREDECESSOR_MODULE_BLOB == "c9731cae7e386f09b2a190b045532591c4fa00be"
    assert _blob(BASE_MODULE) == "c9731cae7e386f09b2a190b045532591c4fa00be"
    assert _blob(V26) == "268644ac4edee15a4cc4e29d3fed7f61eeb3caa7"
    assert _blob(READ_BACKEND) == "263f68eb1186a8ac14a08fc4b4df1fc5b292c711"
    assert _blob(WRITE_BACKEND) == "e54570292489bd17570038dca7518419ac00418c"
    assert _blob(STABLE_REFERENCE) == "4e336b6e1a6238dac782fa320751d68281493ee1"


def test_issue503_reuses_every_issue501_systems_constant_and_threshold() -> None:
    assert base.SYSTEM_BATCH_SIZES == (8, 64)
    assert (
        base.SYSTEM_WARMUP_CALLS,
        base.SYSTEM_TIMED_CALLS_PER_ROUND,
        base.SYSTEM_ROUNDS,
    ) == (3, 20, 5)
    assert base.BATCH8_MIN_FULL_SPEED_RATIO == 0.25
    assert base.BATCH64_MIN_FULL_SPEED_RATIO == 1.25
    assert (base.INTEGRATED_ATOL, base.INTEGRATED_RTOL) == (1e-2, 1e-2)
    assert base.EXPECTED_STATE_BYTES == 77_760
    assert (
        base.EXPECTED_SELECTED_WRITES,
        base.EXPECTED_CANDIDATES,
        base.EXPECTED_VECTOR_UPDATES,
    ) == (16, 255, 1)
    assert base.MAX_GPU_SECONDS == 600
    protocol = repair1.repair1_protocol()
    assert protocol["batch_sizes"] == [8, 64]
    assert protocol["random_token_seed_rule"] == "138471 + 10000 + batch_size"
    assert protocol["timing_order"] == "rotated interleaved conditions per issue381"
    assert protocol["timing_clock"] == "CUDA events with synchronize before/after"
    assert protocol["hard"] is True
    assert protocol["route_mode"] == "hard_sparse"
    assert protocol["physically_real_sparse_required"] is True
    assert protocol["dense_masked_sparse_credit"] is False


def test_issue503_successor_reuses_issue501_helpers_instead_of_redefining_architecture() -> None:
    source = REPAIR_MODULE.read_text()
    assert "from . import aera_v26_5_end_to_end_systems as base" in source
    for required in (
        "base.load_models(",
        "base._timed_summaries(",
        "base._reset_execution_counters(",
        "base._route_signature(",
        "base._logit_equivalence(",
        "base._state_equivalence(",
        "base._physical_sparse_proof(",
        "base._write_geometry(",
        "base._episodic_state_bytes_per_session(",
        "base._threshold_for_batch(",
        "base._peak_vram_mb(",
        "base._profile_candidate(",
    ):
        assert required in source
    for forbidden in (
        "HardwareAwareAERATextLMV26(",
        "TritonFICEMReadWriteBackend(",
        "TorchFICEMReferenceBackend(",
        "torch.optim",
        ".backward(",
        "optimizer.step",
        "torch.save(",
    ):
        assert forbidden not in source


def test_issue503_top_level_is_not_inference_decorated_and_measurement_region_is() -> None:
    source = inspect.getsource(repair1.run_end_to_end_systems_repair1)
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
    assert function.decorator_list == []

    inference_withs: list[ast.With] = []
    for node in ast.walk(function):
        if not isinstance(node, ast.With):
            continue
        names = [
            _qualified_call_name(item.context_expr)
            for item in node.items
            if isinstance(item.context_expr, ast.Call)
        ]
        if "torch.inference_mode" in names:
            inference_withs.append(node)
    assert len(inference_withs) == 1
    measurement = inference_withs[0]

    load_call = next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and _qualified_call_name(node) == "base.load_models"
    )
    version_calls = sorted(
        (
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and _qualified_call_name(node) == "base._parameter_versions"
        ),
        key=lambda node: node.lineno,
    )
    assert len(version_calls) == 6
    assert load_call.lineno < measurement.lineno
    assert all(call.lineno < measurement.lineno for call in version_calls[:3])
    assert all(call.lineno > measurement.end_lineno for call in version_calls[3:])

    measured_call_names = {
        _qualified_call_name(node)
        for node in ast.walk(measurement)
        if isinstance(node, ast.Call)
    }
    for required in (
        "base._timed_summaries",
        "base._logit_equivalence",
        "base._state_equivalence",
        "base._physical_sparse_proof",
        "base._peak_vram_mb",
        "base._profile_candidate",
    ):
        assert required in measured_call_names


def test_issue503_normal_parameters_keep_version_counters_across_inference_only_work() -> None:
    torch.manual_seed(503)
    model = torch.nn.Linear(4, 4)
    versions_before = tuple(int(parameter._version) for parameter in model.parameters())
    with torch.inference_mode():
        output = model(torch.ones(2, 4))
        assert output.shape == (2, 4)
    versions_after = tuple(int(parameter._version) for parameter in model.parameters())
    assert versions_after == versions_before


def test_issue503_cpu_preflight_authorizes_no_gpu_or_science() -> None:
    check = repair1.cpu_contract_preflight_repair1()
    assert check["gpu_authorized_by_issue503"] is False
    assert check["model_construction_performed"] is False
    assert check["checkpoint_loaded"] is False
    assert check["scientific_seed_consumed"] is False
    assert check["architecture_freeze_authorized"] is False
    assert check["100m_authorized"] is False
    assert check["breakthrough_proven"] is False
    assert check["predecessor"]["gpu_authorized_by_cpu_preflight"] is False

    protocol = repair1.repair1_protocol()
    assert protocol["top_level_inference_decorated"] is False
    assert protocol["model_construction_outside_inference_mode"] is True
    assert protocol["parameter_version_snapshots_outside_inference_mode"] is True
    assert protocol["measurements_inside_explicit_inference_mode"] is True
    assert protocol["historical_issue501_module_mutated"] is False
    assert protocol["gpu_authorized_by_issue503"] is False
    for key in (
        "architecture_freeze_authorized",
        "s2_authorized",
        "fresh_scientific_seed_authorized",
        "independent_replication_credit",
        "100m_authorized",
        "breakthrough_proven",
    ):
        assert protocol[key] is False


def test_issue503_adds_no_launcher_workflow_or_gpu_execution_path() -> None:
    source = REPAIR_MODULE.read_text().lower()
    for forbidden in (
        "import modal",
        "modal.",
        "workflow_dispatch",
        "github_run_attempt",
        "modal run",
    ):
        assert forbidden not in source
    assert "requires one nvidia l4" in source
    assert "gpu_authorized_by_issue503\": false" not in source
    assert repair1.cpu_contract_preflight_repair1()["gpu_authorized_by_issue503"] is False
