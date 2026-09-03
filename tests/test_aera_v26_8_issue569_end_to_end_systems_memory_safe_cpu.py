from __future__ import annotations

import hashlib
import inspect
import math
from pathlib import Path

import pytest
import torch

from tam_research import aera_v26_5_end_to_end_systems as base
from tam_research import aera_v26_6_issue530_end_to_end_systems as frozen530
from tam_research import aera_v26_8_issue562_end_to_end_systems as issue562
from tam_research import aera_v26_8_issue569_end_to_end_systems_memory_safe as adapter
from tam_research.aera import AERAState


ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "tam_research/aera_v26_8_issue569_end_to_end_systems_memory_safe.py"
CPU_TEST = ROOT / "tests/test_aera_v26_8_issue569_end_to_end_systems_memory_safe_cpu.py"
BASE_SYSTEMS = ROOT / "tam_research/aera_v26_5_end_to_end_systems.py"
REPAIR1_SYSTEMS = ROOT / "tam_research/aera_v26_5_end_to_end_systems_repair1.py"
ISSUE530_SYSTEMS = ROOT / "tam_research/aera_v26_6_issue530_end_to_end_systems.py"
ISSUE562_ADAPTER = ROOT / "tam_research/aera_v26_8_issue562_end_to_end_systems.py"
ISSUE562_CPU_TEST = ROOT / "tests/test_aera_v26_8_issue562_end_to_end_systems_cpu.py"
V26_8 = ROOT / "tam_research/aera_hardware_core_v26_8_ficem_read_mixed_strength_precision.py"
ISSUE564_LAUNCHER = ROOT / "modal_aera_v26_8_issue564_end_to_end_systems_l4_app.py"
ISSUE564_WORKFLOW = ROOT / ".github/workflows/aera-v26-8-issue564-e2e-systems-l4.yml"
ISSUE564_CPU_TEST = ROOT / "tests/test_aera_v26_8_issue564_end_to_end_systems_l4_cpu.py"

SOURCE_MAIN = "6e9471ca86fed0438bd25dd314040a16e637f2be"
SOURCE_TREE = "64f452228c58d5a33549dcf601912f6568f5701c"
EXPECTED_BLOBS = {
    BASE_SYSTEMS: "c9731cae7e386f09b2a190b045532591c4fa00be",
    REPAIR1_SYSTEMS: "b3f7082b188644007b873db3733492f424d4941a",
    ISSUE530_SYSTEMS: "9d5a3c31f4a3862f96b957540baa2e0ec6a84c6b",
    ISSUE562_ADAPTER: "3534103eea21f7c4d9d31798ad34601fd47090d6",
    ISSUE562_CPU_TEST: "2893a86cbdf767cdfa92601503d107d5ca3912fb",
    V26_8: "3575c58d1cd730be77649f087908c51dbf3e6088",
    ISSUE564_LAUNCHER: "3b6268a905b4fd2707d1deaf5950c7a89682b4bd",
    ISSUE564_WORKFLOW: "376e693b2116c027d29211374928a8379667fa87",
    ISSUE564_CPU_TEST: "fdfd1d120651c92d4678a073d8e6dc67ea4c8b05",
    ADAPTER: "1eeaa80adb2ae960e2a8dad06031c4ed5ca99203",
}


def _git_blob(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def _state(*, dtype: torch.dtype = torch.float32) -> base.HardwareAERAState:
    memory = base.ContextualEpisodicMemoryState(
        keys=torch.zeros(2, 3, 4, dtype=dtype),
        values=torch.ones(2, 3, 4, dtype=dtype),
        strengths=torch.full((2, 3), 0.5, dtype=dtype),
        valid=torch.ones(2, 3, dtype=torch.bool),
    )
    stage = AERAState(stream=torch.zeros(2, 4, dtype=dtype), memory=memory)
    return base.HardwareAERAState(stages=[stage])


def _assert_logit_results_equal(expected: dict[str, object], actual: dict[str, object]) -> None:
    assert actual.keys() == expected.keys()
    for key in expected:
        if key == "max_abs" and isinstance(expected[key], float) and math.isnan(expected[key]):
            assert isinstance(actual[key], float) and math.isnan(actual[key])
        else:
            assert actual[key] == expected[key]


def test_issue569_exact_source_frozen_lineage_and_two_file_namespace() -> None:
    assert adapter.SOURCE_MAIN == SOURCE_MAIN
    assert adapter.SOURCE_TREE == SOURCE_TREE
    assert all(path.exists() for path in EXPECTED_BLOBS)
    for path, expected in EXPECTED_BLOBS.items():
        assert _git_blob(path) == expected

    issue569_files = sorted(
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*issue569*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    )
    assert issue569_files == [
        "tam_research/aera_v26_8_issue569_end_to_end_systems_memory_safe.py",
        "tests/test_aera_v26_8_issue569_end_to_end_systems_memory_safe_cpu.py",
    ]


def test_issue569_freezes_consumed_566_and_inert_567_boundaries() -> None:
    assert adapter.ISSUE566_TRIGGER == 566
    assert adapter.ISSUE566_BOUND_MAIN == SOURCE_MAIN
    assert adapter.ISSUE566_RUN == 33737873193
    assert adapter.ISSUE566_JOB == 100592625418
    assert adapter.ISSUE566_ATTEMPT == 1
    assert adapter.ISSUE566_AUTHORITATIVE_RESULT_EMITTED is False
    assert adapter.ISSUE566_L4_STARTED is True
    assert adapter.ISSUE566_FAILURE_TYPE == "torch.OutOfMemoryError"
    assert adapter.ISSUE566_FAILURE == "CUDA out of memory. Tried to allocate 6.14 GiB."
    assert adapter.ISSUE566_FAILURE_SITE.endswith("::_logit_equivalence")
    assert adapter.ISSUE566_FAILURE_EXPRESSION == (
        "float((reference.float() - candidate.float()).abs().max())"
    )

    assert adapter.ISSUE567_TRIGGER == 567
    assert adapter.ISSUE567_RUN == 33737887818
    assert adapter.ISSUE567_JOB == 100592679436
    assert adapter.ISSUE567_CANONICAL is False
    assert adapter.ISSUE567_GPU_STARTED is False

    assert adapter.CHECKPOINT_HASHES == {
        "aera": "f8aa92421801e8f190247e420632be5f0c20bc5ea8bf6bdeefe06686b3a31b30",
        "transformer": "cdd5cab4439a709468d6607d45d82081b33e876b2e40d91d4a38ba139b219dd7",
    }


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_issue569_chunked_logit_equivalence_matches_frozen_small_tensors(
    dtype: torch.dtype,
) -> None:
    reference = torch.linspace(-1.0, 1.0, 3 * 5 * 7, dtype=torch.float32).reshape(3, 5, 7)
    candidate = reference.clone()
    candidate[0, 0, 0] += 0.001
    candidate[1, 2, 3] -= 0.02
    reference = reference.to(dtype)
    candidate = candidate.to(dtype)

    expected = adapter._FROZEN_LOGIT_EQUIVALENCE(reference, candidate)
    actual = adapter.memory_safe_logit_equivalence(reference, candidate)
    _assert_logit_results_equal(expected, actual)


def test_issue569_chunked_logit_equivalence_preserves_metadata_and_nan_semantics() -> None:
    reference = torch.zeros(3, 2, 4, dtype=torch.float32)
    candidate = reference.to(torch.bfloat16)
    expected = adapter._FROZEN_LOGIT_EQUIVALENCE(reference, candidate)
    actual = adapter.memory_safe_logit_equivalence(reference, candidate)
    _assert_logit_results_equal(expected, actual)
    assert actual["dtype_device_shape_exact"] is False
    assert actual["pass"] is False

    reference_nan = torch.zeros(3, 2, 4)
    candidate_nan = reference_nan.clone()
    candidate_nan[2, 1, 3] = float("nan")
    expected_nan = adapter._FROZEN_LOGIT_EQUIVALENCE(reference_nan, candidate_nan)
    actual_nan = adapter.memory_safe_logit_equivalence(reference_nan, candidate_nan)
    _assert_logit_results_equal(expected_nan, actual_nan)


def test_issue569_chunking_is_one_leading_row_and_visits_all_rows(monkeypatch) -> None:
    reference = torch.arange(4 * 3 * 2, dtype=torch.float32).reshape(4, 3, 2)
    candidate = reference.clone()
    seen: list[tuple[int, ...]] = []
    original_allclose = torch.allclose

    def recording_allclose(left: torch.Tensor, right: torch.Tensor, **kwargs):
        seen.append(tuple(left.shape))
        return original_allclose(left, right, **kwargs)

    monkeypatch.setattr(torch, "allclose", recording_allclose)
    result = adapter.memory_safe_logit_equivalence(reference, candidate)

    assert result["pass"] is True
    assert adapter.CHUNK_BATCH_ROWS == 1
    assert seen == [(1, 3, 2)] * 4


def test_issue569_memory_safe_finite_output_matches_frozen_finite_and_nonfinite() -> None:
    finite = {"logits": torch.zeros(2, 5, 7), "state": _state()}
    assert adapter.memory_safe_finite_output(finite) == adapter._FROZEN_FINITE_OUTPUT(finite)
    assert adapter.memory_safe_finite_output(finite) is True

    nonfinite_logits = {"logits": torch.zeros(2, 5, 7), "state": _state()}
    nonfinite_logits["logits"][1, 4, 6] = float("inf")
    assert adapter.memory_safe_finite_output(nonfinite_logits) == adapter._FROZEN_FINITE_OUTPUT(
        nonfinite_logits
    )
    assert adapter.memory_safe_finite_output(nonfinite_logits) is False

    bad_state = _state()
    bad_state.stages[0].memory.keys[0, 0, 0] = float("nan")
    nonfinite_state = {"logits": torch.zeros(2, 5, 7), "state": bad_state}
    assert adapter.memory_safe_finite_output(nonfinite_state) == adapter._FROZEN_FINITE_OUTPUT(
        nonfinite_state
    )
    assert adapter.memory_safe_finite_output(nonfinite_state) is False


def test_issue569_cpu_preflight_preserves_frozen_systems_surface() -> None:
    contract = adapter.cpu_contract_preflight_issue569()
    protocol = contract["protocol"]

    assert contract["gpu_authorized_by_cpu_preflight"] is False
    assert contract["model_construction_performed"] is False
    assert contract["checkpoint_loaded"] is False
    assert contract["systems_measurement_performed"] is False
    assert contract["scientific_seed_consumed"] is False

    assert adapter.CHUNK_BATCH_ROWS == 1
    assert base.SYSTEM_BATCH_SIZES == (8, 64)
    assert (
        base.SYSTEM_WARMUP_CALLS,
        base.SYSTEM_TIMED_CALLS_PER_ROUND,
        base.SYSTEM_ROUNDS,
    ) == (3, 20, 5)
    assert (base.BATCH8_MIN_FULL_SPEED_RATIO, base.BATCH64_MIN_FULL_SPEED_RATIO) == (
        0.25,
        1.25,
    )
    assert (base.INTEGRATED_ATOL, base.INTEGRATED_RTOL) == (1e-2, 1e-2)
    assert base.EXPECTED_STATE_BYTES == 77_760
    assert (
        base.EXPECTED_SELECTED_WRITES,
        base.EXPECTED_CANDIDATES,
        base.EXPECTED_VECTOR_UPDATES,
    ) == (16, 255, 1)
    assert protocol["sampling_or_approximation"] is False
    assert protocol["result_dependent_chunk_sizing"] is False
    assert protocol["all_elements_covered"] is True


def test_issue569_run_substitutes_only_two_helpers_around_one_issue562_call(
    monkeypatch,
) -> None:
    original_logit = adapter._FROZEN_LOGIT_EQUIVALENCE
    original_finite = adapter._FROZEN_FINITE_OUTPUT
    calls = 0

    def fake_run(*, run_dir: str):
        nonlocal calls
        calls += 1
        assert run_dir == "/tmp/never-loaded"
        assert base._logit_equivalence is adapter.memory_safe_logit_equivalence
        assert base._finite_output is adapter.memory_safe_finite_output
        return {"scope": "frozen562"}

    monkeypatch.setattr(adapter, "_FROZEN_ISSUE562_RUN", fake_run)
    monkeypatch.setattr(issue562, "run_end_to_end_systems_v26_8", fake_run)

    result = adapter.run_end_to_end_systems_v26_8_memory_safe(run_dir="/tmp/never-loaded")

    assert calls == 1
    assert base._logit_equivalence is original_logit
    assert base._finite_output is original_finite
    assert result["issue569_adapter_metadata"]["frozen_issue562_scope"] == "frozen562"
    assert result["issue569_adapter_metadata"]["chunk_batch_rows"] == 1
    assert result["issue569_adapter_metadata"]["scientific_seed_consumed"] is False


def test_issue569_static_wrapper_contains_no_copied_systems_equations_or_gpu_surface() -> None:
    run_source = inspect.getsource(adapter.run_end_to_end_systems_v26_8_memory_safe)
    assert run_source.count("_FROZEN_ISSUE562_RUN(run_dir=run_dir)") == 1
    assert "base._logit_equivalence = memory_safe_logit_equivalence" in run_source
    assert "base._finite_output = memory_safe_finite_output" in run_source
    assert "finally:" in run_source
    assert "base._logit_equivalence = _FROZEN_LOGIT_EQUIVALENCE" in run_source
    assert "base._finite_output = _FROZEN_FINITE_OUTPUT" in run_source

    for forbidden in (
        "_timed_summaries",
        "_route_signature",
        "_state_equivalence",
        "_physical_sparse_proof",
        "_write_geometry",
        "torch.randint",
        "load_models",
    ):
        assert forbidden not in run_source

    module_source = ADAPTER.read_text()
    assert "import modal" not in module_source
    assert "modal.App" not in module_source
    assert 'gpu="L4"' not in module_source
    assert ".remote()" not in module_source
    assert "workflow_dispatch" not in module_source
    assert "torch.cuda" not in module_source
    assert not (ROOT / "modal_aera_v26_8_issue569_end_to_end_systems_l4_app.py").exists()
    assert not (
        ROOT / ".github/workflows/aera-v26-8-issue569-end-to-end-systems-l4.yml"
    ).exists()


def test_issue569_only_chunked_helpers_change_and_higher_authorizations_are_false() -> None:
    logit_source = inspect.getsource(adapter.memory_safe_logit_equivalence)
    finite_source = inspect.getsource(adapter.memory_safe_finite_output)
    assert "range(0, reference.size(0), CHUNK_BATCH_ROWS)" in logit_source
    assert "reference[start:stop].float()" in logit_source
    assert "candidate[start:stop].float()" in logit_source
    assert "torch.allclose(" in logit_source
    assert "equal_nan" not in logit_source
    assert "range(0, logits.size(0), CHUNK_BATCH_ROWS)" in finite_source
    assert "torch.isfinite(logits[start:stop]).all()" in finite_source

    protocol = adapter.issue569_systems_protocol()
    assert protocol["issue566_authoritative_result_emitted"] is False
    assert protocol["issue566_l4_started"] is True
    assert protocol["issue567_canonical"] is False
    assert protocol["issue567_gpu_started"] is False
    for key in (
        "gpu_authorized_by_issue569",
        "end_to_end_systems_executed_by_issue569",
        "architecture_freeze_authorized",
        "s2_authorized",
        "fresh_scientific_seed_authorized",
        "independent_replication_credit",
        "100m_authorized",
        "breakthrough_proven",
    ):
        assert protocol[key] is False
