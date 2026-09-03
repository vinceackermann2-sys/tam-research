from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

from tam_research import aera_v26_6_issue530_end_to_end_systems as frozen530
from tam_research import aera_v26_8_issue562_end_to_end_systems as adapter
from tam_research import aera_hardware_core_v26_8_ficem_read_mixed_strength_precision as v26_8
from tam_research.aera_hardware_core_v26 import TorchFICEMReferenceBackend
from tam_research.aera_hardware_core_v26_6_ficem_write_materialize_cast import (
    MaterializeCastTritonFICEMReadWriteBackend,
)


ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "tam_research/aera_v26_8_issue562_end_to_end_systems.py"
BASE_SYSTEMS = ROOT / "tam_research/aera_v26_5_end_to_end_systems.py"
REPAIR1_SYSTEMS = ROOT / "tam_research/aera_v26_5_end_to_end_systems_repair1.py"
ISSUE530_SYSTEMS = ROOT / "tam_research/aera_v26_6_issue530_end_to_end_systems.py"
V26_8 = ROOT / "tam_research/aera_hardware_core_v26_8_ficem_read_mixed_strength_precision.py"
V26_7 = ROOT / "tam_research/aera_hardware_core_v26_7_ficem_read_mixed_dtype.py"
REPAIR5 = ROOT / "tam_research/aera_hardware_core_v26_3_ficem_read_triton.py"
WRITE_V26_6 = ROOT / "tam_research/aera_hardware_core_v26_6_ficem_write_materialize_cast.py"
V26_INTERFACE = ROOT / "tam_research/aera_hardware_core_v26.py"
STABLE_REFERENCE = ROOT / "tam_research/aera_hardware_core_v25_1_compact.py"

SOURCE_MAIN = "75987bfb7976c6a970d63801c6e81b5b4993f544"
SOURCE_TREE = "8f493dbdfe53392d47bbd1addfe2e61aa8dd132d"
EXPECTED_BLOBS = {
    BASE_SYSTEMS: "c9731cae7e386f09b2a190b045532591c4fa00be",
    REPAIR1_SYSTEMS: "b3f7082b188644007b873db3733492f424d4941a",
    ISSUE530_SYSTEMS: "9d5a3c31f4a3862f96b957540baa2e0ec6a84c6b",
    V26_8: "3575c58d1cd730be77649f087908c51dbf3e6088",
    V26_7: "d8133c6b204b1ee5f23955255fb2fb09d09bd723",
    REPAIR5: "263f68eb1186a8ac14a08fc4b4df1fc5b292c711",
    WRITE_V26_6: "d45c262314a0b4691f26812a279937a225043ad9",
    V26_INTERFACE: "268644ac4edee15a4cc4e29d3fed7f61eeb3caa7",
    STABLE_REFERENCE: "4e336b6e1a6238dac782fa320751d68281493ee1",
    ADAPTER: "3534103eea21f7c4d9d31798ad34601fd47090d6",
}


def _git_blob(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def test_issue562_exact_source_and_frozen_blob_lineage() -> None:
    assert all(path.exists() for path in EXPECTED_BLOBS)
    for path, expected in EXPECTED_BLOBS.items():
        assert _git_blob(path) == expected
    assert adapter.SOURCE_MAIN == SOURCE_MAIN
    assert adapter.SOURCE_TREE == SOURCE_TREE
    assert adapter.BASE_SYSTEMS_BLOB == EXPECTED_BLOBS[BASE_SYSTEMS]
    assert adapter.REPAIR1_SYSTEMS_BLOB == EXPECTED_BLOBS[REPAIR1_SYSTEMS]
    assert adapter.ISSUE530_SYSTEMS_BLOB == EXPECTED_BLOBS[ISSUE530_SYSTEMS]
    assert adapter.V26_8_CANDIDATE_BLOB == EXPECTED_BLOBS[V26_8]
    assert adapter.V26_7_PREDECESSOR_BLOB == EXPECTED_BLOBS[V26_7]
    assert adapter.REPAIR5_READ_BLOB == EXPECTED_BLOBS[REPAIR5]
    assert adapter.V26_6_WRITE_BLOB == EXPECTED_BLOBS[WRITE_V26_6]
    assert adapter.V26_INTERFACE_BLOB == EXPECTED_BLOBS[V26_INTERFACE]
    assert adapter.STABLE_REFERENCE_BLOB == EXPECTED_BLOBS[STABLE_REFERENCE]


def test_issue562_freezes_authoritative_primitive_and_consumed_systems_evidence() -> None:
    assert adapter.ISSUE558_TRIGGER == 561
    assert adapter.ISSUE558_RUN == 33733085825
    assert adapter.ISSUE558_JOB == 100577290103
    assert adapter.ISSUE558_ATTEMPT == 1
    assert adapter.ISSUE558_BOUND_MAIN == SOURCE_MAIN
    assert adapter.ISSUE558_RESULT_PATH == (
        "/vol/aera-v26/issue558-ficem-read-mixed-strength-precision/result.json"
    )
    assert adapter.ISSUE558_RESULT_SHA256 == (
        "e1fdc7e6b69a33084ca4b419b5489e755d7a98b12c367775ef19d1127700aa7e"
    )
    assert adapter.ISSUE558_DECISION == "PASS"

    assert adapter.ISSUE553_TRIGGER == 555
    assert adapter.ISSUE553_RUN == 33727540468
    assert adapter.ISSUE553_JOB == 100559866985
    assert adapter.ISSUE553_DECISION == "FAIL"
    assert adapter.ISSUE553_RESULT_SHA256 == (
        "009af31baf70e46eb93b6e7489d62f356a02b727521d3fabe4a7dab2dcf5ab47"
    )

    assert adapter.ISSUE545_TRIGGER == 550
    assert adapter.ISSUE545_RUN == 33686037672
    assert adapter.ISSUE545_JOB == 100433658768
    assert adapter.ISSUE545_ATTEMPT == 1
    assert adapter.ISSUE545_AUTHORITATIVE_RESULT_EMITTED is False
    assert adapter.ISSUE545_FAILURE == "FICEM read-tail floating dtypes must match"

    assert adapter.ISSUE529_TRIGGER == 529
    assert adapter.ISSUE529_RUN == 33680028132
    assert adapter.ISSUE529_JOB == 100414089065
    assert adapter.ISSUE529_RESULT_SHA256 == (
        "a07790c2d55d7a696baf0e903a05dbdf925e3ef2a08b84c784c77d1fbdd31874"
    )


def test_issue562_cpu_preflight_preserves_exact_systems_decision_surface() -> None:
    contract = adapter.cpu_contract_preflight_issue562()
    protocol = contract["protocol"]
    base = frozen530.base

    assert contract["gpu_authorized_by_cpu_preflight"] is False
    assert contract["model_construction_performed"] is False
    assert contract["checkpoint_loaded"] is False
    assert contract["systems_measurement_performed"] is False
    assert contract["scientific_seed_consumed"] is False

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

    assert protocol["frozen_issue530_run_function_reused"] is True
    assert protocol["frozen_issue530_loader_replaced_before_parameter_snapshot"] is True
    assert protocol["frozen_issue530_loader_replaced_before_any_model_call"] is True
    assert protocol["frozen_issue530_candidate_timing_label_retained"] is True
    assert protocol["candidate_backend"] == adapter.StrengthPrecisionTritonFICEMReadWriteBackend.name
    assert protocol["reference_backend"] == TorchFICEMReferenceBackend.name


def test_issue562_v26_8_candidate_is_exact_read_successor_with_inherited_v26_6_write() -> None:
    assert issubclass(
        adapter.StrengthPrecisionTritonFICEMReadWriteBackend,
        MaterializeCastTritonFICEMReadWriteBackend,
    )
    protocol = v26_8.mixed_strength_precision_v26_8_protocol()
    assert protocol["same_dtype_dispatch"] == "historical-repair5"
    assert protocol["same_dtype_arithmetic_changed_by_v26_8"] is False
    assert protocol["same_dtype_kernel_changed_by_v26_8"] is False
    assert protocol["mixed_new_triton_kernels"] == 1
    assert protocol["mixed_tail_triton_launches_target"] == 1
    assert protocol["mixed_strengths_values_dtype_equality_required"] is True
    assert protocol["mixed_host_pre_tail_cast_kernels"] == 0
    assert protocol["write_backend_changed_by_v26_8"] is False
    assert protocol["training_backend_changed_by_v26_8"] is False


def test_issue562_loader_substitutes_backend_immediately_after_frozen_loader() -> None:
    source = inspect.getsource(adapter.load_models_v26_8)
    load_index = source.index("frozen530.base.load_models(")
    install_index = source.index("_install_v26_8_candidate_backend(candidate)")
    reference_check_index = source.index("for stage in reference.stages:")
    assert load_index < install_index < reference_check_index
    assert "_parameter_versions" not in source
    assert "_model_call" not in source
    assert "_transformer_call" not in source
    assert "checkpoint_hashes" not in source
    assert "TorchFICEMReferenceBackend.name" in source

    installer = inspect.getsource(adapter._install_v26_8_candidate_backend)
    assert "candidate.stages" not in installer  # retrieved through getattr, preserving explicit validation
    assert 'getattr(candidate, "stages", None)' in installer
    assert "isinstance(memory, CoalescedFICEMMemory)" in installer
    assert "StrengthPrecisionTritonFICEMReadWriteBackend()" in installer


def test_issue562_run_reuses_byte_frozen_issue530_equation_body_and_restores_globals() -> None:
    assert frozen530.run_end_to_end_systems_v26_6 is adapter._FROZEN_ISSUE530_RUN
    assert frozen530.load_models_v26_6 is adapter._FROZEN_ISSUE530_LOADER
    assert frozen530.issue530_systems_protocol is adapter._FROZEN_ISSUE530_PROTOCOL

    source = inspect.getsource(adapter.run_end_to_end_systems_v26_8)
    assert source.count("_FROZEN_ISSUE530_RUN(run_dir=run_dir)") == 1
    assert "frozen530.load_models_v26_6 = load_models_v26_8" in source
    assert "frozen530.issue530_systems_protocol = issue562_systems_protocol" in source
    assert "finally:" in source
    assert "frozen530.load_models_v26_6 = _FROZEN_ISSUE530_LOADER" in source
    assert "frozen530.issue530_systems_protocol = _FROZEN_ISSUE530_PROTOCOL" in source
    assert "_timed_summaries" not in source
    assert "_route_signature" not in source
    assert "_logit_equivalence" not in source
    assert "_state_equivalence" not in source
    assert "_physical_sparse_proof" not in source
    assert "_write_geometry" not in source
    assert "torch.randint" not in source


def test_issue562_introduces_no_gpu_harness_or_trigger_surface() -> None:
    source = ADAPTER.read_text()
    assert "import modal" not in source
    assert "modal.App" not in source
    assert 'gpu="L4"' not in source
    assert ".remote()" not in source
    assert "workflow_dispatch" not in source
    assert "RESULT_MARKER" not in source
    assert "L4_START_MARKER" not in source

    assert not (ROOT / "modal_aera_v26_8_issue562_end_to_end_systems_l4_app.py").exists()
    assert not (
        ROOT / ".github/workflows/aera-v26-8-issue562-end-to-end-systems-l4.yml"
    ).exists()


def test_issue562_all_higher_authorizations_remain_false() -> None:
    protocol = adapter.issue562_systems_protocol()
    assert protocol["issue558_decision"] == "PASS"
    assert protocol["issue558_overall_pass"] is True
    assert protocol["issue558_mixed_rows_pass"] == [8, 8]
    assert protocol["issue553_decision"] == "FAIL"
    assert protocol["issue545_authoritative_result_emitted"] is False
    for key in (
        "systems_gpu_authorized_by_issue562",
        "end_to_end_systems_executed_by_issue562",
        "architecture_freeze_authorized",
        "s2_authorized",
        "fresh_scientific_seed_authorized",
        "independent_replication_credit",
        "100m_authorized",
        "breakthrough_proven",
    ):
        assert protocol[key] is False
