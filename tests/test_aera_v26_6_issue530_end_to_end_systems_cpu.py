from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import torch

from tam_research import aera_v26_5_end_to_end_systems as base
from tam_research import aera_v26_5_end_to_end_systems_repair1 as repair1
from tam_research import aera_v26_6_issue530_end_to_end_systems as gate
from tam_research.aera_hardware_core_v26 import TorchFICEMReferenceBackend
from tam_research.aera_hardware_core_v26_6_ficem_write_materialize_cast import (
    MaterializeCastTritonFICEMReadWriteBackend,
    materialize_cast_ficem_read_write_v26_6_protocol,
)


def _git_blob_sha(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def test_issue530_freezes_authoritative_lineage_and_source_blobs() -> None:
    assert gate.RESEARCH_ISSUE == 530
    assert gate.SOURCE_MAIN == "2c0c28005bff8d9b4f36a96de86144dd74107e39"
    assert gate.BASE_SYSTEMS_BLOB == "c9731cae7e386f09b2a190b045532591c4fa00be"
    assert gate.REPAIR1_SYSTEMS_BLOB == "b3f7082b188644007b873db3733492f424d4941a"
    assert gate.V26_INTERFACE_BLOB == "268644ac4edee15a4cc4e29d3fed7f61eeb3caa7"
    assert gate.READ_BACKEND_BLOB == "263f68eb1186a8ac14a08fc4b4df1fc5b292c711"
    assert (
        gate.HISTORICAL_V26_4_WRITE_BACKEND_BLOB
        == "e54570292489bd17570038dca7518419ac00418c"
    )
    assert gate.STABLE_REFERENCE_BLOB == "4e336b6e1a6238dac782fa320751d68281493ee1"
    assert gate.V26_6_CANDIDATE_BLOB == "d45c262314a0b4691f26812a279937a225043ad9"

    assert _git_blob_sha(Path(base.__file__)) == gate.BASE_SYSTEMS_BLOB
    assert _git_blob_sha(Path(repair1.__file__)) == gate.REPAIR1_SYSTEMS_BLOB
    candidate_module = inspect.getmodule(MaterializeCastTritonFICEMReadWriteBackend)
    assert candidate_module is not None
    assert _git_blob_sha(Path(candidate_module.__file__)) == gate.V26_6_CANDIDATE_BLOB

    assert gate.ISSUE527_TRIGGER == 529
    assert gate.ISSUE527_RUN == 33680028132
    assert gate.ISSUE527_JOB == 100414089065
    assert gate.ISSUE527_ATTEMPT == 1
    assert gate.ISSUE527_BOUND_MAIN == gate.SOURCE_MAIN
    assert (
        gate.ISSUE527_RESULT_SHA256
        == "a07790c2d55d7a696baf0e903a05dbdf925e3ef2a08b84c784c77d1fbdd31874"
    )
    assert gate.ISSUE527_ORACLE_BLOB == "8f472451af4024bb3faacb56d814f7d6bdb25cc9"
    assert gate.ISSUE527_PROBE_BLOB == "bcfeb6a93ed062b7d00359603dc9fbc7aca5767f"

    assert gate.ISSUE508_TRIGGER == 510
    assert gate.ISSUE508_RUN == 33661498305
    assert gate.ISSUE508_JOB == 100352870198
    assert gate.ISSUE508_ATTEMPT == 1
    assert gate.ISSUE508_BOUND_MAIN == "1d475a199cfd2b14d5e94e5cffa29e05ac868ab1"
    assert gate.ISSUE508_FAILURE == "FICEM write state/value floating dtypes must match"


def test_issue530_reuses_entire_frozen_issue501_systems_surface() -> None:
    base_protocol = base.systems_protocol()
    assert base.SYSTEM_BATCH_SIZES == (8, 64)
    assert base.SYSTEM_WARMUP_CALLS == 3
    assert base.SYSTEM_TIMED_CALLS_PER_ROUND == 20
    assert base.SYSTEM_ROUNDS == 5
    assert base.BATCH8_MIN_FULL_SPEED_RATIO == 0.25
    assert base.BATCH64_MIN_FULL_SPEED_RATIO == 1.25
    assert (base.INTEGRATED_ATOL, base.INTEGRATED_RTOL) == (1e-2, 1e-2)
    assert base.EXPECTED_STATE_BYTES == 77_760
    assert (
        base.EXPECTED_SELECTED_WRITES,
        base.EXPECTED_CANDIDATES,
        base.EXPECTED_VECTOR_UPDATES,
    ) == (16, 255, 1)
    assert base.SOURCE_CHECKPOINT_SEED == 8471
    assert base.CHECKPOINT_RELATIVE_DIR == "/vol/aera-real-language/v25-dev-seed8471"
    assert base_protocol["random_token_seed_rule"] == "138471 + 10000 + batch_size"
    assert base_protocol["timing_order"] == "rotated interleaved conditions per issue381"
    assert base_protocol["timing_clock"] == "CUDA events with synchronize before/after"
    assert base_protocol["hard"] is True
    assert base_protocol["route_mode"] == "hard_sparse"
    assert base_protocol["physically_real_sparse_required"] is True
    assert base_protocol["dense_masked_sparse_credit"] is False

    protocol = gate.issue530_systems_protocol()
    assert protocol["batch_sizes"] == [8, 64]
    assert protocol["warmup_calls"] == 3
    assert protocol["timed_calls_per_round"] == 20
    assert protocol["rounds"] == 5
    assert protocol["batch8_min_full_speed_ratio"] == 0.25
    assert protocol["batch64_min_full_speed_ratio"] == 1.25
    assert protocol["integrated_atol"] == 1e-2
    assert protocol["integrated_rtol"] == 1e-2
    assert protocol["persistent_state_bytes_per_session"] == 77_760
    assert protocol["production_write_geometry"] == [16, 255, 1]
    assert protocol["source_checkpoint_seed"] == 8471
    assert protocol["checkpoint_relative_dir"] == base.CHECKPOINT_RELATIVE_DIR
    assert protocol["random_token_seed_rule"] == base_protocol["random_token_seed_rule"]
    assert protocol["timing_order"] == base_protocol["timing_order"]
    assert protocol["timing_clock"] == base_protocol["timing_clock"]
    assert protocol["hard"] is True
    assert protocol["route_mode"] == "hard_sparse"
    assert protocol["physically_real_sparse_required"] is True
    assert protocol["dense_masked_sparse_credit"] is False


def test_issue530_candidate_is_exact_v26_6_fieldwise_backend() -> None:
    protocol = gate.issue530_systems_protocol()
    candidate = materialize_cast_ficem_read_write_v26_6_protocol()

    assert protocol["reference_backend"] == TorchFICEMReferenceBackend.name
    assert protocol["candidate_backend"] == MaterializeCastTritonFICEMReadWriteBackend.name
    assert protocol["historical_v26_4_candidate_backend_decision_bearing"] is False
    assert protocol["frozen_loader_v26_4_backend_replaced_before_parameter_snapshot"] is True
    assert protocol["frozen_loader_v26_4_backend_replaced_before_any_model_call"] is True
    assert protocol["only_candidate_semantic_change"] == (
        "execution_backend_v26_4_to_v26_6_materialize_cast"
    )

    assert candidate["backend_name"] == MaterializeCastTritonFICEMReadWriteBackend.name
    assert candidate["repair5_read_backend_blob"] == gate.READ_BACKEND_BLOB
    assert candidate["predecessor_write_backend_blob"] == gate.HISTORICAL_V26_4_WRITE_BACKEND_BLOB
    assert candidate["write_global_cross_field_dtype_equality_required"] is False
    assert candidate["write_supported_float_dtypes"] == ["float32", "bfloat16"]
    assert candidate["write_fieldwise_mixed_dtype_supported"] is True
    assert candidate["write_materialization_output_follows_durable_state_field_dtype"] is True
    assert candidate["write_duplicate_decisions_before_materialization"] is True
    assert candidate["write_materialize_both_branches_cast_to_output_element_type"] is True
    assert candidate["write_materialize_cast_numeric_not_bitcast"] is True
    assert candidate["write_explicit_pre_tail_cast_kernels"] == 0
    assert candidate["write_adjudicate_kernel_changed_by_v26_6"] is False
    assert candidate["write_tail_triton_launches_target"] == 2
    assert candidate["read_backend_changed_by_v26_6"] is False
    assert candidate["write_similarity_einsums_changed_by_v26_6"] is False
    assert candidate["write_value_projection_changed_by_v26_6"] is False
    assert candidate["write_strength_semantics_changed_by_v26_6"] is False
    assert candidate["write_duplicate_semantics_changed_by_v26_6"] is False
    assert candidate["write_state_schema_changed_by_v26_6"] is False
    assert candidate["write_persistent_state_changed_by_v26_6"] is False
    assert candidate["write_training_backend_changed_by_v26_6"] is False


def test_issue530_cpu_preflight_authorizes_no_gpu_or_science() -> None:
    result = gate.cpu_contract_preflight_issue530()
    assert result["gpu_authorized_by_cpu_preflight"] is False
    assert result["model_construction_performed"] is False
    assert result["checkpoint_loaded"] is False
    assert result["scientific_seed_consumed"] is False
    assert result["architecture_freeze_authorized"] is False
    assert result["100m_authorized"] is False
    assert result["breakthrough_proven"] is False

    protocol = result["protocol"]
    assert protocol["issue527_decision"] == "PASS"
    assert protocol["issue527_direct_pass"] == [256, 256]
    assert protocol["issue527_edge_pass"] == [32, 32]
    assert protocol["issue527_public_pass"] == [6, 6]
    assert protocol["issue527_topology_pass"] == [4, 4]
    assert protocol["issue508_authoritative_result_emitted"] is False
    assert protocol["issue508_integrated_failure"] == gate.ISSUE508_FAILURE
    assert protocol["systems_gpu_authorized_by_issue530"] is False
    assert protocol["architecture_freeze_authorized"] is False
    assert protocol["s2_authorized"] is False
    assert protocol["fresh_scientific_seed_authorized"] is False
    assert protocol["independent_replication_credit"] is False
    assert protocol["100m_authorized"] is False
    assert protocol["breakthrough_proven"] is False


def test_issue530_backend_replacement_precedes_any_measurement_without_cuda(monkeypatch) -> None:
    events: list[str] = []

    class FakeReferenceBackend:
        name = TorchFICEMReferenceBackend.name

    class FakeV26Backend:
        name = MaterializeCastTritonFICEMReadWriteBackend.name

    class FakeMemory:
        def __init__(self, backend) -> None:
            self._execution_backend = backend

        @property
        def execution_backend_name(self) -> str:
            return self._execution_backend.name

    class FakeStage:
        def __init__(self, backend) -> None:
            self.memory = FakeMemory(backend)

    class FakeModel:
        def __init__(self, backend) -> None:
            self.stages = [FakeStage(backend), FakeStage(backend)]

        def __call__(self, *args, **kwargs):
            events.append("model_called")
            raise AssertionError("no model call is allowed during issue530 loading")

    reference = FakeModel(FakeReferenceBackend())
    historical_candidate = FakeModel(type("Historical", (), {"name": "historical-v26.4"})())
    transformer = object()

    def fake_load_models(*, run_dir: str, device: torch.device):
        events.append("frozen_loader_returned")
        return reference, historical_candidate, transformer

    monkeypatch.setattr(gate.base, "load_models", fake_load_models)
    monkeypatch.setattr(gate, "CoalescedFICEMMemory", FakeMemory)
    monkeypatch.setattr(gate, "MaterializeCastTritonFICEMReadWriteBackend", FakeV26Backend)

    ref, cand, trans, names = gate.load_models_v26_6(
        run_dir="unused", device=torch.device("cpu")
    )
    assert ref is reference
    assert cand is historical_candidate
    assert trans is transformer
    assert names == (FakeV26Backend.name, FakeV26Backend.name)
    assert all(stage.memory.execution_backend_name == FakeV26Backend.name for stage in cand.stages)
    assert all(
        stage.memory.execution_backend_name == TorchFICEMReferenceBackend.name
        for stage in ref.stages
    )
    assert events == ["frozen_loader_returned"]


def test_issue530_source_preserves_issue503_orchestration_and_frozen_decision_gates() -> None:
    source = inspect.getsource(gate)
    run_source = inspect.getsource(gate.run_end_to_end_systems_v26_6)
    load_source = inspect.getsource(gate.load_models_v26_6)

    assert "@torch.inference_mode" not in run_source
    assert "with torch.inference_mode():" in run_source
    assert "base.load_models(run_dir=run_dir, device=device)" in load_source
    assert "_install_v26_6_candidate_backend(candidate)" in load_source

    load_pos = run_source.index("load_models_v26_6")
    reference_version_pos = run_source.index("reference_versions_before")
    inference_pos = run_source.index("with torch.inference_mode():")
    timed_pos = run_source.index("base._timed_summaries")
    assert load_pos < reference_version_pos < inference_pos < timed_pos

    required_fragments = (
        "triage.DIAGNOSTIC_SEED + 10_000 + batch_size",
        "base.SYSTEM_BATCH_SIZES",
        "base._timed_summaries(calls, batch_size=batch_size)",
        "base._route_signature",
        "triage._routing_accounting",
        "base._logit_equivalence",
        "base._state_equivalence",
        "base._physical_sparse_proof",
        "base._write_geometry",
        "base._finite_output",
        "base._episodic_state_bytes_per_session",
        "base._threshold_for_batch",
        "candidate_ms <= reference_ms",
        "base._peak_vram_mb",
        "base._profile_candidate",
        "base._parameter_versions",
        "base.checkpoint_hashes",
    )
    for fragment in required_fragments:
        assert fragment in run_source

    assert '"v26_6_triton_full_ficem"' in run_source
    assert "memory._execution_backend = TritonFICEMReadWriteBackend()" not in source
    assert "torch.optim" not in source
    assert ".backward(" not in source
    assert "workflow_dispatch" not in source
    assert "modal." not in source
