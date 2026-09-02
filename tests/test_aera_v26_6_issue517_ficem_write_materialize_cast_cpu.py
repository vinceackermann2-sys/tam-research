from __future__ import annotations

import hashlib
import inspect
import itertools
from pathlib import Path

import torch

from tam_research import aera_hardware_core_v26_4_ficem_write_triton as historical
from tam_research import aera_hardware_core_v26_5_ficem_write_mixed_dtype as failed_v26_5
from tam_research import aera_hardware_core_v26_6_ficem_write_materialize_cast as repair
from tam_research.aera_hardware_core_v26_3_ficem_read_triton import TritonFICEMReadBackend

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL = ROOT / "tam_research" / "aera_hardware_core_v26_4_ficem_write_triton.py"
FAILED_V26_5 = ROOT / "tam_research" / "aera_hardware_core_v26_5_ficem_write_mixed_dtype.py"
REPAIR = ROOT / "tam_research" / "aera_hardware_core_v26_6_ficem_write_materialize_cast.py"
READ_BACKEND = ROOT / "tam_research" / "aera_hardware_core_v26_3_ficem_read_triton.py"
V26 = ROOT / "tam_research" / "aera_hardware_core_v26.py"
STABLE_REFERENCE = ROOT / "tam_research" / "aera_hardware_core_v25_1_compact.py"
HISTORICAL_PROBE = ROOT / "tam_research" / "aera_v26_4_ficem_write_probe.py"
ISSUE514_PROBE = ROOT / "tam_research" / "aera_v26_5_issue514_ficem_write_mixed_dtype_probe.py"
ISSUE514_LAUNCHER = ROOT / "modal_aera_v26_5_issue514_ficem_write_mixed_dtype_app.py"
ISSUE514_WORKFLOW = ROOT / ".github" / "workflows" / "aera-v26-5-issue514-ficem-write-mixed-dtype-l4.yml"


def _blob(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def _function_source(path: Path, name: str) -> str:
    source = path.read_text()
    start = source.index(f"def {name}(")
    next_def = source.find("\ndef ", start + 4)
    next_class = source.find("\nclass ", start + 4)
    ends = [position for position in (next_def, next_class) if position >= 0]
    end = min(ends) if ends else len(source)
    return source[start:end]


def test_issue517_freezes_authoritative_514_and_all_historical_inputs() -> None:
    assert repair.RESEARCH_ISSUE == 517
    assert repair.SOURCE_MAIN == "a67ba825cd71ed78cc7294c7c9fed7532a5100ca"
    assert repair.ISSUE514_RUN == 33664645415
    assert repair.ISSUE514_JOB == 100363263710
    assert repair.ISSUE514_RESULT_SHA256 == "c1a8936458c57e975787a27288d3caf494e360ec8ae8acb8d0f5742aef6bf505"
    assert _blob(HISTORICAL) == "e54570292489bd17570038dca7518419ac00418c"
    assert _blob(FAILED_V26_5) == "dab24c733eff7aa08e5f818614f7504eaac48dc3"
    assert _blob(READ_BACKEND) == "263f68eb1186a8ac14a08fc4b4df1fc5b292c711"
    assert _blob(V26) == "268644ac4edee15a4cc4e29d3fed7f61eeb3caa7"
    assert _blob(STABLE_REFERENCE) == "4e336b6e1a6238dac782fa320751d68281493ee1"
    assert _blob(HISTORICAL_PROBE) == "7d8c2c4990beb4c7b4a719d02d009ffefe94671f"
    assert _blob(ISSUE514_PROBE) == "e48dc308bef3b4ef9b6590ab0324db4a50c0f7af"
    assert _blob(ISSUE514_LAUNCHER) == "1ab50f7b184feda61a6f6e1c7553296bed8863a6"
    assert _blob(ISSUE514_WORKFLOW) == "5871b0a12e6168f16b59a1e7f1895feea6e8426c"


def test_issue517_keeps_full_independent_fp32_bf16_field_contract() -> None:
    choices = (torch.float32, torch.bfloat16)
    assignments = tuple(itertools.product(choices, repeat=8))
    assert len(assignments) == 256
    assert all(repair.supported_write_field_dtypes(case) for case in assignments)
    assert not repair.supported_write_field_dtypes(())
    assert not repair.supported_write_field_dtypes((torch.float16,))
    assert not repair.supported_write_field_dtypes((torch.float64,))
    assert not repair.supported_write_field_dtypes((torch.int64,))


def test_issue517_versions_only_materializer_and_reuses_exact_adjudication() -> None:
    assert issubclass(repair.MaterializeCastTritonFICEMReadWriteBackend, historical.TritonFICEMReadWriteBackend)
    assert issubclass(repair.MaterializeCastTritonFICEMReadWriteBackend, TritonFICEMReadBackend)
    assert repair._write_adjudicate_map_kernel is historical._write_adjudicate_map_kernel
    source = REPAIR.read_text()
    assert source.count("@triton.jit") == 1
    assert "_write_materialize_kernel" not in source
    assert source.count("_write_adjudicate_map_kernel[(batch,)](") == 1
    assert source.count("_write_materialize_cast_kernel[(batch * WRITE_CAPACITY,)](") == 1
    assert repair.WRITE_COUNT == historical.WRITE_COUNT == 16
    assert repair.WRITE_CAPACITY == historical.WRITE_CAPACITY == 48
    assert repair.WRITE_MEMORY_DIM == historical.WRITE_MEMORY_DIM == 50
    assert repair.WRITE_DUPLICATE_SIMILARITY == historical.WRITE_DUPLICATE_SIMILARITY == 0.95


def test_issue517_casts_both_materialization_branches_before_where() -> None:
    kernel = _function_source(REPAIR, "_write_materialize_cast_kernel")
    assert kernel.count(".to(out_keys_ptr.dtype.element_ty)") == 2
    assert kernel.count(".to(out_values_ptr.dtype.element_ty)") == 2
    assert kernel.count(".to(out_strengths_ptr.dtype.element_ty)") == 2
    assert "bitcast=True" not in kernel
    assert "bitcast = True" not in kernel

    key_where = kernel.index("tl.where(from_new, new_key, old_key)")
    value_where = kernel.index("tl.where(from_new, new_value, old_value)")
    strength_where = kernel.index("tl.where(from_new, new_strength, old_strength)")
    assert kernel.rindex(".to(out_keys_ptr.dtype.element_ty)", 0, key_where) < key_where
    assert kernel.rindex(".to(out_values_ptr.dtype.element_ty)", 0, value_where) < value_where
    assert kernel.rindex(".to(out_strengths_ptr.dtype.element_ty)", 0, strength_where) < strength_where


def test_issue517_has_no_host_or_extra_kernel_dtype_conversion() -> None:
    tail = _function_source(REPAIR, "fused_ficem_write_tail_materialize_cast")
    assert "out_keys = torch.empty_like(state.keys)" in tail
    assert "out_values = torch.empty_like(state.values)" in tail
    assert "out_strengths = torch.empty_like(state.strengths)" in tail
    assert ".to(" not in tail
    assert ".copy_(" not in tail
    assert "torch.cat" not in tail
    assert "torch.stack" not in tail
    assert tail.count("_write_adjudicate_map_kernel[(batch,)](") == 1
    assert tail.count("_write_materialize_cast_kernel[(batch * WRITE_CAPACITY,)](") == 1


def test_issue517_preserves_v26_5_learned_math_and_duplicate_inputs() -> None:
    failed_cls = inspect.getsource(failed_v26_5.MixedDtypeTritonFICEMReadWriteBackend)
    repair_cls = inspect.getsource(repair.MaterializeCastTritonFICEMReadWriteBackend)
    required_math = (
        "new_values = torch.tanh(memory.v(payload))",
        "new_strengths = strength[..., 0].clamp(0.0, 1.0)",
        "new_valid = new_strengths > 0.0",
        'incoming_similarity = torch.einsum("bkd,bjd->bkj", new_keys, new_keys)',
        'old_similarity = torch.einsum("bkd,bsd->bks", new_keys, normalized_old)',
    )
    for expression in required_math:
        assert expression in failed_cls
        assert expression in repair_cls

    tail_call = repair_cls.index("fused_ficem_write_tail_materialize_cast(")
    for expression in required_math:
        assert repair_cls.index(expression) < tail_call

    inference_source = inspect.getsource(
        repair.MaterializeCastTritonFICEMReadWriteBackend._inference_update_from_projected
    )
    assert ".to(" not in inference_source
    assert ".copy_(" not in inference_source


def test_issue517_inherits_training_reference_delegation_without_redefinition() -> None:
    repair_cls = inspect.getsource(repair.MaterializeCastTritonFICEMReadWriteBackend)
    assert "def update_from_projected(" not in repair_cls
    assert "def update(" not in repair_cls
    assert "def read(" not in repair_cls
    predecessor_cls = inspect.getsource(historical.TritonFICEMReadWriteBackend)
    assert predecessor_cls.count("torch.is_grad_enabled() or memory.differentiable_pretraining") == 2
    assert "return self._reference.update_from_projected(" in predecessor_cls
    assert "return self._reference.update(" in predecessor_cls


def test_issue517_protocol_freezes_repair_scope_and_all_higher_gates_false() -> None:
    protocol = repair.materialize_cast_ficem_read_write_v26_6_protocol()
    assert protocol["research_issue"] == 517
    assert protocol["predecessor_write_backend_blob"] == "e54570292489bd17570038dca7518419ac00418c"
    assert protocol["failed_v26_5_backend_blob"] == "dab24c733eff7aa08e5f818614f7504eaac48dc3"
    assert protocol["issue514_probe_blob"] == "e48dc308bef3b4ef9b6590ab0324db4a50c0f7af"
    assert protocol["issue514_result_sha256"] == "c1a8936458c57e975787a27288d3caf494e360ec8ae8acb8d0f5742aef6bf505"
    assert protocol["historical_v26_4_backend_mutated"] is False
    assert protocol["failed_v26_5_backend_mutated"] is False
    assert protocol["write_supported_float_dtypes"] == ["float32", "bfloat16"]
    assert protocol["write_fieldwise_mixed_dtype_supported"] is True
    assert protocol["write_materialize_both_branches_cast_to_output_element_type"] is True
    assert protocol["write_materialize_cast_numeric_not_bitcast"] is True
    assert protocol["write_explicit_pre_tail_cast_kernels"] == 0
    assert protocol["write_new_triton_kernels"] == 1
    assert protocol["write_adjudicate_kernel_changed_by_v26_6"] is False
    assert protocol["write_materialize_kernel_versioned_by_v26_6"] is True
    assert protocol["write_tail_triton_launches_target"] == 2
    for key in (
        "read_backend_changed_by_v26_6",
        "write_similarity_einsums_changed_by_v26_6",
        "write_value_projection_changed_by_v26_6",
        "write_strength_semantics_changed_by_v26_6",
        "write_duplicate_semantics_changed_by_v26_6",
        "write_state_schema_changed_by_v26_6",
        "write_persistent_state_changed_by_v26_6",
        "write_training_backend_changed_by_v26_6",
        "mixed_dtype_gpu_gate_authorized",
        "end_to_end_systems_authorized",
        "architecture_freeze_authorized",
        "s2_authorized",
        "fresh_scientific_seed_authorized",
        "independent_replication_credit",
        "100m_authorized",
        "breakthrough_proven",
    ):
        assert protocol[key] is False

    source = REPAIR.read_text().lower()
    for forbidden in (
        "modal.",
        "workflow_dispatch",
        "torch.optim",
        ".backward(",
        "seed8471",
    ):
        assert forbidden not in source
