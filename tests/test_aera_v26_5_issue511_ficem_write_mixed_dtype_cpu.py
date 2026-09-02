from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import torch

from tam_research import aera_hardware_core_v26_4_ficem_write_triton as predecessor
from tam_research import aera_hardware_core_v26_5_ficem_write_mixed_dtype as mixed
from tam_research.aera_hardware_core_v26_3_ficem_read_triton import TritonFICEMReadBackend

ROOT = Path(__file__).resolve().parents[1]
PREDECESSOR = ROOT / "tam_research" / "aera_hardware_core_v26_4_ficem_write_triton.py"
MIXED = ROOT / "tam_research" / "aera_hardware_core_v26_5_ficem_write_mixed_dtype.py"
V26 = ROOT / "tam_research" / "aera_hardware_core_v26.py"
READ_BACKEND = ROOT / "tam_research" / "aera_hardware_core_v26_3_ficem_read_triton.py"
STABLE_REFERENCE = ROOT / "tam_research" / "aera_hardware_core_v25_1_compact.py"
HISTORICAL_SYSTEMS = ROOT / "tam_research" / "aera_v26_5_end_to_end_systems.py"
REPAIRED_SYSTEMS = ROOT / "tam_research" / "aera_v26_5_end_to_end_systems_repair1.py"
EXHAUSTED_LAUNCHER = ROOT / "modal_aera_v26_5_issue508_end_to_end_systems_repair1_app.py"
EXHAUSTED_WORKFLOW = ROOT / ".github" / "workflows" / "aera-v26-5-issue508-e2e-systems-l4-repair1.yml"


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


def test_issue511_freezes_all_governing_historical_blobs() -> None:
    assert mixed.RESEARCH_ISSUE == 511
    assert mixed.SOURCE_MAIN == "1d475a199cfd2b14d5e94e5cffa29e05ac868ab1"
    assert _blob(PREDECESSOR) == "e54570292489bd17570038dca7518419ac00418c"
    assert _blob(V26) == "268644ac4edee15a4cc4e29d3fed7f61eeb3caa7"
    assert _blob(READ_BACKEND) == "263f68eb1186a8ac14a08fc4b4df1fc5b292c711"
    assert _blob(STABLE_REFERENCE) == "4e336b6e1a6238dac782fa320751d68281493ee1"
    assert _blob(HISTORICAL_SYSTEMS) == "c9731cae7e386f09b2a190b045532591c4fa00be"
    assert _blob(REPAIRED_SYSTEMS) == "b3f7082b188644007b873db3733492f424d4941a"
    assert _blob(EXHAUSTED_LAUNCHER) == "5597dbbd79c782420d48ed538ef2669aebfe5fae"
    assert _blob(EXHAUSTED_WORKFLOW) == "556ea59ebc2d95272caa774a9fef62efbf66a302"
    assert _blob(MIXED) == "dab24c733eff7aa08e5f818614f7504eaac48dc3"


def test_issue511_preserves_predecessor_global_equality_as_historical_evidence() -> None:
    predecessor_source = PREDECESSOR.read_text()
    assert "new_keys.dtype\n        == new_values.dtype\n        == new_strengths.dtype\n        == state.keys.dtype" in predecessor_source
    assert 'raise TypeError("FICEM write state/value floating dtypes must match")' in predecessor_source

    successor_source = MIXED.read_text()
    assert "new_keys.dtype\n        == new_values.dtype" not in successor_source
    assert "FICEM write state/value floating dtypes must match" not in successor_source
    assert "supported_write_field_dtypes(t.dtype for t in floating)" in successor_source


def test_issue511_cpu_dtype_contract_accepts_same_and_mixed_fp32_bf16_only() -> None:
    same_fp32 = (torch.float32,) * 8
    same_bf16 = (torch.bfloat16,) * 8
    mixed_fields = (
        torch.bfloat16,
        torch.bfloat16,
        torch.bfloat16,
        torch.bfloat16,
        torch.float32,
        torch.float32,
        torch.float32,
        torch.float32,
    )
    inverse_mixed = tuple(reversed(mixed_fields))
    assert mixed.supported_write_field_dtypes(same_fp32)
    assert mixed.supported_write_field_dtypes(same_bf16)
    assert mixed.supported_write_field_dtypes(mixed_fields)
    assert mixed.supported_write_field_dtypes(inverse_mixed)
    assert not mixed.supported_write_field_dtypes(())
    assert not mixed.supported_write_field_dtypes((torch.float16,))
    assert not mixed.supported_write_field_dtypes((torch.float64,))
    assert not mixed.supported_write_field_dtypes((torch.int64,))


def test_issue511_reuses_exact_historical_read_and_write_kernels() -> None:
    assert issubclass(mixed.MixedDtypeTritonFICEMReadWriteBackend, predecessor.TritonFICEMReadWriteBackend)
    assert issubclass(mixed.MixedDtypeTritonFICEMReadWriteBackend, TritonFICEMReadBackend)
    assert mixed._write_adjudicate_map_kernel is predecessor._write_adjudicate_map_kernel
    assert mixed._write_materialize_kernel is predecessor._write_materialize_kernel

    source = MIXED.read_text()
    assert "@triton.jit" not in source
    assert source.count("_write_adjudicate_map_kernel[(batch,)](") == 1
    assert source.count("_write_materialize_kernel[(batch * WRITE_CAPACITY,)](") == 1
    assert mixed.WRITE_COUNT == predecessor.WRITE_COUNT == 16
    assert mixed.WRITE_CAPACITY == predecessor.WRITE_CAPACITY == 48
    assert mixed.WRITE_MEMORY_DIM == predecessor.WRITE_MEMORY_DIM == 50
    assert mixed.WRITE_DUPLICATE_SIMILARITY == predecessor.WRITE_DUPLICATE_SIMILARITY == 0.95


def test_issue511_materializes_directly_to_each_durable_state_field_dtype() -> None:
    tail = _function_source(MIXED, "fused_ficem_write_tail_mixed_dtype")
    assert "out_keys = torch.empty_like(state.keys)" in tail
    assert "out_values = torch.empty_like(state.values)" in tail
    assert "out_strengths = torch.empty_like(state.strengths)" in tail
    assert "new_keys," in tail
    assert "new_values," in tail
    assert "new_strengths," in tail
    assert "state.keys," in tail
    assert "state.values," in tail
    assert "state.strengths," in tail
    assert ".to(" not in tail
    assert ".copy_(" not in tail


def test_issue511_preserves_predecessor_learned_math_and_duplicate_decisions_before_tail() -> None:
    predecessor_cls = inspect.getsource(predecessor.TritonFICEMReadWriteBackend)
    successor_cls = inspect.getsource(mixed.MixedDtypeTritonFICEMReadWriteBackend)

    required_math = (
        "new_values = torch.tanh(memory.v(payload))",
        "new_strengths = strength[..., 0].clamp(0.0, 1.0)",
        "new_valid = new_strengths > 0.0",
        'incoming_similarity = torch.einsum("bkd,bjd->bkj", new_keys, new_keys)',
        'old_similarity = torch.einsum("bkd,bsd->bks", new_keys, normalized_old)',
    )
    for expression in required_math:
        assert expression in predecessor_cls
        assert expression in successor_cls

    tail_call = successor_cls.index("fused_ficem_write_tail_mixed_dtype(")
    for expression in required_math:
        assert successor_cls.index(expression) < tail_call

    inference_source = inspect.getsource(
        mixed.MixedDtypeTritonFICEMReadWriteBackend._inference_update_from_projected
    )
    assert ".to(" not in inference_source
    assert ".copy_(" not in inference_source
    assert "torch.einsum" in inference_source


def test_issue511_inherits_training_and_update_delegation_without_redefinition() -> None:
    successor_cls = inspect.getsource(mixed.MixedDtypeTritonFICEMReadWriteBackend)
    assert "def update_from_projected(" not in successor_cls
    assert "def update(" not in successor_cls
    assert "def read(" not in successor_cls

    predecessor_cls = inspect.getsource(predecessor.TritonFICEMReadWriteBackend)
    assert predecessor_cls.count("torch.is_grad_enabled() or memory.differentiable_pretraining") == 2
    assert "return self._reference.update_from_projected(" in predecessor_cls
    assert "return self._reference.update(" in predecessor_cls


def test_issue511_protocol_freezes_narrow_repair_and_no_higher_authorization() -> None:
    protocol = mixed.mixed_dtype_ficem_read_write_v26_5_protocol()
    assert protocol["research_issue"] == 511
    assert protocol["predecessor_write_backend_blob"] == "e54570292489bd17570038dca7518419ac00418c"
    assert protocol["historical_v26_4_backend_mutated"] is False
    assert protocol["write_global_cross_field_dtype_equality_required"] is False
    assert protocol["write_supported_float_dtypes"] == ["float32", "bfloat16"]
    assert protocol["write_fieldwise_mixed_dtype_supported"] is True
    assert protocol["write_materialization_output_follows_durable_state_field_dtype"] is True
    assert protocol["write_duplicate_decisions_before_materialization"] is True
    assert protocol["write_explicit_pre_tail_cast_kernels"] == 0
    assert protocol["write_triton_kernel_bodies_changed"] is False
    assert protocol["write_tail_triton_launches_target"] == 2
    for key in (
        "read_backend_changed_by_v26_5",
        "write_similarity_einsums_changed_by_v26_5",
        "write_value_projection_changed_by_v26_5",
        "write_strength_semantics_changed_by_v26_5",
        "write_duplicate_semantics_changed_by_v26_5",
        "write_state_schema_changed_by_v26_5",
        "write_persistent_state_changed_by_v26_5",
        "write_training_backend_changed_by_v26_5",
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

    source = MIXED.read_text().lower()
    for forbidden in (
        "modal.",
        "workflow_dispatch",
        "torch.optim",
        ".backward(",
        "checkpoint",
        "seed8471",
    ):
        assert forbidden not in source
