from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

from tam_research import aera_hardware_core_v26_4_ficem_write_triton as write_backend

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "tam_research" / "aera_hardware_core_v26_4_ficem_write_triton.py"
READ_BACKEND = ROOT / "tam_research" / "aera_hardware_core_v26_3_ficem_read_triton.py"
V26_INTERFACE = ROOT / "tam_research" / "aera_hardware_core_v26.py"
STABLE_REFERENCE = ROOT / "tam_research" / "aera_hardware_core_v25_1_compact.py"
ISSUE488_PROBE = ROOT / "tam_research" / "aera_v26_4_ficem_write_probe.py"

SOURCE_MAIN = "f5338575df16d4c44fd117cc8f1a3d910a60c4e4"
PRE_REPAIR_BACKEND_BLOB = "5d703bbba296328ca2f49407e56192d10541349d"
READ_BACKEND_BLOB = "263f68eb1186a8ac14a08fc4b4df1fc5b292c711"
V26_INTERFACE_BLOB = "268644ac4edee15a4cc4e29d3fed7f61eeb3caa7"
STABLE_REFERENCE_BLOB = "4e336b6e1a6238dac782fa320751d68281493ee1"
ISSUE488_PROBE_BLOB = "7d8c2c4990beb4c7b4a719d02d009ffefe94671f"
EXHAUSTED_GATE_RUN = 33638047466
EXHAUSTED_GATE_JOB = 100273784137


def _blob(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def _adjudicate_source() -> str:
    source = MODULE.read_text()
    start = source.index("def _write_adjudicate_map_kernel(")
    end = source.index("\n    @triton.jit\n    def _write_materialize_kernel(", start)
    return source[start:end]


def test_issue491_freezes_governing_predecessor_and_reference_blobs() -> None:
    assert SOURCE_MAIN == "f5338575df16d4c44fd117cc8f1a3d910a60c4e4"
    assert PRE_REPAIR_BACKEND_BLOB == "5d703bbba296328ca2f49407e56192d10541349d"
    assert _blob(READ_BACKEND) == READ_BACKEND_BLOB
    assert _blob(V26_INTERFACE) == V26_INTERFACE_BLOB
    assert _blob(STABLE_REFERENCE) == STABLE_REFERENCE_BLOB
    assert _blob(ISSUE488_PROBE) == ISSUE488_PROBE_BLOB
    assert EXHAUSTED_GATE_RUN == 33638047466
    assert EXHAUSTED_GATE_JOB == 100273784137


def test_issue491_numeric_threshold_and_fixed_geometry_are_unchanged() -> None:
    assert write_backend.WRITE_COUNT == 16
    assert write_backend.WRITE_CAPACITY == 48
    assert write_backend.WRITE_MEMORY_DIM == 50
    assert write_backend.WRITE_DUPLICATE_SIMILARITY == 0.95
    source = MODULE.read_text()
    assert source.count("@triton.jit") == 2
    assert source.count("_write_adjudicate_map_kernel[(batch,)](") == 1
    assert source.count("_write_materialize_kernel[(batch * WRITE_CAPACITY,)](") == 1
    assert "DUPLICATE_THRESHOLD=WRITE_DUPLICATE_SIMILARITY" in source


def test_issue491_both_duplicate_comparisons_use_input_dtype_visible_threshold() -> None:
    source = _adjudicate_source()
    assert source.count("duplicate_threshold = tl.full(") == 1
    assert "(1,), DUPLICATE_THRESHOLD, incoming_similarity.dtype" in source
    assert source.count(">= duplicate_threshold") == 2
    assert ">= DUPLICATE_THRESHOLD" not in source
    assert "incoming_similarity >= duplicate_threshold" in source
    assert "new_old_similarity >= duplicate_threshold" in source


def test_issue491_preserves_newest_wins_stable_compaction_and_materialization() -> None:
    source = MODULE.read_text()
    assert "later = other_incoming[None, :] > incoming[:, None]" in source
    assert "surviving_new = incoming_valid & ~shadowed" in source
    assert "& surviving_new[:, None]" in source
    assert "keep_old = old_valid & ~duplicate_old" in source
    assert "source_new_order = other_incoming[None, :] >= incoming[:, None]" in source
    assert "prefix = old_other[None, :] <= old_slot[:, None]" in source
    assert "tl.where(from_new, new_key, old_key)" in source
    assert "tl.where(from_new, new_value, old_value)" in source
    assert "tl.where(from_new, new_strength, old_strength)" in source


def test_issue491_keeps_fp32_source_math_and_pytorch_similarity_paths_unchanged() -> None:
    cls_source = inspect.getsource(write_backend.TritonFICEMReadWriteBackend)
    assert cls_source.count("new_values = torch.tanh(memory.v(payload))") == 1
    assert cls_source.count('torch.einsum("bkd,bjd->bkj", new_keys, new_keys)') == 1
    assert cls_source.count('torch.einsum("bkd,bsd->bks", new_keys, normalized_old)') == 1
    assert "new_strengths = strength[..., 0].clamp(0.0, 1.0)" in cls_source
    assert "new_valid = new_strengths > 0.0" in cls_source
    assert "F.normalize(state.keys.detach(), dim=-1)" in cls_source
    assert "float32" not in _adjudicate_source().lower()


def test_issue491_protocol_records_only_threshold_visibility_repair() -> None:
    protocol = write_backend.fused_ficem_read_write_v26_4_protocol()
    assert protocol["duplicate_similarity"] == 0.95
    assert protocol["write_tail_triton_launches_target"] == 2
    assert protocol["write_threshold_input_dtype_visibility_repair1"] is True
    assert protocol["write_numeric_duplicate_threshold_changed_by_repair1"] is False
    assert protocol["float32_write_threshold_semantics_changed_by_repair1"] is False
    assert protocol["write_tail_kernel_count_changed_by_repair1"] is False
    assert protocol["write_threshold_repair1_issue"] == 491
    assert protocol["write_pre_repair_backend_blob"] == PRE_REPAIR_BACKEND_BLOB
    for key in (
        "read_backend_changed_by_v26_4",
        "write_similarity_einsums_changed",
        "write_value_projection_changed",
        "write_strength_semantics_changed",
        "write_duplicate_semantics_changed",
        "write_incoming_order_changed",
        "write_stable_compaction_semantics_changed",
        "write_invalid_storage_semantics_changed",
        "write_training_backend_changed",
        "write_persistent_state_changed",
        "write_persistent_cache",
        "write_gpu_gate_authorized",
        "end_to_end_systems_authorized",
        "architecture_freeze_authorized",
        "s2_authorized",
        "fresh_scientific_seed_authorized",
        "100m_authorized",
        "breakthrough_proven",
    ):
        assert protocol[key] is False


def test_issue491_has_no_fallback_science_or_gate_path() -> None:
    source = MODULE.read_text().lower()
    for forbidden in (
        "workflow_dispatch",
        "modal.",
        "torch.optim",
        ".backward(",
        "checkpoint",
        "resample",
        "retry",
        "seed8471",
    ):
        assert forbidden not in source
    cls_source = inspect.getsource(write_backend.TritonFICEMReadWriteBackend)
    assert cls_source.count("torch.is_grad_enabled() or memory.differentiable_pretraining") == 2
    assert "return self._reference.update_from_projected(" in cls_source
    assert "return self._reference.update(" in cls_source
