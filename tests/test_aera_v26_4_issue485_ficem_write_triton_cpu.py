from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

from tam_research import aera_hardware_core_v26_4_ficem_write_triton as write_backend
from tam_research.aera_hardware_core_v26_3_ficem_read_triton import (
    TritonFICEMReadBackend,
    fused_ficem_read_v26_3_protocol,
)

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "tam_research" / "aera_hardware_core_v26_4_ficem_write_triton.py"
READ_BACKEND = ROOT / "tam_research" / "aera_hardware_core_v26_3_ficem_read_triton.py"
V26_INTERFACE = ROOT / "tam_research" / "aera_hardware_core_v26.py"
STABLE_REFERENCE = ROOT / "tam_research" / "aera_hardware_core_v25_1_compact.py"

SOURCE_MAIN = "8ab27e55270a4d0ff26e9d21674c58ac3a4ce612"
READ_BACKEND_BLOB = "263f68eb1186a8ac14a08fc4b4df1fc5b292c711"
V26_INTERFACE_BLOB = "268644ac4edee15a4cc4e29d3fed7f61eeb3caa7"
STABLE_REFERENCE_BLOB = "4e336b6e1a6238dac782fa320751d68281493ee1"


def _blob(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def _function_source(name: str) -> str:
    source = MODULE.read_text()
    start = source.index(f"def {name}(")
    next_def = source.find("\ndef ", start + 4)
    next_class = source.find("\nclass ", start + 4)
    ends = [p for p in (next_def, next_class) if p >= 0]
    end = min(ends) if ends else len(source)
    return source[start:end]


def test_issue485_freezes_source_main_and_semantic_reference_blobs() -> None:
    assert SOURCE_MAIN == "8ab27e55270a4d0ff26e9d21674c58ac3a4ce612"
    assert _blob(READ_BACKEND) == READ_BACKEND_BLOB
    assert _blob(V26_INTERFACE) == V26_INTERFACE_BLOB
    assert _blob(STABLE_REFERENCE) == STABLE_REFERENCE_BLOB


def test_issue485_composes_exact_repair5_read_backend_without_reimplementing_read() -> None:
    assert issubclass(write_backend.TritonFICEMReadWriteBackend, TritonFICEMReadBackend)
    cls_source = inspect.getsource(write_backend.TritonFICEMReadWriteBackend)
    assert "def read(" not in cls_source
    assert "def update(" in cls_source
    assert "def update_from_projected(" in cls_source

    read_protocol = fused_ficem_read_v26_3_protocol()
    protocol = write_backend.fused_ficem_read_write_v26_4_protocol()
    for key in (
        "bf16_actual_autocast_tail_repair5",
        "bf16_strength_bias_fp32_repair5",
        "bf16_logits_fp32_repair5",
        "bf16_final_weights_fp32_repair5",
        "bf16_recalled_fp32_repair5",
        "bf16_product_rounding_active_after_repair5",
        "float32_path_changed_by_repair5",
        "read_top_k",
        "read_temperature",
        "min_strength",
    ):
        assert protocol[key] == read_protocol[key]
    assert protocol["read_backend_changed_by_v26_4"] is False


def test_issue485_has_exactly_two_fixed_geometry_write_triton_kernels() -> None:
    source = MODULE.read_text()
    assert source.count("@triton.jit") == 2
    assert "def _write_adjudicate_map_kernel(" in source
    assert "def _write_materialize_kernel(" in source
    assert write_backend.WRITE_COUNT == 16
    assert write_backend.WRITE_CAPACITY == 48
    assert write_backend.WRITE_MEMORY_DIM == 50
    assert write_backend.WRITE_SOURCE_COUNT == 64
    assert write_backend.WRITE_DUPLICATE_SIMILARITY == 0.95
    assert "K_BLOCK=16" in source
    assert "CAPACITY_BLOCK=64" in source
    assert "DIM_BLOCK=64" in source
    assert source.count("_write_adjudicate_map_kernel[(batch,)](") == 1
    assert source.count("_write_materialize_kernel[(batch * WRITE_CAPACITY,)](") == 1


def test_issue485_newest_wins_and_old_suppression_direction_are_explicit() -> None:
    source = MODULE.read_text()
    assert "later = other_incoming[None, :] > incoming[:, None]" in source
    assert "& incoming_valid[:, None]" in source
    assert "& other_valid[None, :]" in source
    assert "surviving_new = incoming_valid & ~shadowed" in source
    assert "& surviving_new[:, None]" in source
    assert "duplicate_old" in source
    assert "keep_old = old_valid & ~duplicate_old" in source
    assert "DUPLICATE_THRESHOLD=WRITE_DUPLICATE_SIMILARITY" in source


def test_issue485_encodes_reverse_new_order_and_stable_valid_then_invalid_compaction() -> None:
    source = MODULE.read_text()
    # Original incoming index i is ranked in conceptual reversed order using j >= i.
    assert "source_new_order = other_incoming[None, :] >= incoming[:, None]" in source
    assert "new_valid_rank" in source
    assert "new_invalid_rank" in source
    assert "total_valid + new_invalid_rank" in source

    # Old slots retain their original source order after all reversed new slots.
    assert "prefix = old_other[None, :] <= old_slot[:, None]" in source
    assert "new_valid_count + old_valid_prefix" in source
    assert "total_valid + new_invalid_count + old_invalid_prefix" in source

    # Every retained output gets a source identity. Invalid retained storage is copied
    # from that source; materialization does not synthesize zero durable slots.
    assert "source_map_ptr + batch_row * CAPACITY + new_destination" in source
    assert "source_map_ptr + batch_row * CAPACITY + old_destination" in source
    materialize = source[source.index("def _write_materialize_kernel(") :]
    assert "encoded_source" in materialize
    assert "tl.where(from_new, new_key, old_key)" in materialize
    assert "tl.where(from_new, new_value, old_value)" in materialize
    assert "tl.where(from_new, new_strength, old_strength)" in materialize
    assert "torch.zeros" not in _function_source("fused_ficem_write_tail")


def test_issue485_keeps_learned_projection_and_similarity_math_in_pytorch() -> None:
    cls_source = inspect.getsource(write_backend.TritonFICEMReadWriteBackend)
    assert cls_source.count("new_values = torch.tanh(memory.v(payload))") == 1
    assert cls_source.count('torch.einsum("bkd,bjd->bkj", new_keys, new_keys)') == 1
    assert cls_source.count('torch.einsum("bkd,bsd->bks", new_keys, normalized_old)') == 1
    assert "new_strengths = strength[..., 0].clamp(0.0, 1.0)" in cls_source
    assert "new_valid = new_strengths > 0.0" in cls_source
    assert "memory.address_factors(identity, context)" in cls_source
    assert "F.normalize(state.keys.detach(), dim=-1)" in cls_source


def test_issue485_accelerated_post_similarity_tail_has_no_fragmented_pytorch_rebuild() -> None:
    tail_source = _function_source("fused_ficem_write_tail")
    for forbidden in (
        "torch.topk",
        "torch.cat",
        "torch.stack",
        "torch.cumsum",
        "scatter_add",
        "torch.einsum",
    ):
        assert forbidden not in tail_source
    assert "for " not in tail_source
    assert "while " not in tail_source


def test_issue485_training_and_differentiable_writes_delegate_to_reference() -> None:
    cls_source = inspect.getsource(write_backend.TritonFICEMReadWriteBackend)
    assert cls_source.count("torch.is_grad_enabled() or memory.differentiable_pretraining") == 2
    assert "return self._reference.update_from_projected(" in cls_source
    assert "return self._reference.update(" in cls_source
    assert "_set_known_empty_hint(next_state, False)" in cls_source


def test_issue485_has_no_persistent_state_or_scientific_gpu_authorization() -> None:
    protocol = write_backend.fused_ficem_read_write_v26_4_protocol()
    assert protocol["research_issue"] == 485
    assert protocol["source_main"] == SOURCE_MAIN
    assert protocol["write_count"] == 16
    assert protocol["capacity"] == 48
    assert protocol["memory_dim"] == 50
    assert protocol["duplicate_similarity"] == 0.95
    assert protocol["write_tail_triton_launches_target"] == 2
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

    source = MODULE.read_text().lower()
    for forbidden in (
        "modal.",
        "workflow_dispatch",
        "torch.optim",
        ".backward(",
        "checkpoint",
        "seed8471",
    ):
        assert forbidden not in source
