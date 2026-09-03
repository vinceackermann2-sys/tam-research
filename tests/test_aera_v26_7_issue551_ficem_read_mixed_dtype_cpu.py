from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import torch

from tam_research import aera_hardware_core_v26_3_ficem_read_triton as repair5
from tam_research import aera_hardware_core_v26_6_ficem_write_materialize_cast as v26_6
from tam_research import aera_hardware_core_v26_7_ficem_read_mixed_dtype as successor


ROOT = Path(__file__).resolve().parents[1]
REPAIR5_PATH = ROOT / "tam_research" / "aera_hardware_core_v26_3_ficem_read_triton.py"
SUCCESSOR_PATH = ROOT / "tam_research" / "aera_hardware_core_v26_7_ficem_read_mixed_dtype.py"


def _git_blob_sha(path: Path) -> str:
    payload = path.read_bytes()
    header = f"blob {len(payload)}\0".encode()
    return hashlib.sha1(header + payload).hexdigest()


def test_issue551_frozen_source_and_consumed_systems_evidence() -> None:
    assert successor.RESEARCH_ISSUE == 551
    assert successor.SOURCE_MAIN == "383444afa414fa955c46f98f11cf733ddcef656f"
    assert successor.HISTORICAL_REPAIR5_READ_BLOB == (
        "263f68eb1186a8ac14a08fc4b4df1fc5b292c711"
    )
    assert successor.V26_6_WRITE_BACKEND_BLOB == (
        "d45c262314a0b4691f26812a279937a225043ad9"
    )
    assert successor.ISSUE530_SYSTEMS_EVALUATOR_BLOB == (
        "9d5a3c31f4a3862f96b957540baa2e0ec6a84c6b"
    )
    assert successor.ISSUE545_TRIGGER == 550
    assert successor.ISSUE545_RUN == 33686037672
    assert successor.ISSUE545_JOB == 100433658768
    assert successor.ISSUE545_FAILURE == "FICEM read-tail floating dtypes must match"
    assert successor.ISSUE545_AUTHORITATIVE_RESULT_EMITTED is False
    assert successor.ISSUE545_SCIENTIFIC_SEED_CONSUMED is False


def test_historical_repair5_blob_and_overconstraint_remain_untouched() -> None:
    assert _git_blob_sha(REPAIR5_PATH) == successor.HISTORICAL_REPAIR5_READ_BLOB
    source = REPAIR5_PATH.read_text()
    assert (
        "if strengths.dtype != values.dtype or similarity.dtype != values.dtype:"
        in source
    )
    assert 'raise TypeError("FICEM read-tail floating dtypes must match")' in source


def test_issue551_dtype_layout_contract_is_narrow_and_cpu_checkable() -> None:
    allowed = successor.supported_read_dtype_layout

    # Frozen same-dtype routes.
    assert allowed(torch.float32, torch.float32, torch.float32)
    assert allowed(torch.bfloat16, torch.bfloat16, torch.bfloat16)
    assert allowed(torch.float16, torch.float16, torch.float16)

    # Exactly the newly preregistered compute-versus-durable FP32/BF16 layouts.
    assert allowed(torch.bfloat16, torch.float32, torch.float32)
    assert allowed(torch.float32, torch.bfloat16, torch.bfloat16)

    # State strengths and values do not gain an arbitrary mixed-field contract.
    assert not allowed(torch.float32, torch.float32, torch.bfloat16)
    assert not allowed(torch.bfloat16, torch.float32, torch.bfloat16)
    assert not allowed(torch.float16, torch.float32, torch.float32)
    assert not allowed(torch.float32, torch.float16, torch.float16)
    assert not allowed(torch.float64, torch.float32, torch.float32)


def test_successor_reuses_exact_repair5_kernel_and_defines_no_triton_kernel() -> None:
    assert successor._ficem_read_tail_kernel is repair5._ficem_read_tail_kernel
    source = SUCCESSOR_PATH.read_text()
    assert "@triton.jit" not in source
    assert "_ficem_read_tail_kernel[(batch * time,)]" in source
    assert "IS_BF16=similarity.dtype is torch.bfloat16" in source
    assert "SLOT_BLOCK=64" in source
    assert "DIM_BLOCK=64" in source
    assert "num_warps=4" in source


def test_tail_wrapper_adds_no_host_cast_or_copy_preprocessing() -> None:
    source = inspect.getsource(successor.fused_ficem_read_tail_mixed_dtype)
    assert ".to(" not in source
    assert "_to_copy" not in source
    assert "copy_(" not in source
    assert "torch.cat" not in source
    assert "torch.stack" not in source
    assert "_ficem_read_tail_kernel[(batch * time,)]" in source
    assert source.count("_ficem_read_tail_kernel[") == 1


def test_recalled_precision_is_driven_by_repair5_compute_contract() -> None:
    source = inspect.getsource(successor.fused_ficem_read_tail_mixed_dtype)
    assert "similarity.dtype is torch.bfloat16" in source
    assert "values.dtype is torch.bfloat16" in source
    assert "else values.dtype" in source
    assert "IS_BF16=similarity.dtype is torch.bfloat16" in source


def test_backend_changes_only_read_and_preserves_v26_6_write() -> None:
    backend = successor.MixedDtypeTritonFICEMReadWriteBackend
    assert issubclass(backend, v26_6.MaterializeCastTritonFICEMReadWriteBackend)
    assert (
        backend._inference_update_from_projected
        is v26_6.MaterializeCastTritonFICEMReadWriteBackend._inference_update_from_projected
    )
    read_source = inspect.getsource(backend.read)
    assert "fused_ficem_read_tail_mixed_dtype" in read_source
    assert "memory.address_factors" in read_source
    assert 'torch.einsum("btd,bsd->bts", query, keys)' in read_source
    assert "memory.out(recalled)" in read_source
    assert "_known_empty_hint(state)" in read_source


def test_repair5_arithmetic_and_protocol_are_inherited_not_reimplemented() -> None:
    successor_source = SUCCESSOR_PATH.read_text()
    repair5_source = REPAIR5_PATH.read_text()

    # The v26.7 module does not contain the Triton arithmetic implementation.
    assert "strength_bias = tl.log" not in successor_source
    assert "weight0_visible = soft0.to(tl.bfloat16)" not in successor_source
    assert "product0 = weight0.to(tl.float32)" not in successor_source

    # Those repair5 checkpoints remain in the untouched historical kernel.
    assert "strength_bias = tl.log" in repair5_source
    assert "weight0_visible = soft0.to(tl.bfloat16)" in repair5_source
    assert "product0 = weight0.to(tl.float32)" in repair5_source


def test_issue551_protocol_keeps_every_higher_authorization_false() -> None:
    protocol = successor.mixed_dtype_ficem_read_write_v26_7_protocol()
    assert protocol["version"] == "aera-v26.7-ficem-read-mixed-dtype"
    assert protocol["research_issue"] == 551
    assert protocol["source_main"] == successor.SOURCE_MAIN
    assert protocol["historical_repair5_backend_mutated"] is False
    assert protocol["read_kernel_reused_from_repair5"] is True
    assert protocol["read_new_triton_kernels"] == 0
    assert protocol["read_tail_triton_launches_target"] == 1
    assert protocol["read_global_cross_field_dtype_equality_required"] is False
    assert protocol["read_strengths_values_dtype_equality_required"] is True
    assert protocol["read_supported_new_mixed_dtypes"] == ["float32", "bfloat16"]
    assert protocol["read_historical_same_dtype_float16_preserved"] is True
    assert protocol["read_arbitrary_strengths_values_mixing_authorized"] is False
    assert protocol["read_bf16_mode_selected_by_similarity_dtype"] is True
    assert protocol["read_host_pre_tail_cast_kernels"] == 0
    assert protocol["read_arithmetic_changed_by_v26_7"] is False
    assert protocol["read_topology_changed_by_v26_7"] is False
    assert protocol["read_training_backend_changed_by_v26_7"] is False
    assert protocol["write_backend_changed_by_v26_7"] is False
    assert protocol["repair5_bf16_actual_autocast_tail_preserved"] is True
    assert protocol["issue545_authoritative_result_emitted"] is False
    assert protocol["issue545_scientific_seed_consumed"] is False

    for key in (
        "mixed_dtype_read_gpu_gate_authorized",
        "end_to_end_systems_authorized",
        "architecture_freeze_authorized",
        "s2_authorized",
        "fresh_scientific_seed_authorized",
        "independent_replication_credit",
        "100m_authorized",
        "breakthrough_proven",
    ):
        assert protocol[key] is False


def test_issue551_files_contain_no_gpu_or_workflow_authorization_surface() -> None:
    source = SUCCESSOR_PATH.read_text()
    assert "gpu=\"L4\"" not in source
    assert "modal.App" not in source
    assert "workflow_dispatch" not in source
    assert "run_end_to_end_systems" not in source
    assert "scientific_seed" not in inspect.getsource(
        successor.MixedDtypeTritonFICEMReadWriteBackend.read
    )
