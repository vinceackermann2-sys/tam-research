from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import torch

from tam_research.aera_hardware_core_v24 import (
    MIN_STRENGTH,
    READ_TEMPERATURE,
    READ_TOP_K,
)
from tam_research.aera_hardware_core_v26_3_ficem_read_triton import (
    TritonFICEMReadBackend,
    fused_ficem_read_tail,
    fused_ficem_read_v26_3_protocol,
)

ROOT = Path(__file__).resolve().parents[1]
BACKEND_PATH = ROOT / "tam_research" / "aera_hardware_core_v26_3_ficem_read_triton.py"
PROBE_PATH = ROOT / "tam_research" / "aera_v26_3_ficem_read_probe.py"
FROZEN_ISSUE418_PROBE_BLOB = "16aa99b9f6f0a1d11bd7bf5f36b2f0b1fb97047b"


def _git_blob_sha(path: Path) -> str:
    data = path.read_bytes()
    header = f"blob {len(data)}\0".encode()
    return hashlib.sha1(header + data).hexdigest()


def _kernel_source() -> str:
    source = BACKEND_PATH.read_text()
    return source.split("def _ficem_read_tail_kernel(", 1)[1].split(
        "def triton_ficem_read_available", 1
    )[0]


def _bf16_branch(kernel: str) -> str:
    return kernel.split("        if IS_BF16:\n", 1)[1].split(
        "        else:\n            # Frozen pre-#426", 1
    )[0]


def test_issue466_protocol_freezes_repair5_scope_and_authorization():
    protocol = fused_ficem_read_v26_3_protocol()
    assert protocol["bf16_reference_rounding_repair3"] is True
    assert protocol["float32_path_changed_by_repair3"] is False
    assert protocol["bf16_product_rounding_repair4"] is True
    assert protocol["float32_path_changed_by_repair4"] is False
    assert protocol["bf16_actual_autocast_tail_repair5"] is True
    assert protocol["bf16_strength_bias_fp32_repair5"] is True
    assert protocol["bf16_logits_fp32_repair5"] is True
    assert protocol["bf16_final_weights_fp32_repair5"] is True
    assert protocol["bf16_recalled_fp32_repair5"] is True
    assert protocol["bf16_product_rounding_active_after_repair5"] is False
    assert protocol["float32_path_changed_by_repair5"] is False
    assert protocol["read_tail_triton_launches_target"] == 1
    assert protocol["capacity"] == 48
    assert protocol["memory_dim"] == 50
    assert protocol["read_top_k"] == 4
    assert protocol["read_temperature"] == 0.10
    assert protocol["min_strength"] == 1e-4
    assert protocol["address_projection_changed"] is False
    assert protocol["key_normalization_changed"] is False
    assert protocol["similarity_einsum_changed"] is False
    assert protocol["learned_out_projection_changed"] is False
    assert protocol["write_backend_changed"] is False
    assert protocol["training_backend_changed"] is False
    assert protocol["persistent_state_changed"] is False
    assert protocol["gpu_authorized_by_module"] is False
    assert protocol["scientific_training_authorized"] is False
    assert protocol["end_to_end_systems_authorized"] is False
    assert protocol["architecture_freeze_authorized"] is False
    assert protocol["s2_authorized"] is False
    assert protocol["fresh_scientific_seed_authorized"] is False
    assert protocol["100m_authorized"] is False
    assert protocol["breakthrough_proven"] is False


def test_issue466_keeps_frozen_geometry_and_one_kernel_launch_topology():
    source = BACKEND_PATH.read_text()
    kernel = _kernel_source()
    launch = inspect.getsource(fused_ficem_read_tail)
    assert READ_TOP_K == 4
    assert READ_TEMPERATURE == 0.10
    assert MIN_STRENGTH == 1e-4
    assert source.count("@triton.jit") == 1
    assert source.count("def _ficem_read_tail_kernel(") == 1
    assert kernel.count("tl.argmax(") == 4
    for token in (
        "TIME: tl.constexpr",
        "CAPACITY: tl.constexpr",
        "MEMORY_DIM: tl.constexpr",
        "SLOT_BLOCK: tl.constexpr",
        "DIM_BLOCK: tl.constexpr",
        "WRITE_INDICES: tl.constexpr",
        "IS_BF16: tl.constexpr",
        "MIN_STRENGTH: tl.constexpr",
        "READ_TEMPERATURE: tl.constexpr",
        "READ_TOP_K: tl.constexpr",
    ):
        assert token in kernel
    assert "_ficem_read_tail_kernel[(batch * time,)](" in launch
    assert "SLOT_BLOCK=64" in launch
    assert "DIM_BLOCK=64" in launch
    assert "num_warps=4" in launch


def test_issue466_bf16_logit_path_matches_actual_autocast_visibility():
    kernel = _kernel_source()
    bf16 = _bf16_branch(kernel)
    assert "similarity_visible = similarity.to(tl.bfloat16)" in bf16
    assert "clamped_visible = clamped_strengths.to(tl.bfloat16)" in bf16
    assert "strength_bias = tl.log(clamped_visible.to(tl.float32))" in bf16
    assert "similarity_visible.to(tl.float32) + strength_bias" in bf16
    assert ") / READ_TEMPERATURE" in bf16
    assert "strength_bias = tl.log(clamped_visible.to(tl.float32)).to(tl.bfloat16)" not in bf16
    assert "(similarity_visible + strength_bias).to(tl.bfloat16)" not in bf16

    selected = kernel.split(
        "            # #460 actual reference already has FP32 top logits here.", 1
    )[1].split("        else:\n            safe0 =", 1)[0]
    for i in range(4):
        assert f"safe{i} = tl.where(valid{i}, logit{i}, -1.0e9).to(tl.float32)" in selected
        assert f"safe{i} = tl.where(valid{i}, logit{i}, -1.0e9).to(tl.bfloat16)" not in selected


def test_issue466_bf16_weights_have_one_visible_boundary_then_fp32_renormalization():
    kernel = _kernel_source()
    section = kernel.split(
        "            # Actual reference: softmax is FP32, then `.to(identity_source.dtype)`", 1
    )[1].split("        else:\n            weight0 = tl.where", 1)[0]
    for i in range(4):
        assert f"weight{i}_visible = soft{i}.to(tl.bfloat16)" in section
        assert f"weight{i}_valid = tl.where(valid{i}, weight{i}_visible, 0.0).to(tl.bfloat16)" in section
        assert f"weight{i} = weight{i}_valid.to(tl.float32) / denominator" in section
    assert "valid_weight_sum = (" in section
    assert "weight0_valid.to(tl.float32)" in section
    assert "weight3_valid.to(tl.float32)" in section
    assert "denominator = tl.maximum(valid_weight_sum, 1.0e-9)" in section
    assert "valid_weight_sum = (\n" in section
    assert ").to(tl.bfloat16)" not in section
    assert "denominator.to(tl.bfloat16)" not in section


def test_issue466_bf16_products_and_recalled_are_fp32_without_repair4_rounding():
    kernel = _kernel_source()
    section = kernel.split(
        "            # #460 actual in-autocast reference multiplies FP32 normalized weights", 1
    )[1].split("        else:\n            recalled = (", 1)[0]
    for i in range(4):
        assert f"product{i} = weight{i}.to(tl.float32) * value{i}.to(tl.float32)" in section
    assert "recalled = product0 + product1 + product2 + product3" in section
    assert ".to(tl.bfloat16).to(tl.float32)" not in section

    launch = inspect.getsource(fused_ficem_read_tail)
    assert "recalled_dtype = torch.float32 if values.dtype is torch.bfloat16 else values.dtype" in launch
    assert "dtype=recalled_dtype" in launch


def test_issue466_preserves_float32_path_and_learned_out_projection_boundary():
    source = BACKEND_PATH.read_text()
    kernel = _kernel_source()
    float_branch = kernel.split(
        "        else:\n            # Frozen pre-#426 float32/float16 execution behavior", 1
    )[1].split("\n        logits = tl.where", 1)[0]
    assert "similarity_visible = similarity.to(tl.float32)" in float_branch
    assert "clamped_visible = clamped_strengths.to(tl.float32)" in float_branch
    assert "strength_bias = tl.log(clamped_visible)" in float_branch
    assert "logits = (similarity_visible + strength_bias) / READ_TEMPERATURE" in float_branch
    assert "tl.bfloat16" not in float_branch

    read_source = source.split("    def read(\n", 1)[1].split("    def update(\n", 1)[0]
    assert "recalled, _ = fused_ficem_read_tail(" in read_source
    assert "recalled=memory.out(recalled)" in read_source
    assert "memory.out(" not in kernel


def test_issue466_has_no_hidden_fallback_state_growth_or_training_path():
    source = BACKEND_PATH.read_text()
    forbidden = (
        "torch.topk(",
        "torch.softmax(",
        ".gather(",
        "torch.cat(",
        "torch.stack(",
        "register_buffer",
        "nn.Parameter",
        "torch.load(",
        ".backward(",
        "torch.optim",
        "seed8471",
    )
    for token in forbidden:
        assert token not in source
    assert '"persistent_cache": False' in source
    assert '"persistent_packed_state": False' in source
    assert "if torch.is_grad_enabled() or memory.differentiable_pretraining:" in source
    assert "return self._reference.read(memory, identity_source, context_source, state)" in source
    assert "return self._reference.update(" in source
    assert "return self._reference.update_from_projected(" in source
    assert TritonFICEMReadBackend.name == "triton-fused-ficem-read-tail-v26.3"


def test_issue466_preserves_frozen_probe_blob_and_no_probe_edit():
    assert _git_blob_sha(PROBE_PATH) == FROZEN_ISSUE418_PROBE_BLOB
