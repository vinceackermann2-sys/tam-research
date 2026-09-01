from __future__ import annotations

import hashlib
from pathlib import Path

from tam_research.aera_hardware_core_v24 import (
    MIN_STRENGTH,
    READ_TEMPERATURE,
    READ_TOP_K,
)
from tam_research.aera_hardware_core_v26_3_ficem_read_triton import (
    TritonFICEMReadBackend,
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


def _repair5_active(protocol: dict[str, object]) -> bool:
    return protocol.get("bf16_actual_autocast_tail_repair5") is True


def _assert_repair5_successor_contract(protocol: dict[str, object]) -> None:
    # #464 intentionally recognizes only the explicit preregistered repair5 marker.
    assert protocol["bf16_actual_autocast_tail_repair5"] is True
    assert protocol["bf16_reference_rounding_repair3"] is True
    assert protocol["bf16_product_rounding_repair4"] is True
    assert protocol["bf16_product_rounding_active_after_repair5"] is False
    assert protocol["float32_path_changed_by_repair5"] is False
    assert protocol["read_tail_triton_launches_target"] == 1
    assert protocol["capacity"] == 48
    assert protocol["memory_dim"] == 50
    assert protocol["read_top_k"] == 4
    assert protocol["read_temperature"] == 0.10
    assert protocol["min_strength"] == 1e-4
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


def test_issue445_protocol_freezes_repair4_scope_and_authorization():
    protocol = fused_ficem_read_v26_3_protocol()
    assert protocol["bf16_reference_rounding_repair3"] is True
    assert protocol["float32_path_changed_by_repair3"] is False
    assert protocol["bf16_product_rounding_repair4"] is True
    assert protocol["float32_path_changed_by_repair4"] is False
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


def test_issue445_keeps_frozen_geometry_constants_and_single_kernel():
    source = BACKEND_PATH.read_text()
    kernel = _kernel_source()
    assert READ_TOP_K == 4
    assert READ_TEMPERATURE == 0.10
    assert MIN_STRENGTH == 1e-4
    assert source.count("@triton.jit") == 1
    assert source.count("def _ficem_read_tail_kernel(") == 1
    assert kernel.count("tl.argmax(") == 4
    assert "CAPACITY: tl.constexpr" in kernel
    assert "MEMORY_DIM: tl.constexpr" in kernel
    assert "READ_TOP_K: tl.constexpr" in kernel
    assert "READ_TEMPERATURE: tl.constexpr" in kernel
    assert "MIN_STRENGTH: tl.constexpr" in kernel


def test_issue445_bf16_rounds_each_selected_product_before_fp32_reduction():
    protocol = fused_ficem_read_v26_3_protocol()
    if _repair5_active(protocol):
        _assert_repair5_successor_contract(protocol)
        return

    kernel = _kernel_source()
    assert "# #439/#442: the reference rounds each BF16-visible selected product" in kernel
    for i in range(4):
        expected = (
            f"product{i} = (\n"
            f"                weight{i}.to(tl.float32) * value{i}.to(tl.float32)\n"
            f"            ).to(tl.bfloat16).to(tl.float32)"
        )
        assert expected in kernel
    assert "recalled = product0 + product1 + product2 + product3" in kernel


def test_issue445_non_bf16_path_keeps_direct_fp32_products_without_forced_rounding():
    protocol = fused_ficem_read_v26_3_protocol()
    if _repair5_active(protocol):
        _assert_repair5_successor_contract(protocol)
        kernel = _kernel_source()
        # Repair5 is BF16-only. The historical direct non-BF16 product path remains.
        float_tail = kernel.rsplit("        else:\n            recalled = (", 1)[1]
        for i in range(4):
            assert f"weight{i}.to(tl.float32) * value{i}.to(tl.float32)" in float_tail
        return

    kernel = _kernel_source()
    tail = kernel.split("if IS_BF16:\n            # #439/#442", 1)[1]
    bf16_part, non_bf16_part = tail.split("        else:\n            recalled = (", 1)
    assert bf16_part.count(".to(tl.bfloat16).to(tl.float32)") == 4
    direct = "        else:\n            recalled = (" + non_bf16_part
    for i in range(4):
        assert f"weight{i}.to(tl.float32) * value{i}.to(tl.float32)" in direct
    direct_until_store = direct.split("        tl.store(", 1)[0]
    assert ".to(tl.bfloat16)" not in direct_until_store


def test_issue445_preserves_prior_repair3_visible_checkpoints():
    protocol = fused_ficem_read_v26_3_protocol()
    if _repair5_active(protocol):
        _assert_repair5_successor_contract(protocol)
        return

    kernel = _kernel_source()
    required = (
        "similarity_visible = similarity.to(tl.bfloat16)",
        "clamped_visible = clamped_strengths.to(tl.bfloat16)",
        "tl.log(clamped_visible.to(tl.float32)).to(tl.bfloat16)",
        "(similarity_visible + strength_bias).to(tl.bfloat16)",
        "safe0 = tl.where(valid0, logit0, -1.0e9).to(tl.bfloat16).to(tl.float32)",
        "weight0_visible = weight0.to(tl.bfloat16)",
        "weight3_visible = weight3.to(tl.bfloat16)",
        "valid_weight_sum = (",
        "denominator = tl.maximum(",
        "weight0_visible.to(tl.float32) / denominator.to(tl.float32)",
        "weight3_visible.to(tl.float32) / denominator.to(tl.float32)",
    )
    for token in required:
        assert token in kernel


def test_issue445_preserves_probe_and_no_hidden_fallback_or_state_growth():
    assert _git_blob_sha(PROBE_PATH) == FROZEN_ISSUE418_PROBE_BLOB
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


def test_issue445_write_and_training_paths_still_delegate_to_reference():
    source = BACKEND_PATH.read_text()
    assert "if torch.is_grad_enabled() or memory.differentiable_pretraining:" in source
    assert "return self._reference.read(memory, identity_source, context_source, state)" in source
    assert "return self._reference.update(" in source
    assert "return self._reference.update_from_projected(" in source
    assert TritonFICEMReadBackend.name == "triton-fused-ficem-read-tail-v26.3"
