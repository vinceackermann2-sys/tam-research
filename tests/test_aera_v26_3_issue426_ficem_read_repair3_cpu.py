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


def test_issue426_frozen_ficem_constants_and_geometry_remain_exact():
    protocol = fused_ficem_read_v26_3_protocol()
    assert READ_TOP_K == 4
    assert READ_TEMPERATURE == 0.10
    assert MIN_STRENGTH == 1e-4
    assert protocol["capacity"] == 48
    assert protocol["memory_dim"] == 50
    assert protocol["read_top_k"] == 4
    assert protocol["read_temperature"] == 0.10
    assert protocol["min_strength"] == 1e-4
    assert protocol["read_tail_triton_launches_target"] == 1
    assert protocol["bf16_reference_rounding_repair3"] is True
    assert protocol["float32_path_changed_by_repair3"] is False


def test_issue426_keeps_exactly_one_fused_triton_kernel():
    source = BACKEND_PATH.read_text()
    kernel = _kernel_source()
    assert source.count("@triton.jit") == 1
    assert source.count("def _ficem_read_tail_kernel(") == 1
    assert "IS_BF16: tl.constexpr" in kernel
    assert "TIME: tl.constexpr" in kernel
    assert "CAPACITY: tl.constexpr" in kernel
    assert "MEMORY_DIM: tl.constexpr" in kernel
    assert "READ_TOP_K: tl.constexpr" in kernel
    assert kernel.count("tl.argmax(") == 4


def test_issue426_bf16_has_explicit_reference_visible_rounding_checkpoints():
    kernel = _kernel_source()
    assert "if IS_BF16:" in kernel
    assert "similarity_visible = similarity.to(tl.bfloat16)" in kernel
    assert "clamped_visible = clamped_strengths.to(tl.bfloat16)" in kernel
    assert "tl.log(clamped_visible.to(tl.float32)).to(tl.bfloat16)" in kernel
    assert "(similarity_visible + strength_bias).to(tl.bfloat16)" in kernel
    assert "/ READ_TEMPERATURE" in kernel
    assert ").to(tl.bfloat16)" in kernel
    assert "safe0 = tl.where(valid0, logit0, -1.0e9).to(tl.bfloat16).to(tl.float32)" in kernel
    assert "weight0_visible = weight0.to(tl.bfloat16)" in kernel
    assert "weight3_visible = weight3.to(tl.bfloat16)" in kernel
    assert "valid_weight_sum = (" in kernel
    assert ").to(tl.bfloat16)" in kernel
    assert "denominator = tl.maximum(" in kernel
    assert "weight0_visible.to(tl.float32) / denominator.to(tl.float32)" in kernel
    assert "weight3_visible.to(tl.float32) / denominator.to(tl.float32)" in kernel
    assert "weight0.to(tl.float32) * value0.to(tl.float32)" in kernel
    assert "weight3.to(tl.float32) * value3.to(tl.float32)" in kernel


def test_issue426_float32_branch_preserves_pre_repair_execution_shape():
    kernel = _kernel_source()
    assert "else:\n            # Frozen pre-#426 float32/float16 execution behavior" in kernel
    assert "similarity_visible = similarity.to(tl.float32)" in kernel
    assert "clamped_visible = clamped_strengths.to(tl.float32)" in kernel
    assert "strength_bias = tl.log(clamped_visible)" in kernel
    assert "logits = (similarity_visible + strength_bias) / READ_TEMPERATURE" in kernel
    assert "valid_weight_sum = weight0 + weight1 + weight2 + weight3" in kernel
    assert "denominator = tl.maximum(valid_weight_sum, 1.0e-9)" in kernel


def test_issue426_launch_selects_bf16_at_compile_time_without_extra_kernel():
    source = inspect.getsource(fused_ficem_read_tail)
    assert "IS_BF16=similarity.dtype is torch.bfloat16" in source
    assert "SLOT_BLOCK=64" in source
    assert "DIM_BLOCK=64" in source
    assert "num_warps=4" in source
    assert "MIN_STRENGTH=MIN_STRENGTH" in source
    assert "READ_TEMPERATURE=READ_TEMPERATURE" in source
    assert "READ_TOP_K=READ_TOP_K" in source


def test_issue426_accelerated_backend_has_no_reference_tail_fallback_or_state_growth():
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
    assert "persistent_cache" in source
    assert '"persistent_cache": False' in source
    assert '"persistent_packed_state": False' in source


def test_issue426_write_and_training_paths_still_delegate_to_reference():
    source = BACKEND_PATH.read_text()
    assert "if torch.is_grad_enabled() or memory.differentiable_pretraining:" in source
    assert "return self._reference.read(memory, identity_source, context_source, state)" in source
    assert "return self._reference.update(" in source
    assert "return self._reference.update_from_projected(" in source
    assert TritonFICEMReadBackend.name == "triton-fused-ficem-read-tail-v26.3"


def test_issue426_does_not_mutate_issue418_probe_contract_or_fixtures():
    assert _git_blob_sha(PROBE_PATH) == FROZEN_ISSUE418_PROBE_BLOB
    probe = PROBE_PATH.read_text()
    assert "DESIGN_SEED = 408_411" in probe
    assert "BF16_ATOL = 1e-2" in probe
    assert "BF16_RTOL = 1e-2" in probe
    assert "MAX_GEOMEAN_LATENCY_RATIO = 0.90" in probe
    assert "MAX_ROW_LATENCY_RATIO = 1.05" in probe
    assert "MAX_FULL_EVENT_RATIO = 0.75" in probe
    assert "WARMUP_CALLS = 10" in probe
    assert "TIMED_ROUNDS = 5" in probe
    assert "CALLS_PER_ROUND = 100" in probe
