from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F

from architectures.cortex_s.language_model import CortexSLM, CortexSLMConfig, TrulySparseMoE, parameter_count
from architectures.cortex_s.experiments.scale100m_2b.systems_microbench_v1_repair2 import (
    CLASSIFICATION,
    CONSUMED_ENGINEERING_SEEDS,
    ENGINEERING_SEED,
    FORBIDDEN_SCIENTIFIC_SEEDS,
    GROUPED_MM_ALIGNMENT_BYTES,
    MEASURED_STEPS,
    MIN_PROMISING_SPEEDUP,
    STOP_GROUPED_BELOW_SPEEDUP,
    StrideAlignedBF16GroupedSparseMoE,
    TRIGGER_TITLE,
    _aligned_row_view,
    _grouped_mm_layout_ok,
    convert_to_stride_aligned_grouped,
    validate_microbench_protocol,
    zero_gpu_layout_contract_probe,
)


def _repo() -> Path:
    return Path(__file__).resolve().parents[3]


def _tiny_config() -> CortexSLMConfig:
    return CortexSLMConfig(
        vocab_size=64,
        d_model=32,
        n_layers=4,
        n_heads=4,
        max_seq_len=32,
        state_size=8,
        num_experts=4,
        top_k=2,
        expert_hidden=24,
        attention_every=2,
    )


def test_repair2_seed_and_namespace_are_fresh_and_engineering_only():
    protocol = validate_microbench_protocol()
    assert protocol["classification"] == CLASSIFICATION == "ENGINEERING_SYSTEMS_MICROBENCH_ONLY"
    assert ENGINEERING_SEED == 2_026_090_902
    assert 2_026_090_901 in CONSUMED_ENGINEERING_SEEDS
    assert 910_001 in CONSUMED_ENGINEERING_SEEDS
    assert ENGINEERING_SEED not in set(CONSUMED_ENGINEERING_SEEDS)
    assert ENGINEERING_SEED not in set(FORBIDDEN_SCIENTIFIC_SEEDS)
    assert {8_100, 48_131, 48_132, 48_133}.issubset(set(FORBIDDEN_SCIENTIFIC_SEEDS))
    assert TRIGGER_TITLE == "[modal-cortex-s-100m-systems-microbench-v1-repair2]"
    assert protocol["full_training_authorized"] is False
    assert protocol["next_stage_authorized"] is False
    assert MEASURED_STEPS == 20
    assert STOP_GROUPED_BELOW_SPEEDUP == 1.10
    assert MIN_PROMISING_SPEEDUP == 1.20


def test_exact_production_stride_failure_is_reproduced_and_repaired_on_cpu():
    probe = zero_gpu_layout_contract_probe()
    assert probe["status"] == "PASS"
    assert probe["logical_hidden"] == 338
    assert probe["old_repair1_w1_stride"] == [512 * 338, 338, 1]
    assert probe["old_repair1_w1_valid"] is False
    # W1 transpose view: [E, 512, 338] with unit K stride and 512-element N stride.
    assert probe["repair2_w1_logical_stride"] == [338 * 512, 1, 512]
    assert probe["repair2_w1_valid"] is True
    # Physical activation rows pad 338 BF16 values to 344: 344*2=688 bytes.
    assert probe["repair2_post_gelu_stride"] == [344, 1]
    assert probe["physical_hidden_row_stride"] == 344
    assert probe["repair2_post_gelu_valid"] is True
    assert probe["repair2_w2_valid"] is True
    assert GROUPED_MM_ALIGNMENT_BYTES == 16


def test_actual_fp32_transpose_then_bf16_cast_keeps_grouped_mm_valid_stride():
    candidate = StrideAlignedBF16GroupedSparseMoE(d_model=512, num_experts=8, top_k=2, hidden=338)
    logical_w1 = candidate.expert_w1.transpose(-2, -1).to(torch.bfloat16)
    assert logical_w1.shape == (8, 512, 338)
    assert logical_w1.stride() == (338 * 512, 1, 512)
    assert _grouped_mm_layout_ok(logical_w1)


def test_gelu_contiguous_hidden338_is_invalid_then_alignment_view_is_valid_and_exact():
    torch.manual_seed(7001)
    x = torch.randn(29, 338, dtype=torch.float32).to(torch.bfloat16)
    gelu = F.gelu(x)
    assert gelu.stride() == (338, 1)
    assert not _grouped_mm_layout_ok(gelu)
    repaired = _aligned_row_view(gelu)
    assert repaired.shape == gelu.shape
    assert repaired.stride() == (344, 1)
    assert _grouped_mm_layout_ok(repaired)
    torch.testing.assert_close(repaired, gelu, atol=0, rtol=0)


def test_repair2_candidate_matches_legacy_forward_backward_on_cpu():
    torch.manual_seed(7002)
    legacy = TrulySparseMoE(d_model=24, num_experts=8, top_k=2, hidden=16)
    candidate = StrideAlignedBF16GroupedSparseMoE(d_model=24, num_experts=8, top_k=2, hidden=16)
    candidate.copy_from_legacy(legacy)
    assert parameter_count(candidate) == parameter_count(legacy)

    x_legacy = torch.randn(3, 11, 24, requires_grad=True)
    x_candidate = x_legacy.detach().clone().requires_grad_(True)
    y_legacy = legacy(x_legacy)
    y_candidate = candidate(x_candidate)
    torch.testing.assert_close(y_candidate, y_legacy, atol=2e-6, rtol=2e-6)
    y_legacy.square().mean().backward()
    y_candidate.square().mean().backward()
    torch.testing.assert_close(x_candidate.grad, x_legacy.grad, atol=3e-6, rtol=3e-6)

    for index, expert in enumerate(legacy.experts):
        # W1 is now stored exactly in native Linear orientation.
        torch.testing.assert_close(candidate.expert_w1.grad[index], expert[0].weight.grad, atol=3e-6, rtol=3e-6)
        # W2 remains multiplication-oriented.
        torch.testing.assert_close(candidate.expert_w2.grad[index], expert[2].weight.grad.T, atol=3e-6, rtol=3e-6)
    torch.testing.assert_close(candidate.router.weight.grad, legacy.router.weight.grad, atol=3e-6, rtol=3e-6)
    torch.testing.assert_close(candidate.router.bias.grad, legacy.router.bias.grad, atol=3e-6, rtol=3e-6)


def test_full_model_repair2_conversion_preserves_count_logits_and_state_on_cpu():
    torch.manual_seed(7003)
    legacy = CortexSLM(_tiny_config()).eval()
    candidate = deepcopy(legacy)
    before = parameter_count(candidate)
    candidate = convert_to_stride_aligned_grouped(candidate).eval()
    assert parameter_count(candidate) == before == parameter_count(legacy)
    assert all(isinstance(block.moe, StrideAlignedBF16GroupedSparseMoE) for block in candidate.blocks)

    tokens = torch.randint(0, 64, (2, 12))
    with torch.no_grad():
        legacy_logits, legacy_state = legacy(tokens, return_state=True)
        candidate_logits, candidate_state = candidate(tokens, return_state=True)
    torch.testing.assert_close(candidate_logits, legacy_logits, atol=3e-6, rtol=3e-6)
    for candidate_layer, legacy_layer in zip(candidate_state, legacy_state):
        torch.testing.assert_close(candidate_layer, legacy_layer, atol=3e-6, rtol=3e-6)


def test_grouped_mm_call_rejects_invalid_layout_before_kernel(monkeypatch):
    candidate = StrideAlignedBF16GroupedSparseMoE(d_model=16, num_experts=2, top_k=1, hidden=10)
    packed = torch.randn(7, 16, dtype=torch.float32)
    # Old repair1-style logical W1: BF16 row stride 10 -> 20 bytes, invalid.
    invalid = torch.randn(2, 16, 10, dtype=torch.float32)
    offsets = torch.tensor([3, 7], dtype=torch.int32)
    monkeypatch.setattr(candidate, "grouped_cuda_available", lambda _: True)
    called = {"value": False}

    def fake_grouped_mm(left, right, *, offs):
        called["value"] = True
        return torch.zeros(left.size(0), right.size(-1), dtype=torch.bfloat16)

    monkeypatch.setattr(F, "grouped_mm", fake_grouped_mm, raising=False)
    try:
        candidate._grouped_mm(packed, invalid, offsets)
    except RuntimeError as exc:
        assert "16-byte stride contract" in str(exc)
    else:
        raise AssertionError("invalid grouped layout was not rejected")
    assert called["value"] is False


def test_repair2_files_are_bounded_and_have_no_full_training_path():
    app = (_repo() / "modal_cortex_s_100m_systems_microbench_v1_repair2.py").read_text(encoding="utf-8")
    workflow = (_repo() / ".github/workflows/modal-cortex-s-100m-systems-microbench-v1-repair2.yml").read_text(encoding="utf-8")
    doc = (_repo() / "architectures/cortex_s/SYSTEMS_MICROBENCH_100M_V1_REPAIR2.md").read_text(encoding="utf-8")

    assert 'ENGINEERING_SEED = 2_026_090_902' in app
    assert 'APP_NAME = "cortex-s-v0-100m-systems-microbench-v1-repair2"' in app
    assert 'RESULT_ROOT = "/vol/cortex-s-v0/100m-systems-microbench-v1-repair2"' in app
    assert "H100_TIMEOUT_SECONDS = 15 * 60" in app
    assert "def full_2b(" not in app
    assert "train_full_2b" not in app
    assert "workflow_dispatch" not in workflow
    assert "github.event.issue.user.login == github.repository_owner" in workflow
    assert "[modal-cortex-s-100m-systems-microbench-v1-repair2]" in workflow
    assert "from architectures." not in workflow
    assert "34326234104" in doc
    assert "strides should be multiple of 16 bytes" in doc
    assert "2026090901" in doc and "consumed" in doc.lower()
    assert "cannot authorize the 2B run" in doc
