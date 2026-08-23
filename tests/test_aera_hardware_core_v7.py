from __future__ import annotations

import torch
import torch.nn.functional as F

from tam_research.aera_hardware_core import HardwareAERAConfig
from tam_research.aera_hardware_core_v6 import BMMHardSparseExpertBank
from tam_research.aera_hardware_core_v7 import NativeGroupedMMSparseExpertBank


def cfg() -> HardwareAERAConfig:
    return HardwareAERAConfig(
        d_model=32,
        n_stages=1,
        n_heads=4,
        chunk_size=8,
        n_experts=4,
        max_active_experts=2,
        expert_mult=2,
        memory_dim=8,
        max_reason_steps=2,
    )


def _fake_grouped_mm(mat_a, mat_b, *, offs=None, bias=None, out_dtype=None):
    """CPU stand-in matching native grouped_mm's physical-row output shape.

    Rows after offs[-1] are physical sentinels: grouped computation ignores them,
    but the returned tensor still has the physical input row count. This behavior
    reproduces the L4 failure from issue #196 and prevents sentinel rows from being
    accidentally reshaped into routed assignments again.
    """
    assert offs is not None
    ends = offs.tolist()
    start = 0
    pieces = []
    for group, end in enumerate(ends):
        a = mat_a[start:end]
        y = a @ mat_b[group]
        if bias is not None:
            y = y + bias[group]
        pieces.append(y)
        start = end
    logical = torch.cat(pieces, dim=0) if pieces else mat_a.new_empty((0, mat_b.size(-1)))
    trailing = mat_a.size(0) - start
    if trailing > 0:
        logical = torch.cat((logical, logical.new_zeros((trailing, mat_b.size(-1)))), dim=0)
    return logical.to(out_dtype) if out_dtype is not None else logical


def test_native_grouped_algorithm_matches_v6_reference(monkeypatch):
    torch.manual_seed(71)
    c = cfg()
    ref = BMMHardSparseExpertBank(c)
    grouped = NativeGroupedMMSparseExpertBank(c)
    grouped.load_state_dict(ref.state_dict())

    monkeypatch.setattr(F, "grouped_mm", _fake_grouped_mm, raising=False)
    monkeypatch.setattr(
        NativeGroupedMMSparseExpertBank,
        "native_grouped_mm_available",
        staticmethod(lambda _x: True),
    )

    x = torch.randn(7, 8, c.d_model)
    logits = torch.randn(7, c.n_experts)
    count_logits = torch.tensor(
        [
            [8.0, -8.0],
            [-8.0, 8.0],
            [8.0, -8.0],
            [-8.0, 8.0],
            [-8.0, 8.0],
            [8.0, -8.0],
            [8.0, -8.0],
        ]
    )
    with torch.no_grad():
        expected = ref(x, logits, count_logits, hard=True)
        actual = grouped(x, logits, count_logits, hard=True)
    assert actual.dtype == x.dtype
    assert actual.shape == x.shape
    assert torch.allclose(expected, actual, atol=3e-5, rtol=3e-5)
    stats = grouped.stats()
    assert stats is not None
    assert stats["hard_kernel"] == "native_grouped_mm"
    assert stats["hard_input_dtype"] == str(x.dtype)
    assert stats["hard_compute_dtype"] == str(x.dtype)  # CPU forced-native test keeps dtype.


def test_grouped_sentinel_row_never_becomes_assignment(monkeypatch):
    torch.manual_seed(711)
    c = cfg()
    grouped = NativeGroupedMMSparseExpertBank(c)
    monkeypatch.setattr(F, "grouped_mm", _fake_grouped_mm, raising=False)
    monkeypatch.setattr(
        NativeGroupedMMSparseExpertBank,
        "native_grouped_mm_available",
        staticmethod(lambda _x: True),
    )
    x = torch.randn(1, c.chunk_size, c.d_model)
    logits = torch.tensor([[9.0, -9.0, -9.0, -9.0]])
    count_logits = torch.tensor([[9.0, -9.0]])
    with torch.no_grad():
        out = grouped(x, logits, count_logits, hard=True)
    # The exact #196 failure would produce T+1 physical rows and fail this shape.
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_cpu_hard_path_records_bmm_fallback():
    torch.manual_seed(72)
    c = cfg()
    bank = NativeGroupedMMSparseExpertBank(c)
    x = torch.randn(4, 8, c.d_model)
    logits = torch.randn(4, c.n_experts)
    count_logits = torch.tensor([[8.0, -8.0], [-8.0, 8.0], [8.0, -8.0], [-8.0, 8.0]])
    with torch.no_grad():
        out = bank(x, logits, count_logits, hard=True)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    stats = bank.stats()
    assert stats is not None
    assert stats["hard_kernel"] == "bmm_fallback"
    assert stats["hard_input_dtype"] == str(x.dtype)
    assert stats["hard_compute_dtype"] is None


def test_native_path_never_materializes_dense_all_experts(monkeypatch):
    torch.manual_seed(73)
    c = cfg()
    bank = NativeGroupedMMSparseExpertBank(c)
    monkeypatch.setattr(F, "grouped_mm", _fake_grouped_mm, raising=False)
    monkeypatch.setattr(
        NativeGroupedMMSparseExpertBank,
        "native_grouped_mm_available",
        staticmethod(lambda _x: True),
    )

    seen_group_counts: list[int] = []
    original = F.grouped_mm

    def counted(mat_a, mat_b, **kwargs):
        seen_group_counts.append(int(mat_b.size(0)))
        return original(mat_a, mat_b, **kwargs)

    monkeypatch.setattr(F, "grouped_mm", counted)
    x = torch.randn(6, 8, c.d_model)
    # Route all examples only to experts 1 or 3; unused experts must not enter grouped GEMM.
    logits = torch.full((6, c.n_experts), -20.0)
    logits[:3, 1] = 20.0
    logits[3:, 3] = 20.0
    count_logits = torch.tensor([[8.0, -8.0]]).expand(6, -1)
    with torch.no_grad():
        bank(x, logits, count_logits, hard=True)
    assert seen_group_counts == [2, 2]


def test_native_eligibility_does_not_require_residual_bf16(monkeypatch):
    # The full model may keep residual activations FP32 under autocast. Eligibility
    # depends on CUDA + grouped_mm + SM80+, while the expert-local path owns the BF16 cast.
    class FakeTensor:
        is_cuda = True
        dtype = torch.float32

        @staticmethod
        def is_floating_point():
            return True

        device = torch.device("cuda")

    monkeypatch.setattr(F, "grouped_mm", _fake_grouped_mm, raising=False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device=None: (8, 9))
    assert NativeGroupedMMSparseExpertBank.native_grouped_mm_available(FakeTensor())
