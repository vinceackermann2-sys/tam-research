from __future__ import annotations

from pathlib import Path

import torch

from architectures.cortex_s.affine_scan_triton_candidate import (
    affine_scan_triton_candidate,
    candidate_status,
)
from architectures.cortex_s.language_model import affine_scan


REPO_ROOT = Path(__file__).resolve().parents[3]


def _grads(fn, *, length: int, initial: bool):
    generator = torch.Generator(device="cpu").manual_seed(91_901 + length)
    a = torch.sigmoid(torch.randn(3, length, 5, generator=generator, dtype=torch.float64)).requires_grad_(True)
    b = torch.randn(3, length, 5, generator=generator, dtype=torch.float64).requires_grad_(True)
    initial_tensor = (
        torch.randn(3, 5, generator=generator, dtype=torch.float64).requires_grad_(True)
        if initial
        else None
    )
    upstream = torch.randn(3, length, 5, generator=generator, dtype=torch.float64)
    out = fn(a, b, initial_tensor)
    loss = (out * upstream).sum()
    loss.backward()
    return (
        out.detach(),
        a.grad.detach(),
        b.grad.detach(),
        None if initial_tensor is None else initial_tensor.grad.detach(),
    )


def test_cpu_candidate_matches_production_forward_and_backward() -> None:
    for length in (1, 2, 7, 16, 31):
        for with_initial in (False, True):
            expected = _grads(affine_scan, length=length, initial=with_initial)
            actual = _grads(affine_scan_triton_candidate, length=length, initial=with_initial)
            for left, right in zip(expected[:3], actual[:3]):
                assert torch.allclose(left, right, atol=1e-10, rtol=1e-10)
            if with_initial:
                assert expected[3] is not None and actual[3] is not None
                assert torch.allclose(expected[3], actual[3], atol=1e-10, rtol=1e-10)
            else:
                assert expected[3] is None and actual[3] is None


def test_cpu_candidate_preserves_chunk_carry() -> None:
    generator = torch.Generator(device="cpu").manual_seed(91_955)
    a = torch.sigmoid(torch.randn(2, 19, 4, generator=generator, dtype=torch.float64))
    b = torch.randn(2, 19, 4, generator=generator, dtype=torch.float64)
    initial = torch.randn(2, 4, generator=generator, dtype=torch.float64)

    whole = affine_scan_triton_candidate(a, b, initial)
    first = affine_scan_triton_candidate(a[:, :8], b[:, :8], initial)
    second = affine_scan_triton_candidate(a[:, 8:], b[:, 8:], first[:, -1])
    joined = torch.cat((first, second), dim=1)
    assert torch.allclose(whole, joined, atol=1e-10, rtol=1e-10)


def test_candidate_remains_non_authorizing_and_not_production_wired() -> None:
    status = candidate_status()
    assert status["production_wired"] is False
    assert status["gpu_benchmark_authorized"] is False
    assert status["parameter_count_delta"] == 0
    assert status["scientific_semantics_changed"] is False
    assert status["intended_shape"] == [64, 512, 128]
    assert status["kernels_per_scan"] == 2

    production_source = (REPO_ROOT / "architectures/cortex_s/language_model.py").read_text(encoding="utf-8")
    assert "affine_scan_triton_candidate" not in production_source
    assert "affine_scan_triton" not in production_source


def test_candidate_rejects_shape_or_dtype_drift() -> None:
    a = torch.ones(2, 4, 3, dtype=torch.float32)
    b = torch.ones(2, 4, 3, dtype=torch.float32)
    try:
        affine_scan_triton_candidate(a, b[:, :3])
    except ValueError as exc:
        assert "matching" in str(exc)
    else:
        raise AssertionError("shape mismatch must fail")

    try:
        affine_scan_triton_candidate(a, b.to(torch.float64))
    except ValueError as exc:
        assert "device and dtype" in str(exc)
    else:
        raise AssertionError("dtype mismatch must fail")
