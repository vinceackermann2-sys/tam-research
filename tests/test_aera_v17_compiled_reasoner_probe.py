from __future__ import annotations

import torch

from tam_research.aera_hardware_core_v5 import DtypeSafeChunkLatentReasoner
from tam_research.aera_v17_compiled_reasoner_probe import (
    CompileFriendlyHardSparseReasoner,
    SELECTED_SIZES,
    copy_reference_weights,
    run_probe,
    validate_probe_protocol,
)


def _logits(chosen: torch.Tensor, steps: int = 4) -> torch.Tensor:
    out = torch.full((chosen.numel(), steps), -6.0)
    out.scatter_(1, (chosen - 1)[:, None], 6.0)
    return out


def test_compiled_reasoner_probe_protocol_is_measurement_only() -> None:
    p = validate_probe_protocol()
    assert SELECTED_SIZES == (4, 8, 16, 32, 64, 128)
    assert p["training_performed"] is False
    assert p["architecture_changed"] is False
    assert p["hard_selected_depth_changed"] is False
    assert p["dense_masked_is_latency_upper_bound_only"] is True


def test_run_probe_does_not_shadow_module_torch_binding() -> None:
    # Regression for issue #256: `import torch._dynamo` inside run_probe caused
    # Python to mark `torch` as local, raising UnboundLocalError before CUDA work.
    assert "torch" not in run_probe.__code__.co_varnames


def test_compile_friendly_reasoner_matches_reference_mixed_depths_cpu() -> None:
    torch.manual_seed(11)
    ref = DtypeSafeChunkLatentReasoner(12, 4).eval()
    cand = CompileFriendlyHardSparseReasoner(12, 4).eval()
    copy_reference_weights(ref, cand)
    summary = torch.randn(8, 12)
    logits = _logits(torch.tensor([1, 2, 3, 4, 1, 3, 2, 4]))
    with torch.no_grad():
        expected = ref(summary, logits, hard=True)
        actual = cand(summary, logits)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_compile_friendly_reasoner_handles_empty_later_steps_cpu() -> None:
    torch.manual_seed(12)
    ref = DtypeSafeChunkLatentReasoner(10, 4).eval()
    cand = CompileFriendlyHardSparseReasoner(10, 4).eval()
    copy_reference_weights(ref, cand)
    summary = torch.randn(6, 10)
    logits = _logits(torch.ones(6, dtype=torch.long))
    with torch.no_grad():
        expected = ref(summary, logits, hard=True)
        actual = cand(summary, logits)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
