from __future__ import annotations

import torch

from tam_research.aera_hardware_core_v5 import DtypeSafeChunkLatentReasoner
from tam_research.aera_v17_kernel_probe import (
    D_MODEL,
    SELECTED_SIZES,
    dense_masked_reasoner_reference,
    production_shape_config,
    validate_probe_protocol,
)


def test_kernel_probe_protocol_is_frozen_to_production_shapes() -> None:
    p = validate_probe_protocol()
    assert SELECTED_SIZES == (1, 2, 4, 8, 16, 32, 64, 128)
    assert D_MODEL == 200
    assert p["shape"] == {
        "chunk_size": 256,
        "d_model": 200,
        "n_experts": 8,
        "expert_hidden": 800,
        "max_reason_steps": 4,
    }
    assert p["training_performed"] is False
    assert p["checkpoint_mutated"] is False
    assert p["dense_reasoner_is_speed_upper_bound_only"] is True


def test_production_shape_config_matches_v17_core_shape() -> None:
    cfg = production_shape_config()
    assert cfg.d_model == 200
    assert cfg.chunk_size == 256
    assert cfg.n_experts == 8
    assert cfg.expert_mult == 4
    assert cfg.max_reason_steps == 4


def test_dense_masked_reasoner_matches_sparse_semantics_on_cpu() -> None:
    torch.manual_seed(7)
    reasoner = DtypeSafeChunkLatentReasoner(12, 4).eval()
    summary = torch.randn(8, 12)
    chosen = torch.arange(8) % 4
    logits = torch.full((8, 4), -6.0)
    logits.scatter_(1, chosen[:, None], 6.0)
    with torch.no_grad():
        sparse = reasoner(summary, logits, hard=True)
        dense = dense_masked_reasoner_reference(reasoner, summary, logits)
    torch.testing.assert_close(sparse, dense, rtol=1e-5, atol=1e-6)
