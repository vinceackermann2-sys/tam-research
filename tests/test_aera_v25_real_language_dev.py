from __future__ import annotations

import torch
import pytest

from tam_research import aera_real_language_v25 as v25
from tam_research import aera_real_language_v25_gpu as gpu
from tam_research.aera_hardware_core import HardwareAERAConfig
from tam_research.aera_hardware_core_v25 import HardwareAwareAERATextLMV25


def _tiny_model() -> HardwareAwareAERATextLMV25:
    cfg = HardwareAERAConfig(
        vocab_size=97,
        d_model=32,
        n_stages=4,
        n_heads=4,
        chunk_size=256,
        n_experts=4,
        max_active_experts=2,
        expert_mult=2,
        memory_dim=16,
        max_reason_steps=2,
        block_size=4,
    )
    return HardwareAwareAERATextLMV25(cfg)


def test_frozen_v25_real_language_protocol_summary() -> None:
    p = gpu.frozen_protocol_summary()
    assert p["research_issue"] == 366
    assert p["seed"] == 8471
    assert p["development_only"] is True
    assert p["token_budget_per_model"] == 8_388_608
    assert p["memory_dim"] == 50
    assert p["chunk_size"] == 256
    assert p["episodic_capacity_per_stage"] == 48
    assert p["state_bytes_per_session"] == 77_760
    assert p["candidates_per_chunk"] == 255
    assert p["selected_writes_per_chunk"] == 16
    assert p["vectorized_updates_per_completed_stage_chunk"] == 1
    assert p["payload_token_weight"] == 1.0
    assert p["address_auxiliary_weight"] == 0.0
    assert p["payload_events_per_microbatch"] == 256
    assert p["payload_events_per_optimizer_step"] == 1024
    assert p["thresholds_identical_to_issue324_v18"] is True
    assert p["gpu_authorized_by_module"] is False
    assert p["counts_toward_independent_replication"] is False
    assert p["100m_authorized"] is False


def test_v25_payload_budget_refuses_more_than_256() -> None:
    torch.manual_seed(1)
    model = _tiny_model()
    tokens = torch.randint(0, model.cfg.vocab_size, (1, 256))
    with pytest.raises(ValueError, match="exceeds frozen per-microbatch cap"):
        v25.payload_teaching_terms(model, tokens, step=0, max_events=257)


def test_sparse_audit_snapshot_survives_later_memory_disabled_forward() -> None:
    torch.manual_seed(2)
    model = _tiny_model().eval()
    stage = model.stages[0]
    events = torch.randn(2, 256, model.cfg.d_model)

    with torch.no_grad():
        stage.forward_chunk(events, None, hard=True, update_memory=True)

    rows: list[dict[str, object]] = []
    pair: list[torch.Tensor] = []
    strength: list[torch.Tensor] = []
    gpu._capture_sparse_write_stats(
        model,
        batch_index=0,
        rows=rows,
        pair_gate_values=pair,
        selected_strength_values=strength,
    )
    assert rows
    assert rows[0]["candidates"] == 255
    assert rows[0]["selected_writes"] == 16
    assert rows[0]["vectorized_updates"] == 1

    with torch.no_grad():
        stage.forward_chunk(events, None, hard=True, update_memory=False)

    # The stage-local diagnostics are allowed to reset on the later no-write call;
    # the held-out evaluator must already have captured immutable CPU snapshots.
    assert rows[0]["candidates"] == 255
    assert rows[0]["selected_writes"] == 16
    assert rows[0]["vectorized_updates"] == 1
    assert pair and strength
    assert pair[0].device.type == "cpu"
    assert strength[0].device.type == "cpu"


def test_v25_binding_keeps_inherited_thresholds_exact() -> None:
    assert gpu.QUALITY_GAP_MAX_NLL == 0.50
    assert gpu.MEMORY_SECOND_CHUNK_MIN_ADVANTAGE_NLL == 0.005
    assert gpu.MEMORY_OVERALL_MIN_ADVANTAGE_NLL == 0.0
    assert gpu.WRITE_MEAN_MIN == 0.01
    assert gpu.WRITE_MEAN_MAX == 0.95
    assert gpu.WRITE_SPREAD_MIN == 0.01
    assert gpu.OPTIONAL_STAGE_TARGET_MAE_MAX == 0.12
    assert gpu.OPTIONAL_STAGE_MIN_RUN_FRACTION == 0.05
    assert gpu.TOTAL_STAGE_EXEC_MIN == 0.35
    assert gpu.TOTAL_STAGE_EXEC_MAX == 0.70
    assert gpu.BATCH8_MIN_SPEED_RATIO == 0.25
    assert gpu.BATCH64_MIN_SPEED_RATIO == 1.25
