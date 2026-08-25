from __future__ import annotations

import torch

from tam_research import aera_v23_posthoc_diagnosis as diag
from tam_research.aera_hardware_core import HardwareAERAConfig
from tam_research.aera_hardware_core_v23 import HardwareAwareAERATextLMV23


def _tiny_model() -> HardwareAwareAERATextLMV23:
    cfg = HardwareAERAConfig(
        vocab_size=97,
        d_model=24,
        n_stages=4,
        n_heads=4,
        chunk_size=diag.EXPECTED_CHUNK_SIZE,
        n_experts=4,
        max_active_experts=2,
        expert_mult=2,
        memory_dim=16,
        max_reason_steps=2,
        block_size=2,
    )
    return HardwareAwareAERATextLMV23(cfg)


def test_posthoc_protocol_is_frozen_and_cannot_authorize_scale():
    p = diag.frozen_protocol()
    assert p["research_issue"] == 341
    assert p["source_scientific_result"] == 340
    assert p["checkpoint_seed"] == 8461
    assert p["diagnostic_seed"] == 128461
    assert p["posthoc_only"] is True
    assert p["training_performed"] is False
    assert p["optimizer_created"] is False
    assert p["checkpoint_mutation_authorized"] is False
    assert p["alphas"] == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert p["diagnostic_batches"] == 8
    assert p["diagnostic_batch_size"] == 8
    assert p["chunk_size"] == 256
    assert p["candidates_per_completed_stage_chunk"] == 255
    assert p["selected_writes_per_completed_stage_chunk"] == 16
    assert p["claims"] == {
        "v24_authorized": False,
        "architecture_freeze_authorized": False,
        "independent_replication": False,
        "100m_authorized": False,
        "breakthrough_proven": False,
    }


def test_read_scaling_changes_only_read_output_and_restores_method():
    torch.manual_seed(9101)
    model = _tiny_model().eval()
    stage = model.stages[0]
    events = torch.randn(2, 5, model.cfg.d_model)
    state = stage.empty_state(events)
    state.memory.matrix.normal_()
    original = stage.memory.read(events, state.memory)
    versions = diag.parameter_versions(model)

    with diag.scaled_memory_reads(model, [0.25, 1.0, 1.0, 1.0]):
        scaled = stage.memory.read(events, state.memory)
        torch.testing.assert_close(scaled, original * 0.25)

    restored = stage.memory.read(events, state.memory)
    torch.testing.assert_close(restored, original)
    assert diag.parameter_versions(model) == versions


def test_zero_reads_preserve_exact_16_of_255_sparse_writes_without_parameter_mutation():
    torch.manual_seed(9102)
    model = _tiny_model().eval()
    model.set_memory_pretraining_mode(False)
    stage = model.stages[0]
    events = torch.randn(2, diag.EXPECTED_CHUNK_SIZE, model.cfg.d_model)
    versions = diag.parameter_versions(model)

    with diag.scaled_memory_reads(model, 0.0), torch.no_grad():
        _, state, _ = stage.forward_chunk(
            events,
            None,
            hard=True,
            update_memory=True,
        )

    assert stage.last_candidate_count == 255
    assert stage.last_selected_count == 16
    assert stage.last_selected_indices is not None
    assert stage.last_selected_indices.shape == (2, 16)
    assert float(state.memory.matrix.abs().sum()) > 0.0
    assert diag.parameter_versions(model) == versions


def test_repeat_usefulness_reports_selector_enrichment_and_payload_match():
    chunk0 = torch.arange(256, dtype=torch.long).view(1, 256)
    chunk1 = torch.full((1, 256), 9999, dtype=torch.long)
    # Only addresses 0..15 repeat; each repeated address is followed by its
    # original adjacent payload, so selecting exactly those 16 is maximally useful.
    for i in range(16):
        chunk1[0, 2 * i] = i
        chunk1[0, 2 * i + 1] = i + 1
    selected = torch.arange(16, dtype=torch.long).view(1, 16)
    r = diag.repeat_usefulness_metrics(chunk0, chunk1, selected)
    assert r["selected_repeat_rate"] == 1.0
    assert 0.0 < r["all_candidate_repeat_rate"] < 1.0
    assert r["repeat_enrichment_ratio"] > 1.0
    assert r["selected_payload_match_given_repeat"] == 1.0


def test_scale_validation_rejects_bad_stage_count_and_negative_scale():
    model = _tiny_model()
    try:
        with diag.scaled_memory_reads(model, [1.0, 1.0]):
            pass
    except ValueError as exc:
        assert "expected 4" in str(exc)
    else:
        raise AssertionError("stage-count mismatch was accepted")

    try:
        with diag.scaled_memory_reads(model, -0.1):
            pass
    except ValueError as exc:
        assert "nonnegative" in str(exc)
    else:
        raise AssertionError("negative read scale was accepted")
