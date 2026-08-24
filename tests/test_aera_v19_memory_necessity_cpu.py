from __future__ import annotations

import torch

import aera_v19_memory_necessity_cpu as diag


def test_associative_batch_is_valid_and_overwrites_are_real() -> None:
    batch = diag.make_batch(4, 123)
    assert diag.oracle_accuracy(batch) == 1.0
    assert batch.query_positions.shape == (4, diag.N_QUERY_KEYS)
    assert batch.query_targets.shape == (4, diag.N_QUERY_KEYS)
    assert batch.overwrite_mask.sum(dim=1).tolist() == [diag.N_OVERWRITES] * 4
    overwritten_stale = batch.stale_targets[batch.overwrite_mask]
    overwritten_current = batch.query_targets[batch.overwrite_mask]
    assert torch.all(overwritten_stale >= diag.VALUE_START)
    assert torch.all(overwritten_stale != overwritten_current)


def test_matched_controls_start_bit_exact_and_all_stages_are_forced_on() -> None:
    full = diag.build_model(diag.SEED)
    stream = diag.build_model(diag.SEED)
    assert set(full.state_dict()) == set(stream.state_dict())
    for key, value in full.state_dict().items():
        torch.testing.assert_close(stream.state_dict()[key], value, atol=0.0, rtol=0.0)
    for model in (full, stream):
        for router in model.stage_routers:
            assert not any(p.requires_grad for p in router.parameters())
            torch.testing.assert_close(router.proj.weight, torch.zeros_like(router.proj.weight))
            assert torch.all(router.proj.bias > 10.0)


def test_query_loss_reaches_integrated_memory_parameters() -> None:
    model = diag.build_model(321).train()
    batch = diag.make_batch(2, 456)
    loss, accuracy = diag._loss_and_accuracy(
        model,
        batch,
        update_memory=True,
        differentiable_memory=True,
    )
    assert torch.isfinite(loss)
    assert 0.0 <= accuracy <= 1.0
    loss.backward()
    memory_grad = 0.0
    for stage in model.stages:
        for parameter in (
            stage.memory.q.weight,
            stage.memory.k.weight,
            stage.memory.v.weight,
            stage.memory.out.weight,
        ):
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
            memory_grad += float(parameter.grad.abs().sum())
    assert memory_grad > 0.0


def test_short_smoke_training_is_finite_without_claiming_gate_pass() -> None:
    full, stream, training = diag.train_pair(steps=2)
    assert training["history"]
    for row in training["history"]:
        assert torch.isfinite(torch.tensor(row["full_loss"]))
        assert torch.isfinite(torch.tensor(row["stream_only_loss"]))
    assert sum(p.numel() for p in full.parameters()) == sum(p.numel() for p in stream.parameters())


def test_protocol_is_cpu_only_and_thresholds_match_preregistration() -> None:
    protocol = diag.protocol_summary()
    assert protocol["cpu_only"] is True
    assert protocol["gpu_authorized"] is False
    assert protocol["thresholds"] == {
        "task_validity_min": 0.95,
        "full_accuracy_min": 0.80,
        "full_over_stream_min": 0.15,
        "same_checkpoint_memory_drop_min": 0.15,
        "overwrite_accuracy_min": 0.80,
        "stale_error_max": 0.10,
        "fresh_session_chance_tolerance": 0.10,
    }
