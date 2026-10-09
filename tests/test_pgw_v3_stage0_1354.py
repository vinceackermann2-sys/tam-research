"""PGW-v3 #1354/#1359: CPU only, structural validation, NO training."""

import pytest
import torch

from tam_research.pgw_v3_stage0.model import (
    PGWV3Stage0Config, UtilityAddressedWorkspace,
    instantiated_parameter_count,
)


def _twin_modes(mode: str):
    torch.manual_seed(1354)
    reference = UtilityAddressedWorkspace(PGWV3Stage0Config())
    other = UtilityAddressedWorkspace(PGWV3Stage0Config(route_mode=mode))
    other.load_state_dict(reference.state_dict(), strict=True)
    return reference.eval(), other.eval()


def test_routing_modes_have_equal_instantiated_not_necessarily_active_parameters():
    reference, _ = _twin_modes("hybrid")
    target = instantiated_parameter_count(reference)
    assert target == 2209
    sig = {k: tuple(p.shape) for k, p in reference.named_parameters()}
    for mode in ("surprise_only", "utility_only", "recency", "fixed_random", "no_workspace"):
        _, other = _twin_modes(mode)
        assert instantiated_parameter_count(other) == target
        assert sig == {k: tuple(p.shape) for k, p in other.named_parameters()}


def test_workspace_only_reach_and_no_workspace_invariance():
    on, off = _twin_modes("no_workspace")
    torch.manual_seed(1355)
    x = torch.randn(1, 16, 32)
    changed = x.clone()
    changed[:, :8] += torch.randn_like(changed[:, :8]) * 9
    with torch.no_grad():
        a, b = on(x), on(changed)
        c, d = off(x), off(changed)
    assert torch.equal(a[:, :8], b[:, :8])
    assert (a[:, 8:] - b[:, 8:]).abs().max().item() > 1e-9
    assert torch.equal(c, d)
    assert torch.count_nonzero(c).item() == 0


def test_completed_chunk_lag_and_future_causality():
    on, _ = _twin_modes("recency")
    torch.manual_seed(1356)
    x = torch.randn(1, 24, 32)
    changed = x.clone()
    changed[:, 16:] += 6
    with torch.no_grad():
        a, b = on(x), on(changed)
    assert torch.equal(a[:, :16], b[:, :16])


def test_experiment_modes_select_exact_budget_and_fixed_independent_of_content():
    torch.manual_seed(1357)
    x = torch.randn(2, 24, 32)
    for mode in ("hybrid", "surprise_only", "utility_only", "fixed_random", "recency"):
        _, model = _twin_modes(mode)
        with torch.no_grad():
            model(x)
        stats = model.last_stats
        assert stats is not None
        assert stats["selected_fraction"] == 0.25
        positions = stats["selected_positions"]
        assert positions is not None
        assert tuple(positions.shape) == (2, 3, 2)
        assert int(positions.min()) >= 0 and int(positions.max()) < 8
        assert all(row.unique().numel() == 2 for row in positions.reshape(-1, 2))
        if mode == "recency":
            assert torch.equal(positions[0, 0], torch.tensor([6, 7]))
        if mode == "fixed_random":
            with torch.no_grad():
                model(x * -37)
            assert torch.equal(positions, model.last_stats["selected_positions"])


def test_nonzero_finite_gradient_from_future_only_to_past_and_utility():
    on, _ = _twin_modes("recency")
    torch.manual_seed(1358)
    x = torch.randn(1, 16, 32, requires_grad=True)
    result = on(x)
    loss = result[:, 8:].square().sum()
    loss.backward()
    assert torch.isfinite(loss)
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad[:, :8].abs().sum().item() > 0

    hybrid, _ = _twin_modes("surprise_only")
    x2 = torch.randn(1, 16, 32, requires_grad=True)
    result2 = hybrid(x2)
    result2[:, 8:].square().sum().backward()
    assert hybrid.utility.weight.grad is not None
    assert torch.isfinite(hybrid.utility.weight.grad).all()
    assert hybrid.utility.weight.grad.abs().sum().item() > 0


def test_addressed_overwrite_updates_value_at_same_slot():
    torch.manual_seed(1359)
    cfg = PGWV3Stage0Config(
        d_model=4, key_width=2, value_width=2, predictor_rank=2,
        chunk_size=4, workspace_slots=2, selected_events=1,
    )
    model = UtilityAddressedWorkspace(cfg)
    with torch.no_grad():
        model.event_key.weight.zero_()
        model.event_key.weight[:, :2] = torch.eye(2)
        model.event_value.weight.zero_()
        model.event_value.weight[:, 2:] = torch.eye(2)
        model.write_gate.weight.zero_()
        model.write_gate.bias.fill_(15)
    keys = torch.tensor([[[2.0, 0.0], [-2.0, 0.0]]])
    values = torch.zeros(1, 2, 2)
    first = torch.tensor([[2.0, 0.0, 3.0, 4.0]])
    second = torch.tensor([[2.0, 0.0, -5.0, 6.0]])
    keys, values = model._write(first, keys, values)
    assert torch.allclose(values[0, 0], first[0, 2:], atol=0.003)
    keys, values = model._write(second, keys, values)
    assert torch.allclose(values[0, 0], second[0, 2:], atol=0.003)
    assert torch.count_nonzero(values[0, 1]).item() == 0


def test_bad_configs_and_shape_rejected():
    with pytest.raises(ValueError):
        PGWV3Stage0Config(selected_events=9)
    with pytest.raises(ValueError):
        PGWV3Stage0Config(route_mode="unknown")
    model = UtilityAddressedWorkspace(PGWV3Stage0Config())
    with pytest.raises(ValueError):
        model(torch.randn(1, 9, 32))
    with pytest.raises(ValueError):
        model(torch.randn(1, 16, 31))
