"""Integrity controls for original SI planning-transfer CPU benchmark."""
import torch
from tam_research.si_original_hybrid_v0 import SIConfig, OriginalSIHybrid, TransformerControl
from scripts.si_original_planning_transfer_v3 import (
    FAMILIES, world, true_action_outcomes, model_action_prediction,
    ridge_action_prediction, planning_metrics,
)


def test_world_reproducibility_and_unseen_family_differs():
    a = world(4, 20, 96800, 'direct')
    b = world(4, 20, 96800, 'direct')
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    c = world(4, 20, 96800, 'nonlinear')
    assert not torch.equal(a[2], c[2])


def test_counterfactual_actual_action_equals_simulated_target():
    for family in FAMILIES:
        batch = world(8, 20, 96800, family)
        outcomes = true_action_outcomes(batch, family)
        selected = batch[1][:, -1].argmax(-1)
        chosen_true = outcomes[torch.arange(8), selected]
        assert torch.allclose(chosen_true, batch[2][:, -1], atol=1e-6)


def test_future_observations_do_not_affect_prior_predictions():
    torch.manual_seed(96800)
    original = OriginalSIHybrid(SIConfig()).eval()
    trans = TransformerControl(SIConfig()).eval()
    batch = world(4, 20, 96800, 'switch')
    obs, actions = batch[:2]
    changed = obs.clone()
    changed[:, 13:] += 9
    with torch.no_grad():
        for model in (original, trans):
            y1 = model(obs, actions)['prediction'][:, :13]
            y2 = model(changed, actions)['prediction'][:, :13]
            assert torch.allclose(y1, y2, atol=1e-6)


def test_counterfactual_action_does_not_mutate_prefix():
    torch.manual_seed(96800)
    model = OriginalSIHybrid(SIConfig()).eval()
    batch = world(3, 20, 96800, 'direct')
    obs, action = batch[:2]
    changed = action.clone()
    changed[:, -1] = torch.eye(4)[0]
    with torch.no_grad():
        left = model(obs, action)['prediction'][:, :-1]
        right = model(obs, changed)['prediction'][:, :-1]
        assert torch.allclose(left, right, atol=1e-6)
        planned = model_action_prediction(model, batch)
        assert planned.shape == (3, 4, 3)


def test_linear_ridge_does_not_use_query_target():
    batch = world(16, 20, 96800, 'direct')
    estimated = ridge_action_prediction(batch)
    assert estimated.shape == (16, 4, 3)
    copy = list(batch)
    copy[2] = batch[2].clone()
    copy[2][:, -1] += 300
    assert torch.allclose(estimated, ridge_action_prediction(tuple(copy)), atol=1e-6)


def test_oracle_planning_identity():
    batch = world(8, 20, 96800, 'delayed')
    truth = true_action_outcomes(batch, 'delayed')
    goal = truth[:, 1] + .03
    m = planning_metrics(truth, truth, goal)
    assert m[:, 0].min().item() == 1.0
    assert m[:, 1].abs().max().item() < 1e-7
