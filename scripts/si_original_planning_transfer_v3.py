"""Original SI hybrid: CPU-only action-planning and held-out dynamics probe.

Trains ORIGINAL GitHub tam_research.si_original_hybrid_v0 models, unchanged.
Each world has a private transition and action-effect matrix. Prediction
uses only causal observed state/action sequences; action choice is evaluated
against the real simulator, not a prediction proxy. No paid GPU usage.
"""
from __future__ import annotations
import argparse
import json
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from tam_research.si_original_hybrid_v0 import (
    SIConfig, OriginalSIHybrid, TransformerControl, prediction_loss,
)


FAMILIES = ('direct', 'delayed', 'switch', 'nonlinear')


def advance(state, action, prev_action, mat, effect, family):
    raw = torch.bmm(state[:, None], mat).squeeze(1)
    current = torch.bmm(action[:, None], effect).squeeze(1)
    previous = torch.bmm(prev_action[:, None], effect).squeeze(1)
    if family == 'delayed':
        raw = raw + .35 * current + .75 * previous
    else:
        raw = raw + current
    if family == 'nonlinear':
        raw = raw + .55 * state.roll(1, -1) * state.roll(2, -1)
    return .15 * state + .85 * torch.tanh(raw)


def world(batch, steps, seed, family):
    assert family in FAMILIES
    g = torch.Generator(device='cpu').manual_seed(seed)
    rnd = lambda *s: torch.randn(*s, generator=g)
    mat = .27 * rnd(batch, 3, 3) + .22 * torch.eye(3)[None]
    mat_alt = .27 * rnd(batch, 3, 3) + .22 * torch.eye(3)[None]
    effect = 1.3 * rnd(batch, 4, 3)
    effect_alt = 1.3 * rnd(batch, 4, 3)
    state = .45 * rnd(batch, 3)
    prev_action = torch.zeros(batch, 4)
    observations, actions, targets = [], [], []
    switch_time = torch.randint(steps // 2 - 2, steps // 2 + 3, (batch,), generator=g)
    # Track last-step hidden dynamics only in the evaluator, NOT the models.
    last_mat, last_effect, last_prev = None, None, None
    for t in range(steps):
        action = F.one_hot(torch.randint(0, 4, (batch,), generator=g), 4).float()
        switched = (family == 'switch') & (t >= switch_time)
        m = torch.where(switched[:, None, None], mat_alt, mat)
        e = torch.where(switched[:, None, None], effect_alt, effect)
        observations.append(state)
        actions.append(action)
        last_mat, last_effect, last_prev = m, e, prev_action
        state_next = advance(state, action, prev_action, m, e, family)
        targets.append(state_next)
        state, prev_action = state_next, action
    return (
        torch.stack(observations, 1),
        torch.stack(actions, 1),
        torch.stack(targets, 1),
        last_mat, last_effect, last_prev,
    )


@torch.no_grad()
def true_action_outcomes(batch, family):
    obs, acts, _, mat, effect, prev = batch
    state = obs[:, -1]
    candidates = torch.eye(4)[None].expand(state.size(0), -1, -1)
    outcomes = []
    for choice in range(4):
        outcomes.append(advance(state, candidates[:, choice], prev, mat, effect, family))
    return torch.stack(outcomes, 1)


@torch.no_grad()
def model_action_prediction(model, batch):
    obs, acts = batch[:2]
    batch_size, seq_len, _ = obs.shape
    # One batched forward across all 4 alternative query actions; context
    # observations/actions from t<last are identical in the four copies.
    query = torch.eye(4)[None].expand(batch_size, 4, -1)
    obs_four = obs[:, None].expand(-1, 4, -1, -1).reshape(-1, seq_len, 3)
    act_four = acts[:, None].expand(-1, 4, -1, -1).clone()
    act_four[:, :, -1] = query
    pred = model(obs_four, act_four.reshape(-1, seq_len, 4))['prediction'][:, -1]
    return pred.reshape(batch_size, 4, 3)


@torch.no_grad()
def ridge_action_prediction(batch, recent=None):
    obs, actions = batch[:2]
    # Completed transitions are known from later observations, i.e. the last
    # transition (query action at t=steps-1) must NOT enter regression.
    previous_action = torch.cat((torch.zeros_like(actions[:, :1]), actions[:, :-1]), 1)
    features = torch.cat((obs, actions, previous_action, torch.ones_like(obs[..., :1])), -1)
    ctx = slice(0, -1) if recent is None else slice(-recent-1, -1)
    x = features[:, ctx]
    # For contextual i, target is observation at i+1.
    y = obs[:, 1:]
    if recent is not None:
        y = y[:, -recent:]
    xt = x.transpose(1, 2)
    ridge = torch.eye(x.size(-1))[None] * .2
    weights = torch.linalg.solve(xt @ x + ridge, xt @ y)
    q = features[:, -1:, None, :].expand(-1, 1, 4, -1).squeeze(1).clone()
    q[:, :, 3:7] = torch.eye(4)[None]
    return torch.bmm(q, weights)


@torch.no_grad()
def planning_metrics(predicted, true, goal):
    true_costs = (true - goal[:, None]).square().sum(-1)
    estimate_costs = (predicted - goal[:, None]).square().sum(-1)
    chosen = estimate_costs.argmin(-1)
    optimal = true_costs.argmin(-1)
    achieved = true_costs.gather(-1, chosen[:, None]).squeeze(-1)
    best = true_costs.min(-1).values
    # Choose action with the most positive gain over random, not a proxy score.
    return torch.stack(((chosen == optimal).float(), (achieved - best), achieved), -1)


@torch.no_grad()
def evaluate(model, seed, steps, batch_size=64, draws=3):
    model.eval()
    result = {}
    for family in FAMILIES:
        outcomes = {'model': [], 'ridge_all': [], 'ridge_recent': [], 'random': []}
        predict_mse = []
        for i in range(draws):
            data = world(batch_size, 20, seed + 19*i, family)
            g = torch.Generator().manual_seed(seed + 55555 + 19*i)
            truth = true_action_outcomes(data, family)
            index = torch.randint(0, 4, (batch_size,), generator=g)
            # Goal is produced by one genuinely executable candidate action,
            # with independent noise. Oracle is determined *after* noise.
            goal = truth[torch.arange(batch_size), index] + .08 * torch.randn(batch_size, 3, generator=g)
            estimates = {
                'model': model_action_prediction(model, data),
                'ridge_all': ridge_action_prediction(data, recent=None),
                'ridge_recent': ridge_action_prediction(data, recent=6),
            }
            # Random uses fixed independent generator for all model comparisons.
            random_choice = torch.randint(0, 4, (batch_size,), generator=g)
            # Treat random choices explicitly so no accidental ties in a dummy prediction.
            cost = (truth - goal[:, None]).square().sum(-1)
            opt = cost.min(-1).values
            chosen_cost = cost.gather(1, random_choice[:, None]).squeeze(-1)
            outcomes['random'].append(torch.stack(((random_choice == cost.argmin(-1)).float(), chosen_cost-opt, chosen_cost),-1).mean(0))
            for key, estimation in estimates.items():
                outcomes[key].append(planning_metrics(estimation, truth, goal).mean(0))
            predict_mse.append((estimates['model'] - truth).square().mean())
        result[family] = {
            'prediction_mse': torch.stack(predict_mse).mean().item(),
            'planning': {key: dict(accuracy=float(torch.stack(vals).mean(0)[0]),
                                   regret=float(torch.stack(vals).mean(0)[1]),
                                   realized_cost=float(torch.stack(vals).mean(0)[2]))
                         for key, vals in outcomes.items()},
        }
    return result


def run(seed, steps=450, train_batch=16):
    torch.set_num_threads(2)
    config = SIConfig()
    results = dict(seed=seed, steps=steps, batch_size=train_batch, models={})
    for name, cls in [('original_hybrid', OriginalSIHybrid), ('transformer', TransformerControl)]:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        model = cls(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
        start = time.perf_counter()
        for step in range(steps):
            family = ('direct','delayed','switch')[step % 3]
            data = world(train_batch, 20, seed * 1000000 + step, family)
            model.train(); optimizer.zero_grad(set_to_none=True)
            outputs = model(data[0], data[1]);loss, _ = prediction_loss(outputs, data[2])
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0);optimizer.step()
        elapsed = time.perf_counter() - start
        results['models'][name] = {
            'parameters': sum(p.numel() for p in model.parameters()),
            'training_seconds_cpu': elapsed,
            'heldout': evaluate(model, seed*1000000+75000, steps),
        }
    return results


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,default=96800)
    parser.add_argument('--steps',type=int,default=450)
    parser.add_argument('--train-batch',type=int,default=16)
    args=parser.parse_args()
    print(json.dumps(run(args.seed,args.steps,args.train_batch),indent=2))
