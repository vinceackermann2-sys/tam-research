"""Source-verified original SI: true-feedback vs imagined-open-loop 2-action planning.

The original full 298-line SI model must be attested by git blob SHA before
execution. For each candidate planner: take the same first action under its
own open-loop proposal, then compare sticking to that proposed second action
against choosing a new second action after observing the true first outcome.
No private world law or unobserved future is fed to the planner.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from tam_research.si_original_hybrid_v0 import (
    SIConfig, OriginalSIHybrid, TransformerControl, prediction_loss,
)
from scripts.si_original_fast_memory_v4 import FastEpisodicMemory
from scripts.si_original_planning_transfer_v3 import world, advance, ridge_action_prediction
from scripts.si_original_two_step_v5 import learned_two_step, ridge_two_step, true_two_step

# Exact source content under original SI experiment branch, validated against
# live GitHub connector. Changed base model bytes invalidate this run.
EXPECTED_BASE_BLOB = '575c17dc2f39a29d930d82179aa0c96dcbdc7d0f'
ACTIONS = torch.eye(4)
FAMILIES = ('direct', 'delayed', 'switch', 'nonlinear')


def attested_source():
    import subprocess
    target = Path(__file__).resolve().parent.parent / 'tam_research' / 'si_original_hybrid_v0.py'
    digest = subprocess.check_output(['git', 'hash-object', str(target)], text=True).strip()
    if digest != EXPECTED_BASE_BLOB:
        raise RuntimeError(f'Original SI source mismatch: {digest} != {EXPECTED_BASE_BLOB}')
    return digest


@torch.no_grad()
def second_action_prediction(model, observed, observed_actions):
    """Choose among 4 second actions from the REAL observed x20, no x21."""
    b, t, _ = observed.shape
    four_obs = observed[:, None].expand(-1, 4, -1, -1).reshape(-1, t, 3)
    four_actions = observed_actions[:, None].expand(-1, 4, -1, -1).clone()
    four_actions[:, :, -1] = ACTIONS[None]
    predicted = model(four_obs, four_actions.reshape(-1, t, 4))['prediction'][:, -1]
    return predicted.reshape(b, 4, 3)


@torch.no_grad()
def score_planner(model, data, family, goal, method):
    """Return exact paired open-loop and closed-loop scores for same first action."""
    obs, act, _, mat, effects, prev = data
    b = obs.size(0)
    # The initial decision uses only the original observed episode.
    initial_predictions = (learned_two_step(model, data) if method == 'model'
                           else ridge_two_step(data, None if method == 'ridge_all' else 6))
    costs = (initial_predictions - goal[:, None]).square().sum(-1)
    preferred = costs.argmin(-1)
    first = torch.div(preferred, 4, rounding_mode='floor')
    proposed_second = preferred.remainder(4)
    # Environment executes the chosen first action and reveals the new state.
    executed = ACTIONS[first]
    true_first = advance(obs[:, -1], executed, prev, mat, effects, family)
    observed = torch.cat((obs, true_first[:, None]), 1)
    actions = act.clone()
    actions[:, -1] = executed
    next_action_zeros = torch.zeros(b, 1, 4)
    observed_actions = torch.cat((actions, next_action_zeros), 1)
    # New causal read/write from the completed executed transition.
    second_predictions = (second_action_prediction(model, observed, observed_actions)
                          if method == 'model' else
                          ridge_action_prediction((observed, observed_actions),
                                                  recent=None if method == 'ridge_all' else 6))
    improved_second = (second_predictions - goal[:, None]).square().sum(-1).argmin(-1)
    # Ground truth now scores both original and updated plans.
    true_final_open = advance(true_first, ACTIONS[proposed_second], executed,
                              mat, effects, family)
    true_final_closed = advance(true_first, ACTIONS[improved_second], executed,
                                mat, effects, family)
    truth = true_two_step(data, family)
    truth_costs = (truth - goal[:, None]).square().sum(-1)
    optimal = truth_costs.min(-1).values
    open_real_cost = (true_final_open - goal).square().sum(-1)
    closed_real_cost = (true_final_closed - goal).square().sum(-1)
    closed_pair = first * 4 + improved_second
    return {
        'open_regret': (open_real_cost - optimal).clamp_min(0),
        'closed_regret': (closed_real_cost - optimal).clamp_min(0),
        'open_accuracy': (preferred == truth_costs.argmin(-1)).float(),
        'closed_accuracy': (closed_pair == truth_costs.argmin(-1)).float(),
        'second_action_changed': (proposed_second != improved_second).float(),
        'actual_first_observation': true_first,
    }


@torch.no_grad()
def evaluate(model, seed, draws=3, batch=48):
    model.eval()
    report = {}
    for family in FAMILIES:
        metrics = {m:[] for m in ('model', 'ridge_all', 'ridge_recent')}
        random_scores = []
        for j in range(draws):
            eval_seed = seed + 17*j
            data = world(batch, 20, eval_seed, family)
            truth = true_two_step(data, family)
            g = torch.Generator().manual_seed(eval_seed + 55555)
            picked = torch.randint(0, 16, (batch,), generator=g)
            goal = truth[torch.arange(batch), picked] + .08 * torch.randn(batch, 3, generator=g)
            for key in metrics:
                score = score_planner(model, data, family, goal, key)
                metrics[key].append({name:float(value.mean()) for name,value in score.items()
                                     if name != 'actual_first_observation'})
            oracle_cost = (truth - goal[:,None]).square().sum(-1).min(-1).values
            random_pick = torch.randint(0, 16, (batch,), generator=g)
            random_cost = (truth - goal[:,None]).square().sum(-1).gather(1,random_pick[:,None]).squeeze(1)
            random_scores.append(float((random_cost - oracle_cost).mean()))
        report[family] = {
            'planners':{key:{k:float(np.mean([trial[k] for trial in vals]))
                              for k in vals[0]} for key,vals in metrics.items()},
            'random_open_regret':float(np.mean(random_scores)),
        }
    return report


def run(seed, steps=275, batch=16, eval_draws=3, eval_batch=48):
    torch.set_num_threads(2)
    base_sha = attested_source()
    report={'seed':seed,'steps':steps,'batch':batch,'eval_draws':eval_draws,
            'eval_batch':eval_batch,'verified_original_blob':base_sha,'models':{}}
    for name,cls in [('hybrid_mem',OriginalSIHybrid),('transformer_mem',TransformerControl)]:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        model = FastEpisodicMemory(cls(SIConfig()))
        opt = torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
        start = time.perf_counter()
        for step in range(steps):
            family = ('direct','delayed','switch')[step % 3]
            data = world(batch,20,seed*1000000 + step,family)
            model.train();opt.zero_grad(set_to_none=True)
            loss,_ = prediction_loss(model(data[0],data[1]),data[2])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step()
        training = time.perf_counter() - start
        start_eval = time.perf_counter()
        scores = evaluate(model,seed*1000000+75000,eval_draws,eval_batch)
        report['models'][name]={'params':sum(p.numel() for p in model.parameters()),
            'training_cpu_seconds':training,'evaluation_cpu_seconds':time.perf_counter()-start_eval,
            'scores':scores,'write_rate':float(torch.sigmoid(model.eta_logit).detach())}
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,default=98500)
    parser.add_argument('--steps',type=int,default=275)
    parser.add_argument('--batch',type=int,default=16)
    parser.add_argument('--eval-draws',type=int,default=3)
    parser.add_argument('--eval-batch',type=int,default=48)
    args=parser.parse_args()
    print(json.dumps(run(args.seed,args.steps,args.batch,args.eval_draws,args.eval_batch),indent=2))
