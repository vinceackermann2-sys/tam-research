"""Original SI two-action planning v5: CPU-only action utility test.

Uses unchanged original hybrid and equal-memory Transformer. Neither sees
hidden world matrices, future outcomes, or goal actions. Horizon-2 planning
computes all 16 candidate action pairs. No paid GPU or network use.
"""
from __future__ import annotations
import argparse
import json
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
from tam_research.si_original_hybrid_v0 import SIConfig, OriginalSIHybrid, TransformerControl, prediction_loss
from scripts.si_original_fast_memory_v4 import FastEpisodicMemory
from scripts.si_original_planning_transfer_v3 import world, advance

ACTIONS = torch.eye(4)


def true_two_step(batch, family):
    """Evaluator-only: [B, 16, 3] outcome after actions a1,a2."""
    obs, acts, _, m, effects, previous = batch
    s = obs[:, -1]
    outputs = []
    for a in ACTIONS:
        a0 = a[None, :].expand(s.size(0), -1)
        future = advance(s, a0, previous, m, effects, family)
        for b in ACTIONS:
            a1 = b[None, :].expand(s.size(0), -1)
            outputs.append(advance(future, a1, a0, m, effects, family))
    return torch.stack(outputs, 1)


@torch.no_grad()
def learned_two_step(model, batch):
    obs, acts = batch[:2]
    b, seq, _ = obs.shape
    pairs = torch.cartesian_prod(torch.arange(4), torch.arange(4))
    first, second = ACTIONS[pairs[:, 0]], ACTIONS[pairs[:, 1]]
    obs16 = obs[:, None].expand(-1, 16, -1, -1).reshape(-1, seq, 3)
    actions16 = acts[:, None].expand(-1, 16, -1, -1).clone()
    actions16[:, :, -1, :] = first[None]
    actions16 = actions16.reshape(-1, seq, 4)
    first_pred = model(obs16, actions16)['prediction'][:, -1]
    # The imagined observation is the model's own first-step prediction,
    # not any future truth. Memory updates here receive zero innovation.
    imagined_obs = torch.cat((obs16, first_pred[:, None]), 1)
    imagined_actions = torch.cat((actions16, second[None].expand(b, -1, -1).reshape(-1, 1, 4)), 1)
    return model(imagined_obs, imagined_actions)['prediction'][:, -1].reshape(b, 16, 3)


@torch.no_grad()
def ridge_two_step(batch, recent=None):
    obs, acts = batch[:2]
    prev_acts = torch.cat((torch.zeros_like(acts[:, :1]), acts[:, :-1]), 1)
    features = torch.cat((obs, acts, prev_acts, torch.ones_like(obs[..., :1])), -1)
    slice_ctx = slice(None, -1) if recent is None else slice(-recent - 1, -1)
    x, y = features[:, slice_ctx], obs[:, 1:] if recent is None else obs[:, -recent:]
    xt = x.transpose(1, 2)
    mat = torch.linalg.solve(xt @ x + .2 * torch.eye(x.size(-1))[None], xt @ y)
    b = obs.size(0)
    a1 = ACTIONS.repeat_interleave(4, 0)[None].expand(b, -1, -1)
    a2 = ACTIONS.repeat(4, 1).reshape(16, 4)[None].expand(b, -1, -1)
    last = obs[:, -1:, :].expand(-1, 16, -1)
    prev = prev_acts[:, -1:, :].expand(-1, 16, -1)
    q1 = torch.cat((last, a1, prev, torch.ones(b, 16, 1)), -1)
    s1 = torch.bmm(q1, mat)
    q2 = torch.cat((s1, a2, a1, torch.ones(b, 16, 1)), -1)
    return torch.bmm(q2, mat)


@torch.no_grad()
def score_plan(pred, truth, goal):
    true_costs = (truth - goal[:, None]).square().sum(-1)
    est_costs = (pred - goal[:, None]).square().sum(-1)
    chosen = est_costs.argmin(-1)
    oracle = true_costs.argmin(-1)
    regret = true_costs.gather(1, chosen[:, None]).squeeze(1) - true_costs.min(-1).values
    return torch.stack(((chosen == oracle).float(), regret, true_costs.gather(1, chosen[:, None]).squeeze(1)), -1).mean(0)


@torch.no_grad()
def evaluate(model, seed, n_eval=2, eval_batch=32):
    model.eval()
    results = {}
    for family in ('direct', 'delayed', 'switch', 'nonlinear'):
        outcome = {k: [] for k in ('model', 'ridge_all', 'ridge_recent', 'random')}
        query_mse = []
        for i in range(n_eval):
            sample = world(eval_batch, 20, seed + i * 19, family)
            truth = true_two_step(sample, family)
            generator = torch.Generator().manual_seed(seed + 55555 + i * 19)
            goal_index = torch.randint(0, 16, (eval_batch,), generator=generator)
            goal = truth[torch.arange(eval_batch), goal_index] + .08 * torch.randn(eval_batch, 3, generator=generator)
            estimates = {'model': learned_two_step(model, sample),
                         'ridge_all': ridge_two_step(sample),
                         'ridge_recent': ridge_two_step(sample, 6)}
            cost = (truth - goal[:, None]).square().sum(-1)
            random_index = torch.randint(0, 16, (eval_batch,), generator=generator)
            best = cost.min(-1).values
            rc = cost.gather(1, random_index[:, None]).squeeze(1)
            random_metrics = torch.stack(((random_index == cost.argmin(-1)).float(), rc - best, rc), -1).mean(0)
            outcome['random'].append(random_metrics)
            for key, prediction in estimates.items():
                outcome[key].append(score_plan(prediction, truth, goal))
            query_mse.append((estimates['model'] - truth).square().mean())
        results[family] = {
            'prediction_mse': float(torch.stack(query_mse).mean()),
            'planning': {key: {'accuracy':float(torch.stack(vals).mean(0)[0]),
                              'regret':float(torch.stack(vals).mean(0)[1]),
                              'realized_cost':float(torch.stack(vals).mean(0)[2])} for key,vals in outcome.items()},
        }
    return results


def run(seed, steps=250, batch=16, eval_draws=2):
    torch.set_num_threads(2)
    results = {'seed':seed, 'steps':steps, 'batch':batch, 'eval_draws':eval_draws, 'models':{}}
    for name, class_ in (('original_fast_memory',OriginalSIHybrid),('transformer_fast_memory',TransformerControl)):
        torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
        model = FastEpisodicMemory(class_(SIConfig()))
        opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
        start=time.monotonic()
        for step in range(steps):
            family=('direct','delayed','switch')[step % 3]
            data=world(batch,20,seed * 1000000 + step,family)
            model.train();opt.zero_grad(set_to_none=True)
            l,_ = prediction_loss(model(data[0],data[1]),data[2]);l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        training_time = time.monotonic()-start
        started_eval=time.monotonic()
        results['models'][name] = {'parameters':sum(p.numel() for p in model.parameters()),
              'training_seconds_cpu':training_time,
              'evaluation':evaluate(model,seed * 1000000 + 75000,eval_draws),
              'evaluation_seconds_cpu':time.monotonic()-started_eval,
              'learned_write_rate':float(torch.sigmoid(model.eta_logit).detach())}
    return results


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,default=97900)
    parser.add_argument('--steps',type=int,default=250)
    parser.add_argument('--batch',type=int,default=16)
    parser.add_argument('--eval-draws',type=int,default=2)
    args=parser.parse_args()
    print(json.dumps(run(args.seed,args.steps,args.batch,args.eval_draws),indent=2))
