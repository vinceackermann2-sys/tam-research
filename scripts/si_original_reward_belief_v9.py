"""Original SI v9: belief learning from observations and action feedback only.

No hidden sensor-polarity z targets, no Bayes probe labels, no probe outcome
before probe acquisition. Calibration cross-products are self-supervised
one-step observations; final theta supervision is reconstructed solely from
an executed binary action's correctness feedback (available after action).
Probe policy is learned via an on-policy score-function reward estimator.
Tiny toy task; do not infer general SI / learned world models from it.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from tam_research.si_original_hybrid_v0 import SIConfig, OriginalSIHybrid, TransformerControl
from scripts.si_original_belief_transfer_v8 import (
    BASE_BLOB, N_CAL, make_episode, bayes_policy, posterior_z, assess_baseline,
    tokens, to_original_inputs,
)

FAMILIES = ('base', 'inverted_prior', 'noisy_reference', 'weak_calibration')


def verify_source():
    target = Path(__file__).resolve().parent.parent/'tam_research'/'si_original_hybrid_v0.py'
    current = subprocess.check_output(['git','hash-object',str(target)], text=True).strip()
    if current != BASE_BLOB:
        raise RuntimeError(f'Original model blob mismatch: {current} != {BASE_BLOB}')
    return current


class RewardLearner(nn.Module):
    def __init__(self, arch: str):
        super().__init__()
        assert arch in ('hybrid', 'transformer', 'gru')
        self.arch = arch
        if arch == 'gru':
            self.encoder = nn.Linear(7, 40)
            self.gru = nn.GRU(40, 40, batch_first=True)
            self.head = nn.Linear(40, 3)
        else:
            cls = OriginalSIHybrid if arch == 'hybrid' else TransformerControl
            self.backbone = cls(SIConfig())

    def encode(self, sequence):
        if self.arch == 'gru':
            hidden, _ = self.gru(torch.tanh(self.encoder(sequence)))
            return self.head(hidden)
        obs, actions = to_original_inputs(sequence)
        return self.backbone(obs, actions)['prediction']

    def first(self, episode):
        initial, _ = tokens(episode, torch.zeros_like(episode['z'], dtype=torch.bool))
        predictions = self.encode(initial)
        return predictions[:, -1, 0], predictions[:, :N_CAL-1, 2], predictions[:, -1, 2]

    def finish(self, episode, purchased):
        _, full = tokens(episode, purchased)
        return self.encode(full)[:, -1, 1]


def observed_feedback(episode, purchased, action_positive):
    """Environment produces correctness, reward; NEVER returns latent z/theta.

    Private theta is used by the simulated ENVIRONMENT to score outcomes,
    not handed directly to learning. A binary action + correctness tells the
    agent the correct theta AFTER action execution (allowed feedback).
    """
    correct = (action_positive == episode['truth'].gt(0))
    net = correct.float() - episode['c'] * purchased.float()
    return correct, net


def inferred_outcome_label(action_positive, correct):
    """Observed binary action and binary feedback identify the correct action."""
    return torch.where(correct, action_positive, ~action_positive).float()


def self_supervised_target(episode):
    """Observable NEXT cross-sensor covariance from completed calibration."""
    return torch.tanh(episode['x_cal'][:, 1:] * episode['y_cal'][:, 1:] / 3)


def train_one(seed, arch, steps=240, batch=128):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    model = RewardLearner(arch)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    running_reward_baseline = .5
    clock = time.perf_counter()
    telemetry=[]
    for step in range(steps):
        episode = make_episode(batch, seed*1000000+step, 'base')
        model.train()
        optimizer.zero_grad(set_to_none=True)
        purchase_logits, future_cov, query_cov = model.first(episode)
        # The purchase uses ONLY the pre-purchase input. RNG independent of
        # model initialization and shared across all architectures/steps.
        g=torch.Generator().manual_seed(seed*1000000+step+987001)
        buy = (torch.rand(batch, generator=g) < torch.sigmoid(purchase_logits).detach())
        action_logits = model.finish(episode, buy)
        action_positive = (torch.rand(batch, generator=g) < torch.sigmoid(action_logits).detach())
        correctness, final_reward = observed_feedback(episode, buy, action_positive)
        recovered_target = inferred_outcome_label(action_positive, correctness)
        # Proper next-outcome log loss, using only action/feedback observations.
        action_loss = F.binary_cross_entropy_with_logits(action_logits, recovered_target)
        # Predict next *observable* calibration correlation, NOT latent z.
        cov_loss = F.mse_loss(future_cov, self_supervised_target(episode))
        # Optional second cross-sensor target is observed only when bought.
        if buy.any():
            query_cov_label = torch.tanh(episode['x_query']*episode['y_probe']/3)
            query_loss = F.mse_loss(query_cov[buy], query_cov_label[buy])
        else:
            query_loss = query_cov.sum()*0
        logp = torch.where(buy, F.logsigmoid(purchase_logits), F.logsigmoid(-purchase_logits))
        advantage = final_reward.detach() - running_reward_baseline
        probe_loss = -(advantage * logp).mean()
        entropy = torch.distributions.Bernoulli(logits=purchase_logits).entropy().mean()
        loss = action_loss + .25*cov_loss + .05*query_loss + probe_loss - .005*entropy
        if not torch.isfinite(loss): raise RuntimeError('nonfinite training loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        running_reward_baseline = .975*running_reward_baseline + .025*float(final_reward.mean())
        if step in (0, steps-1):
            telemetry.append({'step':step,'reward':float(final_reward.mean()),
                'probe_rate':float(buy.float().mean()),'next_observation_mse':float(cov_loss.detach()),
                'action_logloss':float(action_loss.detach())})
    return model, time.perf_counter()-clock, telemetry


@torch.no_grad()
def evaluate_model(model, episode):
    model.eval()
    logits, future_cov, _ = model.first(episode)
    purchased = logits >= 0
    action_logits = model.finish(episode, purchased)
    choice_positive = action_logits >= 0
    correct, net = observed_feedback(episode, purchased, choice_positive)
    return dict(reward=float(net.mean()), accuracy=float(correct.float().mean()),
                probe_rate=float(purchased.float().mean()),
                observable_next_cov_mse=float(F.mse_loss(future_cov,self_supervised_target(episode))))


def run(seed, steps=240, batch=128, n_eval=3072):
    torch.set_num_threads(2)
    sha=verify_source()
    report={'seed':seed,'steps':steps,'batch':batch,'n_eval':n_eval,
            'source_blob':sha,'models':{},'baselines':{}}
    evaluation={fam:make_episode(n_eval,seed*1000000+80000+127*j,fam)
                 for j,fam in enumerate(FAMILIES)}
    for fam, ep in evaluation.items():
        report['baselines'][fam]={k:assess_baseline(ep,k)
                                  for k in ('bayes','no_cal_bayes','never_probe','always_probe')}
    for arch in ('hybrid','transformer','gru'):
        m, elapsed, telemetry = train_one(seed, arch, steps, batch)
        report['models'][arch]={
            'parameters':sum(p.numel() for p in m.parameters()),
            'training_seconds_cpu':elapsed,
            'training_telemetry':telemetry,
            'evaluations':{fam:evaluate_model(m,ep) for fam,ep in evaluation.items()},
            'no_calibration':evaluate_model(m,{**evaluation['base'],
                'x_cal':torch.zeros_like(evaluation['base']['x_cal']),
                'y_cal':torch.zeros_like(evaluation['base']['y_cal'])}),
        }
    return report


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--seed',type=int,default=101000)
    p.add_argument('--steps',type=int,default=240)
    p.add_argument('--batch',type=int,default=128)
    p.add_argument('--n-eval',type=int,default=3072)
    a=p.parse_args()
    print(json.dumps(run(a.seed,a.steps,a.batch,a.n_eval),indent=2))
