"""SI original v10: learn conditional probe value from *observed randomized outcomes*.

Two-stage contextual bandit: train the final binary decision from action/correctness
feedback under randomized probe purchases, freeze the decision-maker, then regress
two context-dependent success values using only the *purchased* arm's actual
correctness. Policy purchases when estimated gain exceeds its known cost.

No hidden sensor z, truth labels, analytic Bayes teacher, or unpurchased
probe signal enters an optimizer or action policy. Strictly CPU engineering.
"""
from __future__ import annotations
import argparse
import json
import math
import random
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tam_research.si_original_hybrid_v0 import SIConfig, OriginalSIHybrid, TransformerControl
from scripts.si_original_belief_transfer_v8 import BASE_BLOB, make_episode, tokens, to_original_inputs, assess_baseline

FAMILIES = ('base', 'inverted_prior', 'noisy_reference', 'weak_calibration')

def verify_original():
    path=Path(__file__).resolve().parent.parent/'tam_research'/'si_original_hybrid_v0.py'
    digest=subprocess.check_output(['git','hash-object',str(path)],text=True).strip()
    if digest!=BASE_BLOB: raise RuntimeError('Model source hash mismatch: '+digest)
    return digest

class ContextualController(nn.Module):
    def __init__(self,kind):
        super().__init__()
        self.kind=kind
        if kind=='gru':
            self.encoder=nn.Linear(7,40)
            self.gru=nn.GRU(40,40,batch_first=True)
            self.head=nn.Linear(40,3)
        else:
            self.model=(OriginalSIHybrid if kind=='hybrid' else TransformerControl)(SIConfig())

    def forward_raw(self,tokens_):
        if self.kind=='gru':
            h,_=self.gru(torch.tanh(self.encoder(tokens_)))
            return self.head(h)
        obs,acts=to_original_inputs(tokens_)
        return self.model(obs,acts)['prediction']

    def values(self,e):
        before,_=tokens(e,torch.zeros(len(e['z']),dtype=torch.bool))
        return self.forward_raw(before)[:,-1,1:3]

    def decision_logits(self,e,bought):
        _,after=tokens(e,bought)
        return self.forward_raw(after)[:,-1,0]

    def value_output_parameters(self):
        return self.head.parameters() if self.kind=='gru' else self.model.heads.world.parameters()


def feedback(e,positive_action,bought):
    # Simulator generates correctness only AFTER execution. Correctness plus
    # the agent's own binary action implies correct outcome; no latent inputs.
    correctness=positive_action==e['truth'].gt(0)
    label=torch.where(correctness,positive_action,~positive_action).float()
    net=correctness.float()-e['c']*bought.float()
    return correctness.float(),label,net


def rand_buy(batch,seed):
    g=torch.Generator().manual_seed(int(seed))
    return torch.rand(batch,generator=g)<.5


def train_one(seed,kind,actor_steps=240,value_steps=260,batch=128):
    torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
    net=ContextualController(kind)
    optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01)
    clock=time.perf_counter()
    last_actor=None
    for i in range(actor_steps):
        e=make_episode(batch,seed*1000000+i,'base')
        purchased=rand_buy(batch,seed*1000000+i+987001)
        # Exploration executes random actions and learns using observed feedback.
        executed=rand_buy(batch,seed*1000000+i+1010001)
        correctness,label,_=feedback(e,executed,purchased)
        logits=net.decision_logits(e,purchased)
        loss=F.binary_cross_entropy_with_logits(logits,label)
        optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
        optimizer.step();last_actor=float(loss.detach())
    actor_seconds=time.perf_counter()-clock
    # Freeze the learned action predictor. Value learning must be contextual
    # and conditional only on the arm randomized and actually observed.
    for p in net.parameters():p.requires_grad_(False)
    for p in net.value_output_parameters():p.requires_grad_(True)
    optimizer=torch.optim.AdamW(net.value_output_parameters(),lr=.002,weight_decay=0.)
    start=time.perf_counter()
    last_critic=None
    for i in range(value_steps):
        e=make_episode(batch,seed*1000000+350000+i,'base')
        bought=rand_buy(batch,seed*1000000+350000+i+987001)
        with torch.no_grad():
            predicted_positive=net.decision_logits(e,bought)>=0
            success,_,_=feedback(e,predicted_positive,bought)
        values=net.values(e)  # logit predicted success if skip vs buy
        chosen=values.gather(1,bought.long()[:,None]).squeeze(1)
        # Chosen arm ONLY, never unpurchased outcome or private oracle truth.
        loss=F.binary_cross_entropy_with_logits(chosen,success)
        optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(net.value_output_parameters(),1.)
        optimizer.step();last_critic=float(loss.detach())
    critic_seconds=time.perf_counter()-start
    for p in net.parameters():p.requires_grad_(False)
    return net, dict(actor_seconds=actor_seconds, critic_seconds=critic_seconds,
                     actor_loss=last_actor, critic_loss=last_critic)

@torch.no_grad()
def evaluate(net,e,ablate_calibration=False):
    if ablate_calibration:
        e={**e,'x_cal':torch.zeros_like(e['x_cal']),'y_cal':torch.zeros_like(e['y_cal'])}
    net.eval()
    q=net.values(e).sigmoid()
    estimated_gain=q[:,1]-q[:,0]
    purchase=estimated_gain>e['c']
    positive=net.decision_logits(e,purchase)>=0
    success,_,net_reward=feedback(e,positive,purchase)
    return dict(reward=float(net_reward.mean()),accuracy=float(success.mean()),
                purchase_rate=float(purchase.float().mean()),
                value_gain_mean=float(estimated_gain.mean()),
                value_gain_sd=float(estimated_gain.std()),
                q_skip_mean=float(q[:,0].mean()),q_buy_mean=float(q[:,1].mean()))

@torch.no_grad()
def randomized_value_calibration(net,e,seed):
    buy=rand_buy(len(e['z']),seed)
    logits=net.decision_logits(e,buy)
    true_success,_,_=feedback(e,logits>=0,buy)
    q=net.values(e).sigmoid().gather(1,buy.long()[:,None]).squeeze(1)
    return float((q-true_success).square().mean())


def run(seed,actor_steps=240,value_steps=260,batch=128,n_eval=4096):
    torch.set_num_threads(2)
    verified=verify_original()
    episodes={f:make_episode(n_eval,seed*1000000+88000+j*127,f) for j,f in enumerate(FAMILIES)}
    results=dict(seed=seed,source_blob=verified,actor_steps=actor_steps,value_steps=value_steps,
                 batch=batch,n_eval=n_eval,architectures={},baselines={})
    for f,e in episodes.items():
        results['baselines'][f]={k:assess_baseline(e,k) for k in ('bayes','no_cal_bayes','never_probe','always_probe')}
    for kind in ('hybrid','transformer','gru'):
        net,seconds=train_one(seed,kind,actor_steps,value_steps,batch)
        results['architectures'][kind]={
            'parameters':sum(p.numel() for p in net.parameters()),
            'timing':seconds,
            'families':{f:evaluate(net,e) for f,e in episodes.items()},
            'base_no_calibration':evaluate(net,episodes['base'],True),
            'randomized_value_brier':randomized_value_calibration(net,episodes['base'],seed+90335),
        }
    return results

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--seed',type=int,default=102000)
    p.add_argument('--actor-steps',type=int,default=240)
    p.add_argument('--value-steps',type=int,default=260)
    p.add_argument('--batch',type=int,default=128)
    p.add_argument('--n-eval',type=int,default=4096)
    a=p.parse_args()
    print(json.dumps(run(a.seed,a.actor_steps,a.value_steps,a.batch,a.n_eval),indent=2))
