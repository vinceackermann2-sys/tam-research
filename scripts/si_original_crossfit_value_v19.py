"""SI original v19: independent-arm cross-fitted double-Q with logged feedback.

This is a controlled CPU ablation of v18's same-critic max bootstrap,
NOT independent world-model discovery. Training has no Bayesian decision labels.
Each Q1 critic fits only its own disjoint subset of actually logged arm outcomes.
The cross-fit Q0 target selects a second action using critic A but VALUES that
selected action using critic B. No future unpurchased outcome enters training.
"""
from __future__ import annotations
import argparse
import json
import time
import torch
from torch import nn
import torch.nn.functional as F
from scripts.si_original_active_experiment_v17 import world, subset, score_policy, FAMILIES
from scripts import si_original_logged_value_v18 as v18

EXTENDED_FAMILIES=(*FAMILIES,'strong_discount_ood')


def evaluation_world(n_pairs, seed, family):
    if family != 'strong_discount_ood':
        return world(n_pairs, seed, family)
    e=world(n_pairs,seed,'base')
    # Public prices only: shift the expensive high sensor to a lower fee
    # and the weak sensor to a higher fee, both outside training support.
    e['cost_high']=(e['cost_high']-.14).clamp_min(.015)
    e['cost_low']=e['cost_low']+.04
    return e


class CrossfitDoubleQ(nn.Module):
    def __init__(self, actor: v18.LoggedController, seed: int):
        super().__init__()
        self.actor=actor
        # Construct the additional head only AFTER backbone/actor training.
        # Thus backbone initializations and full-model actor gradient steps are
        # byte-identical to the existing frozen v18 baseline at the same seed.
        with torch.random.fork_rng():
            torch.manual_seed(seed + 733)
            self.q1_b=nn.Linear(3,2)

    def features(self,e,first=None):
        return self.actor.features(e,first)

    def decision(self,e,first,second=None):
        return self.actor.decision(e,first,second)

    def q0(self,e):
        return self.actor.q0(e)

    def critic_pair(self,e,first):
        x=self.features(e,first)
        return self.actor.q1_head(x),self.q1_b(x)

    def q1(self,e,first):
        a,b=self.critic_pair(e,first)
        return (a+b)*.5


def crossfit_selected_losses(q_a,q_b,second,observed):
    """Independent disjoint logged trajectories, with no unobserved-arm target."""
    n=len(second)
    left=torch.arange(n,device=second.device)%2==0
    right=~left
    loss=q_a.sum()*0 + q_b.sum()*0
    if left.any():
        loss=loss+F.mse_loss(q_a[left].gather(1,second[left,None].long()).squeeze(1),observed[left])
    if right.any():
        loss=loss+F.mse_loss(q_b[right].gather(1,second[right,None].long()).squeeze(1),observed[right])
    return loss


def crossfit_bootstrap(q_a,q_b,first_price):
    """Action select from critic A, evaluate from critic B (Double-Q)."""
    with torch.no_grad():
        selected=q_a.argmax(-1)
        return q_b.gather(1,selected[:,None]).squeeze(1)-first_price


def train_one(seed,arch,actor_steps=160,critic_steps=240,batch=96):
    torch.set_num_threads(2)
    # Reuses frozen v18 actor training verbatim; zero critic updates.
    actor,timing=v18.train_one(seed,arch,actor_steps=actor_steps,critic_steps=0,batch=batch)
    net=CrossfitDoubleQ(actor,seed)
    for p in net.parameters():p.requires_grad_(False)
    for p in net.actor.q0_head.parameters():p.requires_grad_(True)
    for p in net.actor.q1_head.parameters():p.requires_grad_(True)
    for p in net.q1_b.parameters():p.requires_grad_(True)
    trained=list(net.actor.q0_head.parameters())+list(net.actor.q1_head.parameters())+list(net.q1_b.parameters())
    opt=torch.optim.AdamW(trained,lr=.003,weight_decay=0)
    t0=time.perf_counter()
    for i in range(critic_steps):
        e=world(batch//2,seed*1000000+310000+i,'base')
        first=v18.random_first(batch,seed*1000000+310000+i+991111)
        q0=net.q0(e)
        active=first!=0
        loss=q0.sum()*0
        if (~active).any():
            no=subset(e,~active)
            executed=v18.random_boolean(len(no['mu_low']),seed*1000000+310000+i+441111)
            correctness=v18.logged_feedback(no,executed).float()
            loss=loss+F.mse_loss(q0[~active,0],correctness)
        if active.any():
            se=subset(e,active); f=first[active];other=3-f;n=len(f)
            second=v18.random_boolean(n,seed*1000000+310000+i+551111)
            with torch.no_grad():
                chosen=torch.zeros(n,dtype=torch.bool)
                if (~second).any():
                    chosen[~second]=net.decision(subset(se,~second),f[~second])>=0
                if second.any():
                    chosen[second]=net.decision(subset(se,second),f[second],other[second])>=0
                correct=v18.logged_feedback(se,chosen).float()
                p2=torch.where(f==1,se['cost_high'],se['cost_low'])
                logged=correct-second.float()*p2
            q_a,q_b=net.critic_pair(se,f)
            loss=loss+crossfit_selected_losses(q_a,q_b,second,logged)
            p1=torch.where(f==1,se['cost_low'],se['cost_high'])
            target=crossfit_bootstrap(q_a,q_b,p1)
            loss=loss+F.mse_loss(q0[active].gather(1,f[:,None]).squeeze(1),target)
        if not torch.isfinite(loss):raise RuntimeError('nonfinite crossfit loss')
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trained,1.)
        opt.step()
    for p in net.parameters():p.requires_grad_(False)
    timing['critic_cpu_seconds']=time.perf_counter()-t0
    return net,timing


@torch.no_grad()
def logged_arm_calibration(model,e,seed):
    """MSE on a randomized *observed* second-stage arm, held out of training."""
    n=len(e['mu_low'])
    first=torch.where(v18.random_boolean(n,seed+100),1,2).long()
    second=v18.random_boolean(n,seed+200)
    other=3-first
    choice=torch.zeros(n,dtype=torch.bool)
    if (~second).any():
        choice[~second]=model.decision(subset(e,~second),first[~second])>=0
    if second.any():
        choice[second]=model.decision(subset(e,second),first[second],other[second])>=0
    feedback=v18.logged_feedback(e,choice).float()
    cost=torch.where(first==1,e['cost_high'],e['cost_low'])
    observed=feedback-second.float()*cost
    q=model.q1(e,first).gather(1,second.long()[:,None]).squeeze(1)
    return float((q-observed).square().mean())


def run(seed,actor_steps=160,critic_steps=240,batch=96,n_pairs=512):
    torch.set_num_threads(2)
    blob=v18.verify_source()
    envs={f:evaluation_world(n_pairs,seed*1000000+81000+131*i,f)
          for i,f in enumerate(EXTENDED_FAMILIES)}
    output={'seed':seed,'source_blob':blob,'actor_steps':actor_steps,'critic_steps':critic_steps,
            'batch':batch,'n_pairs':n_pairs,'controls':{},'architectures':{}}
    for f,e in envs.items():
        output['controls'][f]={p:score_policy(e,p) for p in
                            ('never','always_low','always_high','always_both','greedy','bayes')}
    for arch in ('hybrid','transformer','gru'):
        base,bt=v18.train_one(seed,arch,actor_steps,critic_steps,batch)
        double,dt=train_one(seed,arch,actor_steps,critic_steps,batch)
        output['architectures'][arch]={}
        for name,net,times in (('v18_same_q',base,bt),('v19_crossfit_double_q',double,dt)):
            output['architectures'][arch][name]={
                'params':sum(p.numel() for p in net.parameters()),
                'timing':times,
                'families':{f:{**v18.score_logged(e,net),
                               'randomized_q1_mse':logged_arm_calibration(net,e,seed+191+i*137)}
                            for i,(f,e) in enumerate(envs.items())}}
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--seed',type=int,default=111000)
    p.add_argument('--actor-steps',type=int,default=160)
    p.add_argument('--critic-steps',type=int,default=240)
    p.add_argument('--batch',type=int,default=96)
    p.add_argument('--n-pairs',type=int,default=512)
    a=p.parse_args()
    print(json.dumps(run(a.seed,a.actor_steps,a.critic_steps,a.batch,a.n_pairs),indent=2))
