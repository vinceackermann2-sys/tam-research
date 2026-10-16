"""V18 original SI: learn sequential value-of-information from logged feedback.

No Bayesian purchase-action labels in training. Randomized logging supplies actual
purchased measurements; an executed binary choice plus correctness reconstructs
the outcome label for final-action prediction. Q1 learns observed success less
purchased second cost; Q0 learns stop from executed feedback and purchased-first
options via frozen-target Q1 bootstrap. No unobserved probe values in actions.
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
from scripts.si_original_active_experiment_v17 import (world, token_data, subset,
     score_policy, FAMILIES, SequentialController, train_one as teacher_train)

CANONICAL_BLOB='575c17dc2f39a29d930d82179aa0c96dcbdc7d0f'


def verify_source():
    p=Path(__file__).resolve().parent.parent/'tam_research'/'si_original_hybrid_v0.py'
    digest=subprocess.check_output(['git','hash-object',str(p)],text=True).strip()
    if digest!=CANONICAL_BLOB:raise RuntimeError('Original SI model mismatch '+digest)
    return digest


class LoggedController(nn.Module):
    """Identical tiny task heads attached to each canonical frozen-size backbone."""
    def __init__(self,arch):
        super().__init__()
        self.backbone=SequentialController(arch)
        self.final_head=nn.Linear(3,1)
        self.q1_head=nn.Linear(3,2)
        self.q0_head=nn.Linear(3,3)

    def features(self,e,first=None,second=None):
        return self.backbone(token_data(e,first,second))

    def decision(self,e,first,second=None):
        return self.final_head(self.features(e,first,second)).squeeze(1)

    def q1(self,e,first):
        return self.q1_head(self.features(e,first))

    def q0(self,e):
        return self.q0_head(self.features(e))


def logged_feedback(e,executed):
    """Environment-facing interface exposes correctness, not its private answer.

    In the simulator private truth is read ONLY to generate the feedback; the
    optimizer consumes 'correct', and an executed action can reconstruct the
    correct binary target after observing feedback.
    """
    truth=e['latent_answer']>0
    return executed==truth


def decoded_label(executed,correct):
    return torch.where(correct,executed,~executed).float()


def random_boolean(n,seed):
    return torch.rand(n,generator=torch.Generator().manual_seed(int(seed)))<.5


def random_first(n,seed):
    # Uniform 0=stop, 1=cheap, 2=strong; independent of private answer.
    return torch.randint(0,3,(n,),generator=torch.Generator().manual_seed(int(seed)))


def train_one(seed,arch,actor_steps=160,critic_steps=240,batch=96):
    torch.set_num_threads(2)
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    model=LoggedController(arch)
    opt=torch.optim.AdamW(list(model.backbone.parameters())+list(model.final_head.parameters()),lr=.001,weight_decay=.01)
    t0=time.perf_counter()
    for i in range(actor_steps):
        e=world(batch//2,seed*1000000+i,'base')
        first=torch.where(random_boolean(batch,seed*1000000+i+991111),1,2).long()
        remaining=3-first
        # Two logged interventions, both probes deliberately purchased for
        # classifier learning. Only actual displayed readings reach model.
        # A random executed terminal action yields correctness feedback.
        executed=random_boolean(batch,seed*1000000+i+881111)
        correct=logged_feedback(e,executed)
        learned_target=decoded_label(executed,correct)
        logit_first=model.decision(e,first)
        logit_both=model.decision(e,first,remaining)
        loss=F.binary_cross_entropy_with_logits(logit_first,learned_target)
        loss=loss+.7*F.binary_cross_entropy_with_logits(logit_both,learned_target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.backbone.parameters())+list(model.final_head.parameters()),1.)
        opt.step()
    actor_seconds=time.perf_counter()-t0
    # Freeze action prediction; learn only two Q heads from logged action rewards.
    for p in model.backbone.parameters():p.requires_grad_(False)
    for p in model.final_head.parameters():p.requires_grad_(False)
    opt=torch.optim.AdamW(list(model.q0_head.parameters())+list(model.q1_head.parameters()),lr=.003,weight_decay=0.)
    t1=time.perf_counter()
    for i in range(critic_steps):
        e=world(batch//2,seed*1000000+310000+i,'base')
        first=random_first(batch,seed*1000000+310000+i+991111)
        chosen_q0=model.q0(e)
        first_active=first!=0
        total_loss=chosen_q0.sum()*0
        if (~first_active).any():
            no=subset(e,~first_active)
            executed=random_boolean(len(no['mu_low']),seed*1000000+310000+i+441111)
            correctness=logged_feedback(no,executed).float()
            # observed stop-arm reward, not analytic known-law 0.5
            q0_stop=chosen_q0[~first_active,0]
            total_loss=total_loss+F.mse_loss(q0_stop,correctness)
        if first_active.any():
            selected=subset(e,first_active)
            bought=first[first_active]
            opposite=3-bought
            n=len(bought)
            second=random_boolean(n,seed*1000000+310000+i+551111)
            # Logged actual chosen second-stage action. No evaluation of the
            # unchosen terminal outcome as Q-supervision.
            with torch.no_grad():
                chosen=torch.zeros(n,dtype=torch.bool)
                if (~second).any():
                    chosen[~second]=model.decision(subset(selected,~second),bought[~second])>=0
                if second.any():
                    chosen[second]=model.decision(subset(selected,second),bought[second],opposite[second])>=0
                correct=logged_feedback(selected,chosen).float()
                second_cost=torch.where(bought==1,selected['cost_high'],selected['cost_low'])
                observed_value=correct-second.float()*second_cost
            q1=model.q1(selected,bought)
            fitted_arm=q1.gather(1,second.long()[:,None]).squeeze(1)
            total_loss=total_loss+F.mse_loss(fitted_arm,observed_value)
            # Fitted Q backward induction: Q0(first) is estimated using the
            # best learned Q1 conditional on the ACTUALLY bought first signal.
            # Q1 target is detached, not a Bayes action label or unbought value.
            with torch.no_grad():
                first_cost=torch.where(bought==1,selected['cost_low'],selected['cost_high'])
                bootstrap=q1.detach().max(1).values-first_cost
            q0_first=chosen_q0[first_active].gather(1,bought[:,None]).squeeze(1)
            total_loss=total_loss+F.mse_loss(q0_first,bootstrap)
        if not torch.isfinite(total_loss):raise RuntimeError('nonfinite critic')
        opt.zero_grad(set_to_none=True);total_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.q0_head.parameters())+list(model.q1_head.parameters()),1.)
        opt.step()
    critic_seconds=time.perf_counter()-t1
    for p in model.parameters():p.requires_grad_(False)
    return model,{'actor_cpu_seconds':actor_seconds,'critic_cpu_seconds':critic_seconds}


@torch.no_grad()
def score_logged(e,model):
    model.eval()
    n=len(e['mu_low'])
    first=model.q0(e).argmax(1)
    active=first!=0
    second=torch.zeros(n,dtype=torch.bool)
    guesses=torch.ones_like(e['latent_answer'])
    if active.any():
        chosen=subset(e,active); f=first[active]; remaining=3-f
        q1=model.q1(chosen,f)
        buy_next=q1[:,1]>q1[:,0]
        second[active]=buy_next
        decisions=torch.ones(len(f),dtype=torch.bool)
        if (~buy_next).any():
            decisions[~buy_next]=model.decision(subset(chosen,~buy_next),f[~buy_next])>=0
        if buy_next.any():
            decisions[buy_next]=model.decision(subset(chosen,buy_next),f[buy_next],remaining[buy_next])>=0
        guesses[active]=torch.where(decisions,1.,-1.)
    # no-probe action defaults +1; twin prior guarantees paired 50%.
    cost_first=torch.where(first==1,e['cost_low'],torch.where(first==2,e['cost_high'],0.))
    cost_second=torch.where(first==1,e['cost_high'],e['cost_low'])*second.float()
    correct=guesses==e['latent_answer']
    return {'reward':float((correct.float()-cost_first-cost_second).mean()),
            'accuracy':float(correct.float().mean()),
            'first_low':float((first==1).float().mean()),'first_high':float((first==2).float().mean()),
            'stop':float((first==0).float().mean()),'second_buy':float(second.float().mean()),
            'mean_cost':float((cost_first+cost_second).mean()),
            'mean_q0_stop':float(model.q0(e)[:,0].mean())}


def run(seed,actor_steps=160,critic_steps=240,batch=96,n_pairs=512):
    torch.set_num_threads(2)
    blob=verify_source()
    worlds={f:world(n_pairs,seed*1000000+81000+131*j,f) for j,f in enumerate(FAMILIES)}
    result={'seed':seed,'original_model_blob':blob,'actor_steps':actor_steps,
            'critic_steps':critic_steps,'batch':batch,'n_pairs':n_pairs,
            'references':{f:{p:score_policy(e,p) for p in ('never','always_low','always_high','always_both','greedy','bayes')}
                         for f,e in worlds.items()},'models':{}}
    for arch in ('hybrid','transformer','gru'):
        model,secs=train_one(seed,arch,actor_steps,critic_steps,batch)
        result['models'][arch]={'parameters':sum(p.numel() for p in model.parameters()),
            'timing':secs,'families':{f:score_logged(e,model) for f,e in worlds.items()}}
    # Matched training/evaluation-seed *oracle supervised* v17 positive control.
    result['teacher_models']={}
    for arch in ('hybrid','transformer','gru'):
        oracle,elapsed=teacher_train(seed,arch,240,batch)
        result['teacher_models'][arch]={'parameters':sum(p.numel() for p in oracle.parameters()),
            'cpu_training_seconds':elapsed,
            'families':{fam:score_policy(e,'neural',oracle) for fam,e in worlds.items()}}
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--seed',type=int,default=110000)
    p.add_argument('--actor-steps',type=int,default=160)
    p.add_argument('--critic-steps',type=int,default=240)
    p.add_argument('--batch',type=int,default=96)
    p.add_argument('--n-pairs',type=int,default=512)
    a=p.parse_args()
    print(json.dumps(run(a.seed,a.actor_steps,a.critic_steps,a.batch,a.n_pairs),indent=2))
