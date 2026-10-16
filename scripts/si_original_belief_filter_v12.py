"""Causal change-hypothesis belief as shared input to original SI/Transformer/GRU.

CPU-only toy observation-model test. Model-only inferred posterior is based on
known sensor likelihoods; it is NOT an autonomously learned world law.
Causal neural training uses randomized *observed* buy/skip and action feedback.
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
import torch.nn.functional as F
from scripts.si_original_belief_transfer_v8 import make_episode, tokens, to_original_inputs, BASE_BLOB
from scripts.si_original_paired_drift_v11 import drift_episode, policy_given_polarity_posterior
from scripts.si_original_contextual_value_v10 import ContextualController, feedback, rand_buy

FAMILIES = ('base', 'one_flip', 'early_flip', 'double_flip', 'weak_calibration')
# Three internal candidate change times and a no-change hypothesis.
PATTERNS = ((), (2,), (3,), (4,))


def verify_original():
    target = Path(__file__).resolve().parent.parent/'tam_research/si_original_hybrid_v0.py'
    blob = subprocess.check_output(['git','hash-object',str(target)],text=True).strip()
    if blob != BASE_BLOB: raise RuntimeError('Full original source mismatch')
    return blob


def episode(n,seed,family):
    if family == 'base': return make_episode(n,seed,'base')
    if family == 'weak_calibration': return make_episode(n,seed,'weak_calibration')
    if family == 'one_flip': return drift_episode(n,seed)
    e = make_episode(n,seed,'base')
    if family == 'early_flip':
        g=torch.Generator().manual_seed(seed+3321)
        t=torch.where(torch.rand(n,generator=g)<.5,1,5)
        mask=torch.arange(6)[None]<t[:,None]
        e['private_flip_at']=t
    elif family == 'double_flip':
        # Opposite polarity only in the middle of the episode; two changes.
        mask=(torch.arange(6)[None]>=2)&(torch.arange(6)[None]<5)
        e['private_flip_at']=torch.full((n,),-2)
    else: raise ValueError(family)
    e['x_cal']=torch.where(mask,-e['x_cal'],e['x_cal'])
    return e


def hypothesis_log_joint(e,patterns=PATTERNS,change_mass=.5):
    """log P(calibration cues, final polarity, pattern), marginalizing each theta_i.

    Only noisy observed (x_i,y_i), known sensor strengths are used. Never
    reads e['z'], e['truth'], e['private_flip_at'], or e['y_probe'].
    """
    x,y=e['x_cal'],e['y_cal']
    m,r=e['cal_mu'][:,None],e['mu_ref'][:,None]
    hp=torch.tensor([1-change_mass] + [change_mass/(len(patterns)-1)]*(len(patterns)-1),
                    dtype=x.dtype,device=x.device)
    j=[]
    for z in (1.,-1.):
        for i,changes in enumerate(patterns):
            flip=torch.zeros(6,dtype=torch.bool,device=x.device)
            for cp in changes: flip ^= torch.arange(6,device=x.device)<cp
            signed=torch.where(flip[None],-z*m,z*m)
            lp=-.5*((x-signed)**2+(y-r)**2)
            lm=-.5*((x+signed)**2+(y+r)**2)
            j.append(torch.logaddexp(lp,lm).sum(-1) + (hp[i]/2).log())
    return torch.stack(j,-1).reshape(len(x),2,len(patterns))


def belief(e):
    j=hypothesis_log_joint(e)
    weights=(j-torch.logsumexp(j.flatten(1),dim=-1)[:,None,None]).exp()
    return weights[:,0,:].sum(1),weights[:,:,1:].sum((1,2))


def encoded_tokens(e,purchased,with_belief):
    before,after=tokens(e,purchased)
    if not with_belief: return before,after
    pz,p_change=belief(e)
    # Same 7-channel representation and same model parameter count.
    # Query-time feature slots 3,6 are zero in the v8 protocol; calibration
    # tokens remain unchanged. Bounded centered posterior features.
    zfeat=(pz-.5)*2
    hfeat=p_change
    before=before.clone();after=after.clone()
    before[:,-1,3]=zfeat;before[:,-1,6]=hfeat
    after[:,-2,3]=zfeat;after[:,-2,6]=hfeat
    after[:,-1,3]=zfeat;after[:,-1,6]=hfeat
    return before,after


class Controller(ContextualController):
    def __init__(self,arch,with_belief):
        super().__init__(arch)
        self.with_belief=with_belief

    def values(self,e):
        before,_=encoded_tokens(e,torch.zeros(len(e['z']),dtype=torch.bool),self.with_belief)
        return self.forward_raw(before)[:,-1,1:3]

    def decision_logits(self,e,bought):
        _,after=encoded_tokens(e,bought,self.with_belief)
        return self.forward_raw(after)[:,-1,0]


def train_one(seed,arch,with_belief,actor_steps=180,critic_steps=200,batch=96):
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    model=Controller(arch,with_belief)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    start=time.perf_counter()
    for i in range(actor_steps):
        family='base' if i%2==0 else 'one_flip'
        e=episode(batch,seed*1000000+i,family)
        buy=rand_buy(batch,seed*1000000+i+987001)
        action=rand_buy(batch,seed*1000000+i+1010001)
        _,label,_=feedback(e,action,buy)
        loss=F.binary_cross_entropy_with_logits(model.decision_logits(e,buy),label)
        opt.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        opt.step()
    tactor=time.perf_counter()-start
    for p in model.parameters():p.requires_grad_(False)
    for p in model.value_output_parameters():p.requires_grad_(True)
    critic=torch.optim.AdamW(model.value_output_parameters(),lr=.002,weight_decay=0.)
    start=time.perf_counter()
    for i in range(critic_steps):
        family='one_flip' if i%2==0 else 'base'
        e=episode(batch,seed*1000000+350000+i,family)
        buy=rand_buy(batch,seed*1000000+350000+i+987001)
        with torch.no_grad():
            action=model.decision_logits(e,buy)>=0
            success,_,_=feedback(e,action,buy)
        value=model.values(e).gather(1,buy.long()[:,None]).squeeze(1)
        loss=F.binary_cross_entropy_with_logits(value,success)
        critic.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.value_output_parameters(),1.)
        critic.step()
    tcritic=time.perf_counter()-start
    return model,dict(actor_cpu_seconds=tactor,critic_cpu_seconds=tcritic)

@torch.no_grad()
def evaluate(model,e):
    model.eval()
    q=model.values(e).sigmoid()
    purchased=(q[:,1]-q[:,0])>e['c']
    action=model.decision_logits(e,purchased)>=0
    success,_,reward=feedback(e,action,purchased)
    logbuy=rand_buy(len(e['z']),int(len(e['z'])*4711+29))
    calibration_success,_,_=feedback(e,model.decision_logits(e,logbuy)>=0,logbuy)
    logq=q.gather(1,logbuy.long()[:,None]).squeeze(1)
    return dict(reward=float(reward.mean()),accuracy=float(success.mean()),
                probe_fraction=float(purchased.float().mean()),
                logged_arm_brier=float((logq-calibration_success).square().mean()),
                mean_estimated_gain=float((q[:,1]-q[:,0]).mean()))

@torch.no_grad()
def bayes_reference(e,filter_type):
    if filter_type=='mixture':
        posterior=belief(e)[0]
    elif filter_type=='stationary':
        posterior=hypothesis_log_joint(e,patterns=((),),change_mass=0.)  # not used
    else: raise ValueError(filter_type)
    buy,action,_=policy_given_polarity_posterior(e,posterior)
    success=(action==e['truth'].gt(0)).float()
    return dict(reward=float((success-buy.float()*e['c']).mean()),
                accuracy=float(success.mean()),probe_fraction=float(buy.float().mean()))


def run(seed,actor_steps=180,critic_steps=200,batch=96,n_eval=2048):
    torch.set_num_threads(2)
    source=verify_original()
    worlds={f:episode(n_eval,seed*1000000+83000+131*j,f) for j,f in enumerate(FAMILIES)}
    result=dict(seed=seed,source_blob=source,actor_steps=actor_steps,critic_steps=critic_steps,
                n_eval=n_eval,batch=batch,models={},references={})
    for name,e in worlds.items():
        result['references'][name]={'mixture_filter':bayes_reference(e,'mixture')}
        from scripts.si_original_belief_transfer_v8 import assess_baseline
        result['references'][name]['stationary_bayes']=assess_baseline(e,'bayes')
        result['references'][name]['stationary_always']=assess_baseline(e,'always_probe')
        result['references'][name]['stationary_never']=assess_baseline(e,'never_probe')
    for arch in ('hybrid','transformer','gru'):
        for add in (False,True):
            model,times=train_one(seed,arch,add,actor_steps,critic_steps,batch)
            result['models'][f'{arch}_{"belief" if add else "raw"}']={
                'params':sum(p.numel() for p in model.parameters()),
                'times':times,
                'families':{name:evaluate(model,e) for name,e in worlds.items()}}
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--seed',type=int,default=104000)
    p.add_argument('--actor-steps',type=int,default=180)
    p.add_argument('--critic-steps',type=int,default=200)
    p.add_argument('--batch',type=int,default=96)
    p.add_argument('--n-eval',type=int,default=2048)
    a=p.parse_args()
    print(json.dumps(run(a.seed,a.actor_steps,a.critic_steps,a.batch,a.n_eval),indent=2))
