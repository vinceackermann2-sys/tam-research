"""Original SI architecture v8: learn episode-specific hidden sensor reliability.

Exact canonical OriginalSIHybrid and TransformerControl, identical input
encoding, output usage, training batches/steps; analytic Bayesian and simple
GRU controls. Hidden mode and true query label are never in model INPUTS.
Binary-sensor system is a limited toy POMDP, not a full SI architecture test.
"""
from __future__ import annotations
import argparse
import hashlib
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

BASE_BLOB = '575c17dc2f39a29d930d82179aa0c96dcbdc7d0f'
N_CAL = 6


def verify_source():
    source = Path(__file__).resolve().parent.parent/'tam_research'/'si_original_hybrid_v0.py'
    actual = subprocess.check_output(['git', 'hash-object', str(source)], text=True).strip()
    if actual != BASE_BLOB:
        raise RuntimeError(f'Original model changed: {actual} != {BASE_BLOB}')
    return actual


def make_episode(n:int, seed:int, family:str='base'):
    """Independent calibration truths, fixed hidden sensor polarity per episode."""
    assert family in ('base','inverted_prior','noisy_reference','weak_calibration')
    g=torch.Generator().manual_seed(seed)
    R=lambda *dims:torch.rand(*dims,generator=g)
    Z=lambda *dims:torch.randn(*dims,generator=g)
    inv_prob=0.8 if family=='inverted_prior' else 0.5
    z=torch.where(R(n)<inv_prob, -1., 1.)
    calibration_truth=torch.where(R(n,N_CAL)<.5,-1.,1.)
    truth=torch.where(R(n)<.5,-1.,1.)
    mu0=.35+1.05*R(n)
    mu_ref=(.6+.55*R(n)) if family=='noisy_reference' else (1.55+.2*R(n))
    mu_p=.9+1.25*R(n)
    c=.03+.27*R(n)
    # Weak calibration has less initial evidence but query sensor quality unchanged.
    calibration_mu=mu0*(.38 if family=='weak_calibration' else 1.)
    x_cal=z[:,None]*calibration_truth*calibration_mu[:,None]+Z(n,N_CAL)
    y_cal=calibration_truth*mu_ref[:,None]+Z(n,N_CAL)
    x_query=z*truth*mu0+Z(n)
    y_probe=truth*mu_p+Z(n)
    return dict(z=z,truth=truth,cal_truth=calibration_truth,mu0=mu0,mu_ref=mu_ref,
                cal_mu=calibration_mu,mu_p=mu_p,c=c,x_cal=x_cal,y_cal=y_cal,
                x_query=x_query,y_probe=y_probe)


def posterior_z(ep, use_history=True):
    """Exact marginal calibration likelihood, integrating unknown calibration truths."""
    if not use_history:
        return torch.full_like(ep['z'],.5)
    x,y=ep['x_cal'],ep['y_cal']
    m,r=ep['cal_mu'][:,None],ep['mu_ref'][:,None]
    def likelihood(s):
        lp=-.5*((x-s*m).square()+(y-r).square())
        lm=-.5*((x+s*m).square()+(y+r).square())
        return torch.logaddexp(lp,lm).sum(1)
    logits=likelihood(1.)-likelihood(-1.)
    # Correct prior for the training simulator; shifted-prior evaluation is OOD.
    return torch.sigmoid(logits)


def bayes_policy(ep, use_history=True):
    """Exact Bayes within assumed 50/50 sensor-inversion prior."""
    pz=posterior_z(ep,use_history)
    x,m=ep['x_query'],ep['mu0']
    log_z=torch.stack((torch.log(pz.clamp_min(1e-9)),torch.log((1-pz).clamp_min(1e-9))),-1)
    plus=torch.logsumexp(log_z+torch.stack((-.5*(x-m).square(),-.5*(x+m).square()),-1),-1)
    minus=torch.logsumexp(log_z+torch.stack((-.5*(x+m).square(),-.5*(x-m).square()),-1),-1)
    odds=plus-minus
    p=torch.sigmoid(odds)
    mup=ep['mu_p']
    # y_probe is N(+mup,1) for truth=+; negative mean for truth=-.
    threshold=-odds/(2*mup)
    normal_cdf=lambda v:.5*(1+torch.erf(v/(2**.5)))
    benefit=p*normal_cdf(mup-threshold)+(1-p)*normal_cdf(mup+threshold)-torch.maximum(p,1-p)
    buy=benefit>ep['c']
    updated=odds+buy.float()*2*mup*ep['y_probe']
    choice=torch.where(updated>=0,1.,-1.)
    return dict(buy=buy,choice=choice,pz=pz,ptheta=p,benefit=benefit)


def tokens(ep, buy):
    """Causal 7D token protocol. No hidden truth/sensor polarity in inputs."""
    b=len(ep['z']); z=torch.zeros(b, N_CAL)
    # Calibration tokens: raw x/y, known sensor strengths, calibration flag.
    calibration=torch.stack((ep['x_cal'],ep['y_cal'],ep['cal_mu'][:,None].expand(-1,N_CAL),
         ep['mu_ref'][:,None].expand(-1,N_CAL),z,z,torch.ones_like(z)), -1)
    zeros=torch.zeros_like(ep['z'])
    query=torch.stack((ep['x_query'],zeros,ep['mu0'],zeros,ep['mu_p'],ep['c'],zeros),-1)[:,None]
    probe_token=torch.stack((ep['x_query'],ep['y_probe']*buy.float(),ep['mu0'],zeros,
                        ep['mu_p'],ep['c'],buy.float()),-1)[:,None]
    return torch.cat((calibration,query),1),torch.cat((calibration,query,probe_token),1)


def to_original_inputs(tok):
    # Preserve canonical model's obs_dim=3, action_dim=4 and attention structure.
    return tok[...,:3],tok[...,3:]


class BackboneController(nn.Module):
    def __init__(self,arch):
        super().__init__()
        cls=OriginalSIHybrid if arch=='hybrid' else TransformerControl
        self.base=cls(SIConfig())
    def forward(self,ep,buy):
        stage0,stage1=tokens(ep,buy)
        o0,a0=to_original_inputs(stage0)
        o1,a1=to_original_inputs(stage1)
        logits0=self.base(o0,a0)['prediction'][:,-1]
        logits1=self.base(o1,a1)['prediction'][:,-1]
        return logits0[:,0],logits1[:,1],logits0[:,2]


class GRUController(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Linear(7,40)
        self.gru=nn.GRU(40,40,batch_first=True)
        self.head=nn.Linear(40,3)
    def forward(self,ep,buy):
        a,b=tokens(ep,buy)
        h0,_=self.gru(torch.tanh(self.encoder(a)))
        h1,_=self.gru(torch.tanh(self.encoder(b)))
        f0=self.head(h0[:,-1]);f1=self.head(h1[:,-1])
        return f0[:,0],f1[:,1],f0[:,2]


def train(seed:int, arch:str, steps:int, batch:int):
    torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
    model=GRUController() if arch=='gru' else BackboneController(arch)
    optim=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    start=time.monotonic()
    for j in range(steps):
        e=make_episode(batch,seed*1000000+j,'base')
        with torch.no_grad():
            teacher=bayes_policy(e)
            # Deliberately give both models the same synthetic labels: supervised
            # offline imitation of Bayes purchases, mode labels, and true sensor polarity.
            desired_buy=teacher['buy'].float()
            p=torch.rand(batch,generator=torch.Generator().manual_seed(seed*1000000+j+987001))<.5
        model.train();optim.zero_grad(set_to_none=True)
        probe,action,mode=model(e,p)
        l=(F.binary_cross_entropy_with_logits(probe,desired_buy)
            +F.binary_cross_entropy_with_logits(action,(e['truth']>0).float())
            +.25*F.binary_cross_entropy_with_logits(mode,(e['z']>0).float()))
        if not torch.isfinite(l):raise RuntimeError('nonfinite train loss')
        l.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optim.step()
    return model,time.monotonic()-start


def reward(ep,buy,choice):
    return (choice==ep['truth']).float()-buy.float()*ep['c']

@torch.no_grad()
def evaluate(model, ep):
    model.eval()
    zeros=torch.zeros(len(ep['z']),dtype=torch.bool)
    probe,_,belief=model(ep,zeros)
    buys=probe>0
    _,act,_=model(ep,buys)
    choice=torch.where(act>=0,1.,-1.)
    p=torch.sigmoid(belief)
    return dict(reward=float(reward(ep,buys,choice).mean()),correct=float((choice==ep['truth']).float().mean()),
         probed=float(buys.float().mean()),belief_brier=float(((p-(ep['z']>0).float())**2).mean()),
         belief_accuracy=float(((p>=.5)==(ep['z']>0)).float().mean()))


def assess_baseline(ep, mode):
    optimal=bayes_policy(ep,use_history=(mode!='no_cal_bayes'))
    if mode=='never_probe' or mode=='always_probe':
        buy=torch.zeros_like(optimal['buy']) if mode=='never_probe' else torch.ones_like(optimal['buy'])
        # Use SAME inferred prior, force different information action.
        ptheta=optimal['ptheta'].clamp(1e-9,1-1e-9)
        updated=torch.logit(ptheta)+buy.float()*2*ep['mu_p']*ep['y_probe']
        chosen=torch.where(updated>=0,1.,-1.)
    else:
        buy,chosen=optimal['buy'],optimal['choice']
    return dict(reward=float(reward(ep,buy,chosen).mean()),correct=float((chosen==ep['truth']).float().mean()),
                probed=float(buy.float().mean()),belief_brier=float(((optimal['pz']-(ep['z']>0).float())**2).mean()),
                belief_accuracy=float(((optimal['pz']>=.5)==(ep['z']>0)).float().mean()))


def run(seed:int, steps:int=220, batch:int=64, n_eval:int=8192):
    torch.set_num_threads(2)
    source=verify_source()
    families=('base','inverted_prior','noisy_reference','weak_calibration')
    result=dict(seed=seed,source_blob=source,steps=steps,batch=batch,n_eval=n_eval,architectures={})
    evaluation={f:make_episode(n_eval,seed*1000000+80000+k*127,f) for k,f in enumerate(families)}
    result['baselines']={f:{name:assess_baseline(e,name) for name in ('bayes','no_cal_bayes','never_probe','always_probe')}
                         for f,e in evaluation.items()}
    for arch in ('hybrid','transformer','gru'):
        m,seconds=train(seed,arch,steps,batch)
        result['architectures'][arch]={
            'params':sum(p.numel() for p in m.parameters()),'training_cpu_seconds':seconds,
            'families':{f:evaluate(m,e) for f,e in evaluation.items()},
            'no_calibration_base':evaluate(m,{**evaluation['base'], 'x_cal':torch.zeros_like(evaluation['base']['x_cal']),
                                       'y_cal':torch.zeros_like(evaluation['base']['y_cal'])})}
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,default=100100)
    parser.add_argument('--steps',type=int,default=220)
    parser.add_argument('--batch',type=int,default=64)
    parser.add_argument('--n-eval',type=int,default=8192)
    args=parser.parse_args()
    print(json.dumps(run(args.seed,args.steps,args.batch,args.n_eval),indent=2))
