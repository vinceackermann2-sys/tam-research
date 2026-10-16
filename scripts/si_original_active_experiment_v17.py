"""Original SI v17: two-stage active information acquisition, CPU-only.

A paired, observationally indistinguishable hidden binary query may be probed
sequentially with a cheap-noisy sensor and an expensive-accurate sensor.
An exact known-model Bayes dynamic-programming teacher is contrasted against
matched original hybrid, Transformer and smaller GRU trained to imitate it.
No claim of latent unsupervised discovery: stage0/stage1 labels are oracle policy.
No unpurchased probe signal is used in a policy decision.
"""
from __future__ import annotations
import argparse, json, math, subprocess, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from scripts.si_original_identifiability_v16 import identical_opposite_worlds, assert_twin_certificate
from tam_research.si_original_hybrid_v0 import SIConfig, OriginalSIHybrid, TransformerControl

FAMILIES=('base','high_cost_ood','weak_sensors_ood')
PARAM_HASH='575c17dc2f39a29d930d82179aa0c96dcbdc7d0f'


def verify_original():
    filename=Path(__file__).resolve().parent.parent/'tam_research'/'si_original_hybrid_v0.py'
    h=subprocess.check_output(['git','hash-object',str(filename)],text=True).strip()
    if h!=PARAM_HASH:raise RuntimeError('Canonical original SI hash mismatch: '+h)
    return h


def world(n_pairs,seed,family='base'):
    # Generate counterfactual balanced worlds with identical pre-probe data.
    A,B=identical_opposite_worlds(n_pairs,seed)
    assert_twin_certificate(A,B)
    n=2*n_pairs
    # Seed independent from the v16 Gaussian noise, with probes generated once
    # at environment creation, but ONLY observed after corresponding purchases.
    g=torch.Generator().manual_seed(int(seed)+130019)
    low_mu=.45+.55*torch.rand(n_pairs,generator=g)
    high_mu=1.10+1.05*torch.rand(n_pairs,generator=g)
    low_cost=.025+.115*torch.rand(n_pairs,generator=g)
    high_cost=.13+.20*torch.rand(n_pairs,generator=g)
    if family=='high_cost_ood':
        low_cost=low_cost+.065; high_cost=high_cost+.14
    elif family=='weak_sensors_ood':
        low_mu=.25+.35*torch.rand(n_pairs,generator=g)
        high_mu=.65+.60*torch.rand(n_pairs,generator=g)
    elif family!='base':raise ValueError(family)
    # Shared noise across *paired* worlds, ensuring exactly matched preprobe
    # observations, while true query signs and observed probe means reverse.
    eps0=torch.randn(n_pairs,generator=g)
    eps1=torch.randn(n_pairs,generator=g)
    def catparam(x):return torch.cat([x,x])
    answer=torch.cat([A['truth'],B['truth']])
    e={'latent_answer':answer,'observed_calibration_x':torch.cat([A['x_cal'],B['x_cal']]),
       'observed_calibration_y':torch.cat([A['y_cal'],B['y_cal']]),
       'observed_query':torch.cat([A['x_query'],B['x_query']]),
       'mu_low':catparam(low_mu),'mu_high':catparam(high_mu),
       'cost_low':catparam(low_cost),'cost_high':catparam(high_cost),
       'probe_low':answer*catparam(low_mu)+catparam(eps0),
       'probe_high':answer*catparam(high_mu)+catparam(eps1)}
    return e


def normal_cdf(v):return .5*(1+torch.erf(v/math.sqrt(2.)))


def second_values(log_odds, remaining_mu, remaining_cost):
    """Exact conditional optimal second decision; no future probe observed."""
    p=torch.sigmoid(log_odds)
    threshold=log_odds/(2*remaining_mu)
    after=p*normal_cdf(remaining_mu+threshold)+(1-p)*normal_cdf(remaining_mu-threshold)
    stop=torch.maximum(p,1-p)
    benefit=after-stop-remaining_cost
    return torch.where(benefit>0, after-remaining_cost,stop),benefit>0,benefit


def expected_first_value(first_mu,first_cost,other_mu,other_cost,nodes=24):
    """Gauss-Hermite evaluate E_{theta, first observation}[stage2 optimal value]."""
    pts,weights=np.polynomial.hermite.hermgauss(nodes)
    nodes_t=torch.as_tensor(pts*math.sqrt(2),dtype=first_mu.dtype)
    weights_t=torch.as_tensor(weights/math.sqrt(math.pi),dtype=first_mu.dtype)
    b=len(first_mu)
    # Probability theta=+ and theta=- are 0.5 at stage0.
    up=first_mu[:,None]+nodes_t[None,:]
    down=-first_mu[:,None]+nodes_t[None,:]
    plus_v=second_values(2*first_mu[:,None]*up,other_mu[:,None],other_cost[:,None])[0]
    minus_v=second_values(2*first_mu[:,None]*down,other_mu[:,None],other_cost[:,None])[0]
    return .5*((plus_v+minus_v)*weights_t[None,:]).sum(1)-first_cost


def teacher_first(e,nodes=24):
    q0=torch.full_like(e['mu_low'],.5)
    q1=expected_first_value(e['mu_low'],e['cost_low'],e['mu_high'],e['cost_high'],nodes)
    q2=expected_first_value(e['mu_high'],e['cost_high'],e['mu_low'],e['cost_low'],nodes)
    quality=torch.stack([q0,q1,q2],dim=-1)
    return quality.argmax(-1),quality


def second_purchase(e,first_action):
    """Given actual purchased first reading, should we buy remaining probe?"""
    assert ((first_action==1)|(first_action==2)).all()
    first_mu=torch.where(first_action==1,e['mu_low'],e['mu_high'])
    first_read=torch.where(first_action==1,e['probe_low'],e['probe_high'])
    other_mu=torch.where(first_action==1,e['mu_high'],e['mu_low'])
    other_cost=torch.where(first_action==1,e['cost_high'],e['cost_low'])
    log_odds=2*first_mu*first_read
    _,buy,_=second_values(log_odds,other_mu,other_cost)
    return buy, log_odds


def token_data(e,first=None,second=None):
    """Build seven-dimensional same-input original hybrid/Transformer/GRU tokens.

    Token 0..5 are observed historical calibration (uninformative about latent
    twin identity); query has two costs/strengths. Purchased readings appear
    ONLY as appended tokens, with phase/which-sensor code.
    """
    n=len(e['mu_low']);zero=torch.zeros(n,6)
    calibration=torch.stack([e['observed_calibration_x'],e['observed_calibration_y'],
       zero,zero,zero,zero,torch.ones_like(zero)],dim=-1)
    u=torch.zeros(n)
    query=torch.stack([e['observed_query'],u,u,e['mu_low'],e['cost_low'],e['mu_high'],e['cost_high']],dim=-1)[:,None]
    out=[calibration,query]
    for chosen in (first,second):
        if chosen is None:continue
        if not ((chosen==1)|(chosen==2)).all():raise ValueError('Do not append an unpurchased probe')
        observed=torch.where(chosen==1,e['probe_low'],e['probe_high'])
        token=torch.stack([observed,chosen.float(),torch.ones_like(u),e['mu_low'],e['cost_low'],e['mu_high'],e['cost_high']],dim=-1)[:,None]
        out.append(token)
    return torch.cat(out,dim=1)


class SequentialController(nn.Module):
    def __init__(self,arch):
        super().__init__()
        self.arch=arch
        if arch=='gru':
            self.enc=nn.Linear(7,40)
            self.gru=nn.GRU(40,40,batch_first=True)
            self.head=nn.Linear(40,3)
        else:
            self.backbone=(OriginalSIHybrid if arch=='hybrid' else TransformerControl)(SIConfig())
    def forward(self,tokens):
        if self.arch=='gru':
            x,_=self.gru(torch.tanh(self.enc(tokens)))
            return self.head(x[:,-1])
        return self.backbone(tokens[...,:3],tokens[...,3:])['prediction'][:,-1]


def subset(e,mask):return {k:v[mask] for k,v in e.items()}


def train_one(seed,arch,steps=240,batch=96):
    torch.manual_seed(seed)
    model=SequentialController(arch)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    torch.set_num_threads(2)
    start=time.perf_counter()
    for step in range(steps):
        e=world(batch//2,seed*1000000+step,'base')
        with torch.no_grad():
            optimal_first,_=teacher_first(e)
            # Actual logged first probe selected independently of teacher to
            # ensure both possible sensors observed for stage2 supervision.
            g=torch.Generator().manual_seed(seed*1000000+step+902443)
            random_first=torch.where(torch.rand(batch,generator=g)<.5,1,2).long()
            optimal_second,_=second_purchase(e,random_first)
            # For final answers the simulator executes a random binary action;
            # action+correctness yields the valid terminal answer label.
            executed=torch.rand(batch,generator=g)<.5
            correctness=(executed==(e['latent_answer']>0))
            observed_action_label=torch.where(correctness,executed,~executed).float()
        initial=model(token_data(e))
        stage1=model(token_data(e,random_first))
        loss=F.cross_entropy(initial,optimal_first)+F.binary_cross_entropy_with_logits(stage1[:,0],optimal_second.float())
        loss=loss+.6*F.binary_cross_entropy_with_logits(stage1[:,1],observed_action_label)
        # Also observe/execute second probe in randomized logged two-probe trials.
        # Never use its value in first/second acquisition policy.
        remaining=torch.where(random_first==1,2,1)
        stage2=model(token_data(e,random_first,remaining))
        loss=loss+.3*F.binary_cross_entropy_with_logits(stage2[:,1],observed_action_label)
        optimizer.zero_grad(set_to_none=True)
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
    return model,time.perf_counter()-start


@torch.no_grad()
def score_policy(e,policy,model=None):
    """Simulate actual staged purchasing. All policies use only purchased values."""
    n=len(e['mu_low']); buy0=torch.zeros(n,dtype=torch.long)
    if policy=='bayes':buy0,_=teacher_first(e)
    elif policy=='greedy':
        # Optimizes ONE future probe, ignoring option to adaptively buy other.
        c0=torch.full_like(e['mu_low'],.5)
        c1=normal_cdf(e['mu_low'])-e['cost_low']
        c2=normal_cdf(e['mu_high'])-e['cost_high']
        buy0=torch.stack([c0,c1,c2],-1).argmax(-1)
    elif policy=='always_low':buy0.fill_(1)
    elif policy=='always_high':buy0.fill_(2)
    elif policy=='always_both':buy0.fill_(1)
    elif policy=='never':pass
    elif policy=='neural':
        if model is None:raise ValueError('neural model required')
        buy0=model(token_data(e)).argmax(-1)
    else:raise ValueError(policy)
    active=(buy0!=0)
    buy_second=torch.zeros(n,dtype=torch.bool)
    guessed=torch.ones_like(e['latent_answer'])
    # Stage1 only active on purchased first-probe episodes.
    if active.any():
        idx=active.nonzero().flatten();m=subset(e,active);first=buy0[active]
        if policy=='bayes' or policy=='greedy':
            s,l=second_purchase(m,first)
            buy_second[active]=s
        elif policy=='always_both':buy_second[active]=True
        elif policy=='neural':
            # Final decision after ONLY actually purchased observations.
            logits=model(token_data(m,first))
            s=logits[:,0]>0;buy_second[active]=s
        first_mu=torch.where(first==1,m['mu_low'],m['mu_high'])
        first_read=torch.where(first==1,m['probe_low'],m['probe_high'])
        logged=2*first_mu*first_read
        remaining=torch.where(first==1,2,1)
        if policy=='neural':
            no_second=~s
            decisions=torch.ones_like(m['latent_answer'])
            if no_second.any():
                decisions[no_second]=torch.where(model(token_data(subset(m,no_second),first[no_second]))[:,1]>=0,1.,-1.)
            if s.any():
                decisions[s]=torch.where(model(token_data(subset(m,s),first[s],remaining[s]))[:,1]>=0,1.,-1.)
            guessed[active]=decisions
        else:
            mu_other=torch.where(first==1,m['mu_high'],m['mu_low'])
            value_other=torch.where(first==1,m['probe_high'],m['probe_low'])
            posterior=logged+buy_second[active].float()*2*mu_other*value_other
            guessed[active]=torch.where(posterior>=0,1.,-1.)
    if policy=='neural':
        skipped=~active
        if skipped.any():
            guessed[skipped]=torch.where(model(token_data(subset(e,skipped)))[:,1]>=0,1.,-1.)
    first_cost=torch.where(buy0==1,e['cost_low'],torch.where(buy0==2,e['cost_high'],0.))
    other_cost=torch.where(buy0==1,e['cost_high'],e['cost_low'])
    cost=first_cost+buy_second.float()*other_cost
    acc=(guessed==e['latent_answer']).float()
    return dict(reward=float((acc-cost).mean()),accuracy=float(acc.mean()),
                avg_probe_count=float(active.float().mean()+buy_second.float().mean()),
                first_buy=float(active.float().mean()),second_buy=float(buy_second.float().mean()),
                first_low=float((buy0==1).float().mean()),first_high=float((buy0==2).float().mean()),
                mean_cost=float(cost.mean()))


def run(seed,steps=240,batch=96,n_pairs=512):
    torch.set_num_threads(2)
    attested=verify_original()
    r={'seed':seed,'original_blob':attested,'steps':steps,'batch':batch,'n_pairs':n_pairs,
       'families':{},'models':{}}
    worlds={f:world(n_pairs,seed*1000000+80000+i*131,f) for i,f in enumerate(FAMILIES)}
    for f,w in worlds.items():
        r['families'][f]={p:score_policy(w,p) for p in
             ('never','always_low','always_high','always_both','greedy','bayes')}
        r['families'][f]['first_stage_value']=dict(zip(['stop','low_then_option','high_then_option'],
              [float(x.mean()) for x in teacher_first(w)[1].unbind(-1)]))
    for arch in ('hybrid','transformer','gru'):
        model,seconds=train_one(seed,arch,steps,batch)
        r['models'][arch]={'parameters':sum(p.numel() for p in model.parameters()),
                 'cpu_training_seconds':seconds,
                 'families':{f:score_policy(w,'neural',model) for f,w in worlds.items()}}
    return r


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--seed',type=int,default=109000)
    p.add_argument('--steps',type=int,default=240)
    p.add_argument('--batch',type=int,default=96)
    p.add_argument('--n-pairs',type=int,default=512)
    a=p.parse_args()
    print(json.dumps(run(a.seed,a.steps,a.batch,a.n_pairs),indent=2))
