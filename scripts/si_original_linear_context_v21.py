"""V21 CPU-only simple contextual ridge benchmark, factual logged outcomes.

The terminal action uses the KNOWN Gaussian sensor likelihood ratio, so this
comparator has an explicit privileged structural prior. The *purchase* values
are fit only from actual randomized purchase and correctness feedback; NO
analytical optimal purchase labels are used. Costs are subtracted exactly.
"""
from __future__ import annotations
import torch, time
from scripts.si_original_active_experiment_v17 import subset
from scripts.si_original_crossfit_value_v19 import evaluation_world
from scripts.si_original_logged_value_v18 import random_first, random_boolean, logged_feedback


def q1_features(e,first):
    q=torch.where(first==1,e['mu_low'],e['mu_high'])
    other=torch.where(first==1,e['mu_high'],e['mu_low'])
    obs=torch.where(first==1,e['probe_low'],e['probe_high'])
    strength=(q*obs).abs().clamp(max=6)
    n=len(strength);ones=torch.ones(n)
    return torch.stack((ones,strength,strength.square(),q,other,other.square(),strength*other,q*other),-1)


def q0_features(e):
    low=e['mu_low'];high=e['mu_high'];ones=torch.ones_like(low)
    return torch.stack((ones,low,high,low.square(),high.square(),low*high),-1)


def fit_ridge(X,y,penalty=10.0):
    # Use true observed labels, equal propensity by arm in logged data.
    x=X.double();target=y.double()
    reg=torch.eye(x.size(1),dtype=torch.float64)*penalty
    reg[0,0]=1e-5
    return torch.linalg.solve(x.T@x+reg,x.T@target).float()


class ContextualRidge:
    def __init__(self):
        self.q1_weights=None;self.q0_weights=None

    def action(self,e,first=None,second=None):
        if first is None:return torch.ones(len(e['mu_low']),dtype=torch.bool)
        evidence=torch.where(first==1,e['mu_low']*e['probe_low'],e['mu_high']*e['probe_high'])
        if second is not None:
            other=torch.where(first==1,e['mu_high']*e['probe_high'],e['mu_low']*e['probe_low'])
            evidence=evidence+other
        return evidence>=0

    def q1(self,e,first):
        X=q1_features(e,first)
        success=torch.stack([X@w for w in self.q1_weights],-1).clamp(0,1)
        other_cost=torch.where(first==1,e['cost_high'],e['cost_low'])
        return success-torch.stack([torch.zeros_like(other_cost),other_cost],-1)

    def q0(self,e):
        X=q0_features(e)
        success=torch.stack([X@w for w in self.q0_weights],-1)
        return success-torch.stack((torch.zeros_like(e['cost_low']),e['cost_low'],e['cost_high']),-1)


def train(seed,n_train=6000):
    start=time.perf_counter()
    model=ContextualRidge()
    first_data=evaluation_world(n_train//2,seed*1000000+900000,'base')
    n=len(first_data['mu_low'])
    f=torch.where(random_boolean(n,seed+1198),1,2).long()
    second=random_boolean(n,seed+2358)
    # Separately calculate actually executed actions on logged purchased arms.
    action=torch.zeros(n,dtype=torch.bool)
    if (~second).any():action[~second]=model.action(subset(first_data,~second),f[~second])
    if second.any():action[second]=model.action(subset(first_data,second),f[second],3-f[second])
    correct=logged_feedback(first_data,action).float()
    X=q1_features(first_data,f)
    model.q1_weights=[fit_ridge(X[~second],correct[~second]),fit_ridge(X[second],correct[second])]

    second_data=evaluation_world(n_train//2,seed*1000000+910000,'base')
    n=len(second_data['mu_low']);f=random_first(n,seed+3198)
    random_second=random_boolean(n,seed+4358)
    factual=torch.zeros(n);valid=f==0
    # Log stop reward from executed constant action (+1), average exactly 0.5.
    if valid.any(): factual[valid]=logged_feedback(subset(second_data,valid),torch.ones(int(valid.sum()),dtype=torch.bool)).float()
    active=f!=0
    if active.any():
        m=subset(second_data,active);fs=f[active];ss=random_second[active]
        deployed=model.q1(m,fs).argmax(-1).bool()
        valid[active]=ss==deployed
        action=torch.zeros(len(fs),dtype=torch.bool)
        if (~ss).any():action[~ss]=model.action(subset(m,~ss),fs[~ss])
        if ss.any():action[ss]=model.action(subset(m,ss),fs[ss],3-fs[ss])
        correct=logged_feedback(m,action).float()
        cost_first=torch.where(fs==1,m['cost_low'],m['cost_high'])
        cost_second=torch.where(fs==1,m['cost_high'],m['cost_low'])
        # first Q fit predicts net value AFTER deducting first fee at decision.
        factual[active]=correct-ss.float()*cost_second
    X=q0_features(second_data)
    model.q0_weights=[fit_ridge(X[(f==i)&valid],factual[(f==i)&valid],penalty=4) for i in range(3)]
    return model,{'seconds':time.perf_counter()-start,'q1_logged_buy':int(second.sum()),'q0_accepted':int(valid.sum()),'q0_total':n}


@torch.no_grad()
def score(model,e):
    first=model.q0(e).argmax(-1)
    active=first!=0
    buy_next=torch.zeros(len(first),dtype=torch.bool)
    answer=torch.ones(len(first),dtype=torch.bool)
    if active.any():
        s=subset(e,active);f=first[active]
        next_buy=model.q1(s,f).argmax(-1).bool()
        buy_next[active]=next_buy
        action=torch.zeros(len(f),dtype=torch.bool)
        if (~next_buy).any():action[~next_buy]=model.action(subset(s,~next_buy),f[~next_buy])
        if next_buy.any():action[next_buy]=model.action(subset(s,next_buy),f[next_buy],3-f[next_buy])
        answer[active]=action
    correct=logged_feedback(e,answer).float()
    price1=torch.where(first==1,e['cost_low'],torch.where(first==2,e['cost_high'],0.))
    price2=torch.where(first==1,e['cost_high'],e['cost_low'])*buy_next.float()
    return {'reward':float((correct-price1-price2).mean()),'first_low':float((first==1).float().mean()),'first_high':float((first==2).float().mean()),'second_buy':float(buy_next.float().mean()),'mean_cost':float((price1+price2).mean())}
